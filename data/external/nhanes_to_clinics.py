"""NHANES -> V1-schema clinics (`data/clinics_nhanes*`).

Maps continuous NHANES cycles 2011–2018 (G/H/I/J — raw XPTs in `data/external/nhanes/`,
provenance in `data/external/SOURCES.md`) onto the frozen V1 synthetic schema
(`FEATURE_COLS` + `ckd_stage3plus`), then partitions participants into pseudo-practices so the
EXISTING pipeline (`--clinics-dir`, `load_clinic_frames`, private sweeps composed in notebook 04)
runs on real data untouched.

Why NHANES carries the load V1 cannot: real laboratories (serum creatinine -> eGFR, HbA1c, ACR),
real missingness (MNAR — labs/exams only in the mobile-exam subsample), realistic correlated
comorbidity, and — unlike V1 — `sex` + `race_eth`, which unblocks the T2.5 fairness-by-sex audit
(CLAUDE.md live issue 5, blocked on data until now).

Variable mapping (NHANES codebooks, 2011-2018):

  age_years               <- DEMO.RIDAGEYR, restricted to 18..95 (V1 clip range)
  dx_hypertonie           <- BPQ.BPQ020 == 1 (told high BP) OR measured exam mean SBP >= 140 /
                             mean DBP >= 90 (BPX readings averaged over available attempts)
  dx_diabetes             <- DIQ.DIQ010 == 1 (told diabetes) OR GHB.LBXGH >= 6.5%
  dx_khk                  <- MCQ.MCQ160C == 1 (coronary heart disease)
  dx_adipositas           <- BMX.BMXBMI >= 30
  dx_herzinsuffizienz     <- MCQ.MCQ160B == 1 (congestive heart failure)
  dx_hyperurikaemie       <- MCQ.MCQ160N == 1 (gout) OR BIOPRO.LBXSUA >= 7 mg/dL
  years_since_diabetes_dx <- age - DID040 (age when told) when 0 <= DID040 <= age, else 0
  years_since_hypertonie_dx <- age - BPQ030 likewise; years_since_khk_dx = 0 (NHANES asks no
                             CAD onset age) — all per V1 §3a: 0 = not documented.
  label ckd_stage3plus    <- eGFR < 60 mL/min/1.73m² (KDIGO stage >= 3, PREVALENCE — matches the
                             V1 label semantics, not the canonical incidence label)

eGFR: CKD-EPI 2021 creatinine equation, race-free (Inker et al., NEJM 2021):
  eGFR = 142 * min(Scr/k,1)^a * max(Scr/k,1)^-1.200 * 0.9938^Age * (1.012 if female)
  female: k=0.7, a=-0.241; male: k=0.9, a=-0.302.

Questionnaire special values (7/9, 77/99, 77777/99999 = refused/don't-know) map to structural
zero — the V1 rule — and are counted in the meta file's `coding_loss` audit. Pregnant
participants (RIDEXPRG == 1) are excluded (eGFR/BMI distortion). Participants without a serum
creatinine are dropped (label uncomputable) — that drop IS the NHANES MNAR structure.

Practices = cycle x sex, 8 clinics, n ~= 2.5-3k each (`data/clinics_nhanes/`).
`data/clinics_nhanes_s/` subsamples each clinic to a V1-sized band draw n ~ U[150,600] for
DP-run comparability with the V1/V2 federations. Clinic CSVs carry three extra columns
(`sex_female`, `race_eth`, `cycle`) — invisible to `to_xy` (it selects FEATURE_COLS) but required
by the fairness analysis; the full enriched participant table lands in
`data/external/nhanes/participants_mapped.csv`.

KFRE track (`data/clinics_nhanes_kfre/`): the same inclusion frame and clinic key, but the
clinic CSVs carry exactly the 8 raw Tangri-KFRE inputs + the same label — age, male, eGFR, ACR
(= 100 * ALB_CR.URXUMA / URXUCR, mg/g), albumin (BIOPRO.LBXSAL), phosphorus (LBXSPH),
bicarbonate (LBXSC3SI; mmol/L = mEq/L), calcium (LBXSCA) — so the published rule (`kfre.py`)
can be scored against the federated logreg on identical rows. Undrawn labs stay NaN (the loader
median-imputes + flags them for the logreg; the rule scores complete cases only). Provenance
table: `data/external/nhanes/participants_kfre.csv`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..loader import FEATURE_COLS, KFRE_FEATURE_COLS, LABEL_COL

HERE = Path(__file__).resolve().parent
NHANES_RAW = HERE / "nhanes"
CYCLES: dict[str, int] = {"G": 2011, "H": 2013, "I": 2015, "J": 2017}
FILES = ("DEMO", "MCQ", "DIQ", "BPQ", "BPX", "BMX", "BIOPRO", "ALB_CR", "GHB")


def egfr_ckd_epi_2021(scr_mgdl: np.ndarray, age: np.ndarray, female: np.ndarray) -> np.ndarray:
    """Race-free CKD-EPI 2021 creatinine eGFR, vectorised."""
    scr = np.asarray(scr_mgdl, float)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    ratio = scr / k
    return (142 * np.power(np.minimum(ratio, 1.0), a) * np.power(np.maximum(ratio, 1.0), -1.200)
            * np.power(0.9938, age) * np.where(female, 1.012, 1.0))


def _clean_code(s: pd.Series, yes_value: float = 1.0) -> tuple[pd.Series, int]:
    """NHANES questionnaire item -> structural-zero 0/1; returns the series and #invalid rows."""
    invalid = int(s.isna().sum() + s.isin([7.0, 9.0, 77.0, 99.0, 777.0, 999.0]).sum())
    return (s == yes_value).astype(int), invalid


def _duration_years(age: pd.Series, onset_age: pd.Series) -> pd.Series:
    """age - onset_age where plausible (0 <= onset <= age), else 0.0 (V1 §3a: not documented)."""
    yrs = age - onset_age
    ok = onset_age.notna() & (onset_age >= 0) & (onset_age <= age)
    ok &= ~onset_age.isin([77777.0, 99999.0, 777.0, 999.0])
    return np.where(ok, np.clip(yrs, 0, None), 0.0)


def load_cycle(cycle: str, raw_dir: Path = NHANES_RAW) -> dict[str, pd.DataFrame]:
    return {f: pd.read_sas(raw_dir / f"{f}_{cycle}.xpt") for f in FILES}


def build_participants(raw_dir: Path = NHANES_RAW) -> pd.DataFrame:
    """One row per usable NHANES participant, all cycles stacked, every mapped column + aux labs."""
    rows = []
    coding = []
    for cyc, year in CYCLES.items():
        t = load_cycle(cyc, raw_dir)
        demo, diq, bpq, bpx = t["DEMO"], t["DIQ"], t["BPQ"], t["BPX"]
        bmx, biopro, ghb, mcq = t["BMX"], t["BIOPRO"], t["GHB"], t["MCQ"]

        df = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]].copy()
        if "RIDEXPRG" in demo:
            df["_pregnant"] = demo["RIDEXPRG"] == 1
        else:
            df["_pregnant"] = False
        for f, cols in [
            (diq, ["SEQN", "DIQ010", "DID040"]), (bpq, ["SEQN", "BPQ020", "BPQ030"]),
            (bmx, ["SEQN", "BMXBMI"]),
            (biopro, ["SEQN", "LBXSCR", "LBXSUA"]),
            (ghb, ["SEQN", "LBXGH"]),
            (mcq, ["SEQN", "MCQ160B", "MCQ160C", "MCQ160N"]),
        ]:
            have = [c for c in cols if c == "SEQN" or c in f.columns]
            df = df.merge(f[have], on="SEQN", how="left")

        sbp_cols = [c for c in bpx.columns if c.startswith("BPXSY")]
        dbp_cols = [c for c in bpx.columns if c.startswith("BPXDI")]
        sbp = bpx[["SEQN", *sbp_cols]].copy()
        dbp = bpx[["SEQN", *dbp_cols]].copy()
        # NHANES BP values are physical readings; NaN = not measured.
        df["_sbp_mean"] = sbp.set_index("SEQN").mean(axis=1).reindex(df["SEQN"]).to_numpy()
        df["_dbp_mean"] = dbp.set_index("SEQN").mean(axis=1).reindex(df["SEQN"]).to_numpy()

        n0 = len(df)
        df = df[(df.RIDAGEYR >= 18) & (df.RIDAGEYR <= 95) & (~df["_pregnant"])]
        n_label = int(df["LBXSCR"].notna().sum())
        df = df[df["LBXSCR"].notna()].copy()

        age = df["RIDAGEYR"].astype(float).clip(18, 95)
        female = df["RIAGENDR"] == 2
        egfr = egfr_ckd_epi_2021(df["LBXSCR"].astype(float).to_numpy(),
                                 age.to_numpy(), female.to_numpy())

        htn_q, i_htn = _clean_code(df.get("BPQ020", pd.Series(np.nan, index=df.index)))
        dm_q, i_dm = _clean_code(df.get("DIQ010", pd.Series(np.nan, index=df.index)))
        chf, i_chf = _clean_code(df.get("MCQ160B", pd.Series(np.nan, index=df.index)))
        chd, i_chd = _clean_code(df.get("MCQ160C", pd.Series(np.nan, index=df.index)))
        gout, i_gout = _clean_code(df.get("MCQ160N", pd.Series(np.nan, index=df.index)))

        dx_htn = ((htn_q == 1) | (df["_sbp_mean"] >= 140) | (df["_dbp_mean"] >= 90)).astype(int)
        dx_dm = ((dm_q == 1) | (df.get("LBXGH", pd.Series(np.nan, index=df.index)) >= 6.5)).astype(int)
        bmi = df.get("BMXBMI", pd.Series(np.nan, index=df.index)).astype(float)

        out = pd.DataFrame({
            "age_years": age.to_numpy(),
            "dx_hypertonie": dx_htn.to_numpy(),
            "dx_diabetes": dx_dm.to_numpy(),
            "dx_khk": chd.to_numpy(),
            "dx_adipositas": (bmi >= 30).astype(int).to_numpy(),
            "dx_herzinsuffizienz": chf.to_numpy(),
            "dx_hyperurikaemie": ((gout == 1) | (df.get("LBXSUA", pd.Series(np.nan, index=df.index)) >= 7.0)).astype(int).to_numpy(),
            "years_since_hypertonie_dx": _duration_years(age, df.get("BPQ030", pd.Series(np.nan, index=df.index)).astype(float)).astype(float),
            "years_since_diabetes_dx": _duration_years(age, df.get("DID040", pd.Series(np.nan, index=df.index)).astype(float)).astype(float),
            "years_since_khk_dx": 0.0,
            LABEL_COL: (egfr < 60).astype(int),
            "sex_female": female.astype(int).to_numpy(),
            "race_eth": df["RIDRETH3"].astype("Int64").to_numpy(),
            "cycle": year,
            "egfr": np.round(egfr, 1),
            "hba1c": df.get("LBXGH", pd.Series(np.nan, index=df.index)).astype(float).to_numpy(),
            "bmi": np.round(bmi, 1),
        })
        rows.append(out)
        coding.append({"cycle": year, "participants_in": n0, "with_creatinine": n_label,
                       "used": len(out),
                       "invalid_codes": {"htn": i_htn, "dm": i_dm, "chf": i_chf,
                                         "chd": i_chd, "gout": i_gout}})

    participants = pd.concat(rows, ignore_index=True)
    participants.attrs["coding_loss"] = coding
    return participants


def assign_clinics(participants: pd.DataFrame) -> pd.Series:
    """Pseudo-practice key: '<cycle>-female|male' (cycle x sex -> 8 clinics)."""
    sex = np.where(participants["sex_female"] == 1, "female", "male")
    return pd.Series([f"{c}-{s}" for c, s in zip(participants["cycle"], sex)],
                     index=participants.index, name="clinic")


def write_clinics(
    out_full: Path | None = None, out_small: Path | None = None, *,
    raw_dir: Path = NHANES_RAW, small_lo: int = 150, small_hi: int = 600, seed: int = 42,
) -> dict:
    """Write both NHANES federations (full-size and V1-subsampled) + provenance meta."""
    out_full = out_full or HERE.parent / "clinics_nhanes"
    out_small = out_small or HERE.parent / "clinics_nhanes_s"
    participants = build_participants(raw_dir)
    participants.to_csv(raw_dir / "participants_mapped.csv", index=False)
    participants["clinic"] = assign_clinics(participants)

    keep = [*FEATURE_COLS, LABEL_COL, "sex_female", "race_eth", "cycle"]
    rng = np.random.default_rng(seed)
    meta = {"source": "NHANES 2011-2018 (Q/G/H/I cycles)",
            "label": "eGFR<60 (KDIGO stage>=3 prevalence; CKD-EPI 2021 race-free)",
            "n_participants": len(participants),
            "prevalence": round(float(participants[LABEL_COL].mean()), 4),
            "coding_loss": participants.attrs["coding_loss"], "clinics": []}

    for out_dir, size_fn in [(out_full, None),
                             (out_small, lambda n: int(rng.integers(small_lo, small_hi + 1)))]:
        out_dir.mkdir(parents=True, exist_ok=True)
        combined = []
        for k, (key, grp) in enumerate(participants.groupby("clinic", sort=True)):
            g = grp if size_fn is None else grp.sample(n=min(size_fn(len(grp)), len(grp)),
                                                       random_state=seed + k)
            fname = f"clinic_{k:02d}_{key}.csv"
            g[keep].to_csv(out_dir / fname, index=False)
            combined.append(g[keep].assign(clinic_id=k, clinic_name=key))
            meta["clinics"].append({"clinic_id": k, "name": key, "n": len(g),
                                    "ckd_rate": round(float(g[LABEL_COL].mean()), 4),
                                    "dir": out_dir.name})
        pd.concat(combined, ignore_index=True).to_csv(out_dir / "all_clinics.csv", index=False)

    (out_full / "suite_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def build_kfre_participants(raw_dir: Path = NHANES_RAW) -> pd.DataFrame:
    """One row per NHANES participant carrying the 8 Tangri-KFRE inputs (rule-baseline track).

    Same inclusion frame as `build_participants` (18-95, non-pregnant, creatinine observed —
    the label is the same CKD-EPI-2021 eGFR<60). The KFRE variables come straight from the
    local NHANES files, in the equation's units:

    - age (DEMO `RIDAGEYR`), male (DEMO `RIAGENDR` == 1)
    - eGFR (BIOPRO `LBXSCR` -> CKD-EPI 2021, same `egfr_ckd_epi_2021` as the V1 mapping)
    - acr   = 100 * `URXUMA` / `URXUCR`  (urine albumin mg/L over creatinine: mg/dL -> g/L; mg/g)
    - albumin (BIOPRO `LBXSAL`, g/dL), phosphorus (`LBXSPH`, mg/dL),
      bicarbonate (`LBXSC3SI`, mmol/L = mEq/L), calcium (`LBXSCA`, mg/dL)

    Labs that were never drawn stay NaN — the loader median-imputes + flags them for the logreg
    comparators, and kfre.py scores complete cases only.
    """
    rows = []
    coding = []
    for cyc, year in CYCLES.items():
        t = load_cycle(cyc, raw_dir)
        demo, biopro, albcr = t["DEMO"], t["BIOPRO"], t["ALB_CR"]

        df = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
        df["_pregnant"] = demo["RIDEXPRG"] == 1 if "RIDEXPRG" in demo else False
        for f, cols in [
            (biopro, ["SEQN", "LBXSCR", "LBXSAL", "LBXSC3SI", "LBXSCA", "LBXSPH"]),
            (albcr, ["SEQN", "URXUMA", "URXUCR"]),
        ]:
            have = [c for c in cols if c == "SEQN" or c in f.columns]
            df = df.merge(f[have], on="SEQN", how="left")

        n0 = len(df)
        df = df[(df.RIDAGEYR >= 18) & (df.RIDAGEYR <= 95) & (~df["_pregnant"])]
        df = df[df["LBXSCR"].notna()].copy()

        age = df["RIDAGEYR"].astype(float).clip(18, 95)
        male = df["RIAGENDR"] == 1
        egfr = egfr_ckd_epi_2021(df["LBXSCR"].astype(float).to_numpy(),
                                 age.to_numpy(), (~male).to_numpy())
        nan = pd.Series(np.nan, index=df.index)
        uma = df.get("URXUMA", nan).astype(float)      # urine albumin, mg/L
        ucr = df.get("URXUCR", nan).astype(float)      # urine creatinine, mg/dL
        acr = pd.Series(
            np.where((uma > 0) & (ucr > 0), 100.0 * uma / ucr, np.nan),
            index=df.index,
        )  # mg/g

        out = pd.DataFrame({
            "age_years": age.to_numpy(),
            "male": male.astype(int).to_numpy(),
            "egfr": np.round(egfr, 1),
            "acr": np.round(acr.to_numpy(), 1),
            "albumin": df.get("LBXSAL", nan).astype(float).to_numpy(),
            "phosphorus": df.get("LBXSPH", nan).astype(float).to_numpy(),
            "bicarbonate": df.get("LBXSC3SI", nan).astype(float).to_numpy(),
            "calcium": df.get("LBXSCA", nan).astype(float).to_numpy(),
            LABEL_COL: (egfr < 60).astype(int),
            # provenance/assignment helpers — never written into the KFRE clinic CSVs
            "sex_female": (~male).astype(int).to_numpy(),
            "cycle": year,
        })
        rows.append(out)
        coding.append({
            "cycle": year, "participants_in": n0, "used": len(out),
            "labs_observed": {c: int(out[c].notna().sum()) for c in
                              ("egfr", "acr", "albumin", "phosphorus", "bicarbonate", "calcium")},
        })

    participants = pd.concat(rows, ignore_index=True)
    participants.attrs["coding_loss"] = coding
    return participants


def write_kfre_clinics(out_dir: Path | None = None, *, raw_dir: Path = NHANES_RAW) -> dict:
    """Write the KFRE federation (`data/clinics_nhanes_kfre/`): same clinic key as the V1
    NHANES federations (cycle x sex -> 8 practices), columns = the 8 Tangri inputs + label."""
    out_dir = out_dir or HERE.parent / "clinics_nhanes_kfre"
    out_dir.mkdir(parents=True, exist_ok=True)
    participants = build_kfre_participants(raw_dir)
    participants["clinic"] = assign_clinics(participants)
    participants.to_csv(raw_dir / "participants_kfre.csv", index=False)

    keep = [*KFRE_FEATURE_COLS, LABEL_COL]
    complete = participants[KFRE_FEATURE_COLS].notna().all(axis=1)
    meta = {
        "source": "NHANES 2011-2018 (cycles G/H/I/J), raw labs via data/external/nhanes/",
        "schema": "KFRE 8-variable (Tangri et al., JAMA 2011; kfre.py / MDCalc calc/10045)",
        "label": "eGFR<60 (KDIGO stage>=3 prevalence; CKD-EPI 2021 race-free) — see kfre.py "
                 "for why this makes the rule comparison a proxy",
        "n_participants": len(participants),
        "prevalence": round(float(participants[LABEL_COL].mean()), 4),
        "kfre_complete_case_rate": round(float(complete.mean()), 4),
        "coding_loss": participants.attrs["coding_loss"],
        "clinics": [],
    }
    combined = []
    for k, (key, grp) in enumerate(participants.groupby("clinic", sort=True)):
        fname = f"clinic_{k:02d}_{key}.csv"
        grp[keep].to_csv(out_dir / fname, index=False)
        combined.append(grp[keep].assign(clinic_id=k, clinic_name=key))
        meta["clinics"].append({"clinic_id": k, "name": key, "n": len(grp),
                                "ckd_rate": round(float(grp[LABEL_COL].mean()), 4),
                                "kfre_complete_case_rate": round(
                                    float(grp[KFRE_FEATURE_COLS].notna().all(axis=1).mean()), 4)})
    pd.concat(combined, ignore_index=True).to_csv(out_dir / "all_clinics.csv", index=False)
    (out_dir / "suite_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    meta = write_clinics()
    print(f"participants={meta['n_participants']} prevalence={meta['prevalence']:.3f}")
    for c in meta["clinics"][:8]:
        print(f"  {c['clinic_id']} {c['name']:<14} n={c['n']:<5} ckd={c['ckd_rate']:.1%}")
    print("(listed full-size dir; subsampled dir has the same 8 practices)")
    kfre = write_kfre_clinics()
    print(f"kfre participants={kfre['n_participants']} prevalence={kfre['prevalence']:.3f} "
          f"complete-case={kfre['kfre_complete_case_rate']:.3f} -> clinics_nhanes_kfre/")
    for c in kfre["clinics"]:
        print(f"  {c['clinic_id']} {c['name']:<14} n={c['n']:<5} ckd={c['ckd_rate']:.1%} "
              f"complete={c['kfre_complete_case_rate']:.0%}")


if __name__ == "__main__":
    main()
