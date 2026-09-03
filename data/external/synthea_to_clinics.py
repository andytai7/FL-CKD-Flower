"""Synthea 10k COVID-era CSV population -> V1-schema clinics (`data/clinics_synthea/`).

MITRE Synthea, pre-generated (`data/external/SOURCES.md` — 12 353 patients, longitudinal
encounters). Unlike V1/V2 and cross-sectional NHANES, Synthea is a birth-to-death EMR: every
comorbidity has an onset date, so `years_since_*` comes from REAL duration and per-clinic
population = county of care.

Why a CSV mapping rather than the FHIR path: `data/fhir_loader.py` consumes a *live FHIR R4
server*, one practice per URL. The FHIR R4 sample bundles we downloaded
(`data/external/synthea/fhir_sample/`) prove the wire format but a bundle->canonical adapter is
properly scoped as notebook-04 follow-up work; the CSV cut below already exercises the
federation against a third-party generator's clinical logic (conditional disease progression,
treatment records), which is the property V1/V2 lack.

Index date per patient = most recent encounter. Label: CHRONIC CKD stage >= 3 — two eGFR < 60
values at least 90 days apart before index (KDIGO chronicity rule — the same rule
`extract_features.sql` uses; a single low eGFR during e.g. hospitalization is AKI, not CKD).
First build dropped patients without a measured eGFR and produced prevalence 0.68 — pure
indication bias (only kidney patients get measured); instead the cohort is ALL adults, and
unmeasured eGFR -> label 0 with `egfr_measured` recorded per patient (measurement IS
informative — its missing-indicator is the V1+labs hook, kept out of V1 features here).
eGFR observations: LOINC 33914-3 ('Estimated Glomerular Filtration Rate') plus the
'predicted' flavor without a LOINC code (description match).

Comorbidity codes (Synthea conditions.csv, SNOMED-CT):
  dx_hypertonie       = 38341003 'Hypertension' (recorded as plain 'Hypertension' in this sample)
  dx_diabetes         = 44054006 'Diabetes' (+ any 'Diabetes…' subtype)
  dx_khk              = 53741008 'Coronary Heart Disease'
  dx_herzinsuffizienz = descriptions containing 'heart failure' (145 + 670 in this sample)
  dx_adipositas       = obesity condition (DESCRIPTION contains 'obesity', e.g. 162864005)
                        OR last BMI observation (LOINC 39156-5) >= 30
  dx_hyperurikaemie   = 90560007 'Gout'
  years_since_*       = index − first condition START, in years (V1 clip: >= 0, capped 80)
  label               = latest pre-index GFR observation < 60

Clinics = COUNTY of the patient's home address (patients.csv), keeping counties with >= 30
labelled patients and folding the rest into a trailing 'misc' clinic (still deterministic).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..loader import FEATURE_COLS, LABEL_COL

HERE = Path(__file__).resolve().parent
SYNTHEA = HERE / "synthea" / "csv_10k" / "10k_synthea_covid19_csv"

HTN = ("Hypertension",)
DM_CODES = {"44054006"}
KHK_CODES = {"53741008"}
GOUT_CODES = {"90560007"}
GFR_LOINC = {"33914-3"}
GFR_DESC = "glomerular filtration rate"
BMI_LOINC = {"39156-5"}
MIN_CLINIC_N = 30


def build_participants(raw_dir: Path = SYNTHEA) -> pd.DataFrame:
    """One row per adult patient (18–95) with at least one encounter."""
    # Synthea writes encounter/observation timestamps with timezone offsets; birthdates are
    # plain dates. Parse everything UTC then strip tz — one naive timeline throughout.
    enc = pd.read_csv(raw_dir / "encounters.csv", usecols=["PATIENT", "START"])
    enc["START"] = pd.to_datetime(enc["START"], format="mixed", utc=True).dt.tz_localize(None)
    index = enc.groupby("PATIENT")["START"].max().rename("index_date")

    pat = pd.read_csv(raw_dir / "patients.csv",
                      usecols=["Id", "BIRTHDATE", "GENDER", "RACE", "COUNTY"])
    pat["BIRTHDATE"] = pd.to_datetime(pat["BIRTHDATE"], format="mixed", utc=True).dt.tz_localize(None)
    df = pat.merge(index, left_on="Id", right_on="PATIENT", how="inner")
    df["age_years"] = ((df["index_date"] - df["BIRTHDATE"]).dt.days / 365.25).round(1)
    df = df[(df.age_years >= 18) & (df.age_years <= 95)].copy()

    con = pd.read_csv(raw_dir / "conditions.csv", usecols=["PATIENT", "START", "CODE", "DESCRIPTION"])
    con["START"] = pd.to_datetime(con["START"], format="mixed", utc=True).dt.tz_localize(None)

    obs = pd.read_csv(raw_dir / "observations.csv",
                      usecols=["PATIENT", "DATE", "CODE", "DESCRIPTION", "VALUE"])
    obs["DATE"] = pd.to_datetime(obs["DATE"], format="mixed", utc=True).dt.tz_localize(None)
    gfr_mask = obs["CODE"].isin(GFR_LOINC) | obs["DESCRIPTION"].str.lower().str.contains(
        GFR_DESC, na=False)
    gfr = obs[gfr_mask].copy()
    gfr["VALUE"] = pd.to_numeric(gfr["VALUE"], errors="coerce")
    gfr = gfr.dropna(subset=["VALUE"])

    merged = df.merge(gfr, left_on="Id", right_on="PATIENT")

    def _chronic(grp: pd.DataFrame) -> pd.Series:
        """Chronicity per patient: earliest+latest low-eGFR gap >= 90 days; latest low flags now."""
        low = grp[grp["VALUE"] < 60]["DATE"]
        chronic = len(low) >= 2 and (low.max() - low.min()).days >= 90
        latest_val = grp.sort_values("DATE")["VALUE"].iloc[-1] if len(grp) else np.nan
        return pd.Series({"chronic": chronic, "egfr_last": latest_val})

    kidney = (merged[merged["DATE"] <= merged["index_date"]]
              .groupby("Id").apply(_chronic, include_groups=False))
    df["egfr_measured"] = df["Id"].isin(kidney.index).astype(int)
    df = df.merge(kidney.reset_index()[["Id", "egfr_last"]], on="Id", how="left")
    df[LABEL_COL] = df["Id"].map(kidney["chronic"]).fillna(False).astype(int)

    cond_by_pid: dict[str, pd.DataFrame] = {p: g for p, g in con.groupby("PATIENT")}

    def ever(mask_fn) -> pd.Series:  # noqa: ANN001
        res = {}
        for pid, g in cond_by_pid.items():
            sel = mask_fn(g)
            res[pid] = (g.loc[sel, "START"].min() if sel.any() else pd.NaT)
        return pd.Series(res, name="first_onset")

    onset_htn = ever(lambda g: g["DESCRIPTION"] == "Hypertension")
    onset_dm = ever(lambda g: g["DESCRIPTION"].str.contains("Diabetes", na=False))
    onset_khk = ever(lambda g: g["CODE"].isin(KHK_CODES))
    for name, onset in [("onset_htn", onset_htn), ("onset_dm", onset_dm), ("onset_khk", onset_khk)]:
        df[name] = df["Id"].map(onset)
    hf = ever(lambda g: g["DESCRIPTION"].str.contains("heart failure", case=False, na=False))
    gout = ever(lambda g: g["CODE"].isin(GOUT_CODES))
    obese_cond = ever(lambda g: g["DESCRIPTION"].str.contains("obesity", case=False, na=False))

    bmi = obs[obs["CODE"].isin(BMI_LOINC)].copy()
    bmi["VALUE"] = pd.to_numeric(bmi["VALUE"], errors="coerce")
    bmi_last = (bmi.sort_values("DATE").groupby("PATIENT").tail(1)
                .set_index("PATIENT")["VALUE"].rename("bmi_last"))
    df["bmi_last"] = df["Id"].map(bmi_last)
    df["hf_flag"] = df["Id"].map(hf).notna().astype(int)
    df["gout_flag"] = df["Id"].map(gout).notna().astype(int)
    df["obese_cond"] = df["Id"].map(obese_cond).notna().astype(int)

    df["dx_hypertonie"] = df["onset_htn"].notna().astype(int)
    df["dx_diabetes"] = df["onset_dm"].notna().astype(int)
    df["dx_khk"] = df["onset_khk"].notna().astype(int)
    df["dx_adipositas"] = ((df["obese_cond"] == 1) | (df["bmi_last"] >= 30)).astype(int)
    df["dx_herzinsuffizienz"] = df["hf_flag"]
    df["dx_hyperurikaemie"] = df["gout_flag"]

    for onset_c, years_c in [("onset_htn", "years_since_hypertonie_dx"),
                             ("onset_dm", "years_since_diabetes_dx"),
                             ("onset_khk", "years_since_khk_dx")]:
        yrs = (df["index_date"] - pd.to_datetime(df[onset_c])).dt.days / 365.25
        df[years_c] = yrs.clip(0, 80).fillna(0.0).round(1)

    df["sex_female"] = (df["GENDER"] == "F").astype(int)
    df["clinic_county"] = df["COUNTY"].fillna("unknown")
    return df[[*FEATURE_COLS, LABEL_COL, "sex_female", "RACE", "clinic_county",
               "egfr_last", "bmi_last", "egfr_measured"]].rename(columns={"RACE": "race"})


def write_clinics(out_dir: Path | None = None, *, raw_dir: Path = SYNTHEA) -> dict:
    """Partition by county; small counties fold into one trailing 'misc' clinic (deterministic)."""
    out_dir = out_dir or HERE.parent / "clinics_synthea"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stale-file guard: clinic membership and naming can change between builds (county set is
    # data-dependent) — never let yesterday's clinic_*.csv leak into today's federation.
    for stale in out_dir.glob("clinic_*.csv"):
        stale.unlink()
    df = build_participants(raw_dir)
    df.to_csv(HERE / "synthea" / "participants_mapped.csv", index=False)

    counts = df["clinic_county"].value_counts()
    big = set(counts[counts >= MIN_CLINIC_N].index)
    df["clinic_name"] = np.where(df["clinic_county"].isin(big),
                                 df["clinic_county"].str.lower(), "misc")

    meta = {"source": "Synthea 10k COVID-era CSV (MITRE, Apache-2.0)",
            "label": "chronic stage>=3: two eGFR<60 >= 90d apart before index (KDIGO; matches extract_features.sql)",
            "n_participants": len(df),
            "egfr_measured_share": round(float(df["egfr_measured"].mean()), 4),
            "prevalence": round(float(df[LABEL_COL].mean()), 4),
            "clinics": []}
    combined = []
    for k, (name, grp) in enumerate(df.groupby("clinic_name", sort=True)):
        g = grp.drop(columns=["clinic_name", "clinic_county", "egfr_last", "bmi_last"])
        g.to_csv(out_dir / f"clinic_{k:02d}_{name}.csv", index=False)
        combined.append(g.assign(clinic_id=k, clinic_name=name))
        meta["clinics"].append({"clinic_id": k, "name": name, "n": len(g),
                                "ckd_rate": round(float(g[LABEL_COL].mean()), 4)})
    pd.concat(combined, ignore_index=True).to_csv(out_dir / "all_clinics.csv", index=False)
    (out_dir / "suite_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    meta = write_clinics()
    print(f"participants={meta['n_participants']} prevalence={meta['prevalence']:.3f}")
    for c in meta["clinics"]:
        print(f"  {c['clinic_id']} {c['name']:<16} n={c['n']:<5} ckd={c['ckd_rate']:.1%}")


if __name__ == "__main__":
    main()
