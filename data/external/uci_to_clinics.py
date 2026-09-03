"""UCI CKD (dataset 336) -> V1-schema single clinic (`data/clinics_uci/`).

The 400-row Apollo Hospitals (India) set is the project's **preprocessing stress case**: ~12%
missing cells, nominal columns with typos (`\tyes`, ` yes`, `ckd\t`), numeric columns typed as
strings with '?' entries. Provenance: `data/external/SOURCES.md`; raw ARFFs under
`data/external/uci_ckd/Chronic_Kidney_Disease/`.

V1-schema mapping (deficits are DOCUMENTED structural zeros, §3a — under-coding made by reality,
not by a knob):
  age_years           <- age
  dx_hypertonie       <- htn == yes
  dx_diabetes         <- dm == yes   (ARFF carries ' yes'/' yes\t' typos — normalised)
  dx_khk              <- cad == yes  (coronary artery disease ~ I25)
  dx_adipositas       <- 0 (not collected in the study)
  dx_herzinsuffizienz <- 0 (not collected)
  dx_hyperurikaemie   <- 0 (not collected; pot/sod exist but are electrolytes, not gout)
  years_since_*       <- 0.0 (no onset dates collected)
  label ckd_stage3plus<- class == ckd  (study label; chart-review CKD, not stage-graded)

The same row set is additionally exported with native columns in
`data/external/uci_ckd/participants_mapped.csv` for the notebook-04 imputation/messiness audit
(how many cells had to be repaired, per column).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..loader import FEATURE_COLS, LABEL_COL

HERE = Path(__file__).resolve().parent
ARFF = HERE / "uci_ckd" / "Chronic_Kidney_Disease" / "chronic_kidney_disease_full.arff"


def load_raw(path: Path = ARFF) -> pd.DataFrame:
    """Parse the ARFF by hand — 25 attributes, comma data, '?' missing, TyPo'ed nominals.

    Known file pathologies (verified byte-level on 2026-09-03): CRLF endings everywhere;
    2 rows terminate in a **trailing comma** ('…,ckd,' — repaired by dropping the empty last
    field); 1 row carries an extra mid-row comma that misaligns columns irrecoverably —
    QUARANTINED, not guessed. Anything else off-width is quarantined too. Counts are returned.
    """
    data_started, cols, rows = False, [], []
    audit = {"trailing_comma_repaired": 0, "quarantined_rows": []}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if not data_started:
                if low.startswith("@attribute"):
                    parts = line.replace("'", " ").split()
                    cols.append(parts[1])
                elif low.startswith("@data"):
                    data_started = True
                continue
            fields = [f.strip() for f in line.split(",")]
            if len(fields) == len(cols) + 1 and fields[-1] == "":
                fields = fields[:-1]  # trailing comma after the class value
                audit["trailing_comma_repaired"] += 1
            if len(fields) != len(cols):
                audit["quarantined_rows"].append({"line": ln, "n_fields": len(fields)})
                continue
            rows.append(fields)
    df = pd.DataFrame(rows, columns=cols)
    df.attrs["parse_audit"] = audit
    df = df.map(lambda v: np.nan if v is None else str(v).strip().replace("\t", "") or np.nan)
    df = df.replace({"?": np.nan})
    return df


def clean_yes_no(s: pd.Series, audit: dict, col: str) -> pd.Series:
    """Normalise typo'ed yes/no nominals; unanswered/'?' -> structural zero, counted."""
    s = s.str.lower()
    bad = ~s.isin(["yes", "no"])
    audit[col] = {"repaired": int(bad.sum())}
    return (s == "yes").astype(int)


def build_clinic(path: Path = ARFF) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (V1-schema frame, native-schema frame, per-column repair audit)."""
    raw = load_raw(path)
    audit: dict = {"parse": raw.attrs["parse_audit"]}
    native = raw.copy()

    yes_no_map = {"htn": "dx_hypertonie", "dm": "dx_diabetes", "cad": "dx_khk"}
    frame = pd.DataFrame()
    frame["age_years"] = pd.to_numeric(raw["age"], errors="coerce").clip(18, 95).fillna(
        pd.to_numeric(raw["age"], errors="coerce").median())
    for src_col, dst in yes_no_map.items():
        frame[dst] = clean_yes_no(raw[src_col].fillna("no"), audit, src_col)
    frame["dx_adipositas"] = 0
    frame["dx_herzinsuffizienz"] = 0
    frame["dx_hyperurikaemie"] = 0
    frame["years_since_hypertonie_dx"] = 0.0
    frame["years_since_diabetes_dx"] = 0.0
    frame["years_since_khk_dx"] = 0.0
    cls = raw["class"].astype(str).str.lower().str.strip()
    audit["class"] = {"repaired": int((~cls.isin(["ckd", "notckd"])).sum())}
    frame[LABEL_COL] = (cls == "ckd").astype(int)

    # Numeric-as-object repairs, for the native audit frame. Only the columns the codebook
    # declares numeric are audited/converted — rbc/pc/pcc/ba/appet/pe/ane are nominal by design.
    numeric_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
                    "sod", "pot", "hemo", "pcv", "wc", "rc"]
    repairs = {}
    for c in numeric_cols:
        if c not in native.columns:
            continue
        as_num = pd.to_numeric(native[c], errors="coerce")
        bad = native[c].notna() & as_num.isna()
        repairs[c] = {"garbage_values": int(bad.sum()),
                      "missing_share": round(float(as_num.isna().mean()), 4)}
        native[c] = as_num
    audit["numeric_repairs"] = repairs

    return frame[[*FEATURE_COLS, LABEL_COL]], native, audit


def write_clinic(out_dir: Path | None = None) -> dict:
    out = out_dir or HERE.parent / "clinics_uci"
    out.mkdir(parents=True, exist_ok=True)
    frame, native, audit = build_clinic()
    frame.to_csv(out / "clinic_00_uci-apollo.csv", index=False)
    native.to_csv(HERE / "uci_ckd" / "participants_mapped.csv", index=False)
    meta = {"source": "UCI CKD #336 (Apollo Hospitals, Tamil Nadu, ~2 months)",
            "label": "class==ckd (chart review, not stage-graded)",
            "n": len(frame), "ckd_rate": round(float(frame[LABEL_COL].mean()), 4),
            "audit": audit,
            "note": "single clinic by design; missing comorbidity families are structural zeros"}
    (out / "suite_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    meta = write_clinic()
    print(f"UCI CKD -> clinic_00_uci-apollo.csv  n={meta['n']} ckd_rate={meta['ckd_rate']:.1%}")
    print("repairs:", json.dumps(meta["audit"], indent=None)[:400])


if __name__ == "__main__":
    main()
