# External data sources (downloaded 2026-09-03)

Real and realistic-external data used to stress the pipeline beyond V1. Raw files live under
`data/external/` (gitignored, per `.gitignore` `data/**` rule) — this file is the provenance
record that lets anyone rebuild the folder.

| Source | Where it landed | License / terms |
|---|---|---|
| UCI Chronic Kidney Disease (dataset 336) | `data/external/uci_ckd/` | CC-BY 4.0 (UCI ML Repository) |
| NHANES continuous survey, cycles 2011–2012 / 2013–2014 / 2015–2016 / 2017–2018 | `data/external/nhanes/` | CDC public use, no auth |
| Synthea 10k COVID-era population (CSV) | `data/external/synthea/csv_10k/10k_synthea_covid19_csv/` | Apache-2.0 (MITRE) |
| Synthea sample (FHIR R4 bundles) | `data/external/synthea/fhir_sample/` | Apache-2.0 (MITRE) |

## UCI CKD — dataset 336

- URL: <https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip>
- sha256(zip): `3f0d0e5bab0f453165acfa2c77eae695393089c5b338cb4cd98339c96a26a585`
- Contents (after unzip + unrar): `Chronic_Kidney_Disease/chronic_kidney_disease.arff` (train, 400 rows),
  `chronic_kidney_disease_full.arff` (with header/headerless variants note), info txt.
- 400 patients (250 ckd / 150 notckd), 24 features (age, bp, labs, comorbid flags). Very messy:
  ~12% cells missing, categorical typos (`\tyes`, ` yes`, `ckd\t`), numeric-as-object columns.

## NHANES — 36 XPT files (all verified downloaded 200)

Base URL pattern: `https://wwwn.cdc.gov/nchs/data/nhanes/public/<cycle-start>/datafiles/<FILE>_<SUFFIX>.xpt`

- Cycles: 2011→G, 2013→H, 2015→I, 2017→J.
- Files per cycle: `DEMO` (age/sex/race), `MCQ` (told you had CHF/CHD/gout…), `DIQ` (diabetes + age when
  told), `BPQ` (told you had hypertension + when), `BPX` (measured BP), `BMX` (BMI), `BIOPRO`
  (serum creatinine `LBXSCR`, uric acid), `ALB_CR` (urine albumin/creatinine ratio), `GHB` (HbA1c).
- Aggregate fingerprint: sha256 over the 36 per-file sha256s = `8a6ba62b3f01ed66…` (first 16 hex).
  Per-file hashes reproduce via `sha256sum data/external/nhanes/*.xpt`.
- Pandas: `pd.read_sas(path)` reads XPT directly (verified DEMO_J 9 254×46, BIOPRO_J 6 401×41).

## Synthea (MITRE)

- 10k COVID-era CSV population: <https://synthetichealth.github.io/synthea-sample-data/downloads/10k_synthea_covid19_csv.zip>
  — sha256(zip) `559757dc849f4361a328f456d2c0a20c6df72419068321c753c6be787161e937`.
  12 353 patients; 114 545 conditions; 1 659 751 observations; longitudinal (encounters over time).
- FHIR R4 sample bundles: <https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip>
  — sha256(zip) `56cb9e49f7ba6ad4e61c40aa80999f8c10a710823fed1becdf2502053777a521`.
  One JSON bundle per patient — exercises `data/fhir_loader.py`-style parsing without a live FHIR server.
- Codebooks: <https://github.com/synthetichealth/synthea/wiki> (CSV column dictionary per file).
