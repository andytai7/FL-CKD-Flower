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
| MedMNIST DermaMNIST 28/64 (HAM10000-derived) | `data/external/dermamnist/` | CC BY-NC 4.0 (inherited from HAM10000 — research use only) |
| PhysioNet CinC Challenge 2017 (single-lead ECG, AF) | `data/external/ecg_cinc2017/` | PhysioNet open access, no credential required |
| ETT-small ETTh1 / ETTm1 (Informer release) | `data/external/ett/` | CC BY 4.0 per ETDataset repo |
| Weather (Autoformer release, 21 channels) | `data/external/weather/` | BSD-3 per Jena weather / thuml Time-Series-Library mirror |

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

## MedMNIST DermaMNIST — dermoscopy images (downloaded 2026-09-04)

- Zenodo record 10519652 (MedMNIST+ v2): `dermamnist.npz` (19.7 MB, 28x28) sha256
  `1a309fec2e33bb6aba88e7d078e5ccbb9736c84a0b415ac197eb3c8fa331e050`; `dermamnist_64.npz`
  (100.1 MB, 64x64) sha256 `1dcb34f0a67ab8679df5eb4d1070c7db8f1c3e359d3dda76cc4e07080e5e91c6`.
- 10,015 dermatoscopic images, 7 classes (akiec 228, bcc 359, bkl 769, df 80, mel 469, nv 4693,
  vasc 99 in train) — officially split 7007/1003/2005 train/val/test; heavily imbalanced (nv
  dominates), which suits this repo's imbalanced-metrics primary rule.
- Target for the `experiment/image` track: skin-lesion diagnosis (7-class, or binary
  melanoma-vs-rest for the simplest FL cut). Not a V1-schema dataset — the mapper produces image
  clients (label-skewed partition via `data/partition.py`), not clinic CSVs.
- Derived from HAM10000 (Tschandl et al., 2018); the CC BY-NC 4.0 terms carry over: research /
  non-commercial use only, cite MedMNIST v2 (Yang et al., Scientific Data 2023).

## PhysioNet CinC Challenge 2017 — single-lead ECG (downloaded 2026-09-04)

- `training2017.zip` (99.2 MB) from
  `https://physionet.org/files/challenge-2017/1.0.0/training2017.zip` sha256
  `25fdbab36bb6724b66a4d2c26ff89f6b9d6ff87e6aed714d24cf5bf071cd1255`.
  GET on physionet.org requires a non-curl User-Agent (curl's default UA is refused); the
  physionet-open S3 mirror serves the same byte-identical file anonymously.
- 8,528 recordings A00001-A08528, each a 9-60s single-lead (lead I, AliveCor Kardia) ECG at
  300 Hz as a MATLAB `.mat` (`val` int16; reads with `scipy.io.loadmat`) + `.hea` header.
- Labels from the zip's `REFERENCE.csv` (measured): N 5050, O 2456, A 738, ~ 284.
- Target for the `experiment/timeseries` track: atrial-fibrillation screening (binary A vs rest,
  or 4-class rhythm). Primary-care-appropriate modality: Kardia devices are used in GP AF
  screening programmes.

## ETT-small (downloaded 2026-09-04)

- Canonical raw CSVs from `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/`:
  `ETTh1.csv` (2.6 MB, 17 420 hourly rows, 2016-07-01→2018-06-26) sha256
  `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`;
  `ETTm1.csv` (10.4 MB, 69 680 15-min rows) sha256
  `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e`.
- 7 channels (`HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`); benchmark target = `OT` (oil temperature).
- Benchmark role (`research/privacy-dl-ts`): input-96 → horizon-96/192/336 forecasting under the
  event/user-level DP contrast; federation simulated over month-Dirichlet window shards
  (`data/external/forecast_to_seq_clinics.py`).

## Weather (downloaded 2026-09-04)

- Mirror of the Autoformer release at thuml/Time-Series-Library:
  `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv`
  (8.4 MB, 52 696 10-min rows, 21 channels incl. `OT`) sha256
  `34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63`.
  Header has mojibake (`W/m�`, `�mol`) on three columns — passed through as-is, columns addressed
  positionally. Benchmark target = all channels (multivariate forecasting).
- Benchmark role: the seasonality-stress arm of the same window-shard forecasting suites.
