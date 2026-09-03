# Dataset versioning — what "V1" means

**V1 is frozen.** It is exactly the artifact set hashed in `manifest_v1.sha256`:

| V1 artifact | Content |
|---|---|
| `data/synthetic_ckd_data.csv` | 2000-row wiring placeholder (negative control, AUROC ≈ 0.5 by design) |
| `data/clinics/` | 10 non-IID per-practice panels, seed 42 (`uv run ckd-clinics`) — the evaluation federation |
| `data/clinics_ladder/{r5,r20,r100}/` | panel-size ladder variants used by `equity.py` |
| `data/{synthesize,loader,partition}.py` | the generator + preprocessing code that produced them |

Verify integrity any time:

```bash
sha256sum -c data/manifest_v1.sha256
```

Rules:

1. V1 files are never regenerated, renamed, or edited. `ckd-clinics` output overwrites
   `data/clinics/` — do not run it against V1 paths.
2. Everything new lands *alongside* V1, never inside it: new generator modules (`synthesize_v2.py`),
   new clinic directories (`data/clinics_v2_*/`, `data/clinics_nhanes/`, …), external downloads
   under `data/external/`, new notebooks (`notebooks/04_*`, `05_*`, …).
3. Every results figure that names "the V1 dataset" must be reproducible from exactly these hashes.
