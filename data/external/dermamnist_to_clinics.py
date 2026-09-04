"""DermaMNIST -> pseudo-clinic CSVs for the `experiment/image` track.

Raw payload: `data/external/dermamnist/dermamnist{,_64}.npz` (provenance, checksums and the CC
BY-NC 4.0 terms in `data/external/SOURCES.md` — downloaded 2026-09-04, Zenodo record 10519652).

Track target: **binary skin-cancer triage** — melanoma (MedMNIST class 4, HAM10000 `mel`) vs the
other six lesion classes. Binary keeps the repo's model class (logreg) and the imbalanced-data
metric stack unchanged; the 7-class relaxation is a deliberate later re-scoping (rule 2 covers
model class, and `_CLASSES` in models/logreg.py is binary today).

Feature convention: flattened pixels `f0000`.. (row-major RGB), raw uint8 range. `to_xy`'s
generic branch treats every non-label column as a feature; `task.fit_scaler` standardises
per client exactly as in the CKD tracks.

Federation is simulated — DermaMNIST has no site/patient keys at this payload level — so clinics
are a **Dirichlet label-skew partition** (`data.partition.dirichlet_partition`, the same
mechanism as the CKD split, alpha default 0.5) into `num_clinics` pseudo-practices named
`shard-NN`. Label + quantity shift, no covariate keys.

Output (mirrors the external mapper convention): `data/clinics_dermamnist/` with
`clinic_XX_shard-NN.csv` per practice, `all_clinics.csv`, and `suite_meta.json`. CSVs are
gitignored by the repo's `data/**/*.csv` rule; `suite_meta.json` is tracked.

    uv run python -m data.external.dermamnist_to_clinics                 # 28x28, 10 clinics
    uv run python -m data.external.dermamnist_to_clinics --variant 64   # 64x64 payload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.partition import dirichlet_partition

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "dermamnist"
LABEL = "melanoma"
MEL_CLASS = 4  # MedMNIST DermaMNIST class index for HAM10000 'mel'

CLASS_NAMES = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}


def load_frame(raw_dir: Path = RAW_DIR, variant: int = 28) -> pd.DataFrame:
    """Pool train+val+test into one labelled frame of flattened pixels.

    The track is a simulation sandbox, so the official MedMNIST split is intentionally pooled
    before partitioning — clients re-split locally exactly like the CKD clinic frames do.
    """
    npz = np.load(raw_dir / ("dermamnist.npz" if variant == 28 else f"dermamnist_{variant}.npz"))
    images = np.concatenate([npz[f"{s}_images"] for s in ("train", "val", "test")])
    labels = np.concatenate([npz[f"{s}_labels"].ravel() for s in ("train", "val", "test")])
    flat = images.reshape(len(images), -1)
    width = flat.shape[1]
    df = pd.DataFrame({f"f{i:04d}": flat[:, i] for i in range(width)}, dtype="uint8")
    df[LABEL] = (labels == MEL_CLASS).astype("int64")
    return df


def write_clinics(
    out_dir: Path | None = None,
    *,
    num_clinics: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    variant: int = 28,
    min_positives: int = 5,
) -> dict:
    """Write the pseudo-practice federation + provenance meta (mapper-output convention).

    Partition acceptance guard: with ~11% prevalence a pure Dirichlet draw can hand a clinic
    zero positives, which leaves that client's AUROC undefined (one-class eval) and poisons the
    worst-practice aggregate (rule 5). Redraw the partition (seed+1 per attempt) until every
    clinic holds at least `min_positives` positives; the accepted draw is recorded in the meta.
    """
    out_dir = out_dir or HERE.parent / "clinics_dermamnist"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_frame(variant=variant)
    for attempt in range(100):
        parts = dirichlet_partition(df, num_clinics, alpha, seed + attempt, label_col=LABEL)
        if min(int(df[LABEL].to_numpy()[idx].sum()) for idx in parts) >= min_positives:
            if attempt:
                print(f"partition accepted after {attempt + 1} draws (seed={seed + attempt})")
            break
    else:
        raise RuntimeError(f"no Dirichlet draw with >= {min_positives} positives/clinic in 100 tries")
    seed = seed + attempt

    meta = {
        "source": f"MedMNIST DermaMNIST {variant}x{variant} (Zenodo 10519652; HAM10000-derived)",
        "label": "melanoma = dx class 'mel' (binary; other six lesion classes pooled)",
        "task": f"image: {variant}x{variant} RGB, flattened row-major (f0000-f{(variant * variant * 3) - 1:04d})",
        "n_images": len(df),
        "prevalence": round(float(df[LABEL].mean()), 4),
        "partition": {"method": "dirichlet-label", "alpha": alpha, "seed": seed,
                      "num_clinics": num_clinics},
        "note": "No patient/site keys or protected attributes at this payload level; T2.5 "
                "fairness-by-sex not applicable on this track. CC BY-NC 4.0 — research use only.",
        "clinics": [],
    }
    combined = []
    for k, idx in enumerate(parts):
        g = df.iloc[idx].reset_index(drop=True)
        name = f"shard-{k:02d}"
        g.to_csv(out_dir / f"clinic_{k:02d}_{name}.csv", index=False)
        combined.append(g.assign(clinic_id=k, clinic_name=name))
        meta["clinics"].append({"clinic_id": k, "name": name, "n": len(g),
                                "mel_rate": round(float(g[LABEL].mean()), 4),
                                "dir": out_dir.name})
    pd.concat(combined, ignore_index=True).to_csv(out_dir / "all_clinics.csv", index=False)
    (out_dir / "suite_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clinics", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variant", type=int, default=28, choices=[28, 64])
    args = p.parse_args()
    meta = write_clinics(num_clinics=args.clinics, alpha=args.alpha, seed=args.seed,
                         variant=args.variant)
    print(f"wrote {meta['n_images']} images -> {meta['clinics'][0]['dir']} "
          f"({len(meta['clinics'])} clinics, prevalence {meta['prevalence']})")
    for c in meta["clinics"]:
        print(f"  {c['name']}: n={c['n']:>5} mel_rate={c['mel_rate']:.4f}")


if __name__ == "__main__":
    main()
