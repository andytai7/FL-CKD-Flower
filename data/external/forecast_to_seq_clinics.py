"""ETT / Weather forecasting series -> pseudo-clinic sequence suites (`experiment/timeseries`).

The benchmark's forecasting side (research/privacy-dl-ts §0): ETTh1/ETTm1 (Informer release;
univariate target OT, multivariate covariates) and Weather (Autoformer release; multivariate
target = all channels). Each dataset is a single long series, so federation is simulated over
WINDOWS: a sliding (input_len, horizon) window set is Dirichlet-partitioned into pseudo-practices
by calendar month of the window's start — clinics then get seasonally skewed, chronologically
scattered windows, which is the realistic site heterogeneity this track stresses (and a stronger
concept-shift signal than the index-uniform IID partition).

Emission contract (`data/clinics_forecast/<name>_in{I}_h{H}/`):
- per-clinic NPZ: `series` (T, C) float32 (the FULL raw series, duplicated per clinic — MBs, not
  the window tensors), `starts` int64 window-start indices this clinic owns, plus `meta` scalars;
  windows are reconstructed lazily at train time as series[start : start+I+H].
- `all_clinics.npz` + `suite_meta.json` (mapper-output convention). Targets by dataset:
  ETT -> channel OT (last column); Weather -> all C channels. Normalisation is NOT baked in:
  each clinic z-scores on its own train windows only, at fit time (leak-free by construction).

Payloads are gitignored. Provenance + sha256 in SOURCES.md.
Run:  uv run python -m data.external.forecast_to_seq_clinics --dataset etth1 --horizon 96
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.partition import dirichlet_partition

HERE = Path(__file__).resolve().parent
DATASETS = {
    # name -> (csv path, target mode, source label)
    "etth1": (HERE / "ett" / "ETTh1.csv", "ot", "ETT-small ETTh1 (hourly, transformer 1, 2016-07→2018-06)"),
    "ettm1": (HERE / "ett" / "ETTm1.csv", "ot", "ETT-small ETTm1 (15-min, transformer 1)"),
    "weather": (HERE / "weather" / "weather.csv", "all", "Weather (10-min, 21 channels, Jena + OT)"),
}
OUT_ROOT = HERE.parent / "clinics_forecast"


def load_series(csv_path: Path) -> tuple[np.ndarray, pd.Series, list[str]]:
    """(T, C) float32 frame + the parsed datetime index. Weather's mojibake headers pass through."""
    df = pd.read_csv(csv_path)
    dates = pd.to_datetime(df["date"])
    channels = [c for c in df.columns if c != "date"]
    return df[channels].astype("float32").to_numpy(), dates, channels


def write_forecast_clinics(
    dataset: str,
    *,
    horizon: int = 96,
    in_len: int = 96,
    num_clinics: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    min_windows: int = 200,
) -> dict:
    csv_path, target_mode, src_label = DATASETS[dataset]
    series, dates, channels = load_series(csv_path)
    T, C = series.shape
    n_windows = T - in_len - horizon + 1
    if n_windows < num_clinics * min_windows:
        raise ValueError(f"{dataset}: only {n_windows} windows at in={in_len}/h={horizon}")

    # Pseudo-labels for the skewed split: calendar month of the window START.
    months = dates.iloc[:n_windows].dt.month.to_numpy()
    frame = pd.DataFrame({"month": months})
    for attempt in range(100):
        parts = dirichlet_partition(frame, num_clinics, alpha, seed + attempt, label_col="month")
        if min(len(p) for p in parts) >= min_windows:
            break
    else:
        raise RuntimeError(f"no month-Dirichlet draw with >= {min_windows} windows/clinic in 100 tries")
    accepted_seed = seed + attempt

    out_dir = OUT_ROOT / f"{dataset}_in{in_len}_h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    meta = {
        "source": src_label,
        "task": f"forecasting: input {in_len} -> horizon {horizon} steps",
        "target": ("channel OT (last index)" if target_mode == "ot" else "all channels"),
        "series": {"rows": T, "channels": C,
                   "columns": channels,
                   "payload_sha256": sha},
        "windows": {"in_len": in_len, "horizon": horizon, "n": n_windows,
                    "stride": 1, "overlap": "full"},
        "partition": {"method": "dirichlet-on-start-month", "alpha": alpha,
                      "seed": accepted_seed, "num_clinics": num_clinics,
                      "note": "chrono-scattered seasonal skew; guard: every clinic >= "
                              f"{min_windows} windows"},
        "normalisation": "per-clinic z-score on OWN train windows at fit time (leak-free)",
        "files": [],
        "note": "One administrative site per dataset -> federation over window shards is "
                "simulated heterogeneity by construction. Series duplicated per clinic because "
                "it is small; window tensors are materialised lazily.",
    }
    all_starts = []
    for k, idx in enumerate(parts):
        starts = np.sort(idx).astype("int64")
        name = f"shard-{k:02d}"
        np.savez_compressed(
            out_dir / f"clinic_{k:02d}_{name}.npz", series=series, starts=starts,
            months=months[starts],
        )
        month_counts = np.bincount(months[starts], minlength=12).tolist()
        all_starts.append(starts)
        meta["files"].append({"clinic_id": k, "name": name,
                              "file": f"clinic_{k:02d}_{name}.npz", "n_windows": len(starts),
                              "start_month_histogram": month_counts})
    concat = np.concatenate(all_starts)
    assert len(np.unique(concat)) == n_windows, "partition must cover every window exactly once"
    np.savez_compressed(out_dir / "all_clinics.npz", series=series, starts=np.arange(n_windows))
    (out_dir / "suite_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    p.add_argument("--horizon", type=int, default=96, choices=(96, 192, 336))
    p.add_argument("--in-len", type=int, default=96)
    p.add_argument("--clinics", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        meta = write_forecast_clinics(name, horizon=args.horizon, in_len=args.in_len,
                                      num_clinics=args.clinics, alpha=args.alpha,
                                      seed=args.seed)
        w = meta["windows"]
        print(f"{name} in{w['in_len']} h{w['horizon']}: {w['n']} windows -> "
              f"{meta['partition']['num_clinics']} clinics (draw seed {meta['partition']['seed']})")
        for f in meta["files"]:
            print(f"  {f['name']}: {f['n_windows']:>6} windows")


if __name__ == "__main__":
    main()
