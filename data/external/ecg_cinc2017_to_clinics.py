"""PhysioNet CinC Challenge 2017 -> pseudo-clinic CSVs for the `experiment/timeseries` track.

Raw payload: `data/external/ecg_cinc2017/training2017/` (zip provenance, checksum and access
notes in `data/external/SOURCES.md` — downloaded 2026-09-04). 8,528 recordings A00001–A08528:
single-lead (lead I, AliveCor Kardia) ECG at 300 Hz, 9–60 s, MATLAB `.mat` + WFDB `.hea`.

Track target: **binary atrial-fibrillation screening** (`A` vs the rest) — the primary-care AF
check Kardia-style devices are used for. Class `~` (noisy) is dropped from the frames (documented
in suite_meta); 4-class rhythm classification is a deliberate later re-scoping (the repo model
class is binary today).

Feature convention (numpy-scipy only, deterministic, fits the logreg-only model class — rule 2):
  - `w0000`..`w0299` — waveform block profile: each trace resampled to 6,000 points by linear
    interpolation over normalised time, then block-averaged to 300 means;
  - `duration_s`, `amp_mean`, `amp_std`, `amp_rms`, `amp_skew`, `amp_kurt` (excess) — amplitude
    statistics;
  - `dom_freq_hz`, `dom_power_share`, `spec_entropy` — spectrum of the demeaned trace over
    0.5–40 Hz (rfft): dominant frequency, its power share, normalised spectral entropy
    (AF's irregular baseline raises entropy; organized rhythms peak sharply).
All features are computed per recording with no statistics shared across recordings — the frame
is complete numerics, matching `to_xy`'s generic branch (no imputation rules).

Clinics are a Dirichlet label-skew partition (alpha default 0.5) of recordings into
`num_clinics` pseudo-practices — the same mechanism as the CKD split; recordings carry no site
keys, so federation is simulated. Partition acceptance guard: redraw until every clinic holds
>= `min_positives` AF cases (one-class eval would leave a client's AUROC undefined).

Output (mapper convention): `data/clinics_ecg/{clinic_XX_shard-NN.csv, all_clinics.csv,
suite_meta.json}` — CSVs gitignored by `data/**/*.csv`, meta tracked.

    uv run python -m data.external.ecg_cinc2017_to_clinics               # 10 clinics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from data.partition import dirichlet_partition

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "ecg_cinc2017" / "training2017"
LABEL = "afib"
FS = 300.0           # Hz, per the record's .hea files
PROFILE_COLS = 300   # w0000..w0299
PROFILE_RESAMPLE = 6000
BAND = (0.5, 40.0)   # Hz window for the spectral features

STAT_COLS = ["duration_s", "amp_mean", "amp_std", "amp_rms", "amp_skew", "amp_kurt",
             "dom_freq_hz", "dom_power_share", "spec_entropy"]

SEQ_LEN = 4096  # fixed-length resampled trace the LSTM consumes (~13.65 s at 300 Hz native)


def _seq_view(xf: np.ndarray) -> np.ndarray:
    """Fixed-length LSTM input: normalised-time linear resample to SEQ_LEN, per-trace z-norm.

    Traces span 9-60 s at 300 Hz; a normalised-time resample keeps every recording (no crop/pad
    choice to justify) and matches the profile-resample policy in `_waveform_features`.
    Per-trace z-normalisation removes gain/baseline wander between AliveCor devices — rhythm
    carries the AF evidence; raw amplitude statistics remain in STAT_COLS for the feature view.
    """
    z = (xf - xf.mean()) / (xf.std() + 1e-8)
    return np.interp(
        np.linspace(0, len(xf) - 1, SEQ_LEN), np.arange(len(xf)), z
    ).astype("float32")


def _waveform_features(x: np.ndarray) -> dict[str, float]:
    """Deterministic per-trace features (no cross-recording statistics)."""
    n = len(x)
    xf = x.astype("float64")
    mu, sd = xf.mean(), xf.std()
    m3 = float(((xf - mu) ** 3).mean()) / (sd ** 3 + 1e-12)
    m4 = float(((xf - mu) ** 4).mean()) / (sd ** 4 + 1e-12) - 3.0

    # Spectral features over the analysis band (rfft, fs=300).
    spec = np.abs(np.fft.rfft(xf - mu))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    band_mask = (freqs >= BAND[0]) & (freqs <= BAND[1])
    band_spec = spec[band_mask]
    band_freqs = freqs[band_mask]
    psd = band_spec ** 2
    total = psd.sum() + 1e-12
    dom_i = int(psd.argmax())
    p = psd / total
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p)))

    # Fixed-length waveform profile via normalised-time linear resample + block means.
    resampled = np.interp(
        np.linspace(0, n - 1, PROFILE_RESAMPLE), np.arange(n), xf
    )
    profile = resampled.reshape(PROFILE_COLS, PROFILE_RESAMPLE // PROFILE_COLS).mean(axis=1)

    feats = {
        "duration_s": n / FS,
        "amp_mean": float(mu),
        "amp_std": float(sd),
        "amp_rms": float(np.sqrt((xf ** 2).mean())),
        "amp_skew": m3,
        "amp_kurt": m4,
        "dom_freq_hz": float(band_freqs[dom_i]),
        "dom_power_share": float(psd[dom_i] / total),
        "spec_entropy": entropy,
    }
    feats.update({f"w{i:04d}": float(v) for i, v in enumerate(profile)})
    return feats


def load_frame(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, np.ndarray]:
    """Parse every recording + REFERENCE.csv into (labelled feature frame, sequence matrix).

    Labels: A -> 1 (AF); N and O -> 0; ~ (noisy) dropped — a physiological signal quality
    exclusion, documented in suite_meta.json. The sequence matrix stacks one `_seq_view` per
    recording, float32 (n, SEQ_LEN), in the SAME row order as the frame — the Dirichlet
    partition applies to both views identically.
    """
    refs = pd.read_csv(raw_dir / "REFERENCE.csv", header=None, names=["record", "ref"])
    refs = refs[refs["ref"] != "~"].reset_index(drop=True)
    rows, seqs = [], []
    for i, rec in enumerate(refs.itertuples(index=False)):
        x = scipy.io.loadmat(raw_dir / f"{rec.record}.mat")["val"].ravel()
        feats = _waveform_features(x)
        feats[LABEL] = int(rec.ref == "A")
        rows.append(feats)
        seqs.append(_seq_view(x.astype("float64")))
        if (i + 1) % 1000 == 0:
            print(f"  parsed {i + 1}/{len(refs)} recordings")
    cols = [*STAT_COLS, *[f"w{i:04d}" for i in range(PROFILE_COLS)], LABEL]
    return pd.DataFrame(rows)[cols], np.stack(seqs)


def write_clinics(
    out_dir: Path | None = None,
    *,
    num_clinics: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    min_positives: int = 5,
) -> dict:
    """Write the pseudo-practice federation + provenance meta (mapper-output convention).

    Same partition acceptance guard as the image track: redraw (seed+1 per attempt) until every
    clinic holds >= `min_positives` AF cases; the accepted draw lands in the meta.
    """
    out_dir = out_dir or HERE.parent / "clinics_ecg"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = HERE.parent / "clinics_ecg_seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    df, seq = load_frame()
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
        "source": "PhysioNet CinC Challenge 2017 v1.0.0 (single-lead ECG, AliveCor Kardia)",
        "label": "afib = REFERENCE 'A' (binary; N/O pooled as 0; '~' noisy dropped)",
        "task": "time series: 300 Hz lead-I ECG, 9-60 s, waveform block profile + spectral stats",
        "n_recordings": len(df),
        "n_dropped_noisy": 8528 - len(df),
        "prevalence": round(float(df[LABEL].mean()), 4),
        "features": {"stats": STAT_COLS, "profile": f"w0000-w{PROFILE_COLS - 1:04d} "
                     f"({PROFILE_RESAMPLE}-pt resample -> {PROFILE_COLS} block means)"},
        "partition": {"method": "dirichlet-label", "alpha": alpha, "seed": seed,
                      "num_clinics": num_clinics},
        "note": "No patient/site keys at this payload level (one recording per subject); "
                "federation is simulated. PhysioNet open access.",
        "seq_view": "data/clinics_ecg_seq/ (same rows, same partition — LSTM input)",
        "clinics": [],
    }
    combined = []
    seq_parts = []
    for k, idx in enumerate(parts):
        g = df.iloc[idx].reset_index(drop=True)
        name = f"shard-{k:02d}"
        g.to_csv(out_dir / f"clinic_{k:02d}_{name}.csv", index=False)
        combined.append(g.assign(clinic_id=k, clinic_name=name))
        np.savez_compressed(
            seq_dir / f"clinic_{k:02d}_{name}.npz",
            X=seq[idx], y=df[LABEL].to_numpy()[idx].astype("int64"),
        )
        seq_parts.append((k, name, idx))
        meta["clinics"].append({"clinic_id": k, "name": name, "n": len(g),
                                "af_rate": round(float(g[LABEL].mean()), 4),
                                "dir": out_dir.name})
    y_all = df[LABEL].to_numpy().astype("int64")
    pd.concat(combined, ignore_index=True).to_csv(out_dir / "all_clinics.csv", index=False)
    np.savez_compressed(seq_dir / "all_clinics.npz", X=seq, y=y_all)
    (out_dir / "suite_meta.json").write_text(json.dumps(meta, indent=2))

    seq_meta = {
        "source": meta["source"],
        "label": meta["label"],
        "task": "time series: fixed-length sequences for the track's RNN arms (LSTM/GRU)",
        "seq_view": {"length": SEQ_LEN, "dtype": "float32",
                     "resample": "normalised-time linear (np.interp) to SEQ_LEN samples",
                     "normalisation": "per-trace z-score (gain/baseline-wander removed; "
                                      "amplitude stats live in the feature view)"},
        "n_recordings": len(df),
        "prevalence": meta["prevalence"],
        "partition": {**meta["partition"],
                      "note": "BYTE-IDENTICAL row partition to data/clinics_ecg/ (same accepted "
                              "draw in one mapper run) — feature and sequence views align row for row"},
        "files": [{"clinic_id": k, "name": name, "file": f"clinic_{k:02d}_{name}.npz",
                   "n": len(idx)} for k, name, idx in seq_parts],
        "note": meta["note"],
    }
    (seq_dir / "suite_meta.json").write_text(json.dumps(seq_meta, indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clinics", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    meta = write_clinics(num_clinics=args.clinics, alpha=args.alpha, seed=args.seed)
    print(f"wrote {meta['n_recordings']} recordings -> {meta['clinics'][0]['dir']} "
          f"({len(meta['clinics'])} clinics, prevalence {meta['prevalence']}, "
          f"dropped noisy {meta['n_dropped_noisy']})")
    for c in meta["clinics"]:
        print(f"  {c['name']}: n={c['n']:>5} af_rate={c['af_rate']:.4f}")


if __name__ == "__main__":
    main()
