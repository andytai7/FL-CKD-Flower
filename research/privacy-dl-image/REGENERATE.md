# REGENERATE — image track results index

Every file in `results/` is a local, gitignored artifact. This index maps each to the
exact regenerator command (run from the repo root on `experiment/image`; `uv run --extra dl
python -m research.privacy_dl_image.<module>`). Runtime estimates measured on this box
(OMP_NUM_THREADS=4, 2026-09-05 runs). Run the matrix emitter LAST, after its inputs exist.

| Artifact | Regenerator (module) | Runtime (measured) |
|---|---|---|
| `dl_image_p1.json` — seed-42 full ε grid, global clip + MIA per cell | `p1_image` | ~17 min (7 cells incl. MIA) |
| `dl_image_p1_seeds.json` — dispersion sweep {off,1,4} × seeds 43-46 | `sweep_p1_seeds` (calls `run_cell` directly, owns its `_seeds` envelope — see rule below) | ~2.5 h (12 cells) |
| `dl_image_p1_layerwise.json` — per-layer clip arm + filter-health traces | `p1_layerwise` (resume-safe) | ~17 min |
| `dl_image_p2.json` — analytic cost (incl. CNN-M 2.76M wire) + measured cross-ref | `p2_secagg` | seconds |
| `dl_image_p3.json` — FedCT Gaussian-vote grid | `fedct_image` | ~45 min (teachers per ε row) |
| `dl_image_p3_rr.json` — randomized-response votes + gaussian replication verdict | `p3_rr` (resume-safe per seed) | ~10 min |
| `dl_image_p4.json` — verified-hybrid grid + norm proofs | `p4_spine` | ~1 h |
| `dl_image_matrix.json` — THE schema-locked matrix | `matrix` | seconds |

Writer-discipline rule (the incident this guard is for): **`sweep_p1_seeds.py` must never
call `p1_image.run_grid`** — run_grid checkpoints the seed-42 grid file per cell, and on
2026-09-04 that clobbered `dl_image_p1.json` to four sweep rows (matrix + notebook rebuilt
on top of the clobber until a full rerun restored it 2026-09-05; redemption sealed in the
sweep module's docstring). Any future helper that writes results/ gets a checkpoint of ITS
OWN file or none.

Measured P2 row provenance: the shared deployment probe ran on the TS branch
(`research/privacy_dl_ts/p2fab/`); the image P2 file cross-references it — there is no
image-side regenerator for that evidence row.
