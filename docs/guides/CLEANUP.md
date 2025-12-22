# Cleanup roadmap

This project generates artefacts under `outputs/` and caches under `runtime/`.
To keep the repository lean, these directories are **ignored by git** and should
be treated as reproducible build outputs.

## What is generated
- `outputs/reports/`: figures, markdown reports, consolidated metrics.
- `outputs/results/`: experiment runs, baseline metrics, sweeps.
- `outputs/logs/`: structured logs (when enabled).
- `runtime/`: local scratch space for intermediate files.

## Make targets
- `make clean`: remove Python caches + coverage artefacts.
- `make clean-data`: remove processed data files (prompts).
- `make clean-reports`: remove `outputs/reports/` (prompts).
- `make clean-results`: remove `outputs/results/` (prompts).
- `make clean-runtime`: remove `runtime/` (prompts).
- `make clean-outputs`: remove `outputs/` entirely (prompts).

## Notes
- If you want to keep a run, copy the relevant artefacts out of `outputs/`
  before cleaning (e.g., into an external `runs/` archive).

