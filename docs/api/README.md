# API Reference Overview

This folder groups curated entry points for contributors who prefer to browse
the package directly instead of reading through notebooks or runners. The
recommendation is to start from the high-level modules listed below and follow
their docstrings for deeper dives.

- `arara_quant.data`: ingestion facade (`DataLoader`, `DataBundle`) and helpers
  to manipulate cached artefacts in `data/raw` and `data/processed`.
- `arara_quant.estimators`: robust estimators for μ/Σ, including Huber means and
  Ledoit–Wolf shrinkage wrappers.
- `arara_quant.optimization`: the convex solvers (`MeanVarianceConfig`,
  `solve_mean_variance`) alongside the risk constraints toolkit.
- `arara_quant.backtesting`: walk-forward engine, metrics and ledger utilities.
- `arara_quant.portfolio`: execution layer that turns optimizer results into
  rebalancing instructions.

When editing code, keep docstrings descriptive—`pdoc` or `sphinx-apidoc` can be
plugged in later to auto-generate HTML using the structure defined here.
