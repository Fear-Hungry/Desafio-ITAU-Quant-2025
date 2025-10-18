"""Public API for the evaluation toolkit (stats, plots, reporting)."""

from . import plots as _plots
from . import report as _report
from . import stats as _stats
from .plots import *  # noqa: F401,F403 - deliberate re-export
from .report import *  # noqa: F401,F403
from .stats import *  # noqa: F401,F403

__all__ = sorted(set(_stats.__all__ + _plots.__all__ + _report.__all__))
