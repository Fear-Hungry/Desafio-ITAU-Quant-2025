"""Public API for the evaluation toolkit (stats, plots, reporting)."""

from . import plots as _plots
from . import stats as _stats
from .plots import *  # noqa: F401,F403 - deliberate re-export
from .stats import *  # noqa: F401,F403

_report = None
try:
    from . import report as _report  # type: ignore[assignment]
    from .report import *  # noqa: F401,F403
except ModuleNotFoundError as exc:
    if exc.name != "cvxpy":
        raise

if _report is None:
    __all__ = sorted(set(_stats.__all__ + _plots.__all__))
else:
    __all__ = sorted(set(_stats.__all__ + _plots.__all__ + _report.__all__))
