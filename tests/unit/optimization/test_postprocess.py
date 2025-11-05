from __future__ import annotations

import numpy as np
import pandas as pd
from arara_quant.optimization.core import postprocess as post_mod


def test_project_to_simplex_returns_non_negative_and_unit_sum():
    weights = pd.Series([-0.2, 0.3, 0.7], index=["AAA", "BBB", "CCC"])
    projected = post_mod.project_to_simplex(weights)
    assert np.isclose(projected.sum(), 1.0)
    assert (projected >= -1e-12).all()


def test_clip_to_bounds_respects_limits():
    weights = pd.Series([0.6, 0.4], index=["AAA", "BBB"])
    lower = pd.Series([0.2, 0.1], index=weights.index)
    upper = [0.7, 0.6]
    clipped = post_mod.clip_to_bounds(weights, lower, upper)
    assert (clipped >= lower - 1e-12).all()
    assert (clipped <= np.array(upper) + 1e-12).all()
    assert np.isclose(clipped.sum(), 1.0)


def test_enforce_cardinality_trims_to_top_assets():
    weights = pd.Series([0.5, 0.3, 0.2], index=["AAA", "BBB", "CCC"])
    pruned = post_mod.enforce_cardinality(weights, k=2)
    assert (pruned.index[pruned > 0.0].tolist()) == ["AAA", "BBB"]
    assert np.isclose(pruned.sum(), 1.0)


def test_round_to_lots_with_scalar_lot_size():
    weights = pd.Series([0.33, 0.34, 0.33], index=["AAA", "BBB", "CCC"])
    rounded = post_mod.round_to_lots(weights, lot_size=0.1)
    tolerance = 0.1
    assert abs(rounded.sum() - 1.0) <= tolerance
    multiples = rounded / 0.1
    assert np.allclose(multiples, np.round(multiples), atol=1e-8)


def test_postprocess_pipeline_combines_steps():
    weights = pd.Series([0.6, 0.25, 0.15], index=["AAA", "BBB", "CCC"])
    config = {
        "enforce_cardinality": {"k": 2},
        "round_lots": {"lot_size": 0.05},
    }
    processed = post_mod.postprocess_pipeline(weights, config)
    assert processed.gt(0).sum() <= 2
    assert np.isclose(processed.sum(), 1.0)
