import pytest
from itau_quant.config.params_default import (
    DEFAULT_PARAMS,
    StrategyParams,
    default_params,
    merge_params,
)


def test_default_params_returns_copy() -> None:
    params = default_params()
    assert params == DEFAULT_PARAMS
    assert params is not DEFAULT_PARAMS


def test_merge_params_allows_overrides() -> None:
    params = merge_params({"lambda_risk": 8.0, "cvar_alpha": 0.1})
    assert params.lambda_risk == 8.0
    assert params.cvar_alpha == pytest.approx(0.1)


def test_merge_params_rejects_unknown_keys() -> None:
    with pytest.raises(KeyError):
        merge_params({"nonexistent": 1})


def test_strategy_params_from_mapping_partial() -> None:
    params = StrategyParams.from_mapping({"eta_turnover": 0.75})
    assert params.eta_turnover == pytest.approx(0.75)
    assert params.lambda_risk == DEFAULT_PARAMS.lambda_risk
