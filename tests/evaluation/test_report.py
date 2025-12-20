import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from arara_quant.evaluation import (
    AdvancedTearsheetData,
    ReportArtifacts,
    ReportBundle,
    TearsheetFigure,
    build_and_export_report,
    build_report_bundle,
    export_pdf,
    render_html,
)
from arara_quant.evaluation.stats import (
    RiskContributionResult,
    RiskSummary,
    aggregate_risk_metrics,
)
from arara_quant.risk.budgets import RiskBudget


def _bundle_components():
    perf = pd.DataFrame(
        {"strategy": [0.5, 0.4]},
        index=pd.MultiIndex.from_tuples(
            [("performance", "sharpe"), ("risk", "drawdown")],
            names=["category", "metric"],
        ),
    )
    risk_df = pd.DataFrame(
        {"strategy": [0.1]},
        index=pd.MultiIndex.from_tuples([("risk", "volatility")]),
    )
    drawdowns = pd.DataFrame(
        {"strategy": [-0.2, -0.05]}, index=["2024-01-01", "2024-02-01"]
    )
    risk_summary = RiskSummary(
        metrics=risk_df, drawdowns=drawdowns, risk_contribution=None
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    figure = TearsheetFigure(title="Test Figure", figure=fig)
    metadata = {"strategy": "Demo", "start": "2020-01-01"}
    return perf, risk_summary, [figure], metadata


def test_build_report_bundle_normalises_inputs():
    perf, risk_summary, figures, metadata = _bundle_components()
    bundle = build_report_bundle(
        perf, risk_summary, figures, metadata, auto_tearsheet=False
    )
    assert isinstance(bundle, ReportBundle)
    assert bundle.performance.equals(perf)
    assert bundle.risk.equals(risk_summary.metrics)
    assert bundle.drawdowns.equals(risk_summary.drawdowns)
    assert len(bundle.figures) == 1
    assert bundle.risk_contribution is None
    plt.close("all")


def test_build_report_bundle_generates_tearsheet_figures():
    perf, risk_summary, figures, metadata = _bundle_components()
    returns = pd.DataFrame(
        {"strategy": [0.01, -0.005, 0.007]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    cost_breakdown = pd.Series({"slippage": 0.002, "fees": 0.001})
    contrib_index = pd.DatetimeIndex([pd.Timestamp("2024-01-03")])
    risk_contribution = RiskContributionResult(
        component=pd.DataFrame([[0.6]], index=contrib_index, columns=["strategy"]),
        marginal=pd.DataFrame([[0.6]], index=contrib_index, columns=["strategy"]),
        percentage=pd.DataFrame([[1.0]], index=contrib_index, columns=["strategy"]),
        portfolio_volatility=pd.Series(
            [0.1], index=contrib_index, name="portfolio_volatility"
        ),
    )
    risk_summary = RiskSummary(
        metrics=risk_summary.metrics,
        drawdowns=risk_summary.drawdowns,
        risk_contribution=risk_contribution,
    )
    bundle = build_report_bundle(
        perf,
        risk_summary,
        figures,
        metadata,
        returns=returns,
        risk_budgets={"core": ["strategy"]},
        cost_breakdown=cost_breakdown,
    )
    titles = {title for title, _ in bundle.figures}
    assert "Cumulative NAV" in titles
    assert "Drawdown" in titles
    assert "Risk Contribution by Budget" in titles
    assert "Cost Decomposition" in titles
    plt.close("all")


def test_build_report_bundle_uses_metadata_overrides():
    perf, risk_summary, figures, metadata = _bundle_components()
    metadata = {
        **metadata,
        "returns": pd.Series(
            [0.01, -0.004],
            index=pd.date_range("2024-01-01", periods=2),
            name="strategy",
        ),
        "risk_budgets": {"core": ["strategy"]},
    }
    contrib_index = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    risk_contribution = RiskContributionResult(
        component=pd.DataFrame([[0.4]], index=contrib_index, columns=["strategy"]),
        marginal=pd.DataFrame([[0.4]], index=contrib_index, columns=["strategy"]),
        percentage=pd.DataFrame([[1.0]], index=contrib_index, columns=["strategy"]),
        portfolio_volatility=pd.Series(
            [0.1], index=contrib_index, name="portfolio_volatility"
        ),
    )
    risk_summary = RiskSummary(
        metrics=risk_summary.metrics,
        drawdowns=risk_summary.drawdowns,
        risk_contribution=risk_contribution,
    )
    bundle = build_report_bundle(perf, risk_summary, figures, metadata)
    assert bundle.returns is not None
    assert "Cumulative NAV" in {title for title, _ in bundle.figures}
    assert bundle.risk_budgets == {"core": ["strategy"]}
    plt.close("all")


def test_render_html_contains_sections():
    perf, risk_summary, figures, metadata = _bundle_components()
    bundle = build_report_bundle(perf, risk_summary, figures, metadata)
    html = render_html(bundle)
    assert "Performance Metrics" in html
    assert "Risk Metrics" in html
    assert "data:image/png;base64" in html
    plt.close("all")


def test_export_pdf_fallback_to_html(tmp_path):
    html = "<html><body>Hello</body></html>"
    pdf_path = export_pdf(html, tmp_path / "report.pdf", engine="auto")
    assert pdf_path.exists()
    assert pdf_path.suffix in {".pdf", ".html"}
    plt.close("all")


def test_build_and_export_report_creates_files(tmp_path):
    perf, risk_summary, figures, metadata = _bundle_components()
    artifacts = build_and_export_report(
        perf, risk_summary, figures, metadata, tmp_path, filename="demo_report"
    )
    assert isinstance(artifacts, ReportArtifacts)
    assert artifacts.html_path.exists()
    assert artifacts.pdf_path.exists()
    assert artifacts.bundle.metadata["strategy"] == "Demo"
    assert artifacts.table_paths
    for path in artifacts.table_paths.values():
        assert path.exists()
    plt.close("all")


def test_build_and_export_report_respects_auto_toggle(tmp_path):
    perf, risk_summary, figures, metadata = _bundle_components()
    artifacts = build_and_export_report(
        perf,
        risk_summary,
        figures,
        metadata,
        tmp_path,
        filename="no_auto",
        auto_tearsheet=False,
        returns=pd.Series(
            [0.01], index=pd.date_range("2024-01-01", periods=1), name="strategy"
        ),
    )
    assert len(artifacts.bundle.figures) == len(figures)
    plt.close("all")


def test_build_and_export_report_with_advanced_tearsheet(tmp_path):
    dates = pd.date_range("2024-02-01", periods=6, freq="B")
    returns = pd.Series(
        [0.001, -0.0005, 0.0008, 0.0012, -0.0007, 0.0009],
        index=dates,
        name="strategy",
    )

    weights = pd.DataFrame([[0.55, 0.45]], index=[dates[-1]], columns=["EQ", "FI"])
    covariance = pd.DataFrame(
        [[0.03, 0.008], [0.008, 0.018]],
        index=["EQ", "FI"],
        columns=["EQ", "FI"],
    )

    risk_summary = aggregate_risk_metrics(
        returns, weights=weights, covariance=covariance
    )

    performance = pd.DataFrame(
        {"strategy": [returns.mean()]},
        index=pd.MultiIndex.from_tuples(
            [("performance", "mean_return")],
            names=["category", "metric"],
        ),
    )

    metadata = {"strategy": "Advanced Demo"}

    budgets = [
        RiskBudget(name="Equities", tickers=["EQ"]),
        RiskBudget(name="Fixed Income", tickers=["FI"]),
    ]

    cost_breakdown = pd.Series({"linear": 0.0005, "slippage": 0.0002})

    advanced_data = AdvancedTearsheetData(
        returns=returns,
        risk_budgets=budgets,
        cost_breakdown=cost_breakdown,
    )

    artifacts = build_and_export_report(
        performance,
        risk_summary,
        [],
        metadata,
        tmp_path,
        filename="advanced_report",
        advanced_tearsheet=advanced_data,
    )

    assert artifacts.html_path.exists()
    assert artifacts.pdf_path.exists()

    bundle = artifacts.bundle
    titles = {title for title, _ in bundle.figures}
    expected_titles = {
        advanced_data.nav_title,
        advanced_data.drawdown_title,
        advanced_data.risk_budget_title,
        advanced_data.cost_title,
    }
    assert expected_titles <= titles
    assert bundle.risk_contribution is not None
    plt.close("all")
