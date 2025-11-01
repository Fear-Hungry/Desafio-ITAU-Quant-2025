import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from itau_quant.evaluation import (
    AdvancedTearsheetData,
    ReportArtifacts,
    ReportBundle,
    TearsheetFigure,
    build_and_export_report,
    build_report_bundle,
    export_pdf,
    render_html,
)
from itau_quant.evaluation.stats import RiskSummary, aggregate_risk_metrics
from itau_quant.risk.budgets import RiskBudget


def _bundle_components():
    perf = pd.DataFrame({"strategy": [0.5, 0.4]}, index=pd.MultiIndex.from_tuples([("performance", "sharpe"), ("risk", "drawdown")] , names=["category", "metric"]))
    risk_df = pd.DataFrame({"strategy": [0.1]}, index=pd.MultiIndex.from_tuples([("risk", "volatility")]))
    drawdowns = pd.DataFrame({"strategy": [-0.2, -0.05]}, index=["2024-01-01", "2024-02-01"])
    risk_summary = RiskSummary(metrics=risk_df, drawdowns=drawdowns, risk_contribution=None)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    figure = TearsheetFigure(title="Test Figure", figure=fig)
    metadata = {"strategy": "Demo", "start": "2020-01-01"}
    return perf, risk_summary, [figure], metadata


def test_build_report_bundle_normalises_inputs():
    perf, risk_summary, figures, metadata = _bundle_components()
    bundle = build_report_bundle(perf, risk_summary, figures, metadata)
    assert isinstance(bundle, ReportBundle)
    assert bundle.performance.equals(perf)
    assert bundle.risk.equals(risk_summary.metrics)
    assert bundle.drawdowns.equals(risk_summary.drawdowns)
    assert len(bundle.figures) == 1
    assert bundle.risk_contribution is None


def test_render_html_contains_sections():
    perf, risk_summary, figures, metadata = _bundle_components()
    bundle = build_report_bundle(perf, risk_summary, figures, metadata)
    html = render_html(bundle)
    assert "Performance Metrics" in html
    assert "Risk Metrics" in html
    assert "data:image/png;base64" in html


def test_export_pdf_fallback_to_html(tmp_path):
    html = "<html><body>Hello</body></html>"
    pdf_path = export_pdf(html, tmp_path / "report.pdf", engine="auto")
    assert pdf_path.exists()
    assert pdf_path.suffix in {".pdf", ".html"}


def test_build_and_export_report_creates_files(tmp_path):
    perf, risk_summary, figures, metadata = _bundle_components()
    artifacts = build_and_export_report(perf, risk_summary, figures, metadata, tmp_path, filename="demo_report")
    assert isinstance(artifacts, ReportArtifacts)
    assert artifacts.html_path.exists()
    assert artifacts.pdf_path.exists()
    assert artifacts.bundle.metadata["strategy"] == "Demo"


def test_build_and_export_report_with_advanced_tearsheet(tmp_path):
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    returns = pd.Series([0.002, -0.001, 0.0015, 0.0007, -0.0005, 0.0023, -0.0012, 0.0012], index=dates, name="strategy")

    weights = pd.DataFrame([[0.6, 0.4], [0.58, 0.42]], index=[dates[-2], dates[-1]], columns=["EQ", "FI"])
    covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.02]], index=["EQ", "FI"], columns=["EQ", "FI"])

    risk_summary = aggregate_risk_metrics(returns, weights=weights, covariance=covariance)

    performance = pd.DataFrame(
        {"strategy": [returns.mean()]},
        index=pd.MultiIndex.from_tuples([("performance", "mean_return")], names=["category", "metric"]),
    )

    metadata = {"strategy": "Advanced Demo"}

    budgets = [
        RiskBudget(name="Equities", tickers=["EQ"]),
        RiskBudget(name="Fixed Income", tickers=["FI"]),
    ]

    cost_breakdown = pd.Series({"linear": 0.0008, "slippage": 0.0003}, name="cost")

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
