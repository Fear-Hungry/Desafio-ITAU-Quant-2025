import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from itau_quant.evaluation import (
    ReportArtifacts,
    ReportBundle,
    TearsheetFigure,
    build_and_export_report,
    build_report_bundle,
    export_pdf,
    render_html,
)
from itau_quant.evaluation.stats import RiskSummary


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

