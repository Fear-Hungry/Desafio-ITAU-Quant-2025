"""Save pipeline results as JSON and Markdown.

This module handles persisting pipeline execution results to disk in both
machine-readable (JSON) and human-readable (Markdown) formats, with support
for symlinking the latest run for easy access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["save_results", "generate_markdown"]


def generate_markdown(results: dict[str, Any]) -> str:
    """Generate human-readable Markdown report from results.

    Creates a formatted Markdown document summarizing pipeline execution,
    including metadata, stage results, and key metrics.

    Args:
        results: Pipeline results dictionary with metadata and stages

    Returns:
        Formatted Markdown string

    Examples:
        >>> results = {
        ...     "status": "completed",
        ...     "metadata": {"timestamp": "2025-10-28T12:00:00"},
        ...     "stages": {"data": {"status": "completed"}}
        ... }
        >>> md = generate_markdown(results)
        >>> "# Execução Pipeline ARARA" in md
        True
    """
    metadata = results.get("metadata", {})
    timestamp = metadata.get("timestamp", "unknown")
    config_path = metadata.get("config_path", "unknown")
    status = results.get("status", "unknown")
    duration = results.get("duration_seconds", 0.0)

    # Header
    md = f"""# Execução Pipeline ARARA

**Data:** {timestamp}
**Config:** {config_path}
**Status:** {status}
**Duração:** {duration:.2f}s

"""

    # Executive summary (if backtest available)
    stages = results.get("stages", {})
    if "backtest" in stages and stages["backtest"].get("status") == "completed":
        bt_metrics = stages["backtest"].get("metrics", {})
        if bt_metrics:
            md += """## 📊 Resumo Executivo

### Métricas de Performance
"""
            md += f"- **Retorno Total:** {bt_metrics.get('total_return', 0):.2%}\n"
            md += f"- **Retorno Anualizado:** {bt_metrics.get('annualized_return', 0):.2%}\n"
            md += f"- **Sharpe Ratio:** {bt_metrics.get('sharpe_ratio', 0):.3f}\n"
            md += f"- **Max Drawdown:** {bt_metrics.get('max_drawdown', 0):.2%}\n"
            md += f"- **Volatilidade:** {bt_metrics.get('annualized_volatility', 0):.2%}\n\n"

    # Optimization summary (if available)
    if "optimization" in stages:
        opt = stages["optimization"]
        if opt.get("status") == "completed":
            md += """## 🎯 Otimização

"""
            md += f"- **Ativos Selecionados:** {opt.get('n_assets', 0)}\n"
            md += f"- **Retorno Esperado:** {opt.get('expected_return', 0):.2%}\n"
            md += f"- **Volatilidade:** {opt.get('volatility', 0):.2%}\n"
            md += f"- **Sharpe (ex-ante):** {opt.get('sharpe', 0):.2f}\n"
            md += f"- **Risk Aversion (λ):** {opt.get('risk_aversion', 0):.1f}\n\n"

    # Stages executed
    md += """## 🔄 Etapas Executadas

"""
    for stage_name, stage_data in stages.items():
        status_emoji = {
            "completed": "✅",
            "skipped": "⏭️",
            "failed": "❌",
        }.get(stage_data.get("status", "unknown"), "❓")

        stage_status = stage_data.get("status", "unknown")
        duration_s = stage_data.get("duration_seconds", 0.0)

        md += f"{status_emoji} **{stage_name.title()}**: {stage_status} ({duration_s:.2f}s)\n"

        # Add stage-specific details
        if stage_name == "data" and stage_status == "completed":
            md += f"   - {stage_data.get('n_assets', 0)} ativos, {stage_data.get('n_days', 0)} dias\n"
        elif stage_name == "estimation" and stage_status == "completed":
            shrinkage = stage_data.get("shrinkage")
            if shrinkage is not None:
                md += f"   - Shrinkage: {shrinkage:.3f}, window: {stage_data.get('window_used', 0)}\n"
        elif stage_name == "optimization" and stage_status == "completed":
            md += f"   - {stage_data.get('n_assets', 0)} ativos, Sharpe: {stage_data.get('sharpe', 0):.2f}\n"
        elif stage_name == "backtest" and stage_status == "completed":
            n_notes = stage_data.get("n_notes", 0)
            md += f"   - {n_notes} notas registradas\n"

    # Footer with link to JSON
    timestamp_safe = timestamp.replace(":", "-").replace(".", "-")
    md += f"\n---\n\n[Ver JSON completo](run_{timestamp_safe}.json)\n"

    return md


def save_results(
    results: dict[str, Any],
    output_dir: Path,
    *,
    create_symlink: bool = True,
) -> tuple[Path, Path]:
    """Save pipeline results as JSON and Markdown files.

    Persists the results dictionary to disk in both formats and optionally
    creates symbolic links pointing to the latest run for easy access.

    Args:
        results: Pipeline results dictionary (must contain metadata.timestamp)
        output_dir: Directory to save files (will be created if needed)
        create_symlink: If True, create latest_run.{json,md} symlinks

    Returns:
        Tuple of (json_path, md_path) for the created files

    Raises:
        KeyError: If results missing required metadata
        OSError: If file creation or symlink fails

    Examples:
        >>> results = {
        ...     "status": "completed",
        ...     "metadata": {"timestamp": "2025-10-28T12:00:00"},
        ...     "stages": {}
        ... }
        >>> json_path, md_path = save_results(results, Path("/tmp/reports"))
        >>> json_path.exists()
        True
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract timestamp and sanitize for filename
    timestamp = results["metadata"]["timestamp"]
    timestamp_safe = timestamp.replace(":", "-").replace(".", "-")
    base_name = f"run_{timestamp_safe}"

    # Save JSON
    json_path = output_dir / f"{base_name}.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    # Generate and save Markdown
    md_content = generate_markdown(results)
    md_path = output_dir / f"{base_name}.md"
    md_path.write_text(md_content)

    # Create symlinks to latest run
    if create_symlink:
        latest_json = output_dir / "latest_run.json"
        latest_md = output_dir / "latest_run.md"

        # Remove old symlinks if they exist
        latest_json.unlink(missing_ok=True)
        latest_md.unlink(missing_ok=True)

        # Create new symlinks (relative to output_dir)
        latest_json.symlink_to(json_path.name)
        latest_md.symlink_to(md_path.name)

    return json_path, md_path
