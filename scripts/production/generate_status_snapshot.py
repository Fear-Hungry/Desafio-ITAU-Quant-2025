#!/usr/bin/env python
"""
Gerador de Snapshot de Status da Carteira ARARA

Gera automaticamente o arquivo docs/results/CARTEIRA_ARARA_STATUS.md
a partir dos logs de produção.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

# Asset class mappings (expandir conforme necessário)
ASSET_CLASS_MAP = {
    # US Equity
    "SPY": "US Equity",
    "QQQ": "US Equity (Tech)",
    "IWM": "US Equity (Small Cap)",
    "VTV": "US Equity (Value)",
    "VUG": "US Equity (Growth)",
    "VYM": "US Equity (Dividend)",
    "SCHD": "US Equity (Dividend)",
    "SPLV": "US Equity (Low Vol)",
    "USMV": "US Equity (Low Vol)",
    "MTUM": "US Equity (Momentum)",
    "QUAL": "US Equity (Quality)",
    "VLUE": "US Equity (Value)",
    "SIZE": "US Equity (Small Cap)",
    # International Equity
    "EFA": "International Equity",
    "VGK": "International Equity (Europe)",
    "VPL": "International Equity (Pacific)",
    "EEM": "Emerging Markets",
    "VWO": "Emerging Markets",
    # Bonds
    "SHY": "US Treasury 1-3y",
    "IEF": "US Treasury 7-10y",
    "IEI": "US Treasury 3-7y",
    "TLT": "US Treasury 20y+",
    "AGG": "US Aggregate Bonds",
    "VGSH": "US Treasury (curto prazo)",
    "VGIT": "US Treasury (intermediário)",
    "VCSH": "US Corporate Short-Term",
    "BNDX": "International Bonds",
    "LQD": "US Corporate Bonds",
    "HYG": "US High Yield",
    "EMB": "Emerging Market Bonds",
    "EMLC": "EM Local Currency Bonds",
    "TIP": "TIPS (Inflation Protected)",
    # Commodities
    "DBC": "Commodities Broad",
    "USO": "Oil",
    "GLD": "Gold",
    "SLV": "Silver",
    # Crypto
    "IBIT": "Bitcoin ETF",
    "ETHA": "Ethereum ETF",
    "FBTC": "Bitcoin ETF",
    "GBTC": "Bitcoin ETF",
    "ETHE": "Ethereum ETF",
    # FX
    "UUP": "USD Index (FX Hedge)",
    # Cash
    "CASH": "Reserva técnica",
}

# Group definitions (sincronizado com production script)
ASSET_GROUPS = {
    "us_equity": [
        "SPY",
        "QQQ",
        "IWM",
        "VTV",
        "VUG",
        "VYM",
        "SCHD",
        "SPLV",
        "USMV",
        "MTUM",
        "QUAL",
        "VLUE",
        "SIZE",
    ],
    "growth": ["SPY", "QQQ", "VUG", "MTUM"],
    "international": ["EFA", "VGK", "VPL", "EEM", "VWO"],
    "treasuries": ["SHY", "IEF", "IEI", "TLT", "VGIT", "VGSH", "TIP"],
    "corporate_bonds": ["VCSH", "LQD"],
    "all_bonds": [
        "SHY",
        "IEF",
        "IEI",
        "TLT",
        "AGG",
        "VGSH",
        "VGIT",
        "VCSH",
        "BNDX",
        "LQD",
        "HYG",
        "EMB",
        "EMLC",
        "TIP",
    ],
    "commodities": ["DBC", "USO", "GLD", "SLV"],
    "crypto": ["IBIT", "ETHA", "FBTC", "GBTC", "ETHE"],
}


def calculate_group_exposures(weights: pd.Series) -> Dict[str, float]:
    """Calcula exposições por grupo de ativos."""
    exposures = {}
    for group_name, assets in ASSET_GROUPS.items():
        group_weight = weights[[a for a in assets if a in weights.index]].sum()
        exposures[group_name] = group_weight
    return exposures


def generate_status_snapshot(
    output_path: Path = Path("docs/results/CARTEIRA_ARARA_STATUS.md"),
    log_path: Path = Path("results/production/production_log.csv"),
    weights_dir: Path = Path("results/production/weights"),
) -> None:
    """
    Gera snapshot de status da carteira a partir dos logs de produção.

    Parameters
    ----------
    output_path : Path
        Caminho de saída do markdown
    log_path : Path
        Caminho do log CSV de produção
    weights_dir : Path
        Diretório com arquivos de pesos
    """
    print("📄 Gerando snapshot de status...")

    # Read latest log entry
    if not log_path.exists():
        raise FileNotFoundError(f"Log de produção não encontrado: {log_path}")

    log_df = pd.read_csv(log_path)
    if log_df.empty:
        raise ValueError("Log de produção está vazio")

    latest = log_df.iloc[-1]
    log_line_number = len(log_df)

    # Read latest weights
    date_str = latest["date"].replace("-", "")
    weights_file = weights_dir / f"weights_{date_str}.csv"

    if not weights_file.exists():
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_file}")

    weights = pd.read_csv(weights_file, index_col="ticker")["weight"]

    # Calculate metrics
    n_assets = int((weights > 0.001).sum())
    n_effective = float(latest["n_effective"])

    # Calculate group exposures
    exposures = calculate_group_exposures(weights)

    # Build markdown content
    md_lines = []

    # Header
    md_lines.append("# Carteira ARARA – Snapshot de Produção")
    md_lines.append("")
    md_lines.append(f"**Data do rebalance:** {latest['date']}")
    md_lines.append(
        "**Pipeline:** `scripts/production/run_portfolio_production_erc_v2.py`"
    )
    md_lines.append(
        f"**Fontes:** `results/production/production_log.csv` (linha {log_line_number}) e `{weights_file.name}`"
    )
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Metrics table
    md_lines.append("## 📊 Métricas Recentes")
    md_lines.append("")
    md_lines.append(
        "| Métrica (6M ou último rebalance) | Valor | Limite / Observação |"
    )
    md_lines.append(
        "|----------------------------------|-------|---------------------|"
    )

    sharpe = latest["sharpe_6m"]
    vol = latest.get("vol_realized", 0.0)
    if pd.isna(vol) or vol == "":
        vol = 0.0
    max_dd = latest["max_dd"]
    cvar = latest["cvar_95"]
    turnover = latest["turnover_realized"]
    cost_bps = latest["cost_bps"]
    strategy = latest["strategy"]
    fallback = "ativado" if latest["fallback_active"] else "desativado"

    md_lines.append(
        f"| Sharpe Ratio (6M)                | **{sharpe:.2f}** | ≥ 0.80 (conforto) |"
    )
    md_lines.append(
        f"| Volatilidade realizada           | **{vol:.2%}** | Meta: 12% ± 2% |"
    )
    md_lines.append(
        f"| Max Drawdown                     | **{max_dd:.2%}** | Limite: ≥ -15% |"
    )
    md_lines.append(
        f"| CVaR 95%                         | **{cvar:.2%}** | Risco de cauda controlado |"
    )
    md_lines.append(
        f"| N efetivo                        | **{n_effective:.2f}** | Diversificação robusta |"
    )
    md_lines.append(
        f"| Turnover (vs. equal-weight)      | **{turnover:.2%}** | Rebalance realizado |"
    )
    md_lines.append(
        f"| Custo de transação estimado      | **{cost_bps:.2f} bps** | Considerando 15 bps one-way |"
    )
    md_lines.append(
        f"| Estratégia ativa                 | `{strategy}` | Fallback {fallback} |"
    )
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Holdings table
    md_lines.append("## 💰 Alocação Atual")
    md_lines.append("")
    md_lines.append(f"### Top holdings ({n_assets} ativos)")
    md_lines.append("| Ticker | Peso | Classe |")
    md_lines.append("|--------|------|--------|")

    # Sort by weight descending
    sorted_weights = weights.sort_values(ascending=False)
    for ticker, weight in sorted_weights.items():
        if weight < 0.001:  # Skip negligible positions
            continue
        asset_class = ASSET_CLASS_MAP.get(ticker, "Outros")
        md_lines.append(f"| {ticker:6s} | {weight:.2%} | {asset_class} |")

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Asset class exposures
    md_lines.append("## 📈 Exposição por Classe")
    md_lines.append("")
    md_lines.append("| Classe | Exposição | Envelope / Comentário |")
    md_lines.append("|--------|-----------|-----------------------|")

    cash_weight = weights.get("CASH", 0.0)
    md_lines.append(
        f"| Cash | **{cash_weight:.2%}** | Piso normal: 15%, defensivo: 40% |"
    )

    us_equity = exposures.get("us_equity", 0.0)
    md_lines.append(
        f"| US Equity total | **{us_equity:.2%}** | Janela alvo: 10% – 50% |"
    )

    growth = exposures.get("growth", 0.0)
    md_lines.append(f"| Growth (SPY, QQQ, VUG, MTUM) | **{growth:.2%}** | ≥ 5% |")

    intl = exposures.get("international", 0.0)
    md_lines.append(f"| International Equity | **{intl:.2%}** | ≥ 3% |")

    bonds = exposures.get("all_bonds", 0.0)
    md_lines.append(f"| Bonds totais | **{bonds:.2%}** | ≤ 50% |")

    treasuries = exposures.get("treasuries", 0.0)
    md_lines.append(f"| └ Treasuries | {treasuries:.2%} | ≤ 45% |")

    corporate = exposures.get("corporate_bonds", 0.0)
    if corporate > 0:
        md_lines.append(f"| └ Corporate | {corporate:.2%} | dentro do limite geral |")

    commodities = exposures.get("commodities", 0.0)
    crypto = exposures.get("crypto", 0.0)
    md_lines.append(f"| Commodities | **{commodities:.2%}** | ≤ 25% |")
    md_lines.append(f"| Crypto | **{crypto:.2%}** | ≤ 12% |")

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Constraint checks
    md_lines.append("## ✅ Verificações de Constraint")
    md_lines.append("")

    weights_sum = weights.sum()
    md_lines.append(f"- **Soma dos pesos:** {weights_sum:.4f} (incluindo cash)")

    max_pos_ex_cash = weights.drop("CASH", errors="ignore").max()
    md_lines.append(f"- **Max position ex-CASH:** {max_pos_ex_cash:.2%} (limite: 8%)")

    md_lines.append(f"- **Cardinalidade:** {n_assets} ativos")

    md_lines.append(f"- **Cash floor:** {cash_weight:.0%}")

    md_lines.append(f"- **Fallback:** {fallback}")

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Operational notes
    md_lines.append("## 🔍 Observações Operacionais")
    md_lines.append("")

    # Triggers status
    if latest["trigger_sharpe"] or latest["trigger_cvar"] or latest["trigger_dd"]:
        md_lines.append("⚠️ **Triggers ativos:**")
        if latest["trigger_sharpe"]:
            md_lines.append("- Sharpe 6M negativo")
        if latest["trigger_cvar"]:
            md_lines.append("- CVaR abaixo do limite")
        if latest["trigger_dd"]:
            md_lines.append("- Drawdown abaixo do limite")
        md_lines.append("")
    else:
        md_lines.append("✅ Nenhum trigger ativo - sistema operando normalmente")
        md_lines.append("")

    # Volatility check
    if vol > 0:
        if vol < 0.10:
            md_lines.append(
                f"- Volatilidade abaixo do alvo ({vol:.1%} < 12%) pode indicar regime defensivo ou alta alocação em cash/bonds"
            )
        elif vol > 0.14:
            md_lines.append(
                f"- Volatilidade acima do alvo ({vol:.1%} > 12%) - considerar reduzir exposição equity ou aumentar cash floor"
            )

    # Equity allocation check
    if us_equity < 0.10:
        md_lines.append(
            f"- ⚠️ US Equity abaixo do mínimo ({us_equity:.1%} < 10%) - considerar ajustar parâmetros"
        )

    md_lines.append("")
    md_lines.append(
        f"**Última atualização:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    md_lines.append("")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_content = "\n".join(md_lines)
    output_path.write_text(markdown_content, encoding="utf-8")

    print(f"✅ Snapshot gerado: {output_path}")
    print(f"   Data do rebalance: {latest['date']}")
    print(f"   Estratégia: {strategy}")
    print(f"   N ativos: {n_assets}")
    print(f"   N efetivo: {n_effective:.1f}")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = Path("docs/results/CARTEIRA_ARARA_STATUS.md")

    try:
        generate_status_snapshot(output_path=output_path)
    except Exception as e:
        print(f"❌ Erro ao gerar snapshot: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
