#!/usr/bin/env python
"""
Production Logger - Logging estruturado para produção

Registra todos os eventos de rebalance, métricas, triggers e custos.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass
class RebalanceLog:
    """Log de um evento de rebalance"""

    date: str
    strategy: str  # "ERC" ou "1/N"
    turnover_realized: float
    cost_bps: float
    n_active_assets: int
    n_effective: float
    sharpe_6m: float
    cvar_95: float
    max_dd: float
    trigger_status: Dict[str, bool]
    fallback_active: bool
    vol_realized: Optional[float] = None
    regime: Optional[str] = None  # Market regime (calm, neutral, stressed, crash)
    lambda_adjusted: Optional[float] = None  # Risk aversion after regime adjustment
    defensive_mode: Optional[str] = None  # Defensive mode (normal, defensive, critical)

    def to_dict(self) -> Dict:
        return asdict(self)


class ProductionLogger:
    """Logger estruturado para ambiente de produção"""

    def __init__(self, log_dir: Path = Path("outputs/results/production")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "production_log.csv"
        self.weights_dir = self.log_dir / "weights"
        self.weights_dir.mkdir(exist_ok=True)

        # Inicializar arquivo de log se não existir
        if not self.log_file.exists():
            self._initialize_log()

    def _initialize_log(self):
        """Cria header do CSV de log"""
        headers = [
            "date",
            "strategy",
            "turnover_realized",
            "cost_bps",
            "n_active_assets",
            "n_effective",
            "sharpe_6m",
            "cvar_95",
            "max_dd",
            "fallback_active",
            "vol_realized",
            "trigger_sharpe",
            "trigger_cvar",
            "trigger_dd",
            "regime",
            "lambda_adjusted",
            "defensive_mode",
        ]
        pd.DataFrame(columns=headers).to_csv(self.log_file, index=False)

    def log_rebalance(
        self,
        date: datetime,
        weights: pd.Series,
        strategy: str,
        turnover_realized: float,
        cost_bps: float,
        metrics: Dict,
        trigger_status: Dict[str, bool],
        fallback_active: bool,
        force_log: bool = False,
        regime: Optional[str] = None,
        lambda_adjusted: Optional[float] = None,
        defensive_mode: Optional[str] = None,
    ):
        """
        Registra um evento de rebalance.

        Parameters
        ----------
        date : datetime
            Data do rebalance
        weights : pd.Series
            Pesos do portfolio
        strategy : str
            Estratégia ativa ("ERC" ou "1/N")
        turnover_realized : float
            Turnover realizado (ex: 0.15 = 15%)
        cost_bps : float
            Custo em bps (ex: 45.0 = 45 bps)
        metrics : Dict
            Métricas de performance (sharpe_6m, cvar_95, max_dd, vol)
        trigger_status : Dict[str, bool]
            Status dos triggers
        fallback_active : bool
            Se fallback para 1/N está ativo
        force_log : bool, optional
            Se True, força log mesmo se for duplicata (default: False)
        """
        # Calcular métricas de portfolio
        n_active = (weights > 0.001).sum()
        herfindahl = (weights**2).sum()
        n_effective = 1.0 / herfindahl if herfindahl > 0 else 0.0

        # Criar log entry
        log_entry = RebalanceLog(
            date=date.strftime("%Y-%m-%d"),
            strategy=strategy,
            turnover_realized=turnover_realized,
            cost_bps=cost_bps,
            n_active_assets=n_active,
            n_effective=n_effective,
            sharpe_6m=metrics.get("sharpe_6m", 0.0),
            cvar_95=metrics.get("cvar_95", 0.0),
            max_dd=metrics.get("max_dd", 0.0),
            trigger_status=trigger_status,
            fallback_active=fallback_active,
            vol_realized=metrics.get("vol", None),
            regime=regime,
            lambda_adjusted=lambda_adjusted,
            defensive_mode=defensive_mode,
        )

        # Build log dictionary
        log_dict = {
            "date": log_entry.date,
            "strategy": log_entry.strategy,
            "turnover_realized": log_entry.turnover_realized,
            "cost_bps": log_entry.cost_bps,
            "n_active_assets": log_entry.n_active_assets,
            "n_effective": log_entry.n_effective,
            "sharpe_6m": log_entry.sharpe_6m,
            "cvar_95": log_entry.cvar_95,
            "max_dd": log_entry.max_dd,
            "fallback_active": log_entry.fallback_active,
            "vol_realized": (
                log_entry.vol_realized if log_entry.vol_realized is not None else ""
            ),
            "trigger_sharpe": trigger_status.get("sharpe_6m_negative", False),
            "trigger_cvar": trigger_status.get("cvar_breach", False),
            "trigger_dd": trigger_status.get("drawdown_breach", False),
            "regime": log_entry.regime if log_entry.regime is not None else "",
            "lambda_adjusted": (
                log_entry.lambda_adjusted if log_entry.lambda_adjusted is not None else ""
            ),
            "defensive_mode": (
                log_entry.defensive_mode if log_entry.defensive_mode is not None else "normal"
            ),
        }

        # Deduplicate: check if this is identical to last entry
        if not force_log and self.log_file.exists():
            try:
                existing_df = pd.read_csv(self.log_file)
                if not existing_df.empty:
                    last_row = existing_df.iloc[-1].to_dict()

                    # Compare key fields (ignore minor floating point differences)
                    keys_to_compare = [
                        "date",
                        "strategy",
                        "n_active_assets",
                        "fallback_active",
                        "trigger_sharpe",
                        "trigger_cvar",
                        "trigger_dd",
                    ]
                    float_keys_to_compare = [
                        "turnover_realized",
                        "cost_bps",
                        "sharpe_6m",
                        "cvar_95",
                        "max_dd",
                    ]

                    # Check exact matches for categorical fields
                    exact_match = all(
                        log_dict.get(k) == last_row.get(k) for k in keys_to_compare
                    )

                    # Check floating point fields with tolerance
                    float_match = all(
                        abs(float(log_dict.get(k, 0)) - float(last_row.get(k, 0)))
                        < 1e-9
                        for k in float_keys_to_compare
                    )

                    if exact_match and float_match:
                        print(
                            f"⏭️  Rebalance {log_dict['date']} já registrado (pulando duplicata)"
                        )
                        return
            except Exception as e:
                # If deduplication fails, log anyway (safer to have duplicate than miss a log)
                print(f"⚠️  Erro ao verificar duplicatas (continuando): {e}")

        # Salvar pesos
        weights_file = self.weights_dir / f"weights_{date.strftime('%Y%m%d')}.csv"
        weights_df = weights.to_frame(name="weight")
        weights_df.index.name = "ticker"
        weights_df.to_csv(weights_file)

        # Append ao log CSV
        log_df = pd.DataFrame([log_dict])
        log_df.to_csv(self.log_file, mode="a", header=False, index=False)

        print(f"✅ Rebalance registrado: {log_entry.date}")
        print(f"   Estratégia: {strategy}")
        print(f"   Turnover: {turnover_realized:.2%}")
        print(f"   Custo: {cost_bps:.1f} bps")
        print(f"   N_effective: {n_effective:.1f}")
        if fallback_active:
            print("   ⚠️  FALLBACK ATIVO")

    def get_log_history(self) -> pd.DataFrame:
        """Retorna histórico completo de logs"""
        if self.log_file.exists():
            return pd.read_csv(self.log_file, parse_dates=["date"])
        return pd.DataFrame()

    def get_recent_weights(self, n: int = 5) -> Dict[str, pd.Series]:
        """Retorna os N pesos mais recentes"""
        weight_files = sorted(self.weights_dir.glob("weights_*.csv"))
        recent = weight_files[-n:] if len(weight_files) >= n else weight_files

        weights_dict = {}
        for f in recent:
            date_str = f.stem.replace("weights_", "")
            weights = pd.read_csv(f, index_col="ticker")["weight"]
            weights_dict[date_str] = weights

        return weights_dict

    def print_summary(self, last_n: int = 10):
        """Imprime resumo dos últimos N rebalances"""
        log_df = self.get_log_history()

        if log_df.empty:
            print("Nenhum log registrado ainda.")
            return

        print("=" * 80)
        print(f"  RESUMO DE PRODUÇÃO (últimos {last_n} rebalances)")
        print("=" * 80)
        print()

        recent = log_df.tail(last_n)

        print(
            "Data       | Estratégia | Turnover | Custo | N_eff | Sharpe6M | Fallback"
        )
        print("-" * 80)
        for _, row in recent.iterrows():
            fallback_icon = "⚠️ " if row["fallback_active"] else "  "
            print(
                f"{row['date']} | {row['strategy']:10s} | "
                f"{row['turnover_realized']:7.2%} | {row['cost_bps']:5.1f} | "
                f"{row['n_effective']:5.1f} | {row['sharpe_6m']:8.2f} | {fallback_icon}"
            )

        print()
        print("Estatísticas Gerais:")
        print(f"   Turnover médio: {recent['turnover_realized'].mean():.2%}")
        print(f"   Custo médio: {recent['cost_bps'].mean():.1f} bps")
        print(
            f"   Fallbacks ativados: {recent['fallback_active'].sum()} de {len(recent)}"
        )


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    import numpy as np

    print("Testando ProductionLogger...")
    print()

    # Criar logger de teste
    logger = ProductionLogger(log_dir=Path("outputs/results/production_test"))

    # Simular alguns rebalances
    tickers = ["SPY", "QQQ", "IEF", "GLD", "EMB"]

    for i in range(5):
        date = datetime(2025, 10, 1 + i * 7)

        # Pesos aleatórios
        raw_weights = np.random.dirichlet([1] * len(tickers))
        weights = pd.Series(raw_weights, index=tickers)

        # Métricas simuladas
        metrics = {
            "sharpe_6m": np.random.uniform(0.5, 1.5),
            "cvar_95": np.random.uniform(-0.03, -0.01),
            "max_dd": np.random.uniform(-0.15, -0.05),
            "vol": np.random.uniform(0.08, 0.12),
        }

        # Triggers simulados
        triggers = {
            "sharpe_6m_negative": False,
            "cvar_breach": False,
            "drawdown_breach": False,
        }

        logger.log_rebalance(
            date=date,
            weights=weights,
            strategy="ERC",
            turnover_realized=np.random.uniform(0.05, 0.15),
            cost_bps=np.random.uniform(20, 50),
            metrics=metrics,
            trigger_status=triggers,
            fallback_active=False,
        )
        print()

    # Imprimir resumo
    logger.print_summary()

    print()
    print("✅ Teste concluído! Verifique outputs/results/production_test/")
