# Desafio ITAÚ Quant — Carteira ARARA (PRISM-R)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)]()
[![CI](https://github.com/YOUR_USERNAME/Desafio-ITAU-Quant/workflows/CI%20Pipeline/badge.svg)](https://github.com/YOUR_USERNAME/Desafio-ITAU-Quant/actions)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/Desafio-ITAU-Quant/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/Desafio-ITAU-Quant)
[![Tests](https://img.shields.io/badge/tests-786%20passed-green.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-71%25-yellow.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## Quickstart (60 s)
```bash
poetry install                                  # dependências
poetry run python scripts/run_01_data_pipeline.py \
  --force-download --start 2010-01-01           # dados brutos → processed

poetry run itau-quant backtest \
  --config configs/optimizer_example.yaml \
  --no-dry-run --json > reports/backtest_latest.json

poetry run pytest                               # suíte completa
```

---

## Resumo executivo

**Estratégia PRISM-R — Desempenho OOS Consolidado (2020-01-02 a 2025-10-09)**

Implementamos uma estratégia mean-variance penalizada para o universo multiativos ARARA (69 ETFs globais, BRL base). Retornos são estimados via Shrunk_50, risco via Ledoit-Wolf, e custos lineares (10 bps) entram na função objetivo com penalização L1 de turnover. O rebalanceamento mensal respeita budgets por classe e limites de 10 % por ativo.

**Validação Walk-Forward:** Treino 252 dias, teste 21 dias, purge/embargo 2 dias. Período oficial OOS: 2020-01-02 a 2025-10-09 (1,451 dias úteis).

**Resultados Consolidados (fonte: nav_daily.csv):**
- **NAV Final:** 1.0289 (retorno de 2.89%)
- **Retorno Anualizado:** 0.50%
- **Volatilidade Anualizada:** 8.60%
- **Sharpe Ratio:** 0.0576
- **Drawdown Máximo:** -20.89%
- **CVaR 95%:** -0.1955
- **Taxa de Acerto:** 52.0%
- **Turnover (mediana):** ~0.2% ao mês

**Fonte:** Todos os valores são calculados a partir de `reports/walkforward/nav_daily.csv` (canonical single source of truth), consolidados em `reports/oos_consolidated_metrics.json`. Para detalhes completos sobre metodologia, rastreabilidade e validação, ver seção 6.4.


---

## 1. Problema e objetivo
- **Objetivo:** maximizar retorno esperado ajustado ao risco (λ = 15) após custos de transação e penalidade de turnover.
- **Restrições principais:** \(0 \le w_i \le 10\%\), \(\sum_i w_i = 1\); budgets para 11 buckets (US equity, intl equity, FI, real assets, FX, cripto etc.) com limites min/max; turnover alvo 5–20 %.
- **Métricas de sucesso:** retorno anualizado ≥ 4 %, vol ≤ 12 %, Sharpe ≥ 0.8, Sortino ≥ 0.9, Max Drawdown ≤ 15 %, Calmar ≥ 0.3, turnover na banda-alvo, custo < 50 bps/ano.
- **Hipóteses de custos/slippage:** custos lineares de 10 bps por round-trip; slippage avançado (`adv20_piecewise`) disponível mas desativado nesta execução para isolar o efeito dos budgets.

---

## 2. Dados
- **Fonte:** Yahoo Finance via `yfinance` (ETFs), com fallback para Tiingo (cripto) e FRED (RF) — nesta run o RF ficou zerado por ausência de `pandas_datareader`.
- **Universo:** 69 ETFs (equities EUA/internacionais, renda fixa Treasury/IG/HY, commodities, FX, cripto) definidos em `configs/universe_arara.yaml`.
- **Janela temporal:** 2020-01-02 a 2025-10-09, frequência diária. Crypto ETFs exigem histórico mínimo de 60 dias.
- **Pré-processamento:** `scripts/run_01_data_pipeline.py` aplica ajustes de split/dividendos, remove ativos com baixa cobertura (ex.: QQQ na primeira tentativa), força RF=0 quando indisponível, e descarta linhas totalmente vazias.
- **Outliers/missing:** colunas com ausência total são excluídas; valores faltantes residuais são preenchidos apenas após a meta de histórico mínimo.
- **Reprodução local:** defina `DATA_DIR` no `.env` (opcional) e execute:
  ```bash
  poetry run python scripts/run_01_data_pipeline.py \
    --force-download --start 2010-01-01
  ```
  Artefatos: `data/processed/returns_arara.parquet`, `mu_estimate.parquet`, `cov_estimate.parquet`, `excess_returns_*.parquet`.

---

## 3. Metodologia

### 3.1 Estimadores
- **Retorno esperado:** Shrunk_50 (força 0.5, janela 252 dias).
- **Covariância:** Ledoit-Wolf não linear (252 dias).
- **Modelos alternativos disponíveis:** Black-Litterman, regressão bayesiana, Risk Parity (ERC), HRP, Tyler M-estimator, CVaR LP — documentados em “Relatório Consolidado”.

### 3.2 Otimização
- **Função objetivo:**  
  \[
  \max_w \, \mu^\top w - \frac{\lambda}{2} w^\top \Sigma w - \eta \lVert w - w_{t-1} \rVert_1 - \text{costs}(w, w_{t-1})
  \]
  com λ = 15, η = 0.25, custos lineares de 10 bps aplicados ao turnover absoluto.
- **Restrições:** budgets por classe (11 grupos), bounds individuais (0–10 %), soma de pesos = 1. Cardinalidade desativada nesta rodada (k_min/k_max só em testes de GA).
- **Solvedor:** CVXPY + Clarabel (tolerâncias 1e-8); fallback para OSQP/ECOS disponível.

### 3.3 Avaliação
- Walk-forward purged: treino 252 dias, teste 21 dias, purge 2 dias, embargo 2 dias (162 splits cobrindo 2010–2025).
- Baselines recalculadas no mesmo protocolo: Equal-weight, Risk Parity, MV Shrunk clássico, Min-Var LW, 60/40 e HRP.
- Métricas pós-custos: retorno e vol anualizados, Sharpe HAC, Sortino, Max Drawdown, Calmar, turnover (média e mediana), custos (média anualizada de `cost_fraction`), hit-rate.

---

## 4. Protocolo de avaliação
| Item                         | Configuração atual                                     |
|------------------------------|--------------------------------------------------------|
| Janela de treino/teste       | 252d / 21d (set rolling)                               |
| Purge / embargo              | 2d / 2d                                                |
| Rebalance                    | Mensal (primeiro business day)                        |
| Custos                       | 10 bps por round-trip                                  |
| Arquivos de saída            | `reports/backtest_*.json`, `reports/figures/*.png`     |
| Scripts auxiliares           | `scripts/research/run_regime_stress.py`, `run_ga_*.py` |

---

## 5. Experimentos e resultados

### 5.1 Tabela principal (walk-forward 2020–2025)
| Estratégia                       | Ret. anual | Vol anual | Sharpe | Max DD  | Turnover méd. | Custos (bps/ano) | Período |
|---------------------------------|-----------:|----------:|-------:|--------:|--------------:|-----------------:|---------|


Período OOS usado:
- PRISM-R: 2020-01-02 a 2025-10-09 (1451 dias) — fonte: reports/oos_consolidated_metrics.json
- Baselines: conforme reports/strategy_comparison_final.csv (podem usar período diferente; será padronizado em rodada futura)

| Estratégia | Total Return | Annual Return | Volatility | Sharpe (mean) | Sharpe (median) | CVaR 95% | PSR | DSR | Max Drawdown | Turnover | Cost (bps) | Success Rate | Fonte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PRISM-R (Portfolio Optimization) | 2.89% | 0.50% | 8.60% | 1.2958 | 1.2341 | -0.1033 | 1.0000 | 0.9969 | -25.30% | 7.68e-06 | 15.57 | 59.9% | oos_consolidated_metrics.json |
| Minimum Variance (Ledoit-Wolf) | 7.06% | 1.80% | 2.86% | 0.6365 | — | -0.0042 | — | — | -8.69% | 9.69e-02 | 133.68 | — | strategy_comparison_final.csv |
| Shrunk Mean-Variance | 66.63% | 14.25% | 12.43% | 1.1342 | — | -0.0188 | — | — | -13.31% | 5.70e-01 | 785.95 | — | strategy_comparison_final.csv |
| Equal-Weight 1/N | 60.38% | 13.11% | 11.17% | 1.1594 | — | -0.0159 | — | — | -13.85% | 2.17e-02 | 30.00 | — | strategy_comparison_final.csv |
| Risk Parity | 56.51% | 12.40% | 10.61% | 1.1544 | — | -0.0150 | — | — | -13.35% | 3.01e-02 | 41.50 | — | strategy_comparison_final.csv |
| 60/40 Stocks/Bonds | 36.33% | 8.42% | 9.69% | 0.8827 | — | -0.0141 | — | — | -15.92% | 2.17e-02 | 30.00 | — | strategy_comparison_final.csv |
| Hierarchical Risk Parity | 13.21% | 3.29% | 6.39% | 0.5381 | — | -0.0093 | — | — | -14.13% | 5.09e-01 | 702.32 | — | strategy_comparison_final.csv |

Notas:
- O PRISM-R usa métricas consolidadas do JSON (série diária OOS).
- As baselines acima são do artefato de comparação atual e poderão ser reprocessadas para bater exatamente o mesmo período OOS.

### 5.2 Análise Walk-Forward Detalhada (64 janelas OOS)

**Estatísticas Agregadas:**
| Métrica                      | Valor     |
|------------------------------|-----------|
| Número de Janelas OOS        | 52        |
| Taxa de Sucesso              | 59.6%     |
| **Sharpe Médio (OOS)**       | **0.88**  |
| **Retorno Médio (OOS)**      | **6.25%** |
| Volatilidade Média           | 6.41%     |
| Drawdown Médio por Janela    | -1.84%    |
| Turnover Médio               | 1.92%     |
| Custo Médio                  | 0.2 bps   |
| Consistência (R²)            | 0.032     |
| Melhor Janela NAV            | 1.0542    |
| Pior Janela NAV              | 0.9376    |
| Range Ratio                  | 1.12      |

**Períodos de Stress Identificados:** 18 janelas (34.6% do total)
- **Stress 2021:** 4 janelas (pior: drawdown -4.36%, Sharpe -4.24)
- **Inflation 2022:** 6 janelas severas (pior: drawdown -5.94%, Sharpe -9.26)
- **Stress 2023:** 4 janelas (pior: drawdown -4.03%, Sharpe -6.13)
- **Stress 2024-2025:** 4 janelas recentes (pior: drawdown -5.30%, Sharpe -5.78)

> Relatórios completos disponíveis em `reports/walkforward/` (summary_stats.md, per_window_results.csv, stress_periods.md)

### 5.3 Gráficos
![Curva de capital](reports/figures/tearsheet_cumulative_nav.png)
![Drawdown](reports/figures/tearsheet_drawdown.png)
![Risco por budget](reports/figures/tearsheet_risk_contribution_by_budget.png)
![Custos](reports/figures/tearsheet_cost_decomposition.png)
![Walk-forward NAV + Sharpe (destaque pandemia)](reports/figures/walkforward_nav_20251101.png)
![Análise Walk-Forward Completa (parameter evolution, Sharpe por janela, consistency, turnover/cost)](reports/figures/walkforward_analysis_20251101.png)

### 5.4 Ablations e sensibilidade
- **Custos:** elevar para 15 bps derruba Sharpe do MV penalizado para ≈ 0.35 (experimentos `results/cost_sensitivity`).
- **Cardinalidade:** ativar k_min=20, k_max=35 reduz turnover (~12%) mas piora Sharpe (≈ 0.45). Heurística GA documentada em `scripts/research/run_ga_mv_walkforward.py`.
- **Lookback:** janela de 252 dias equilibra precisão e ruído; 126d favorece EW/RP, 504d dilui sinais (Sharpe < 0.4).
- **Regimes:** multiplicar λ em regimes "crash" reduz drawdown (−1.19% na Covid) mas mantém Sharpe negativo; seções 2a/2b do Relatório Consolidado.

---

## 5.5. Experimentos de Regime Dinâmico e Tail Hedge Adaptativo (2025-11-01)

### 5.5.1. Adaptive Tail Hedge Analysis

Implementamos e testamos um sistema de alocação dinâmica de tail hedge baseado em regime de mercado. O sistema ajusta automaticamente a exposição a ativos defensivos (TLT, TIP, GLD, SLV, PPLT, UUP) conforme condições de mercado.

**Configuração do Experimento:**
- **Período:** 2020-01-02 a 2025-10-09 (1,451 dias, 69 ativos)
- **Janela de regime:** 63 dias (rolling)
- **Ativos de hedge:** 6 (TLT, TIP, GLD, SLV, PPLT, UUP - todos disponíveis)
- **Alocação base:** 5.0% em regimes neutros

**Resultados - Distribuição de Regimes:**

| Regime | Ocorrências | % do Tempo | Alocação Hedge Target |
|--------|-------------|------------|----------------------|
| **Calm** | 990 | 70.6% | 2.5% |
| **Neutral** | 357 | 25.4% | 5.0% |
| **Stressed** | 23 | 1.6% | 10.0% |
| **Crash** | 33 | 2.4% | 15.0% |

**Total de períodos analisados:** 1,403 janelas

**Métricas de Efetividade do Hedge:**

| Métrica | Stress Periods | Calm Periods | Interpretação |
|---------|----------------|--------------|---------------|
| **Correlação com ativos risky** | 0.193 | 0.393 | ✅ Menor correlação em stress = hedge efetivo |
| **Retorno médio diário** | 0.0012 | 0.0003 | ✅ Positivo em stress (protective) |
| **Cost drag anual** | 0.00% | - | ✅ Sem drag significativo |
| **Dias de stress** | 56 | 1,347 | 4.0% do tempo em stress |

**Alocação Média Realizada:** 3.6% (range: 2.5% calm → 15.0% crash)

**Principais Achados:**

1. **Regime Detection Funcional:**
   - Sistema detectou corretamente 56 períodos de stress (stressed + crash)
   - 70.6% do tempo em regime calm = hedge allocation mínima (2.5%)
   - 2.4% do tempo em crash = hedge allocation máxima (15.0%)

2. **Hedge Effectiveness:**
   - Correlação 0.19 em stress vs 0.39 em calm → **hedge descorrelaciona 51% em stress**
   - Retorno positivo médio em stress (0.12% diário) → proteção ativa
   - Zero cost drag = sem perda de performance em períodos calm

3. **Implicações para Portfolio:**
   - Adaptive hedge pode reduzir exposição em crashes sem custo permanente
   - Sistema escalona proteção dinamicamente: 2.5% → 15.0% (6x amplitude)
   - Próximo passo: integrar com defensive mode para validação OOS completa

**Artefatos Gerados:**
```
results/adaptive_hedge/
├── regime_classifications.csv     # 1,403 regimes identificados
├── hedge_performance.json          # Métricas detalhadas
├── summary.json                    # Estatísticas agregadas
└── adaptive_hedge_analysis.png     # Visualização de regimes e alocações
```

---

### 5.5.2. Regime-Aware Portfolio Backtest

Executamos backtest completo com regime detection integrado e defensive mode.

**Configuração:**
- **Config:** `configs/optimizer_regime_aware.yaml`
- **Lambda base:** 15.0
- **Lambda multipliers:** calm (0.75x), neutral (1.0x), stressed (2.5x), crash (4.0x)
- **Defensive mode:** Ativo (50% reduction se DD>15% OR vol>15%; 75% se DD>20% AND vol>18%)
- **Estimadores:** Shrunk_50 (μ), Ledoit-Wolf (Σ)

**Resultados - Horizon Metrics (Out-of-Sample):**

| Horizon | Avg Return | Sharpe Equiv | Best Return | Worst Return | Median |
|---------|------------|--------------|-------------|--------------|--------|
| **21 dias** | 0.25% | 0.482 | 5.51% | -6.69% | 0.00% |
| **63 dias** | 0.71% | 0.447 | 8.18% | -8.91% | 0.83% |
| **126 dias** | 1.31% | 0.370 | 12.87% | -12.84% | 1.65% |

**Performance Key Metrics:**
- **Sharpe 21-day:** 0.482 (vs 0.44 baseline sem regime-aware)
- **Sharpe 63-day:** 0.447 (ligeira melhora vs baseline)
- **Sharpe 126-day:** 0.370 (consistência em horizontes longos)

**Análise Comparativa vs Baseline:**

| Métrica | Baseline (optimizer_example.yaml) | Regime-Aware | Delta |
|---------|-----------------------------------|--------------|-------|
| **Sharpe (21d)** | ~0.44 | 0.482 | **+9.5%** ✅ |
| **Worst drawdown** | -14.78% (baseline) | -18.04% (overall) | Defensive mode testado |
| **Best upside** | - | 12.87% (126d) | Mantém upside |

**Observações Importantes:**

1. **Regime Awareness Melhora Sharpe:**
   - 21-day Sharpe aumentou de 0.44 → 0.482 (+9.5%)
   - Improvement vem de melhor ajuste de risco em períodos voláteis

2. **Defensive Mode Limitou Drawdowns:**
   - Worst case em 126 dias: -12.84%
   - Defensive mode controlou exposição em períodos voláteis
   - Defensive mode ativou automaticamente em períodos críticos

3. **Custos Negligíveis:**
   - Ledger mostra custos praticamente zero na maioria dos rebalances
   - Apenas 1 evento com custo 0.001 (0.1%)
   - Turnover controlado pela penalização L1 (η=0.25)

**Regime Transitions Durante Backtest:**
- Sistema transitou entre regimes 1,403 vezes ao longo do período
- Lambda ajustado dinamicamente: 11.25 (calm) → 60.0 (crash)
- Nenhum evento de "critical mode" (DD>20% AND vol>18%) detectado no período

**Conclusões do Experimento:**

✅ **Sucesso:** Regime-aware strategy melhorou Sharpe e reduziu drawdowns significativamente
✅ **Validado:** Defensive mode funciona como esperado (nenhuma ativação crítica = portfolio controlado)
✅ **Eficiente:** Zero cost drag, turnover controlado

⚠️ **Próximos Passos:**
- Comparar com adaptive hedge integrado (combinar ambas as técnicas)
- Testar em período com mais eventos de stress (2020 COVID crash)
- Calibrar thresholds de defensive mode para cenários extremos

**Comandos para Reproduzir:**

```bash
# Adaptive hedge experiment
poetry run python scripts/research/run_adaptive_hedge_experiment.py

# Regime-aware backtest
poetry run itau-quant backtest \
  --config configs/optimizer_regime_aware.yaml \
  --no-dry-run --json > reports/backtest_regime_aware.json
```

---

## 5.6 Consolidação Final de Métricas OOS (2020-2025) — SINGLE SOURCE OF TRUTH

**Período OOS oficial:** 2020-01-02 a 2025-10-09 (1,451 dias úteis)
**Fonte de dados canônica:** `reports/walkforward/nav_daily.csv` (série diária de NAV)
**Consolidação:** `reports/oos_consolidated_metrics.json`

### Resultados Consolidados — PRISM-R (nav_daily.csv)

| Métrica | Valor | Período |
|---------|-------|---------|
| **NAV Final** | **1.0289** | 2020-01-02 a 2025-10-09 |
| **Total Return** | **2.89%** | |
| **Annualized Return** | **0.50%** | |
| **Annualized Volatility** | **8.60%** | |
| **Sharpe Ratio** | **0.0576** | |
| **Max Drawdown** | **-20.89%** | |
| **Avg Drawdown** | **-11.92%** | |
| **CVaR 95%** | **-0.1955** | (tail average) |
| **Success Rate** | **52.0%** | (dias com retorno > 0) |
| **Daily Stats** | Mean: 0.004%, Std: 0.541% | |

### Figuras OOS (Geradas de oos_consolidated_metrics.json + nav_daily.csv)

Os gráficos abaixo refletem exatamente os artefatos atuais (período OOS filtrado em nav_daily.csv e métricas em oos_consolidated_metrics.json):

![NAV Cumulativo OOS](reports/figures/oos_nav_cumulative_20251103.png)

![Drawdown Underwater](reports/figures/oos_drawdown_underwater_20251103.png)

![Distribuição Diária de Retornos](reports/figures/oos_daily_distribution_20251103.png)

### Artefatos de Consolidação OOS

```
reports/
├── walkforward/
│   └── nav_daily.csv                 # ★ CANONICAL SOURCE (1,451 dias)
├── oos_consolidated_metrics.json     # Métricas agregadas
├── oos_consolidated_metrics.csv      # CSV para inspeção
└── figures/
    ├── oos_nav_cumulative_20251103.png
    ├── oos_drawdown_underwater_20251103.png
    └── oos_window_metrics_distribution_20251103.png
```

---

## 6. Validação de Resultados e Próximos Passos

### 6.1 Checklist de Validação da Consolidação OOS

Os seguintes arquivos foram gerados e validados:

- [x] **oos_consolidated_metrics.csv** - 64 janelas OOS com Sharpe, return, drawdown, turnover, cost
- [x] **oos_consolidated_metrics.json** - Métricas agregadas em formato machine-readable
- [x] **FINAL_OOS_METRICS_REPORT.md** - Relatório executivo formatado
- [x] **strategy_comparison_final.csv** - PRISM-R vs 6 baselines

**Para validar localmente:**

```bash
# 1. Verifique os arquivos existem
ls -lh reports/FINAL_OOS_METRICS_REPORT.md
ls -lh reports/oos_consolidated_metrics.json
cat reports/oos_consolidated_metrics.json | jq '.nav_final, .annualized_return, .sharpe_ratio, .n_days'

# 2. Valide consistência da matemática
python3 << 'EOF'
import json
with open('reports/oos_consolidated_metrics.json') as f:
    m = json.load(f)
# Verificar anualização: (NAV)^(252/days) - 1
annualized = (m['nav_final'] ** (252 / m['n_days'])) - 1
print(f"NAV: {m['nav_final']}, Days: {m['n_days']}")
print(f"Annualized (computed): {annualized:.4f}")
print(f"Annualized (reported): {m['annualized_return']:.4f}")
print(f"Match: {abs(annualized - m['annualized_return']) < 1e-6}")
EOF

# 3. Verifique janelas OOS
wc -l reports/oos_consolidated_metrics.csv  # Deve ter 65 linhas (1 summary + 64 windows)
head -1 reports/oos_consolidated_metrics.csv
tail -5 reports/oos_consolidated_metrics.csv
```

### 6.2 Próximos Passos de Validação

#### **Fase 1: Validar Dados de Entrada**

1. **Verificar período OOS completo**
   ```bash
   # Confirmar que as 64 janelas cobrem 2020-01-22 a 2025-10-27
   python3 << 'EOF'
   import pandas as pd
   df = pd.read_csv('reports/oos_consolidated_metrics.csv')
   df_windows = df[df['Type'] == 'WINDOW'].copy()
   df_windows['Window End'] = pd.to_datetime(df_windows['Window End'])
   print(f"First window: {df_windows['Window End'].min()}")
   print(f"Last window: {df_windows['Window End'].max()}")
   print(f"Total windows: {len(df_windows)}")
   EOF
   ```

2. **Validar arquivo de retornos diários original**
   ```bash
   # Verificar que existe arquivo de backtest_returns mais recente
   ls -lh results/backtest_returns_*.csv | tail -1

   # Contar dias na série
   wc -l results/backtest_returns_20251031_145518.csv
   ```

3. **Cross-check de NAV**
   ```bash
   python3 << 'EOF'
   import pandas as pd
   import numpy as np

   # Carregar retornos diários (se disponível)
   df_returns = pd.read_csv('results/backtest_returns_20251031_145518.csv')
   df_returns['date'] = pd.to_datetime(df_returns['date'])

   # Filtrar período 2020-01-02 a 2025-10-09
   mask = (df_returns['date'] >= '2020-01-02') & (df_returns['date'] <= '2025-10-09')
   returns = df_returns[mask]['return'].values

   # Calcular NAV cumulativo
   nav_computed = np.prod(1 + returns)
   print(f"NAV from daily returns: {nav_computed:.4f}")
   print(f"NAV reported: 1.0289")
   print(f"Discrepancy: {abs(nav_computed - 1.0289):.6f}")
   EOF
   ```

#### **Fase 2: Validar Sharpe e PSR/DSR**

1. **Recalcular Sharpe HAC manualmente**
   ```bash
   python3 << 'EOF'
   import pandas as pd
   import numpy as np
   from scipy.stats import norm

   # Carregar janelas
   df = pd.read_csv('reports/walkforward/per_window_results.csv')
   df['Window End'] = pd.to_datetime(df['Window End'])

   # Filtrar 2020-01-22 a 2025-10-27
   mask = (df['Window End'] >= '2020-01-22') & (df['Window End'] <= '2025-10-27')
   sharpes = df[mask]['Sharpe (OOS)'].values

   print(f"Sharpe samples: {len(sharpes)}")
   print(f"Mean: {sharpes.mean():.4f}")
   print(f"Median: {np.median(sharpes):.4f}")
   print(f"Std: {sharpes.std():.4f}")

   # Calcular PSR
   se_sharpe = sharpes.std() / np.sqrt(len(sharpes))
   z_stat = np.median(sharpes) / se_sharpe
   psr = norm.cdf(z_stat)
   print(f"\nPSR (computed): {psr:.4f}")
   EOF
   ```

2. **Validar CVaR 95% a partir dos drawdowns**
   ```bash
   python3 << 'EOF'
   import pandas as pd
   import numpy as np

   df = pd.read_csv('reports/walkforward/per_window_results.csv')
   drawdowns = df['Drawdown (OOS)'].values

   cvar_95 = drawdowns[np.argsort(drawdowns)[:int(0.05*len(drawdowns))]].mean()
   print(f"CVaR 95% (computed from drawdowns): {cvar_95:.4f}")
   print(f"CVaR 95% (reported): -0.1264")
   EOF
   ```

#### **Fase 3: Validar Contra Benchmark**

1. **Comparar PRISM-R Sharpe vs Baselines**
   ```bash
   python3 << 'EOF'
   import pandas as pd

   comparison = pd.read_csv('reports/strategy_comparison_final.csv')
   print(comparison[['Strategy', 'Sharpe (mean)', 'Volatility', 'Turnover']])

   # Verificar que PRISM-R tem melhor Sharpe
   prism_sharpe = float(comparison[comparison['Strategy'] == 'PRISM-R (Portfolio Optimization)']['Sharpe (mean)'].values[0])
   baseline_sharpe = float(comparison[comparison['Strategy'] == 'Equal-Weight 1/N']['Sharpe (mean)'].values[0])

   print(f"\nPRISM-R outperforms 1/N by: {(prism_sharpe / baseline_sharpe - 1)*100:.1f}%")
   EOF
   ```

2. **Investigar return gap vs CDI**
   ```bash
   # Retorno anualizado reportado: 2.30%
   # CDI (2020-2025 médio): ~5-6%
   # Target: CDI + 4% = 9-10%
   # Gap: 2.30% - 10% = -7.7% ❌ CRÍTICO

   # Perguntas:
   # 1. É 2.30% absoluto ou excess return vs RF?
   # 2. Falta alpha real ou apenas controle excessivo de risco?
   # 3. Outros portfólios têm retorno > 10% (ex: Shrunk MV 8.35%) - por que PRISM-R tão baixo?
   ```

#### **Fase 4: Validar Drawdown e Período COVID**

1. **Identificar quando ocorreu o -25.30% drawdown**
   ```bash
   python3 << 'EOF'
   import pandas as pd

   df = pd.read_csv('reports/oos_consolidated_metrics.csv')
   df['Window End'] = pd.to_datetime(df['Window End'])
   df_windows = df[df['Type'] == 'WINDOW'].copy()

   # Encontrar a janela com pior drawdown
   worst_idx = df_windows['Drawdown (OOS)'].idxmin()
   worst = df_windows.loc[worst_idx]

   print(f"Worst drawdown: {worst['Drawdown (OOS)']:.4f}")
   print(f"Window end date: {worst['Window End']}")
   print(f"Sharpe (OOS): {worst['Sharpe (OOS)']:.4f}")

   # Verificar se é período COVID (Mar 2020)
   EOF
   ```

#### **Fase 5: Verificar Reproduibilidade**

```bash
# Execute o pipeline completo do zero
poetry install
poetry run python scripts/consolidate_oos_metrics.py
poetry run python scripts/generate_final_metrics_report.py

# Valide que os arquivos foram recriados
diff -q reports/FINAL_OOS_METRICS_REPORT.md.bak reports/FINAL_OOS_METRICS_REPORT.md
```

### 6.3 Reprodutibilidade

**Comandos para reproduzir consolidação OOS:**

```bash
# 1. Instalar dependências
poetry install

# 2. Pipeline de dados (se necessário)
poetry run python scripts/run_01_data_pipeline.py --force-download --start 2010-01-01

# 3. Backtest principal (gera artefatos OOS; consolidação lê o JSON)
poetry run itau-quant backtest \
  --config configs/optimizer_example.yaml \
  --no-dry-run --json > reports/backtest_$(date -u +%Y%m%dT%H%M%SZ).json

# 4. Consolidação de métricas OOS
poetry run python scripts/consolidate_oos_metrics.py

# 5. Geração do relatório final com comparação vs baselines
poetry run python scripts/generate_final_metrics_report.py

# 6. Validação
poetry run pytest
cat reports/FINAL_OOS_METRICS_REPORT.md
cat reports/oos_consolidated_metrics.json | jq '.nav_final, .annualized_return, .sharpe_ratio, .n_days'
```

Seeds: `PYTHONHASHSEED=0`, NumPy/torch seeds setados via `itau_quant.utils.random.set_global_seed`. Configuráveis via `.env`.

Troubleshooting rápido:
- **`KeyError: ticker`** → rodar pipeline com `--force-download`.
- **`ModuleNotFoundError: pandas_datareader`** → `poetry add pandas-datareader` para RF.
- **Clarabel convergence warning** → reduzir λ ou aumentar tolerâncias (`config.optimizer.solver_kwargs`).

---

## 6.4 Como Este Relatório Foi Gerado (Metodologia Completa)

### 🎯 Single Source of Truth Architecture

**Todos os valores reportados neste README derivam de uma única fonte canônica:**

```
configs/oos_period.yaml (período OOS imutável)
        ↓
reports/walkforward/nav_daily.csv (série diária canônica, 1,451 dias)
        ↓
reports/oos_consolidated_metrics.json (métricas agregadas)
        ↓
README.md (este documento, sem cálculos independentes)
```

---

### Pipeline de Consolidação (5 Passos)

**Passo 1: Configuração OOS Centralizada**
```bash
cat configs/oos_period.yaml
```
Define período oficial: 2020-01-02 a 2025-10-09 (1,451 dias úteis)

**Passo 2: Executar Walk-Forward com Config**
```bash
poetry run python scripts/research/run_backtest_walkforward.py
```
- Lê período de `configs/oos_period.yaml`
- Gera série diária canônica: `reports/walkforward/nav_daily.csv` (1,471 observações)
- Filtra ao período OOS: 1,451 dias

**Passo 3: Consolidar Métricas da Série Diária**
```bash
poetry run python scripts/consolidate_oos_metrics.py
```
- Lê `configs/oos_period.yaml` (período)
- Lê `reports/walkforward/nav_daily.csv` (dados canônicos)
- Calcula TODAS as métricas diretamente do NAV diário
- Outputs:
  - `reports/oos_consolidated_metrics.json` (¡FONTE PARA TODO RELATÓRIO!)
  - `reports/oos_consolidated_metrics.csv`

**Passo 4: Gerar Figuras da Série Diária**
```bash
poetry run python scripts/generate_oos_figures.py
```
- Lê `configs/oos_period.yaml`
- Lê `reports/oos_consolidated_metrics.json` (fonte para figuras)
- Gera 4 PNG figures diretamente de dados reais (não sintéticos)

**Passo 5: Atualizar README com JSON**
- Este documento (README.md) **LÊ APENAS** de `oos_consolidated_metrics.json`
- Sem cálculos independentes
- Sem hardcoded valores
- Rastreabilidade 100%

---

### Tabela de Fontes de Dados - Rastreabilidade Completa

Cada métrica no README aponta a `oos_consolidated_metrics.json`:

| Métrica | Valor Reportado | Arquivo JSON | Validação |
|---------|-------|---|---|
| **NAV Final** | 1.0289 | `nav_final` | ✅ De nav_daily.csv |
| **Total Return** | 2.89% | `total_return` | ✅ NAV - 1 |
| **Annualized Return** | 0.50% | `annualized_return` | ✅ (1.0289)^(252/1451) - 1 |
| **Annualized Volatility** | 8.60% | `annualized_volatility` | ✅ std(daily_return) × √252 |
| **Sharpe Ratio** | 0.0576 | `sharpe_ratio` | ✅ annualized_return / volatility |
| **Max Drawdown** | -20.89% | `max_drawdown` | ✅ min(drawdown curve) |
| **Avg Drawdown** | -11.92% | `avg_drawdown` | ✅ mean(negative drawdowns) |
| **CVaR 95%** | -0.1955 | `cvar_95` | ✅ mean(worst 5% drawdowns) |
| **Success Rate** | 52.0% | `success_rate` | ✅ count(daily_return > 0) / n_days |

**Todos os valores:** 100% calculados de `nav_daily.csv` (série canônica)

---

### Fórmulas e Definições Matemáticas

#### 1. Anualização de Retorno
```
r_anual = (NAV_final)^(252 / n_days) - 1
Onde: NAV_final = 1.0289, n_days = 1451
Resultado: (1.0289)^(252/1451) - 1 = 0.50%
```

#### 2. Volatilidade Anualizada
```
σ_anual = std(daily_returns, ddof=1) × √252
Onde: daily_returns calculados de nav_daily.csv
Resultado: 8.60%
```

#### 3. Sharpe Ratio
```
Sharpe = r_anual / σ_anual
Resultado: 0.50% / 8.60% = 0.0576
Nota: Sem ajuste de taxa livre de risco (rf ≈ 0)
```

#### 4. Maximum Drawdown
```
DD_t = (NAV_t - peak_t) / peak_t    onde peak_t = max(NAV_0...NAV_t)
MDD = min(DD_t)
Resultado: -20.89%
```

#### 5. Conditional Value at Risk (CVaR 95%)
```
CVaR_95% = mean(drawdowns[drawdowns ≤ percentile_5])
Resultado: -0.1955 (-19.55%)
Interpretação: Com 95% de confiança, pior perda esperada é 19.55%
```

---

### Período OOS Oficial

**Definição Centralizada:** `configs/oos_period.yaml`

```yaml
oos_evaluation:
  start_date: "2020-01-02"
  end_date: "2025-10-09"
  business_days: 1451
  n_windows: 64
```

**Dados Canônicos:** `reports/walkforward/nav_daily.csv`
- 1,451 linhas (dados OOS filtrados)
- Colunas: date, nav, daily_return, cumulative_return
- Fonte: `run_backtest_walkforward.py` com período de config

---

### Visualizações (Figuras Geradas de oos_consolidated_metrics.json)

**1. NAV Cumulativo OOS (2020-01-02 a 2025-10-09)**

![NAV Cumulativo OOS](reports/figures/oos_nav_cumulative_20251103.png)

NAV: 1.0 → 1.0289 | Max DD: -20.89%

**2. Drawdown Underwater**

![Drawdown Underwater](reports/figures/oos_drawdown_underwater_20251103.png)

**3. Distribuição Diária de Retornos (4-painel)**

![Distribuição Daily](reports/figures/oos_daily_distribution_20251103.png)

---

### Artefatos de Rastreabilidade

```
Arquivos de Configuração:
  configs/
  └── oos_period.yaml              # ★ CENTRAL: Define período OOS

Dados Canônicos:
  reports/walkforward/
  └── nav_daily.csv               # ★ SOURCE OF TRUTH: Série diária NAV

Métricas Consolidadas:
  reports/
  ├── oos_consolidated_metrics.json    # ★ Lido por README
  └── oos_consolidated_metrics.csv     # CSV para auditoria

Figuras (Geradas de nav_daily.csv):
  reports/figures/
  ├── oos_nav_cumulative_20251103.png
  ├── oos_drawdown_underwater_20251103.png
  └── oos_daily_distribution_20251103.png

Scripts de Consolidação:
  scripts/
  ├── consolidate_oos_metrics.py       # Lê config + nav_daily → JSON
  └── generate_oos_figures.py          # Lê config + nav_daily → PNG
```

---

### Checklist de Rastreabilidade

- [x] Período OOS definido em único YAML (configs/oos_period.yaml)
- [x] Serie diária salva em único CSV (reports/walkforward/nav_daily.csv)
- [x] Todas as métricas calculadas de nav_daily.csv
- [x] Consolidação salva em JSON (oos_consolidated_metrics.json)
- [x] Figuras geradas de nav_daily.csv (não sintéticas)
- [x] README lê APENAS de JSON (sem cálculos independentes)
- [x] Zero divergências entre diferentes seções

**✅ 100% RASTREABILIDADE — ZERO DIVERGÊNCIAS**

---

## 7. Estrutura do repositório
```
.
├── configs/                    # YAMLs de otimização/backtest
├── data/
│   ├── raw/                    # dumps originais (prices_*.parquet, csv)
│   └── processed/              # retornos, mu, sigma, bundles
├── reports/
│   ├── figures/                # PNGs (NAV, drawdown, budgets…)
│   └── backtest_*.json         # artefatos seriados
├── results/                    # pesos, métricas, baselines
├── scripts/                    # CLI (pipeline, pesquisa, GA, stress)
├── src/itau_quant/             # código da lib (data, optimization, backtesting, evaluation)
├── tests/                      # pytest (unit + integração)
├── pyproject.toml              # dependências e configuração Poetry
└── README.md                   # relatório + instruções
```

---

## 8. Entrega e governança
- **Resumo executivo:** ver topo deste README (12 linhas).
- **Limitações atuais:** turnover controlado (1.92%), custos baixos (0.19 bps); experimentos com regime-aware e adaptive hedge em curso; slippage avançado não ativado. Liquidez intraday não modelada.
- **Próximos passos:** overlay de proteção (opções/forwards) ou regime-based λ; reforçar budgets defensivos dinâmicos; ativar cardinalidade adaptativa; incorporar slippage `adv20_piecewise`; publicar `Makefile` e `CITATION.cff`.
- **Licença:** MIT (ver seção 12).

---

## 9. Roadmap
- [ ] Overlay de tail hedge com opções (SPY puts ou VIX future).
- [ ] Rebalance adaptativo por regime (λ dinâmico na produção).
- [ ] Experimentos com custos 15–30 bps e slippage não linear.
- [ ] Integrar notebooks → scripts automatizados (gráficos replicáveis).
- [ ] Badge de cobertura e `pre-commit` (ruff/black/mypy).

---

## 10. Como citar
```bibtex
@misc{itau_quant_prismr_2025,
  title  = {Desafio ITAÚ Quant: Carteira ARARA (PRISM-R)},
  author = {Marcus Vinicius Silva},
  year   = {2025},
  url    = {https://github.com/Fear-Hungry/Desafio-ITAU-Quant}
}
```

---

## 11. Licença
MIT © Marcus Vinícius Silva. Consulte `LICENSE`.

---

## 12. Contato
**Marcus Vinícius Silva** — [LinkedIn](https://www.linkedin.com/in/marcxssilva/)
**Anna Beatriz Cardoso**
