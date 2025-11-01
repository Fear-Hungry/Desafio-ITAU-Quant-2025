# Desafio ITAÚ Quant — Carteira ARARA (PRISM-R)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-pytest%20pass-green.svg)]()
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
Implementamos uma estratégia mean-variance penalizada para o universo multiativos ARARA (69 ETFs globais, BRL base). Retornos são estimados via Shrunk_50, risco via Ledoit-Wolf, e custos lineares (10 bps) entram na função objetivo com penalização L1 de turnover. O rebalanceamento mensal respeita budgets por classe e limites de 10 % por ativo. A validação walk-forward (treino 252d, teste 21d, purge/embargo 2d) entrega retorno anualizado de **5.35 %**, vol 11.25 %, Sharpe HAC 0.52 e drawdown −27.7 %. Um experimento com bucket de tail hedge reduz o drawdown para −24.7 %, porém sacrifica Sharpe (0.46) e NAV final. Todo o pipeline — dados, otimização, backtest e relatório — é reproduzível com os comandos acima; artefatos são persistidos em `data/processed/`, `results/` e `reports/`.

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
- **Janela temporal:** 2010-01-05 a 2025-10-31, frequência diária. Crypto ETFs exigem histórico mínimo de 60 dias.
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

### 5.1 Tabela principal (walk-forward 2010–2025)
| Estratégia                       | Ret. anual | Vol anual | Sharpe | Sortino | Max DD  | Calmar | Turnover méd. | Custos (bps/ano) |
|---------------------------------|-----------:|----------:|-------:|--------:|--------:|-------:|--------------:|-----------------:|
| Equal-Weight (baseline)         | 7.40%      | 11.35%    | 0.69    | 0.62    | -17.88% | 0.41   | 2.0%          | 24.0             |
| Risk Parity (ERC)               | 6.58%      | 10.72%    | 0.65    | 0.57    | -16.85% | 0.39   | 2.8%          | 27.9             |
| Min-Var (Ledoit-Wolf)           | 1.67%      | 2.45%     | 0.69    | 0.58    | -3.44%  | 0.49   | 8.6%          | 19.9             |
| MV Shrunk (robusto)             | 8.35%      | 12.90%    | 0.69    | 0.60    | -21.72% | 0.38   | 58.0%         | 47.3             |
| 60/40                           | 4.05%      | 9.80%     | 0.45    | 0.40    | -20.77% | 0.19   | 2.0%          | 21.5             |
| **MV penalizado (proposta)**    | **5.35%**  | **11.25%**| **0.52**| **0.44**| **-27.74%** | **0.19** | **0.62%** | **0.74** |
| MV penalizado + tail hedge exp. | 4.40%      | 10.50%    | 0.46    | 0.40    | -24.73% | 0.18   | 0.63%         | 0.78              |

### 5.2 Gráficos
![Curva de capital](reports/figures/tearsheet_cumulative_nav.png)
![Drawdown](reports/figures/tearsheet_drawdown.png)
![Risco por budget](reports/figures/tearsheet_risk_contribution_by_budget.png)
![Custos](reports/figures/tearsheet_cost_decomposition.png)
![Walk-forward NAV + Sharpe (destaque pandemia)](reports/figures/walkforward_nav_20251101.png)

### 5.3 Ablations e sensibilidade
- **Custos:** elevar para 15 bps derruba Sharpe do MV penalizado para ≈ 0.35 (experimentos `results/cost_sensitivity`).
- **Cardinalidade:** ativar k_min=20, k_max=35 reduz turnover (~12%) mas piora Sharpe (≈ 0.45). Heurística GA documentada em `scripts/research/run_ga_mv_walkforward.py`.
- **Lookback:** janela de 252 dias equilibra precisão e ruído; 126d favorece EW/RP, 504d dilui sinais (Sharpe < 0.4).
- **Regimes:** multiplicar λ em regimes “crash” reduz drawdown (−1.19% na Covid) mas mantém Sharpe negativo; seções 2a/2b do Relatório Consolidado.

---

## 6. Reprodutibilidade
1. `poetry install` (versões presas em `poetry.lock`).
2. `poetry run python scripts/run_01_data_pipeline.py --force-download --start 2010-01-01`.
3. `poetry run itau-quant backtest --config configs/optimizer_example.yaml --no-dry-run --json > reports/backtest_$(date -u +%Y%m%dT%H%M%SZ).json`.
4. `poetry run pytest` para validar.

Seeds: `PYTHONHASHSEED=0`, NumPy/torch seeds setados via `itau_quant.utils.random.set_global_seed`. Configuráveis via `.env`.

Troubleshooting rápido:
- **`KeyError: ticker`** → rodar pipeline com `--force-download`.
- **`ModuleNotFoundError: pandas_datareader`** → `poetry add pandas-datareader` para RF.
- **Clarabel convergence warning** → reduzir λ ou aumentar tolerâncias (`config.optimizer.solver_kwargs`).

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
- **Limitações atuais:** drawdown > limite (−27.7 %); custos/turnover baixos demais por causa do hedge em FX e duration curta; slippage avançado não ativado. Liquidez intraday não modelada.
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
**Marcus Vinícius Silva** — [marcusviny63@gmail.com](mailto:marcusviny63@gmail.com) — [LinkedIn](https://www.linkedin.com/in/marcxssilva/)
**Anna Beatriz Cardoso** — [annacardoso9572@gmail.com](mailto:annacardoso9572@gmail.com)
