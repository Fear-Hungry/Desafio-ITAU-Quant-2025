# Implementação de Proteção de Cauda e Regime Dinâmico

**Status:** ✅ IMPLEMENTADO
**Data:** 2025-11-01
**Versão:** 1.0

---

## Sumário Executivo

Implementamos com sucesso um sistema completo de proteção de cauda e ajuste dinâmico de risco baseado em regime de mercado, conforme roadmap do projeto. O sistema integra:

1. **Defensive Mode** - Redução automática de exposição em períodos de stress
2. **Regime-Aware Lambda** - Ajuste dinâmico de aversão ao risco
3. **Adaptive Tail Hedge** - Alocação variável em ativos de proteção
4. **Logging Completo** - Rastreamento de regime e decisões

**Componentes Modificados:** 5 arquivos
**Novos Módulos:** 3 arquivos
**Configs Criadas:** 2 YAMLs
**Scripts de Pesquisa:** 1 experimento

---

## Arquitetura Implementada

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE DE REBALANCE                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Detect Regime (risk/regime.py)                      │
│     ├─> calm / neutral / stressed / crash               │
│     └─> Based on: volatility + drawdown                 │
│                                                          │
│  2. Adjust Lambda (rebalancer.py)                       │
│     ├─> λ_adjusted = λ_base × multiplier[regime]       │
│     └─> Example: crash → λ × 4.0                        │
│                                                          │
│  3. Optimize Portfolio (MV-QP)                          │
│     └─> Uses adjusted λ                                 │
│                                                          │
│  4. Apply Defensive Mode (rebalancer.py)                │
│     ├─> Check: DD > 15% OR vol > 15%  → 50% reduction  │
│     ├─> Check: DD > 20% AND vol > 18% → 75% reduction  │
│     └─> Scale risky assets, move to CASH               │
│                                                          │
│  5. Adaptive Hedge (adaptive_hedge.py)                  │
│     ├─> Compute hedge allocation by regime             │
│     └─> Apply hedge rebalance                          │
│                                                          │
│  6. Log Results (production_logger.py)                  │
│     └─> Regime, λ_adjusted, defensive_mode             │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Componentes Implementados

### 1. Defensive Mode (`src/arara_quant/portfolio/rebalancer.py`)

**Função:** `apply_defensive_mode(weights, portfolio_state, config)`

**Lógica:**
```python
# Defensive Mode: DD > 15% OR vol > 15%
if DD < -0.15 or vol > 0.15:
    risky_weights *= 0.50  # Keep 50% exposure
    CASH += freed_capital

# Critical Mode: DD > 20% AND vol > 18%
if DD < -0.20 and vol > 0.18:
    risky_weights *= 0.25  # Keep 25% exposure
    CASH += freed_capital
```

**Configuração (YAML):**
```yaml
defensive_mode:
  enable: true
  defensive_threshold_dd: -0.15
  defensive_threshold_vol: 0.15
  critical_threshold_dd: -0.20
  critical_threshold_vol: 0.18
  safe_assets: [CASH, SHY, TLT, SGOV, BIL]
```

**Integração:** Chamado automaticamente em `rebalance()` após otimização, antes de rounding.

---

### 2. Regime-Aware Lambda (`src/arara_quant/risk/regime.py` + `rebalancer.py`)

**Já Existia:** Módulo `regime.py` com `detect_regime()` e `regime_multiplier()`

**Integração Adicionada:** Linhas 518-534 de `rebalancer.py`

**Fluxo:**
```python
regime_cfg = optimizer_cfg.get("regime_detection")
if regime_cfg:
    regime_snapshot = detect_regime(historical_returns, config=regime_cfg)
    multiplier = regime_multiplier(regime_snapshot, regime_cfg)
    λ_adjusted = λ_base × multiplier
    optimizer_cfg["risk_aversion"] = λ_adjusted
```

**Multipliers Padrão:**
| Regime | Multiplier | Effective λ (base=15) |
|--------|------------|-----------------------|
| calm | 0.75 | 11.25 |
| neutral | 1.0 | 15.0 |
| stressed | 2.5 | 37.5 |
| crash | 4.0 | 60.0 |

---

### 3. Adaptive Tail Hedge (`src/arara_quant/portfolio/adaptive_hedge.py`)

**Funções Principais:**

#### `compute_hedge_allocation(regime, config) -> float`
Calcula alocação-alvo para hedge baseado em regime:
```python
target = base_allocation × regime_multiplier[regime]
target = min(target, max_allocation[regime])
target = max(target, min_allocation)
```

**Exemplo:**
```python
# Calm regime
compute_hedge_allocation("calm", config)
# → 0.025 (2.5%)

# Crash regime
compute_hedge_allocation("crash", config)
# → 0.15 (15.0%, capped)
```

#### `apply_hedge_rebalance(weights, hedge_assets, target_allocation)`
Ajusta pesos do portfólio para atingir alocação-alvo de hedge:
- Escala ativos não-hedge proporcionalmente
- Distribui alocação de hedge igualmente entre ativos de proteção
- Normaliza para soma = 1.0

#### `evaluate_hedge_performance(returns, hedge_assets, regime_labels)`
Avalia efetividade do hedge:
- Correlação hedge vs risky em stress/calm
- Retornos médios por regime
- Custo anual estimado (drag)

---

### 4. Production Logger com Regime (`src/arara_quant/utils/production_logger.py`)

**Campos Adicionados:**
```python
@dataclass
class RebalanceLog:
    ...
    regime: Optional[str]  # calm, neutral, stressed, crash
    lambda_adjusted: Optional[float]  # λ após ajuste de regime
    defensive_mode: Optional[str]  # normal, defensive, critical
```

**CSV Headers:**
```
date, strategy, turnover, cost_bps, ..., regime, lambda_adjusted, defensive_mode
```

**Uso:**
```python
logger.log_rebalance(
    ...
    regime="stressed",
    lambda_adjusted=37.5,
    defensive_mode="defensive"
)
```

---

## Configurações YAML

### Config 1: `configs/optimization/optimizer_regime_aware.yaml`

Portfolio com regime detection e defensive mode:

```yaml
optimizer:
  lambda: 15.0  # Base

  regime_detection:
    enable: true
    window_days: 63
    vol_calm_threshold: 0.12
    vol_stressed_threshold: 0.25
    dd_crash_threshold: -0.15
    multipliers:
      calm: 0.75
      neutral: 1.0
      stressed: 2.5
      crash: 4.0

defensive_mode:
  enable: true
  defensive_threshold_dd: -0.15
  defensive_threshold_vol: 0.15
  critical_threshold_dd: -0.20
  critical_threshold_vol: 0.18
  safe_assets: [CASH, SHY, SGOV, BIL, TLT]
```

### Config 2: `configs/optimization/optimizer_adaptive_hedge.yaml`

Portfolio com tail hedge dinâmico:

```yaml
adaptive_hedge:
  enable: true
  base_allocation: 0.05

  regime_multipliers:
    calm: 0.5        # 2.5% allocation
    neutral: 1.0     # 5.0%
    stressed: 2.0    # 10.0%
    crash: 3.0       # 15.0%

  max_allocation:
    calm: 0.03
    neutral: 0.05
    stressed: 0.12
    crash: 0.15

  min_allocation: 0.02  # Always ≥2%

portfolio:
  risk:
    budgets:
      - name: tail_hedge
        min_weight: 0.02
        max_weight: 0.15
        tickers: [TLT, TIP, GLD, SLV, PPLT, UUP]
```

---

## Scripts de Pesquisa

### `arara_quant.runners.research.run_adaptive_hedge_experiment`

**Objetivo:** Validar efetividade do hedge adaptativo vs estático

**Métricas Computadas:**
- Distribuição de regimes no histórico
- Alocação média de hedge por regime
- Correlação hedge vs risky (stress/calm)
- Custo anual estimado (drag)

**Outputs:**
```
outputs/results/adaptive_hedge/
├── regime_classifications.csv     # Regime por data
├── hedge_performance.json          # Métricas de efetividade
├── summary.json                    # Estatísticas agregadas
└── adaptive_hedge_analysis.png     # Gráficos de regime e alocação
```

**Execução:**
```bash
poetry run python -m arara_quant.runners.research.run_adaptive_hedge_experiment
```

---

## Exemplos de Uso

### Exemplo 1: Backtest com Regime-Aware

```bash
poetry run arara-quant backtest \
  --config configs/optimization/optimizer_regime_aware.yaml \
  --no-dry-run \
  --json > outputs/reports/backtest_regime_aware.json
```

**Comportamento Esperado:**
- Em 2020-03 (COVID crash): regime="crash", λ=60.0, defensive_mode="critical"
- Em 2022 (inflation): regime="stressed", λ=37.5, defensive_mode="defensive"
- Em períodos calm: regime="calm", λ=11.25, defensive_mode="normal"

### Exemplo 2: Production com Adaptive Hedge

```bash
poetry run python -m arara_quant.runners.production.run_portfolio_production_erc_v2 \
  --config configs/optimization/optimizer_adaptive_hedge.yaml
```

**Log Output:**
```
✅ Rebalance registrado: 2025-11-01
   Estratégia: ERC
   Turnover: 8.2%
   Custo: 24.5 bps
   N_effective: 18.5
   Regime: stressed
   Lambda adjusted: 37.5
   Defensive mode: normal
   Hedge allocation: 10.0%
```

---

## Critérios de Sucesso

### Defensive Mode

| Métrica | Target | Como Verificar |
|---------|--------|----------------|
| Trigger em COVID-20 | ✅ Ativo | `outputs/results/production/production_log.csv` → defensive_mode="critical" |
| DD reduction | -25% → -18% | Compare backtest com/sem defensive mode |
| False positives | < 5% do tempo | Count "defensive" != "normal" em períodos calm |

### Regime-Aware Lambda

| Métrica | Target | Como Verificar |
|---------|--------|----------------|
| λ increase em crash | 4.0x | production_log.csv → lambda_adjusted ≈ 60 quando regime="crash" |
| Sharpe em stress | > -1.0 | Compare com baseline Sharpe=-2.88 (COVID) |
| OOS Sharpe overall | ≥ 0.50 | backtest_metrics.csv |

### Adaptive Hedge

| Métrica | Target | Como Verificar |
|---------|--------|----------------|
| DD reduction | -5.7 p.p. | backtest com/sem adaptive hedge |
| Cost drag | < 1.5%/ano | `hedge_performance.json` → cost_drag_annual |
| Correlation (stress) | < -0.2 | `hedge_performance.json` → correlation_stress |

---

## Testes de Validação

### Teste 1: Defensive Mode em COVID-19

```python
# Simular estado de portfolio em 2020-03
portfolio_state = {
    "drawdown": -0.22,  # -22%
    "volatility": 0.35   # 35% vol
}

weights, mode_info = apply_defensive_mode(
    weights=initial_weights,
    portfolio_state=portfolio_state,
    config={"enable": True}
)

assert mode_info["mode"] == "critical"  # DD<-20% AND vol>18%
assert mode_info["scaling"] == 0.25     # 75% reduction
```

### Teste 2: Regime Detection

```python
# COVID crash period
covid_returns = returns.loc["2020-02-01":"2020-04-01"]

regime_snapshot = detect_regime(covid_returns)

assert regime_snapshot.label == "crash"  # DD < -15%
assert regime_snapshot.volatility > 0.25  # High vol
```

### Teste 3: Adaptive Hedge Allocation

```python
# Calm regime → minimal hedge
alloc_calm = compute_hedge_allocation("calm", config=ADAPTIVE_HEDGE_CONFIG)
assert 0.02 <= alloc_calm <= 0.03  # 2-3%

# Crash regime → maximum hedge
alloc_crash = compute_hedge_allocation("crash", config=ADAPTIVE_HEDGE_CONFIG)
assert 0.14 <= alloc_crash <= 0.15  # 14-15%
```

---

## Próximos Passos

### Prioridade Alta

1. **Executar Backtest Completo**
   ```bash
   poetry run arara-quant backtest \
     --config configs/optimization/optimizer_regime_aware.yaml \
     --no-dry-run
   ```

2. **Comparar Métricas OOS**
   - Baseline (sem regime/defensive): `optimizer_example.yaml`
   - Regime-aware: `optimizer_regime_aware.yaml`
   - Adaptive hedge: `optimizer_adaptive_hedge.yaml`

3. **Calibrar Thresholds**
   - Ajustar `vol_calm_threshold`, `vol_stressed_threshold` se necessário
   - Testar `defensive_threshold_dd` ∈ [-0.12, -0.18]
   - Validar `regime_multipliers` em diferentes períodos

### Prioridade Média

4. **Experimento CVaR + Regime**
   - Combinar mean-CVaR com regime detection
   - Testar se CVaR + λ dinâmico > mean-variance

5. **Hysteresis em Defensive Mode**
   - Adicionar cooldown para evitar whipsaw
   - Require confirmation (2-3 dias consecutivos) para entrar/sair

6. **Options Overlay (Exploratório)**
   - Verificar se `yfinance` suporta options data
   - Protótipo: buy SPY puts quando regime="stressed"
   - Decision: go/no-go baseado em custo vs DD reduction

### Prioridade Baixa

7. **Dashboard de Monitoramento**
   - Plotar regime transitions ao longo do tempo
   - Heatmap de λ_adjusted vs retorno realizado
   - Alert system quando regime muda para "crash"

8. **Stress Testing**
   - Simular regime="crash" artificialmente
   - Verificar se defensive mode reduz exposure corretamente
   - Test recovery time após crash

---

## Troubleshooting

### Erro: "Defensive mode not applied"

**Causa:** Config `defensive_mode.enable` não está `true`

**Solução:**
```yaml
defensive_mode:
  enable: true  # ← Must be explicit
```

### Warn: "No hedge assets found in portfolio"

**Causa:** Nenhum asset de `safe_assets` ou `tail_hedge` no universo

**Solução:**
```yaml
# Adicionar ao universe
universe: configs/universe/universe_arara.yaml  # Deve incluir TLT, SHY, etc.

# Ou ajustar safe_assets
defensive_mode:
  safe_assets: [CASH, IEF, AGG]  # Use assets que existem
```

### Erro: "Regime detection failed"

**Causa:** Janela muito curta ou dados insuficientes

**Solução:**
```yaml
regime_detection:
  window_days: 42  # ← Reduce if < 63 days available
```

---

## Referências

**Código:**
- `src/arara_quant/portfolio/rebalancer.py` (linhas 352-461, 707-743)
- `src/arara_quant/portfolio/adaptive_hedge.py`
- `src/arara_quant/risk/regime.py`
- `src/arara_quant/utils/production_logger.py`

**Configs:**
- `configs/optimization/optimizer_regime_aware.yaml`
- `configs/optimization/optimizer_adaptive_hedge.yaml`

**Experimentos:**
- `arara_quant.runners.research.run_adaptive_hedge_experiment`

**Documentação:**
- docs/specs/PRD.md (linhas 302-305: defensive mode spec)
- README.md (linha 184: roadmap)
- RESULTADOS_FINAIS.md (linha 234: regime detection results)

---

## Changelog

**2025-11-01 - v1.0 - Initial Implementation**
- ✅ Defensive mode em `rebalancer.py`
- ✅ Regime logging em `production_logger.py`
- ✅ Adaptive hedge module
- ✅ Configs YAML (regime_aware, adaptive_hedge)
- ✅ Experimento de validação

---

**Implementado por:** Claude (Anthropic)
**Revisado por:** Marcus Vinicius Silva
**Status:** ✅ PRONTO PARA TESTE
