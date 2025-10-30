# PRISM-R - Guia Rápido de Uso (Versão Robusta)

## 🚀 Start Rápido (3 comandos)

```bash
# 1. Otimizar portfolio robusto
poetry run python run_portfolio_arara_robust.py

# 2. Comparar estimadores de μ  
poetry run python run_estimator_comparison.py

# 3. Validar OOS vs baselines
poetry run python run_baselines_comparison.py
```

---

## 📊 O que cada script faz

### 1. `run_portfolio_arara_robust.py`
**Tempo:** ~15 segundos  
**Output:** Portfolio otimizado único

```
Correções vs original:
✅ IBIT/ETHA (spot) em vez de BITO (futuros)
✅ MAX_POSITION = 10% (vs 15%)
✅ Huber robust mean (vs sample mean)
✅ Custos 30 bps + turnover penalty 15 bps
✅ Risk Aversion = 4.0 (mais conservador)
```

**Resultado esperado:**
- N_effective ~ 10-12 (vs 7.4 original)
- Nenhum ativo > 10%
- Sharpe ex-ante ~ 1.5-2.2

**Arquivos gerados:**
- `results/portfolio_weights_robust_TIMESTAMP.csv`
- `results/portfolio_metrics_robust_TIMESTAMP.csv`

---

### 2. `run_estimator_comparison.py`
**Tempo:** ~60 segundos  
**Output:** Comparação de 4 estimadores

**Estimadores testados:**
1. Sample mean (baseline overfit)
2. **Huber robust** (recomendado)
3. Shrunk 50% to zero (conservador)
4. Black-Litterman neutro

**Como escolher:**
- **Sharpe < 2.0** ✅
- **At ceiling < 3** ✅
- **N_eff ≥ 10** ✅

**Arquivos gerados:**
- `results/estimator_comparison_TIMESTAMP.csv`
- `results/weights_{estimator}_TIMESTAMP.csv`

---

### 3. `run_baselines_comparison.py`
**Tempo:** ~5-10 minutos  
**Output:** Métricas OOS de 6 estratégias

**Estratégias comparadas:**
1. 1/N (equal-weight)
2. Min-Variance (Ledoit-Wolf)
3. Risk Parity (ERC)
4. 60/40 (SPY/IEF)
5. HRP (Hierarchical Risk Parity)
6. **MV Robust (Huber)** ← Nossa estratégia

**Critério de sucesso:**
```
Sharpe(MV Robust) ≥ Sharpe(1/N) + 0.2
```

**Arquivos gerados:**
- `results/oos_metrics_comparison_TIMESTAMP.csv`
- `results/oos_returns_all_strategies_TIMESTAMP.csv`
- `results/oos_cumulative_TIMESTAMP.csv`

---

## 🎯 Workflow Recomendado

```
1. Run portfolio robusto
   ↓
2. Verificar métricas:
   - Sharpe < 2.0? ✅
   - N_eff ≥ 10? ✅
   - Nenhum ativo > 10%? ✅
   ↓
3. Se Sharpe > 2.5 → rodar estimator_comparison
   Escolher estimador com Sharpe mais realista
   ↓
4. Rodar baselines_comparison (OOS validation)
   ↓
5. Verificar:
   - MV Robust > 1/N + 0.2? ✅
   - Max DD < 20%? ✅
   - Sharpe OOS < Sharpe ex-ante? ✅ (normal)
   ↓
6. Se passou: USAR EM PRODUÇÃO ✅
   Se falhou: refinar estimadores ou aumentar shrinkage
```

---

## ⚠️ Pontos de Atenção

### 1. Budget Constraints Ativas
**Sintoma esperado:** solver retorna `infeasible` quando limites são incompatíveis.

**Como lidar:** ajuste `min_weight`/`max_weight` ou relaxe budgets conflitantes. Não há mais validação tardia – o modelo bloqueia violações na raiz.

---

### 2. Turnover Cap Reativado
**Uso:** definir `tau` (ou `turnover_cap`) no YAML aplica `∑|w - w_prev| ≤ tau`.

**Se falhar:** garanta `previous_weights` com todos os ativos (preencher ausentes com 0). Erros de dimensão indicam desalinhamento, não bug do solver.

---

### 3. Sharpe Ex-Ante Alto (>2.0)
**Sintoma:** Sharpe = 2.26 mesmo com Huber

**Motivo:** Período recente favorável (bull market crypto)

**Validação:** Rodar OOS - se Sharpe cai para ~0.8-1.2, normal

---

## 📈 Métricas de Referência

### Portfolio Robusto (ex-ante)
- **Sharpe:** 1.5 - 2.2 (vs 2.15 original)
- **Vol:** 12-16% anual
- **N_eff:** 10-12
- **Max position:** ≤ 10%

### Baselines OOS (esperado)
- **1/N:** Sharpe ~ 0.4-0.6
- **Min-Var:** Sharpe ~ 0.5-0.7
- **Risk Parity:** Sharpe ~ 0.6-0.8
- **MV Robust:** Sharpe ~ 0.7-1.0 **← OBJETIVO**

### Red Flags
- ❌ Sharpe ex-ante > 2.5 → overfit severo
- ❌ Sharpe OOS < 1/N → pior que baseline
- ❌ Max DD > 25% → risco excessivo
- ❌ Turnover > 20%/mês → custos altos

---

## 🔧 Customização

### Ajustar Risk Aversion
```python
# Em run_portfolio_arara_robust.py, linha ~70
RISK_AVERSION = 4.0  # Aumentar = mais conservador
```

### Ajustar Max Position
```python
# Linha ~71
MAX_POSITION = 0.10  # Reduzir = mais diversificação
```

### Ajustar Turnover Penalty
```python
# Linha ~74
TURNOVER_PENALTY = 0.0015  # Aumentar = menos trades
```

### Trocar Estimador de μ
```python
# Linha ~169-171
# Opção 1: Huber (default)
mu_huber, weights_eff = huber_mean(recent_returns, c=1.5)
mu_annual = mu_huber * 252

# Opção 2: Shrunk to zero
mu_sample = mean_return(recent_returns) * 252
mu_annual = bayesian_shrinkage_mean(mu_sample, prior_mean=0.0, tau=0.5)

# Opção 3: Black-Litterman neutro
# (ver exemplo completo no script)
```

---

## 💾 Outputs Importantes

### Pesos do Portfolio
```csv
ticker,weight
IBIT,0.1000
GLD,0.1000
QQQ,0.0987
...
```

### Métricas
```csv
sharpe_ratio,2.26
volatility,0.1524
expected_return,0.3447
effective_n,10.6
...
```

### Comparação OOS
```csv
Strategy,Sharpe,Max DD,CVaR 95%
1/N,0.52,-18.2%,-2.1%
MV Robust,0.89,-16.3%,-1.8%
...
```

---

## 📚 Documentação Completa

Ver `IMPLEMENTACAO_ROBUSTA.md` para:
- Detalhes técnicos completos
- Issues conhecidos e soluções
- Arquitetura do código
- Roadmap futuro

---

## 🆘 Troubleshooting

### Script trava ou demora muito
**Solução:** Reduzir universo de ativos ou usar dados em cache

### Solver retorna "inaccurate"
**Solução:** Normal com ECOS, resultado ainda válido

### Sharpe muito diferente entre execuções
**Causa:** Dados de mercado mudaram (yfinance atualiza)

### Budget constraints sempre violadas
**Esperado:** Constraints não integradas ao solver (limitação atual)

---

**Criado:** 2025-10-21  
**Versão:** 1.0  
**Contato:** Ver IMPLEMENTACAO_ROBUSTA.md
