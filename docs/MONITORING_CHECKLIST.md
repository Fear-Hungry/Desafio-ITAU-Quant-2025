# Portfolio Production Monitoring Checklist

**Sistema:** PRISM-R Portfolio Risk Intelligence System  
**Estratégia:** ERC v2.0 (Equal Risk Contribution Calibrado)  
**Configuração:** K=22, Vol Target 7.5%, Max Position 8%  
**Frequência:** Mensal (último dia útil do mês)

---

## 📋 Checklist de Rebalanceamento Mensal

### 1. PRÉ-REBALANCE (T-1)

- [ ] **Dados atualizados**
  - Baixar preços históricos dos últimos 252 dias (1 ano rolling)
  - Verificar ausência de NaNs ou dados faltantes
  - Validar que todos os 69 tickers têm dados recentes (< 5 dias)

- [ ] **Triggers de Fallback**
  - Verificar Sharpe 6M > 0.0 (se ≤ 0, fallback para 1/N)
  - Verificar CVaR 95% > -2% (se < -2%, fallback)
  - Verificar Max DD > -10% (se < -10%, fallback)
  
- [ ] **Configuração**
  - Validar `configs/production_erc_v2.yaml` está com parâmetros corretos
  - K=22, vol_target=0.075, max_position=0.08, min_position=0.02

### 2. EXECUÇÃO DO REBALANCE (T)

- [ ] **Rodar Otimizador**
  ```bash
  poetry run python scripts/production/run_portfolio_production_erc_v2.py
  ```

- [ ] **Validar Output Imediato**
  - Cardinalidade = 22 ativos ✅
  - Vol ex-ante = 6.5-7.5% ✅
  - Max position ≤ 8% ✅
  - Min position ≥ 2% (ativos ativos) ✅
  - Group constraints respeitados ✅

### 3. VALIDAÇÕES PÓS-REBALANCE (T)

- [ ] **Arquivo de Pesos** (`results/production/weights/weights_YYYYMMDD.csv`)
  ```python
  import pandas as pd
  w = pd.read_csv('results/production/weights/weights_YYYYMMDD.csv', index_col=0)
  
  # Check 1: Soma = 100%
  assert abs(w['weight'].sum() - 1.0) < 1e-6
  
  # Check 2: K = 22 ativos ativos
  assert (w['weight'] > 1e-10).sum() == 22
  
  # Check 3: Min weight ≥ 2%
  active = w[w['weight'] > 1e-10]['weight']
  assert active.min() >= 0.02
  
  # Check 4: Max weight ≤ 8%
  assert w['weight'].max() <= 0.08
  ```

- [ ] **Production Log** (`results/production/production_log.csv`)
  ```python
  log = pd.read_csv('results/production/production_log.csv')
  last_row = log.iloc[-1]
  
  # Verificar métricas
  assert last_row['n_active_assets'] == 22
  assert 20.0 <= last_row['n_effective'] <= 24.0  # ~Equal-weight
  assert last_row['vol_realized'] >= 0.06 and last_row['vol_realized'] <= 0.08
  ```

- [ ] **Turnover Mensal**
  - **Primeiro mês (transição):** Esperado ~130-140% ✅ (normal)
  - **Meses seguintes:** Esperado 10-20% ✅ (se > 30%, investigar)
  
  ```python
  # Calcular turnover mensal médio (excluindo primeiro)
  turnover_mean = log.iloc[1:]['turnover_realized'].mean()
  print(f"Turnover médio (pós-transição): {turnover_mean:.2%}")
  
  # Alertar se muito alto
  if turnover_mean > 0.30:
      print("⚠️ ATENÇÃO: Turnover acima de 30% - investigar volatilidade das seleções")
  ```

### 4. ANÁLISE DE PERFORMANCE MENSAL

- [ ] **Métricas Rolling (6 meses)**
  - Sharpe Ratio: Manter > 2.5 (alerta se < 2.0)
  - Volatilidade: 6-8% aa (alerta se > 10% ou < 5%)
  - CVaR 95%: < -2% (alerta se < -3%)
  - Max DD: < -10% (alerta se < -15%)

- [ ] **Exposições Agregadas**
  ```python
  # Verificar exposições por classe de ativo
  groups = {
      'us_equity': ['SPY', 'QQQ', 'IWM', 'VTV', 'VUG', 'VYM', 'SCHD', 'SPLV'],
      'commodities': ['DBC', 'USO', 'GLD', 'SLV', 'CORN'],
      'crypto': ['IBIT', 'ETHA', 'FBTC', 'GBTC', 'ETHE'],
      'treasuries': ['IEF', 'TLT', 'SHY', 'VGSH', 'VGIT'],
  }
  
  for group, assets in groups.items():
      exposure = w[w.index.isin(assets)]['weight'].sum()
      print(f"{group}: {exposure:.2%}")
  
  # Alertas
  # - US Equity: deve estar entre 25-55%
  # - Commodities: ≤ 25%
  # - Crypto: ≤ 12%
  # - Treasuries: ≤ 45%
  ```

- [ ] **Contribuição de Risco**
  ```python
  import numpy as np
  from itau_quant.estimators.cov import ledoit_wolf_shrinkage
  
  # Calcular contribuições marginais de risco
  returns = pd.read_parquet('data/processed/returns_full.parquet').tail(252)
  cov, _ = ledoit_wolf_shrinkage(returns)
  cov_annual = cov * 252
  
  weights_active = w[w['weight'] > 1e-10]['weight']
  assets_active = weights_active.index
  
  cov_sub = cov_annual.loc[assets_active, assets_active]
  w_vec = weights_active.values
  
  # Contribuição marginal: MCR_i = (Σw)_i / sqrt(w'Σw)
  portfolio_vol = np.sqrt(w_vec @ cov_sub @ w_vec)
  mcr = (cov_sub @ w_vec) / portfolio_vol
  contribution = w_vec * mcr
  
  # Top contributors
  contrib_df = pd.DataFrame({
      'weight': weights_active,
      'mcr': mcr,
      'contribution': contribution,
      'contrib_pct': contribution / contribution.sum()
  }).sort_values('contrib_pct', ascending=False)
  
  print(contrib_df.head(10))
  
  # Alerta: Nenhum ativo deve contribuir > 15% do risco total
  if (contrib_df['contrib_pct'] > 0.15).any():
      print("⚠️ Concentração de risco detectada!")
  ```

### 5. COMPARAÇÃO COM EXPECTATIVAS

- [ ] **Benchmark vs. Baseline**
  - Comparar performance vs. Equal-Weight (1/N)
  - Comparar performance vs. Min Variance
  - Validar que Sharpe está acima dos baselines

- [ ] **Custos Realizados**
  ```python
  # Custos mensais acumulados
  costs_monthly = log.groupby(pd.to_datetime(log['date']).dt.to_period('M'))['cost_bps'].sum()
  
  # Alerta se custos > 50 bps em qualquer mês
  if (costs_monthly > 50).any():
      print("⚠️ Custos mensais acima de 50 bps!")
  
  # Custos anualizados (rolling 12M)
  costs_annual = costs_monthly.rolling(12).sum()
  print(f"Custos anualizados (12M): {costs_annual.iloc[-1]:.1f} bps")
  ```

### 6. ALERTAS E GATILHOS

| Métrica | Green | Yellow | Red | Ação |
|---------|-------|--------|-----|------|
| **Sharpe 6M** | > 2.5 | 2.0-2.5 | < 2.0 | Revisar estimadores |
| **Volatilidade** | 6-8% | 5-6% ou 8-10% | < 5% ou > 10% | Ajustar vol_target |
| **Turnover Mensal** | 10-20% | 20-30% | > 30% | Aumentar turnover_penalty |
| **Max DD** | < -10% | -10% a -15% | < -15% | Ativar fallback |
| **Concentração Risco** | < 12% por ativo | 12-15% | > 15% | Reduzir max_position |
| **Custos Anuais** | < 50 bps | 50-75 bps | > 75 bps | Reduzir rebalance freq |

### 7. DOCUMENTAÇÃO MENSAL

- [ ] **Criar Relatório Mensal**
  ```
  results/production/reports/YYYY_MM_monthly_report.md
  ```
  
  Conteúdo mínimo:
  - Data do rebalance
  - Ativos selecionados (22)
  - Turnover realizado
  - Custos do mês
  - Performance MTD
  - Métricas rolling 6M
  - Alertas/gatilhos acionados
  - Mudanças vs. mês anterior

- [ ] **Commit Changes**
  ```bash
  git add results/production/
  git commit -m "chore: monthly rebalance YYYY-MM-DD - K=22, turnover=XX%, vol=XX%"
  git push
  ```

---

## 🚨 Situações Especiais

### Fallback para 1/N
**Quando:** Sharpe ≤ 0, CVaR < -2%, ou DD < -10%  
**Ação:**
1. Implementar equal-weight (1/69 para todos os ativos)
2. Manter por 1 mês
3. Revisar triggers no próximo rebalance
4. Documentar razão do fallback

### Rebalance Extraordinário
**Quando:** 
- DD atinge -12% intra-mês
- Vol realizada > 12% por 5 dias consecutivos
- Trigger manual por evento de mercado

**Ação:**
1. Rodar otimizador com data atual
2. Calcular turnover vs. último rebalance
3. Executar se turnover < 40% (senão, aguardar rebalance regular)
4. Documentar evento extraordinário

---

## 📊 Métricas de Referência (Baseline)

**Esperado após transição (mês 2+):**
- Cardinalidade: 22 ativos
- Vol ex-ante: 6.5-7.5% aa
- Sharpe 6M: 2.5-3.5
- CVaR 95% (anual): -12.7% a -19.0% (equiv. -0.8% a -1.2% diário)
- Max DD: < -8%
- Turnover mensal: 10-20%
- Custos mensais: 5-15 bps
- N_effective: 20-23

**Primeiro mês (transição):**
- Turnover: ~130-140% (esperado e normal)
- Custos: ~20 bps (one-time)

---

## 📅 Calendário de Monitoramento

| Frequência | Tarefa | Responsável |
|-----------|--------|-------------|
| **Mensal** | Rebalance + validações | Portfolio Manager |
| **Mensal** | Relatório de performance | Quant Team |
| **Trimestral** | Revisão de parâmetros (K, vol_target) | CIO |
| **Semestral** | Backtest walk-forward completo | Quant Team |
| **Anual** | Revisão de universo (adicionar/remover ativos) | Investment Committee |

---

## 🔧 Comandos Úteis

```bash
# 1. Rebalance mensal
poetry run python scripts/production/run_portfolio_production_erc_v2.py

# 2. Validar reprodutibilidade
poetry run python scripts/production/run_portfolio_production_erc_v2.py > run1.txt
poetry run python scripts/production/run_portfolio_production_erc_v2.py > run2.txt
diff run1.txt run2.txt  # Deve ser idêntico

# 3. Rodar testes de cardinalidade
poetry run pytest tests/optimization/test_postprocess.py tests/optimization/heuristics/test_cardinality.py

# 4. Backtest walk-forward
poetry run python -m itau_quant.cli walkforward

# 5. Análise rápida de pesos
poetry run python -c "
import pandas as pd
w = pd.read_csv('results/production/weights/weights_$(date +%Y%m%d).csv', index_col=0)
print(w[w['weight'] > 1e-10].sort_values('weight', ascending=False))
"

# 6. Ver log de produção
tail -5 results/production/production_log.csv | column -t -s,
```

---

**Última atualização:** 2025-10-25  
**Versão:** 1.0  
**Próxima revisão:** 2025-11-25
