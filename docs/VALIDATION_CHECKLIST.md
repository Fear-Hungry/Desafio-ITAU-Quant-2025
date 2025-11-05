# Checklist de Validação - README.md Corrigido

**Data:** $(date +%Y-%m-%d)
**Status:** 🔄 Pendente de validação pelo usuário

---

## ✅ Correções Críticas (VERIFICAR)

- [ ] **Moeda base:** Confirmado "USD base" (não BRL) em todo o documento
- [ ] **Parâmetro η:** Confirmado η=0 na execução OOS canônica
- [ ] **Custos:** Confirmado 30 bps em README.md e CLAUDE.md
- [ ] **162 vs 64 splits:** Esclarecido (162 histórico, 64 OOS oficial)
- [ ] **Universo 69 vs 66:** Explicado ETHA/FBTC/IBIT excluídos
- [ ] **Turnover:** Nota de investigação adicionada

---

## 📚 Expansões Técnicas (VERIFICAR COMPLETUDE)

### Seção 2: Dados e Fontes
- [ ] Fontes detalhadas (Yahoo Finance, Tiingo, FRED)
- [ ] Universo completo por classe (66 ativos listados)
- [ ] Pipeline de 7 etapas documentado
- [ ] Comando de reprodução com flags

### Seção 3: Universo e Regras
- [ ] 6 grupos de ativos com caps detalhados
- [ ] Hierarquia hard vs soft explicada
- [ ] Fórmulas de constraints (box, group, budget)
- [ ] Rodapé copy-paste ready

### Seção 4: Metodologia
- [ ] Estimadores: Shrunk_50 com fórmula completa
- [ ] Covariância: Ledoit-Wolf com referência
- [ ] ERC: Definição matemática de RC_i
- [ ] PRISM-R: Função objetivo com η=0 clara
- [ ] Solver: CLARABEL com tolerâncias
- [ ] Modo defensivo: Gatilhos e ajustes
- [ ] Fallback 1/N: Condições explicadas

### Seção 5: Avaliação
- [ ] 8 métricas por janela com fórmulas
- [ ] 8 métricas OOS série completa
- [ ] Turnover: Definição one-way precisa
- [ ] Distinção janela vs série diária (tabela)

---

## 🔍 Rastreabilidade (VERIFICAR ARQUIVOS EXISTEM)

- [ ] `configs/universe_arara.yaml` - Existe e tem 69 tickers
- [ ] `configs/asset_groups.yaml` - Existe e tem 6 grupos
- [ ] `configs/oos_period.yaml` - Existe com período 2020-2025
- [ ] `data/processed/returns_arara.parquet` - Existe
- [ ] `reports/walkforward/nav_daily.csv` - Existe (1,451 linhas)
- [ ] `reports/oos_consolidated_metrics.json` - Existe
- [ ] `reports/walkforward/per_window_results.csv` - Existe (64 janelas)

---

## 🧪 Reprodutibilidade (TESTAR COMANDOS)

```bash
# Teste 1: Data pipeline
poetry run python scripts/run_01_data_pipeline.py \
    --force-download --start 2010-01-01 --end 2025-10-09

# Teste 2: Walk-forward backtest
poetry run python scripts/research/run_backtest_walkforward.py \
    --start-oos 2020-01-02 --end-oos 2025-10-09

# Teste 3: Consolidação
poetry run python scripts/consolidate_oos_metrics.py

# Teste 4: Figuras
poetry run python scripts/generate_oos_figures.py
```

- [ ] Comando 1 executado sem erros
- [ ] Comando 2 executado sem erros
- [ ] Comando 3 gerou `oos_consolidated_metrics.json`
- [ ] Comando 4 gerou figuras em `reports/figures/`
  - Nota: o gráfico de comparação PRISM‑R vs baselines exibe Sharpe em excesso ao T‑Bill (usar `consolidate_oos_metrics.py --riskfree-csv ...`).

---

## 📐 Fórmulas Matemáticas (VALIDAR LATEX)

- [ ] Shrinkage: \hat{\mu}_i = (1-\delta)\bar{r}_i + \delta\mu_{\text{prior}}
- [ ] Ledoit-Wolf: \hat{\Sigma} = \delta F + (1-\delta)S
- [ ] RC_i: w_i \cdot (\Sigma w)_i
- [ ] Turnover: \text{TO} = \frac{1}{2}\sum_i |w_i - w_{i,t-1}|
- [ ] CVaR: -\mathbb{E}[r \mid r \leq Q_{0.05}]
- [ ] Sharpe: r_{\text{ann}} / \sigma_{\text{ann}}

---

## 🐛 Bugs Conhecidos (VERIFICAR STATUS)

- [ ] **Turnover PRISM-R:** Bug documentado em `BUG_TURNOVER_PRISM_R.md`
- [ ] Valores ~1e-05 vs baselines ~0.04-0.07% (2000x diferença)
- [ ] Investigação em andamento?

---

## 📊 Consistência Numérica (VALIDAR)

Da seção 5.6 (Consolidação OOS):
- [ ] NAV final = 1.0289
- [ ] Total return = 2.89%
- [ ] Retorno anualizado = 0.50%
- [ ] Volatilidade = 8.60%
- [ ] Sharpe = 0.0576
- [ ] Max Drawdown = -20.89%
- [ ] N_days = 1,451

Fórmula de validação:
\`\`\`python
nav_final = 1.028866
n_days = 1451
ann_return = (nav_final ** (252 / n_days)) - 1
# Deve dar ~0.004954 (0.50%)
\`\`\`

- [ ] Fórmula validada numericamente

---

## 🎯 Próximas Ações

- [ ] Executar todos os comandos de reprodução
- [ ] Verificar métricas batem com artefatos
- [ ] Resolver bug de turnover (se confirmado)
- [ ] Adicionar seções 8 (Discussão) e 9 (Governança) se solicitado

---

**Gerado por:** validation_checklist_generator.sh
**Data:** $(date)
