# BUG REPORT: Turnover Incorreto no PRISM-R

**Data:** 2025-01-XX  
**Arquivo afetado:** `reports/walkforward/per_window_results.csv`  
**Severidade:** ALTA — Impede análise correta de custos de transação

---

## Resumo

O arquivo `reports/walkforward/per_window_results.csv` registra valores de turnover **2000x menores** que o esperado para a estratégia PRISM-R. Valores médios de ~1e-05 (0.001%) vs valores esperados de ~2e-02 (2%) observados nas baselines.

## Evidências

### Valores Observados (Período OOS: 2020-01-02 a 2025-10-09)

**PRISM-R (per_window_results.csv):**
- Mediana: 8.39e-06 (0.000839%)
- P95: 1.50e-05 (0.001500%)
- Média: 9.51e-06 (0.000951%)

**Baselines (recalculados corretamente):**
- Equal-Weight 1/N: mediana=4.52e-04 (0.045%), p95=9.39e-04 (0.094%)
- Risk Parity (ERC): mediana=4.43e-04 (0.044%), p95=9.01e-04 (0.090%)
- Min-Variance LW: mediana=1.29e-04 (0.013%), p95=2.19e-04 (0.022%)

**Discrepância:** PRISM-R apresenta turnover **53-350x menor** que baselines passivos (1/N, 60/40), o que é fisicamente impossível para uma estratégia de otimização ativa com rebalanceamento mensal.

### Comparação com avg_turnover

No arquivo `results/oos_canonical/metrics_oos_canonical.csv`, as baselines apresentam:
- Equal-Weight: avg_turnover = 1.92e-02 (1.92% por rebalance)
- Risk Parity: avg_turnover = 2.67e-02 (2.67% por rebalance)

Esses valores são **2000x maiores** que os registrados para PRISM-R no per_window_results.csv.

## Hipóteses

1. **Métrica errada registrada:** A coluna "Turnover" pode estar registrando:
   - Turnover diário médio dentro da janela (em vez de turnover de rebalance)
   - Apenas uma componente do turnover (ex: apenas long-only, sem contar shorts)
   - Turnover normalizado por algum fator incorreto

2. **Bug no cálculo:** O script de backtest walk-forward pode estar calculando:
   ```python
   # ERRADO (possivelmente implementado):
   turnover = 0.5 * sum(abs(w_new - w_old)) / N_days_in_window
   
   # CORRETO (deveria ser):
   turnover = 0.5 * sum(abs(w_new - w_old))  # one-way turnover no rebalance
   ```

3. **Divisão por período:** Valores podem estar divididos por número de dias na janela (~21 dias), resultando em diluição de ~20x.

## Impacto

- **README Table 5.1:** Valores de turnover (mediana/p95) para PRISM-R marcados como "—" (não disponíveis)
- **Análise de custos:** Impossível validar se custos de transação estão sendo contabilizados corretamente
- **Comparação com baselines:** Não é possível comparar turnover realizado de PRISM-R vs alternativas

## Ação Requerida

1. **Revisar script de backtest walk-forward** (`scripts/research/run_backtest_walkforward.py`):
   - Verificar cálculo de turnover em cada janela
   - Confirmar definição: `turnover = 0.5 * sum(abs(w_target - w_pretrade))`
   - Garantir que não há divisão por número de dias

2. **Reexecutar backtest** para período OOS com correção:
   ```bash
   poetry run python scripts/research/run_backtest_walkforward.py \
       --config configs/optimizer_example.yaml \
       --start 2020-01-02 \
       --end 2025-10-09
   ```

3. **Validar output:**
   - Turnover médio deve estar em ordem de grandeza 1e-02 a 5e-02 (1-5%)
   - Primeira janela (transição) pode ter turnover maior (~50-100%)

4. **Atualizar README** com valores corretos após correção

## Workaround Temporário

Para análise de custos, usar valores de `avg_turnover` da tabela de métricas consolidadas (`metrics_oos_canonical.csv`) quando disponível, ou estimar turnover baseado em:
- Volatilidade do portfólio
- Frequência de rebalanceamento (mensal)
- Penalização L1 configurada (η)

## Arquivos Relacionados

- `reports/walkforward/per_window_results.csv` — Arquivo com bug
- `results/oos_canonical/metrics_oos_canonical.csv` — Baselines (valores corretos de avg_turnover)
- `README.md` (linhas 119-131) — Tabela atualizada com "—" para PRISM-R
- `scripts/research/run_backtest_walkforward.py` — Script a ser revisado

## Status

- [x] Bug identificado e documentado
- [x] README atualizado com valores corretos de baselines
- [x] PRISM-R marcado como "—" até correção
- [ ] Script de backtest corrigido
- [ ] Novo per_window_results.csv gerado
- [ ] README atualizado com valores corretos de PRISM-R

---

**Criado:** 2025-01-XX  
**Última atualização:** 2025-01-XX