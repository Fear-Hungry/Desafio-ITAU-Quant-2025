# BUG REPORT: Turnover Incorreto no PRISM-R

**Data de abertura:** 2025-01-XX  
**Situação:** ✅ **RESOLVIDO em 2025-02-XX**  
**Arquivos afetados:** `reports/walkforward/per_window_results.csv`, `reports/walkforward/trades.csv`, `README.md`  
**Severidade original:** ALTA — Impedia análise correta de custos de transação

---

## Resumo

O arquivo `reports/walkforward/per_window_results.csv` gerado pelo script legada `scripts/research/run_backtest_walkforward.py` registrava valores de turnover **2000x menores** que o esperado para a estratégia PRISM-R. Valores médios de ~1e-05 (0.001%) vs valores esperados de ~2e-02 (2%) observados nas baselines.

## Evidências

### Valores Observados (Período OOS: 2020-01-02 a 2025-10-09)

**PRISM-R (per_window_results.csv) — *ANTES DA CORREÇÃO*:**
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

## Impacto (antes da correção)

- **README Table 5.1:** valores de turnover do PRISM-R estavam vazios/errados
- **Análise de custos:** impossível validar custos transacionais
- **Comparação com baselines:** métricas incoerentes inviabilizavam benchmarking

## Correção implementada (2025-02-XX)

1. **Pipeline unificado:** `scripts/research/run_backtest_walkforward.py` foi substituído por um *wrapper* fino que delega para `itau_quant.backtesting.run_backtest` (mesma engine usada pelo CLI).  
2. **Instrumentação completa:** `_generate_wf_report` (CLI) agora salva:
   - `per_window_results_raw.csv` (split_metrics sem arredondamento)
   - `trades.csv` (turnover one-way e custos por rebalance)
   - `weights_history.csv` (matriz de pesos executados)
3. **Reexecução OOS canônica:**  
   ```bash
   poetry run python scripts/research/run_backtest_walkforward.py \
       --config configs/optimizer_example.yaml \
       --output-dir reports/walkforward
   ```
   Resultados (2020‑01‑02 → 2025‑10‑09):
   - Turnover median (trades): **9.48e-04** (0.0948%)
   - Turnover p95 (trades): **8.36e-03** (0.8356%)
   - Média: 6.04e-03 (0.604%) — coerente com `avg_turnover` do resumo
4. **README atualizado:** `scripts/update_readme_turnover_stats.py` agora consome `reports/walkforward/trades.csv` como fonte primária e preenche a Tabela 5.1 com os novos valores.

## Situação Atual

- `reports/walkforward/per_window_results.csv` e `per_window_results_raw.csv` trazem os valores corretos (mesmos da coluna `turnover` em `trades.csv`).  
- `README.md` mostra `Turnover mediano` = 9.48e-04 e `Turnover p95` = 8.36e-03 para PRISM-R.  
- Scripts de consolidação e validação (`scripts/augment_oos_metrics.py`, `scripts/update_readme_turnover_stats.py`) já usam a nova instrumentação.

## Lições / Workaround histórico

- Enquanto o bug existia, o `avg_turnover` agregado (retirado de `metrics_oos_canonical.csv`) era a única proxy confiável.  
- A nova instrumentação elimina essa necessidade ao expor diretamente cada rebalanceamento.

## Arquivos Relacionados

- `reports/walkforward/per_window_results.csv` — Arquivo com bug
- `results/oos_canonical/metrics_oos_canonical.csv` — Baselines (valores corretos de avg_turnover)
- `README.md` (linhas 119-131) — Tabela atualizada com "—" para PRISM-R
- `scripts/research/run_backtest_walkforward.py` — Script a ser revisado

## Status

- [x] Bug identificado e documentado
- [x] README atualizado com valores corretos de baselines
- [x] PRISM-R marcado como "-" até correção
- [x] Script de backtest substituído pelo pipeline oficial
- [x] Novos artefatos (`per_window_results_raw.csv`, `trades.csv`) gerados via `poetry run python scripts/research/run_backtest_walkforward.py`
- [x] README/Tabela 5.1 atualizados com turnovers precisos via `scripts/update_readme_turnover_stats.py`
- [x] Monitoramento contínuo: `trades.csv` passa a ser a fonte única de verdade para métricas de turnover/custo

---

**Criado:** 2025-01-XX  
**Atualizado:** 2025-02-XX — bug resolvido e documentação revisada
