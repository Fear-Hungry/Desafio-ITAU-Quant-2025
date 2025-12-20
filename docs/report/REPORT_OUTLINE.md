# PRISM-R — Draft Report Outline (≤ 10 páginas)

> Objetivo: organizar conteúdo e métricas para o relatório final institucional, incluindo seção explícita de **Uso de IA Generativa**.

## 1. Resumo Executivo (0.5 págs)
- Propósito da carteira ARARA e principais guardrails (vol ≤ 12%, DD ≤ 15%, CVaR5% ≤ 8% anual).
- Destaque para resultados mais recentes (`configs/optimizer_example_trimmed.yaml`, backtest em 2020-2025):
  - Sharpe OOS: 0.41 (excesso ao T‑Bill quando fornecida série RF diária)
  - Max Drawdown: -14.8%
  - Vol anualizada: 6.1%
  - Retorno total: 14.1%
- Três bullet points sobre diferenciais (custos explícitos, cardinalidade dinâmica, fallback ERC/1/N).

## 2. Universo & Dados (1 pág)
- Universo base: 69 ETFs globais (lista produzida por `get_arara_universe()`); atualização diária via yfinance.
- Pipeline principal gera `data/processed/returns_full.parquet` como cache, mas rodadas finais usaram download direto com `BASELINES_FORCE_DOWNLOAD=1`.
- Tratamento: forward-fill, corte de histórico para garantir ≥252 dias + 45 dias de buffer.
- Limitações: cripto ETFs recentes ainda têm janela curta; monitorar impacto em métricas de regime.

## 3. Metodologia de Estimação (1 pág)
- Estimadores de retorno (`shrunk_50`, janela 252d) e covariância (Ledoit-Wolf não linear).
- Calibração de custos: 10 bps lineares + slippage `adv20_piecewise`.
- Seeds e reprodutibilidade (`Settings.from_env`, `PYTHONPATH=src`).

## 4. Otimizador Principal (1.5 págs)
- Estrutura da função objetivo: MV + L1 custos + penalidade de turnover.
- Parâmetros testados:
  - `lambda` = 15.0 (baseline), 8.0, 6.0.
  - `eta` = 0.25, 0.15, 0.30.
  - Cardinalidade `[20, 35]`, `[18, 32]`, `[22, 34]`.
- Resultado comparativo (ver `outputs/reports/backtest_optimizer_*`):
  - `example_trimmed`: Sharpe 0.41, DD -14.8%.
  - `tuning_a`: Sharpe 0.31, DD -19.0%.
  - `tuning_b`: Sharpe 0.37, DD -20.3%.
- Conclusão: manter lambda alto preserva limite de drawdown; combinações agressivas melhoram retorno porém pioram cauda.

## 5. Backtest Walk-Forward (1 pág)
- Configuração: janela 252/21, purge/embargo 2, 60 splits (2021-2025).
- Destaque para warmup:
  - Turnover 1ª janela ≈ 100%; subsequentes ≈ 0% (ver JSON `walkforward[...]`).
  - Volatilidade estabiliza entre 4% e 9% após o warmup.
- Métricas agregadas (Sharpe 0.41 em excesso ao T‑Bill quando RF diária é usada, NAV final 1.14, CVaR95% anual ≈ -20% – calcular de `metrics["cvar_95_annual"]` ou `cvar_95 × √252`).
- Discussão sobre gap p/ baseline: equal-weight Sharpe ~0.43 (janela curta 2024.07+ no comparativo local).

## 6. Regime Stress & Mitigações (1 pág)
- Thresholds revisados (`vol calm 6%`, `stressed 10%`, `drawdown crash -8%`, multiplicadores 0.75/1.0/2.5/4.0).
- Resultados (ver `outputs/results/regime_stress/*_metrics.csv`):
  - Covid 2020: Sharpe -3.25, DD -1.19%, vol 4.3% (reduziu dd vs versão anterior, porém performance ainda negativa).
  - Inflação 2022: Sharpe -0.47, DD -1.12%, vol 3.6% (melhora sobre MV robusto, ainda abaixo de EW/RP).
- Próximas ações: incorporar views/tail hedges para evitar Sharpe negativo mesmo com λ elevado.

## 7. Comparação com Baselines (1 pág)
- **Convenção CVaR:** Todos os valores reportados são **anualizados** (CVaR_diário × √252) para consistência com volatilidade e retorno. CVaR diário disponível em `cvar_95` para debug/monitoramento.
- Script agora baixa dados reais com `BASELINES_FORCE_DOWNLOAD=1` e `BASELINES_DOWNLOAD_SLEEP=1`, carregando os 69 tickers de `get_arara_universe()`.
- Amostra OOS: 2019-10-01 a 2025-10-09 (`outputs/results/baselines/baseline_metrics_oos.csv`).
  - Sharpe: Min-Var (0.69), Equal-Weight (0.69), Shrunk MV (0.69), ERC (0.65); drawdowns variam de -3.4% (Min-Var) a -21.7% (Shrunk MV).
  - Stress tests mostram Shrunk MV positivo em 2022 (+7.4%), demais estratégias negativas; 2023 banking stress favorável às carteiras defensivas.
- Snapshot curto anterior (2024-07 → 2025-10) permanece no README apenas como smoke test; explicitar baixa confiabilidade.

## 8. Operação & Monitoramento (1 pág)
- Descrever produção ERC v2: cash floor dinâmico 15%-40%, triggers (Sharpe 6M ≤0, CVaR diário <-2% ou ~-32% anual, DD<-10%).
- Processo mensal + fallback 1/N, logs em `outputs/results/production/`.
- Checklist warmup: rodar 3-6 rebalanceamentos usando pesos persistidos, verificar turnover médio <12%.

## 9. Uso de IA Generativa (0.5–1 pág)
- Descrever como LLMs auxiliaram:
  - Exploração de parâmetros (`lambda`, `eta`, cardinalidade) e interpretação de métricas.
  - Ajuste de scripts (`run_regime_stress.py`, `run_baselines_comparison.py`) para ambientes offline.
  - Estruturação do próprio relatório (este outline) e identificação de gaps (drawdown, baselines).
- Registrar controles de revisão humana e validações (execução via `poetry`, inspeção manual de métricas).

## 10. Próximos Passos & Riscos (0.5 pág)
- Necessidade de painel mais longo para baselines/bootstraps.
- Ajustar overlay para reduzir DD > 15%.
- Automatizar pipeline PDF (WeasyPrint/LaTeX) mantendo ≤10 páginas.
- Revisar operação antes de scale-up (broker API, alertas).

---

### Artefatos de Suporte
- Backtests: `outputs/reports/backtest_optimizer_example_trimmed_*.json`, `outputs/reports/backtest_optimizer_tuning_*.json`.
- Regime stress: `outputs/results/regime_stress/*.csv`.
- Baselines: `outputs/results/baselines/baseline_metrics_oos.csv`.
- Produção: `outputs/results/production/production_log.csv`.

### Checklist de Conteúdo
- [ ] Inserir gráficos NAV/vol (usar `outputs/reports/figures/*` ou regenerar).
- [ ] Exportar tabelas de métricas para LaTeX/Markdown.
- [ ] Garantir seções ≤10 páginas (contagem aproximada acima = 8.5–9 págs).
- [ ] Validar narrativa de riscos + seção GenAI com compliance interno.
