# Adaptive Robust Optimization: Enhancing Mean-Variance Portfolios with Regime-Aware Risk Control

Resumo (draft) — Propomos um overlay interpretável de regimes (volatilidade/drawdown) sobre um otimizador mean-variance robusto com custos e budgets conservadores. Ele ajusta a aversão a risco e limites por regime e visa recuperar a performance perdida por versões estáticas em 2020–2025 (Sharpe -0.21, DD -20.9%) sem abandonar convexidade nem rastreabilidade operacional.

## 1. Introdução
- Problema: estimativas de retorno/covariância instáveis e custos ignorados levam a MV frágil. Mesmo com shrinkage (Ledoit-Wolf, Shrunk_50) e bounds, a versão estática teve Sharpe -0.21 no OOS 2020–2025.
- Lacuna: métodos “robustos” tradicionais assumem λ fixo/estacionaridade; HMMs são caros e propensos a overfitting.
- Contribuição: overlay de regime simples (gatilhos de vol e drawdown observáveis) que ajusta λ, bounds e caixa mantendo a convexidade. Hipótese: reduzir DD e melhorar Sharpe/CVaR vs. MV estática, com custos/turnover explícitos.

## 2. Dados e cenário
- Universo: 66 ETFs (USD) conforme `configs/universe/universe_arara.yaml` (excluídos ETHA/FBTC/IBIT por histórico curto).
- Períodos: Treino rolling 252d, teste 21d (purge/embargo 2d); OOS oficial 2020-01-02 a 2025-10-09. Histórico carregado desde 2010-01-01.
- Fontes: Yahoo Finance (ajustado), fallback Tiingo para cripto, T-Bill diário via FRED (quando disponível). Série RF≈0 usada em artefatos canônicos; recalcular com T-Bill: `poetry run python -m arara_quant.runners.data.fetch_tbill_fred` + `poetry run python -m arara_quant.runners.reporting.consolidate_oos_metrics --riskfree-csv data/processed/riskfree_tbill_daily.csv`.
- Artefatos: retornos/cov em `data/processed/*`, NAV e métricas em `outputs/reports/oos_consolidated_metrics.json`, janelas WF em `outputs/reports/walkforward/*`, baselines em `outputs/results/baselines/`, stress regimes em `outputs/results/regime_stress/`.

## 3. Metodologia base (PRISM-R estática)
- Estimadores: μ Shrunk_50 (força 0.5, janela 252d), Σ Ledoit-Wolf não linear.
- Otimizador: `max_w μ^T w - (λ/2) w^T Σ w - costs(w,w_{t-1})`; λ=15 baseline; custos lineares 30 bps round-trip aplicados ao turnover one-way.
- Restrições: 0 ≤ w_i ≤ 10%; budgets por 11 buckets; soma = 1. Cardinalidade desativada na rodada canônica. Solver CVXPY+Clarabel.
- Convenções: CVaR reportado anualizado (ver `docs/standards/CVAR_CONVENTION.md`); turnover como ‖Δw‖₁ one-way (pré-trade vs pós-drift).

## 4. Overlay regime-aware (proposto)
- Sinalização (exemplo, ajustar conforme experimento):
  - Calm: vol_21d_ann < 6% → λ = 0.75 λ_base; bounds padrão.
  - Stressed: vol_21d_ann > 10% ou DD rolling < -8% → λ = 2.5 λ_base; reduzir limites por classe/ativo; opcional caixa mínima 5–10%.
  - Crash: DD < -15% → λ = 4.0 λ_base; ampliar caixa/afrouxar orçamentos defensivos.
- Implementação: overlay antes de cada rebalance mensal (usa NAV/vol/DD das janelas WF), mantendo problema convexo. Seeds explícitos: numpy RNG 42 (smoke), 0 (tests), 777 (GA).
- Variante opcional: ajustar também penalização de turnover/custos (c) por regime para evitar saltos.

## 5. Protocolos experimentais
- Repro OOS canônico (baseline estático): `poetry install --sync && make reproduce-oos` → `outputs/reports/oos_consolidated_metrics.json`, `outputs/reports/walkforward/per_window_results.csv`, figuras em `outputs/reports/figures/`.
- Regenerar figuras: `poetry run python -m arara_quant.runners.reporting.generate_oos_figures`.
- Comparar baselines: `outputs/results/baselines/baseline_metrics_oos.csv` (Equal Weight, 60/40, MV shrink, ERC, Min-Var).
- Stress slices: `outputs/results/regime_stress/*.csv` (covid 2020, inflação 2022). Adaptar overlay para rodar nos mesmos recortes.
- Validação rápida offline: `poetry run pytest tests/backtesting/test_engine_walkforward.py::test_turnover_matches_half_l1_pretrade -q`.

## 6. Resultados preliminares
Tabela OOS 2020–2025 (preencher após rodar overlay):

| Estratégia                | Ret Ann | Vol Ann | Sharpe (ex RF) | Max DD | CVaR95 Ann | Turnover med | Custos bps/ano |
| ---                       | ---     | ---     | ---            | ---    | ---        | ---          | ---            |
| MV robusta estática (λ=15)| 0.50%   | 8.60%   | -0.21          | -20.89%| -20.23%    | 0.00067      | 8.79           |
| Equal Weight              | 4.32%   | 11.18%  | 0.26           | -25.88%| (preencher)| 0.0192       | (preencher)    |
| 60/40                     | (preencher) | ( ) | ( )           | ( )    | ( )        | ( )          | ( )            |
| Regime-aware (overlay)    | TODO    | TODO    | TODO           | TODO   | TODO       | TODO         | TODO           |

Notas:
- Valores da linha estática vêm de `outputs/reports/oos_consolidated_metrics.json` e `outputs/reports/walkforward/per_window_results.csv`.
- Complete Equal Weight/60-40 a partir de `outputs/results/baselines/baseline_metrics_oos.csv`.
- Após rodar overlay, documentar config, seed e artefato gerado (salvar JSON/CSV em `outputs/reports/` ou `outputs/results/regime_stress/`).

## 7. Ablations e sensitividade
- Variação de thresholds de vol (6/8/10%) e multiplicadores de λ (0.75/1.0/2.5/4.0).
- Overlay só em λ vs. overlay em λ + bounds + caixa.
- Penalização de custos dinâmica vs. fixa.
- Comparação com HMM simples (2 estados) para mostrar risco de overfitting vs. gatilhos observáveis.

## 8. Robustez e reproducibilidade
- Caminho único para dados/artefatos: `outputs/reports/walkforward/nav_daily.csv` é a fonte canônica; métricas consolidadas em `outputs/reports/oos_consolidated_metrics.json`.
- Scripts versionados (Poetry), comandos explícitos (make/poetry). Sem segredos em repo; `.env` local.
- Determinismo: seeds documentados; solver tolerâncias padrão (Clarabel 1e-8).
- Checklist de validação: `docs/guides/VALIDATION_CHECKLIST.md`; convenção CVaR em `docs/standards/CVAR_CONVENTION.md`.

## 9. Trabalhos relacionados (posicionar contribuição)
- Markowitz (1952); Ledoit & Wolf (2004); Jagannathan & Ma (2003).
- Regime switching via HMM (destacar simplicidade/parsimonia do overlay).
- Tail risk parity / modelos de cauda: contraste com custo computacional e falta de custos/turnover.

## 10. Conclusão (alvo)
- Declarar ganho absoluto/relativo do overlay (Sharpe/CVaR/DD/turnover) vs. MV estática e baselines.
- Limitações: thresholds heurísticos, dependência de indicadores de vol/DD; impacto quando RF real é usada.
- Próximos passos: calibrar λ/η e budgets, overlay defensivo para cash, incorporar estimadores de μ alternativos (BL, regressão bayesiana) e slippage `adv20_piecewise`.

---

# Análise e sugestões para publicação científica (roadmap)

## Pontos fortes atuais
- Problema bem delimitado: MV frágil + custos ignorados em modelos estáticos.
- Reprodutibilidade sólida: seeds, artefatos versionados, comandos explícitos (make/poetry).
- Baseline robusto: shrinkage (LW + Shrunk_50), custos explícitos, budgets conservadores.
- Simplicidade interpretável: gatilhos observáveis de vol/DD ao invés de HMMs complexos.

## Inovações propostas (para ficar publicável)

### A. Contribuições teóricas
- Provar que o overlay regime-aware preserva convexidade; bound de suboptimalidade vs. fronteira estática.
- Analisar estabilidade dinâmica e condições de não-arbitragem sob switching de λ; conectar com controle estocástico.
- Formular framework unificado
  ```
  J(w) = μᵀw - λ(regime)·wᵀΣw - c(regime)·TC(w) + g(regime,constraints)
  ```
  e provar existência/unicidade sob hipóteses brandas.

### B. Experimentos diferenciadores
- Backtests de crises históricas (1987, 2008, 2011, 2015, 2018) com overlay vs. estático; organizar em `src/arara_quant/runners/crisis_analysis/` com `crisis_periods.yaml`.
- Custos realistas: implementar `adv20_piecewise` e comparar com 30 bps lineares; “cost-aware regime switching” (ajustar thresholds considerando fricção).
  ```python
  def cost_nonlinear(trade_value, adv20):
      return linear_bps + impact_factor * (trade_value / adv20)**power
  ```
- Out-of-sample em outros universos: internacional, cripto nativo, commodities; estruturar em `experiments/` por universo.

### C. Comparações com SOTA
- Baselines adicionais: DRO (Delage & Ye), HMM-MV, LSTM/Transformer para cov/regime, Graphical Lasso, tail risk parity.
- Tabela obrigatória com Sharpe, CVaR95, turnover e custo computacional (O(n²)/GPU etc.).

### D. Sensibilidade rigorosa
- Grid search documentado para thresholds de vol/DD e multiplicadores de λ (heatmaps Sharpe × thresholds × λ).
  ```python
  # arara_quant.runners.sensitivity.grid_search
  lambda_mults = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
  vol_thresholds = [0.04, 0.06, 0.08, 0.10, 0.12]
  dd_thresholds = [-0.05, -0.08, -0.10, -0.15, -0.20]
  ```
- Decompor ganhos: (a) λ dinâmico; (b) bounds dinâmicos; (c) cash overlay. Avaliar look-ahead bias simulando delay na detecção de regime.

### E. Aspectos práticos
- Implementação realista: latência/capacidade, modelo de fills; análise de drawdown (duração, recuperação) em `src/arara_quant/runners/drawdown_analysis.py`.
- Risk budgeting dinâmico: visualizar heatmap temporal de budgets por bucket/regime.
- Forward-looking signals: VIX, credit spreads como proxy de regime; suavização de transições via “regime confidence weighting”.
  ```python
  confidence = compute_mahalanobis_distance(current_vol, threshold)
  lambda_t = (1 - confidence) * lambda_calm + confidence * lambda_stress
  ```

## Estrutura sugerida por tier de journal
- Tier 2 recomendado: Journal of Portfolio Management / Quantitative Finance — equilíbrio entre teoria e prática; requer comparativos SOTA + robustez OOS.
- Tier 1 (Finance/OR): precisa provas formais e ≥5 métodos SOTA.
- Tier 3 (aplicado): Computational Economics / Journal of Financial Data Science — ciclo de revisão mais rápido.

## Roadmap (3–6 meses)
- Mês 1–2: teoria (convexidade, estabilidade), montar `theory/` com LaTeX e simulações Monte Carlo.
- Mês 2–3: experimentos SOTA (DRO, HMM, LSTM-cov) e consolidação em `outputs/results/comparisons/all_methods_oos.csv`.
- Mês 3–4: sensibilidade (200+ combinações) e crises históricas; gerar figuras.
- Mês 4–5: generalização para universos alternativos; seção de validade externa.
- Mês 5–6: draft completo e submissão; preparar container/make para “reproduce-paper”.

## Inovações específicas a destacar
- Adaptive cost penalty: penalização de turnover por regime para balancear fricção vs. proteção.
- Regime confidence weighting: transições suaves para evitar flapping.
- Forward-looking regime proxy: VIX/credit spreads para antecipar regimes.

## Checklist pré-submissão
- ≥2 teoremas formais; ≥5 métodos comparativos; OOS em ≥3 períodos e ≥2 universos; grid de sensibilidade; custos não-lineares; pacote de reprodutibilidade (Docker + make).

## Nota integrada do PDF “1. Contribuições Originais Presentes no Repositório”

**Contribuições já disponíveis no código**
- Otimização com custos/turnover internalizados na função objetivo (MV penalizado), não apenas pós-otimização.
- Restrições realistas: budgets por classe, limites 0–10%, cardinalidade (20–35) com heurísticas (GA/variáveis binárias opcionais), hedge cambial.
- Estimadores robustos: média Huber, shrinkage bayesiano (Shrunk_50), Black-Litterman; covariância Ledoit-Wolf (linear/não linear), Tyler/Student-t; experimento indicou Shrunk_50 como mais realista.
- Validação purged walk-forward (Lopez de Prado) com KPIs claros e baselines (1/N, min-var, RP) e critérios de sucesso (Sharpe OOS vs baseline).
- Overlays de risco: modos defensivos/críticos por gatilhos de vol/DD; experimentos de λ dinâmico em crises (COVID/inflação 2022).
- Pipeline modular e replicável: YAML + Pydantic, runners reprodutíveis, testes/lint; pronto para Colab/CLI.

**Oportunidades teóricas**
- Provar convexidade/existência de ótimo (KKT) do problema contínuo; penalização L1 como relaxação convexa da cardinalidade.
- Contrastar penalidade vs. restrição de turnover; formular mean-CVaR equivalente (LP/SOCP) e discutir quando converge para 1/N.
- Análise de sensibilidade a μ, Σ, λ, η e condicionamento; limites teóricos de desempenho OOS vs. 1/N (à la DeMiguel/Michaud).
- Formular MILP da cardinalidade e limites de subótimo das heurísticas.

**Extensões computacionais (Colab)**
- Ampliar meta-heurísticas (GA, PSO, simulated annealing, NSGA-II) para cardinalidade e tuning de λ/η/τ; comparar com solver exato MIQP.
- ML leve para regimes (classificadores) e μ (ridge/lasso/XGBoost); overlay formalizado e reproduzível.
- Híbridos: clustering/HRP para reduzir dimensionalidade antes do MV; AutoML/BO para hiperparâmetros.
- Escalabilidade: Monte Carlo/bootstrap massivo (Numba/JAX), paralelizar backtests; avaliar solvers alternativos.

**Comparações e SOTA**
- Incluir DRO (Delage & Ye), HMM-MV, LSTM/Transformer para cov/regime, Graphical Lasso, tail risk parity; tabela com Sharpe/CVaR/turnover/custo computacional.
- Documentar achados de mean-CVaR (colapso para 1/N) e GA (overfit in-sample, defensivo OOS); evidenciar trade-offs de shrinkage agressivo vs. infactibilidade de budgets.

**Marcos resumidos (4 meses)**
- Mês 1: revisar literatura, fixar escopo, validar números finais, rascunho de intro/metodologia.
- Mês 2: rodar extensões escolhidas, analisar resultados, escrever demos teóricas.
- Mês 3: completar discussão/conclusão, formatar no template do veículo, revisão por pares.
- Mês 4: ajustes finais, pacote suplementar (repo sanitizado + apêndice), submissão.

## Plano de múltiplos papers (versão realista)

Eixos principais do repositório/contribuições:
- Núcleo + backtest sério (custos, turnover, budgets, purged walk-forward).
- Modelagem de covariância (estática vs. dinâmica).
- Regimes de mercado e controle adaptativo (λ/bounds/cash dinâmicos).
- ML/IA para previsão/controle (RL, LSTM, generativos).
- Guia de engenharia/prática (pipeline, reprodutibilidade, operações).

Divisão sugerida:
- **Paper 1 — Core framework + backtests**: problema, formulação MV/M-CVaR com custos e restrições, convexidade básica, walk-forward purged, baselines (1/N, 60/40, ERC, min-var, HRP). Referência fundacional.
- **Paper 2 — Covariância dinâmica**: comparação LW/shrinkage vs. EWMA/DCC; quando Σ(t) dinâmica ajuda (crises vs. regime calmo); métricas de previsão de risco e impacto na carteira.
- **Paper 3 — Regimes + controle**: definição de regimes via vol/DD (ou sinais macro), ajuste dinâmico de λ/bounds/cash, comparação estático vs. adaptativo em crises/recuperação.
- **Paper 4 — ML/IA**: RL para alocação, LSTM/Transformer para μ/Σ ou regime; foco em “quando ML supera o robusto clássico” (aceita resultados negativos).
- **Paper 5 — Practitioner/engenharia**: pipeline de dados, setup de solver, seeds/Poetry/Docker, lições de operação; alvo revistas técnicas ou capítulos.

Prioridade pragmática (IC + projetos em paralelo):
- Curto/médio (1–2 anos): Paper 1 (obrigatório), mais Paper 2 **ou** Paper 3 (escolher eixo: covariância se quiser econometria; regimes/controle se quiser aplicação/originalidade), e um Paper 5 leve/prático.
- Longo prazo (3–5 anos): o eixo não escolhido (cov vs. regimes) e, se fizer sentido, o Paper 4 de ML/IA.

Pergunta de decisão imediata:
- Paper 1 já está claro. Escolher para Paper 2: **covariância dinâmica** (mais teórico/estatístico) ou **regimes + controle** (mais aplicado/controle). A partir daí, montar esqueleto e encher com resultados existentes/novos.

## Questões / checklist por paper

### Paper 1 — Core framework + backtests
- Pergunta central: o MV/M-CVaR com custos, turnover e budgets (convexo) entrega valor vs. baselines simples no OOS 2020–2025?
- Dados/experimentos: reproducão canônica (`make reproduce-oos`), figuras de `outputs/reports/figures/*`, baselines de `outputs/results/baselines`. Métricas completas (Sharpe, Sortino, CVaR anual, DD, turnover, custos).
- Teoria mínima: convexidade/único ótimo; convenção CVaR; definição de turnover; por que purged walk-forward.
- Comparações: 1/N, 60/40, ERC, min-var, HRP. Sensibilidade curta de λ/η/c.
- Riscos: overfit de μ; custo fricção; seeds/ determinismo. Entregáveis: tabela LaTeX, NAV/DD plots, apêndice com configs YAML.

### Paper 2 — Covariância dinâmica
- Pergunta central: modelos dinâmicos de Σ (EWMA/DCC) melhoram risco/retorno vs shrinkage estático em crises?
- Dados/experimentos: janelas 252/21 com Σ dinâmico; métricas de previsão de vol/cov; impacto em Sharpe/DD/CVaR. Crises (2020, 2022) e períodos calmos.
- Teoria: propriedades de estabilidade/condicionamento de Σ(t); quando κ explode; implicações para solver. Métrica de erro de previsão.
- Comparações: LW linear/não-linear vs EWMA vs DCC (se implementado). Controlar custo/turnover constante.
- Riscos: custo computacional DCC; parametrização sensível; dados insuficientes pré-2010 para long history. Entregáveis: heatmaps de κ vs tempo, tabelas de performance por regime.

### Paper 3 — Regimes + controle
- Pergunta central: overlay de regimes (vol/DD ou macro) ajustando λ/bounds/cash reduz DD e melhora Sharpe/CVaR vs otimização estática?
- Dados/experimentos: regras calm/stress/crash; vol_21d_ann, DD rolling; comparação estático vs adaptativo no OOS e crises; teste de delay (look-ahead).
- Teoria/controle: provar preservação de convexidade; discutir estabilidade de transições; possivelmente controlador (PID/vol target) simples.
- Comparações: overlay vs nenhum overlay; overlay + Σ dinâmico opcional. Medir flapping e turnover adicional.
- Riscos: thresholds heurísticos; over-defensivo; dependência de sinais reativos. Entregáveis: tabela Sharpe/DD/CVaR, gráfico de NAV/underwater com regimes, configs claras.

### Paper 4 — ML/IA (RL, DL, generativos)
- Pergunta central: ML (RL ou DL) supera o framework robusto clássico em risco/retorno líquido e estabilidade?
- Dados/experimentos: agente RL (PPO/DQN) vs otimizador; LSTM/Transformer para μ/Σ ou regime; GAN para stress testing. Baseline: Paper 1 e 3.
- Teoria/metodologia: evitar look-ahead; divisão treino/val/test OOS; regularização; métricas de estabilidade (turnover, overfit gaps).
- Comparações: métodos clássicos vs ML; custo computacional (CPU/GPU); resultados negativos são válidos.
- Riscos: tempo de treino, overfit, hiperparâmetros. Entregáveis: tabelas de performance, análise de overfit, runners reproduzíveis.

### Paper 5 — Practitioner / engenharia
- Pergunta central: como operar e reproduzir o framework de ponta a ponta de forma confiável?
- Conteúdo: pipeline de dados, configs YAML (Pydantic), runners e comandos make/poetry, seeds, logging, monitoração, custos/latência, limites operacionais (AUM/capacidade).
- Experimentos: smokes e validações (`make validate`, testes-chave), exemplos de falhas comuns e mitigação.
- Comparações: escolhas de solver (Clarabel/OSQP/ECOS), custo vs precisão, cardinalidade heurística vs exata.
- Entregáveis: guia passo-a-passo, checklist de produção, contêiner/`make reproduce-paper`, artefatos referências (outputs/reports/ outputs/results/ docs/).
