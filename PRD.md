# PRD — PRISM-R (carteira ARARA)

**Objetivo:** implementar, validar e documentar uma carteira multi-ativos ARARA com **retorno ajustado a risco** e **custos/turnover** na função-objetivo, sob **cardinalidade** e **restrições realistas**, entregando **relatório final** no padrão do edital (até 10 páginas + seção “Uso de IA Generativa”). Metas e cronograma abaixo amarram com as datas do desafio e um “adiantamento” interno para fechamento em 01/11. O edital pede 10 páginas e seção obrigatória de GenAI; a versão V2 indica **04/11/2025** para entrega final, e a versão anterior indicava **07/11/2025**. Vamos cravar **internamente 01/11/2025** para ter folga.

---

## 1) Escopo

**Universo padrão (ARARA):** ETFs globais previamente definidos (Equity US amplo: SPY, QQQ, IWM; Desenvolvidos ex-US: EFA; Emergentes: EEM; Setores EUA: XLC…XLU; Fatores: USMV, MTUM, QUAL, VLUE, SIZE; REITs: VNQ, VNQI; Treasuries: SHY, IEI, IEF, TLT; TIPS: TIP; Crédito IG: LQD; HY: HYG; EM: EMB, EMLC; Commodities: GLD, DBC; USD: UUP; Cripto spot: IBIT, ETHA).
**Frequência:** rebalance **mensal no 1º pregão útil**; monitoramento diário de risco; rebalance extraordinário por gatilho (drawdown > 10% ou CVaR acima do limite).
**Benchmark:** MSCI ACWI em USD e também 60/40 US (SPY/TLT) para leitura.
**Dados mínimos:** OHLCV + close adjusted; calendário EUA.
**Entrega formatada conforme edital:** relatório até 10 páginas, com seções exigidas e **“Uso de IA Generativa”** obrigatório (vale 15% da nota).

---

## 2) Requisitos de negócio e KPIs

**North-star:** maximizar retorno ajustado a risco com custo e fricção na conta.
**KPIs primários (out-of-sample, com custos):**

* Sharpe HAC e Sortino; Vol anualizada; **CVaR 5%** e Drawdown máx; **Turnover realizado** e **custo realizado**; Hit-rate mensal; Tracking-error vs benchmark.
* **Baselines obrigatórios:** 1/N, min-var com shrinkage, risk-parity. (Referências padrão em portfólios: MVP, mean-CVaR, RP, etc.)

**Critérios de aceite (tem que bater):**

* Sharpe OOS da carteira principal ≥ Sharpe do melhor baseline − 0.10 e **CVaR 5%** ≤ pior baseline em CVaR.
* Turnover médio mensal ≤ 20% e custo anualizado ≤ 50 bps (parametrizável).
* Relatório em 10 páginas com seção GenAI conforme edital.

---

## 3) Formulação técnica

**Objetivo padrão (MV com custos/turnover):**
[
\max_w\ \mu^\top w - \lambda, w^\top \Sigma w\ -\ \eta,\lVert w - w_{t-1}\rVert_1\ -\ c^\top|w - w_{t-1}|
]
s.a. ( \sum_i w_i = 1), (0 \le w_i \le u_i), grupos/segmentos com limites, alavancagem opcional, tracking-error opcional. **Cardinalidade**: (|w|*0 \le K) via (w_i \le u_i z_i), (z_i\in{0,1}) ou heurística.
**Alternativa robusta:** **CVaR**: minimizar (\text{CVaR}*\alpha(w)) com retorno-alvo, ou maximizar retorno com (\text{CVaR}_\alpha) limitado. (Portfólios mean-CVaR e medidas de cauda são tratadas no texto do Palomar; também lista turnover e custos como elementos de modelagem.)

---

## Resumo executivo p/ stakeholders

Analisando o PRISM-R, a carteira ARARA é uma **solução institucional de alocação global multiativos** com as seguintes características fundamentais:

### Definição core da carteira

Carteira de **retorno absoluto** sem alavancagem, composta por 20–35 ETFs globais líquidos, otimizada mensalmente via QP/SOCP com penalização explícita de custos e turnover.

### Objetivos quantitativos rigorosos

```
Target pós-custos:
• Retorno: CDI + 4 p.p. a.a.
• Volatilidade: ≤ 12% a.a.
• Max Drawdown: ≤ 15%
• CVaR(5%): ≤ 8%
• Sharpe Ratio: ≥ 0.80
• Turnover mensal: 5–20%
• Custos totais: ≤ 50 bps a.a.
```

### Alocação estratégica por buckets

| Bucket                | Alocação central | Banda tática | Função principal |
|-----------------------|------------------|--------------|------------------|
| Núcleo Ações EUA      | 25%              | ±10 p.p.     | Motor de crescimento, liquidez profunda |
| Desenvolvidos ex-US   | 15%              | ±7 p.p.      | Diversificação geográfica |
| Emergentes            | 8%               | ±5 p.p.      | Beta controlado ao ciclo global |
| Fatores Smart Beta    | 12%              | ±6 p.p.      | Redução de volatilidade via USMV/QUAL |
| Crédito Global        | 15%              | ±7 p.p.      | Carry estrutural |
| Treasuries            | 15%              | ±10 p.p.     | Hedge de cauda, correlação negativa |
| Real Assets           | 8%               | ±5 p.p.      | Proteção inflacionária |
| Alternativos líquidos | 2%               | 0 a +3 p.p.  | Cripto via ETFs spot (≤ 5% hard cap) |

### Framework de otimização

Função objetivo com custos internos:

```math
\max_w  \mu^T w - \lambda w^T \Sigma w - \eta \lVert w - w_{t-1} \rVert_1 - c^T \lvert w - w_{t-1} \rvert
```

Restrições operacionais:
- **Cardinalidade**: `20 ≤ Σ z_i ≤ 35` ativos.
- **Exposição cambial**: `|FX líquido vs BRL| ≤ 30%` (hedge dinâmico 30% → 70%).
- **Limites por ativo/grupo**: obedecem liquidity caps (`ADV20`).
- **Custos modelados**: 10 bps linear + slippage não linear.

### Estimadores robustos

- `μ`: média robusta (Huber) com janela adaptativa e opção Black-Litterman para views.
- `Σ`: Ledoit-Wolf shrinkage não linear.
- Extensão futura: mean-CVaR com retorno-alvo ou CVaR limitado.

### Processo de execução

1. **Rebalanceamento base:** 1º dia útil de cada mês.
2. **Modo defensivo:** reduzir 50% do risco se `DD > 15%` **ou** `vol > 15%`.
3. **Modo crítico:** reduzir 75% se `DD > 20%` **e** `vol > 18%`.
4. **Controle de turnover:** penalização L1 mantendo entre 5–20%.
5. **Validação:** walk-forward com purging/embargo (López de Prado).

### Diferenciais técnicos

- **Otimizador híbrido:** núcleo convexo (CVXPY) + meta-heurística para cardinalidade.
- **Custos reais** integrados na função objetivo.
- **Backtesting rigoroso:** sem look-ahead bias, bootstrap em blocos.
- **Baselines:** 1/N, min-var (shrinkage), risk-parity.

### Em síntese

Portfólio institucional global que entrega **CDI + 4% com risco controlado** usando ETFs líquidos, otimização robusta com custos embutidos e governança transparente. Custos e execução são tratados dentro do modelo — não como ajuste posterior — garantindo promessas realistas e execução consistente.

**Estimadores:**

* (\Sigma): **shrinkage** (Ledoit-Wolf; versão não linear quando útil).
* (\mu): **média robusta** (Huber/t-robusta) ou **Black-Litterman** quando houver views.
* Validação: walk-forward com **purging + embargo**; **bootstrap em blocos** para IC.
  **Backtest:** atenção a vieses e overfitting; usar randomizações, stress e walk-forward, como recomendado em literatura moderna de backtesting.

**Otimizador híbrido:**

* **Núcleo convexo (cvxpy)**: QP/SOCP/LP-CVaR para conjunto fixo (S).
* **Camada meta-heurística** (GA/PSO/SA/GRASP): busca (S), (K), (\lambda,\eta,\tau).
* Seleção por métricas OOS com custo e penalidade de turnover.

---

## 4) Requisitos funcionais

1. **Ingestão de dados:** yfinance ou equivalente; OHLCV e close adj; consolidar feriados; mapear divisões e splits.
2. **Pré-processo:** alinhamento, winsorize outliers, imputação estrita, construção de retornos, máscaras de liquidez.
3. **Estimadores robustos:** (\mu,\Sigma) com logs e seeds fixas.
4. **Solver convexo:** módulo `solve_convex(...)` com custos/turnover, limites, grupos, CVaR opcional.
5. **Meta-heurística externa:** `metaheuristic_outer(...)` com busca de subset e hiperparâmetros, orçamento computacional fixo.
6. **Backtest walk-forward:** `rebalance_backtest(...)` mensal, custos, gatilhos de risco, logs de decisão.
7. **Relato & gráficos:** tabelas de métricas OOS, curvas com IC via bootstrap, comparação com baselines.
8. **Conformidade edital:** relatório 10 páginas, seção **“Uso de IA Generativa”** obrigatória; pode usar GenAI em qualquer etapa do processo.

---

## 5) Requisitos não funcionais

* **Reprodutibilidade:** seeds fixas, ambiente `requirements.txt`, logs estruturados, versionamento de dados.
* **Performance:** rebalance mensal resolve < 30 s por rodada no universo ARARA; backtest YTD < 5 min.
* **Qualidade:** testes mínimos `pytest` para cada módulo.
* **Portabilidade:** Linux/Windows, Python 3.11+, `cvxpy`, `numpy/pandas/polars`.
* **Código próprio do modelo:** ferramentas prontas apenas para estudo/validação; o **modelo principal implementado pela equipe** (sim, isso é exigido).

---

## 6) Arquitetura de alto nível

* **Camada Dados:** conectores, calendário, normalização e cache parquet.
* **Camada Research:** estimadores, núcleo convexo, meta-heurística, riscos, relatórios.
* **Camada Execução Simulada:** motor de rebalance, custos, gatilhos, logs.
* **Camada Relatório:** notebooks e export PDF; seção GenAI descrevendo onde foi usada (planejado: ajuda na engenharia de features/limpeza, ablação e redação final).

---

## 7) Restrições e regras

* **Soma de pesos = 1; long-only por padrão**; limites por ativo e por classe; exposição a fatores e TE opcionais.
* **Cardinalidade** (K) alvo: 10–30.
* **Turnover** alvo: 5–20% por rebalance; **custos lineares** 20–50 bps round-trip como default.
* **Cripto via ETFs spot** IBIT/ETHA, com limite agregado.
* **Rebalance mensal**; extraordinário por gatilho de risco.
* **Medidas de risco**: Var, **CVaR**, downside, drawdown disponíveis; escolha principal: MV com shrinkage; alternativa: **mean-CVaR**. (Todas presentes no escopo do Palomar, inclusive custos/turnover transparecendo no índice/chapters.)

---

## 8) Cronograma e marcos até 01/11/2025

> Datas no fuso America/Cuiabá. “Entregas do edital” abaixo são para amarrar expectativas. O edital V2 marca pré-relatório **30/09** e final **04/11**; nossa **data-alvo interna** é **01/11** para dar lastro.

* **26–30/09** · Kickoff técnico, repo, checklist de dados; rascunho pré-relatório do edital (sumário executivo + plano). **Marco do edital: pré-relatório 30/09.**
* **01–08/10** · Estimadores robustos prontos; baselines 1/N, min-var (shrinkage), risk-parity; backtest básico.
* **09–16/10** · Núcleo convexo com custos/turnover; restrições realistas e grupos; bateria de testes.
* **17–22/10** · Meta-heurística (PSO/GA) para (S,K,\lambda,\eta,\tau); validação cruzada walk-forward.
* **23–27/10** · Avaliação OOS, IC via bootstrap de blocos; análise de sensibilidade.
* **28–31/10** · Redação final 10 páginas, gráficos, revisão e checagens.
* **01/11** · **Fechamento interno** do relatório completo (versão “congelada”).
* **Buffer 02–03/11** · Ajustes finos;
* **04/11** · Entrega no sistema do desafio, se mantido o cronograma V2.

---

## 9) Entregáveis

1. **Código** com módulos: `estimate_mu_cov()`, `solve_convex()`, `metaheuristic_outer()`, `rebalance_backtest()`; `pytest` básico; `README`.
2. **Relatório (≤10 págs)**: hipótese, modelagem, backtest, análise, conclusão, **Uso de IA Generativa**.
3. **Pacote de evidências**: logs, seeds, configs, tabelas OOS com custos.
4. **Apêndice opcional**: material estendido, se permitido.

---

## 10) Riscos e mitigação

* **Overfitting/backtest feliz:** randomizações, stress, walk-forward, embargo; reportar IC.
* **Mudança de cronograma do edital:** manter **01/11** como D-1; acompanhar canal do desafio.
* **Ruído/cauda pesada:** uso de **CVaR** e estimadores robustos; análise de cenários.
* **Custos e fricção mais altos que o previsto:** varrer grid de (c), tolerância de turnover; relatório com sensibilidade.
* **Cardinalidade MIQP pesado:** heurísticas de subset + núcleo convexo.

---

## 11) Governança e papéis

* **Marcus:** dono do produto, pesquisa, implementação, escrita; aprovação final.
* **Assistência (eu):** especificação técnica, revisão crítica, geração de artefatos e gráficos, poda de vieses, revisão do PDF.
* **Critério de “done”:** KPIs e critérios de aceite atendidos, relatório ≤10 págs com seção GenAI e baselines comparados, reprodutibilidade verificada.

---

## 12) Sumário do relatório (10 páginas, guia)

1. **Resumo executivo**
2. **Hipótese e racional**
3. **Dados e pré-processo**
4. **Modelagem** (MV com custos/turnover; alternativa CVaR)
5. **Restrições e execução**
6. **Backtest e validação** (walk-forward, IC, stress)
7. **Resultados e ablações**
8. **Comparação com baselines**
9. **Uso de IA Generativa** (onde ajudou de verdade)
10. **Conclusão e próximos passos**
11. **Referências essenciais** (se couber no limite)

---

## 13) Stack e padrões

* **Linguagem:** Python 3.11+
* **Libs:** `cvxpy`, `numpy`, `pandas/polars`, `scipy`, `statsmodels`; plots simples. Palomar cita `CVXPY` e bibliotecas correlatas no ecossistema de portfólios.
* **Qualidade:** `pytest`, `black/isort`, `pre-commit`.
* **Reprodutibilidade:** versão de dados congelada, seeds fixas, logs.

---

---

## Resumo executivo p/ stakeholders

Analisando o PRISM-R, a carteira ARARA é uma **solução institucional de alocação global multiativos** com as seguintes características fundamentais:

### Definição core da carteira

Carteira de **retorno absoluto** sem alavancagem, composta por 20–35 ETFs globais líquidos, otimizada mensalmente via QP/SOCP com penalização explícita de custos e turnover.

### Objetivos quantitativos rigorosos

```
Target pós-custos:
• Retorno: CDI + 4 p.p. a.a.
• Volatilidade: ≤ 12% a.a.
• Max Drawdown: ≤ 15%
• CVaR(5%): ≤ 8%
• Sharpe Ratio: ≥ 0.80
• Turnover mensal: 5–20%
• Custos totais: ≤ 50 bps a.a.
```

### Alocação estratégica por buckets

| Bucket                | Alocação central | Banda tática | Função principal |
|-----------------------|------------------|--------------|------------------|
| Núcleo Ações EUA      | 25%              | ±10 p.p.     | Motor de crescimento, liquidez profunda |
| Desenvolvidos ex-US   | 15%              | ±7 p.p.      | Diversificação geográfica |
| Emergentes            | 8%               | ±5 p.p.      | Beta controlado ao ciclo global |
| Fatores Smart Beta    | 12%              | ±6 p.p.      | Redução de volatilidade via USMV/QUAL |
| Crédito Global        | 15%              | ±7 p.p.      | Carry estrutural |
| Treasuries            | 15%              | ±10 p.p.     | Hedge de cauda, correlação negativa |
| Real Assets           | 8%               | ±5 p.p.      | Proteção inflacionária |
| Alternativos líquidos | 2%               | 0 a +3 p.p.  | Cripto via ETFs spot (≤ 5% hard cap) |

### Framework de otimização

Função objetivo com custos internos:

```math
\max_w  \mu^T w - \lambda w^T \Sigma w - \eta \lVert w - w_{t-1} \rVert_1 - c^T \lvert w - w_{t-1} \rvert
```

Restrições operacionais:
- **Cardinalidade**: `20 ≤ Σ z_i ≤ 35` ativos.
- **Exposição cambial**: `|FX líquido vs BRL| ≤ 30%` (hedge dinâmico 30% → 70%).
- **Limites por ativo/grupo**: obedecem liquidity caps (`ADV20`).
- **Custos modelados**: 10 bps linear + slippage não linear.

### Estimadores robustos

- `μ`: média robusta (Huber) com janela adaptativa e opção Black-Litterman para views.
- `Σ`: Ledoit-Wolf shrinkage não linear.
- Extensão futura: mean-CVaR com retorno-alvo ou CVaR limitado.

### Processo de execução

1. **Rebalanceamento base:** 1º dia útil de cada mês.
2. **Modo defensivo:** reduzir 50% do risco se `DD > 15%` **ou** `vol > 15%`.
3. **Modo crítico:** reduzir 75% se `DD > 20%` **e** `vol > 18%`.
4. **Controle de turnover:** penalização L1 mantendo entre 5–20%.
5. **Validação:** walk-forward com purging/embargo (López de Prado).

### Diferenciais técnicos

- **Otimizador híbrido:** núcleo convexo (CVXPY) + meta-heurística para cardinalidade.
- **Custos reais** integrados na função objetivo.
- **Backtesting rigoroso:** sem look-ahead bias, bootstrap em blocos.
- **Baselines:** 1/N, min-var (shrinkage), risk-parity.

### Em síntese

Portfólio institucional global que entrega **CDI + 4% com risco controlado** usando ETFs líquidos, otimização robusta com custos embutidos e governança transparente. Custos e execução são tratados dentro do modelo — não como ajuste posterior — garantindo promessas realistas e execução consistente.

