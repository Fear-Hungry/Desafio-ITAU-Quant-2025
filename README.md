# PRISM-R — Portfolio Risk Intelligence System (Carteira ARARA)

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Build](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org)
[![Style](https://img.shields.io/badge/code%20style-ruff%20%7C%20black-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**Plataforma quantitativa multiativos focada em otimização robusta, custos de transação
reais e validação walk-forward para a carteira ARARA.**

## 📑 Navegação Rápida

- [Executive Brief](#-executive-brief)
- [O que é a Carteira ARARA](#-o-que-é-exatamente-a-nossa-carteira) — Resumo técnico (5 min)
- [Explicação Completa para Iniciantes](#-carteira-arara---explicação-completa-para-iniciantes) — Tutorial detalhado (15 min)
- [Arquitetura e Código](#-arquitetura-funcional) — Para desenvolvedores

## 🎯 Executive Brief

- Optimiza um universo de 40+ ETFs globais com rebalanceamento mensal e limites por classe.
- Incorpora custos, turnover e cardinalidade diretamente na função objetivo do portfólio.
- Utiliza estimadores robustos (Huber, Ledoit-Wolf) e prevê extensão para Black-Litterman.
- Backtesting desenhado com *purging/embargo*, métricas pós-custos e comparação com baselines.
- Roadmap direcionado ao relatório de 10 páginas exigido pelo edital, com rastreabilidade completa.

## 📌 Guardrails de Performance (alvo OOS)

| Métrica                 | Target         | Observação                                 |
|------------------------|----------------|---------------------------------------------|
| Sharpe Ratio           | ≥ 0.80         | Estimado com correção HAC                   |
| Max Drawdown           | ≤ 15%          | Janela 2010+ simulada com custos            |
| CVaR 5%                | ≤ 8%           | Histórico com bootstrap em blocos           |
| Turnover mensal        | 5% – 20%       | Controle via penalidade L1 e cap hard       |
| Custos anuais          | ≤ 50 bps       | Inclui taxas lineares e slippage opcional   |

> Métricas reais serão publicadas após validação completa; hoje servem como norte de
> design e critérios de aceite.

## 🚀 Onboarding Rápido

### 1. Preparar ambiente

```bash
git clone https://github.com/your-org/ITAU-Quant.git
cd ITAU-Quant
poetry install
```

### 2. Validar instalação

```bash
poetry run pytest
poetry run ruff check src tests
```

### 3. Pipeline mínimo de dados

```python
from itau_quant.data.loader import preprocess_data

returns = preprocess_data(
    raw_file_name="prices_arara.csv",
    processed_file_name="returns_arara.parquet",
)
print(returns.tail())
```

1. Coloque o CSV bruto em `data/raw/` com a coluna de data como índice.
2. O pipeline salva retornos em `data/processed/`, prontos para os estimadores.

### 4. Executar backtest (quando o motor estiver ativo)

```bash
poetry run python -m itau_quant.backtesting.engine \
  --config configs/optimizer_example.yaml \
  --oos-start 2018-01-01
```

> O módulo `backtesting.engine` está em rascunho. Verifique o roadmap para prioridade
> de implementação.

### Configuração de exemplo (`configs/optimizer_example.yaml`)

```yaml
universe: configs/universe_arara.yaml
base_currency: BRL
benchmark:
  name: ACWI60_AGG40_BRUnhedged
rebalancing:
  frequency: monthly
  day_rule: first_business_day
  turnover_target: [0.05, 0.20]
risk_limits:
  vol_annual_max: 0.12
  cvar_alpha: 0.05
  cvar_max: 0.08
  max_drawdown: 0.15
fx:
  net_exposure_abs_max: 0.30
  hedge_ratio_default: 0.30
  hedge_ratio_defensive: 0.70
optimizer:
  objective: mean_variance_l1_costs
  lambda: 6.0
  eta: 0.50
  tau: 0.20
  cardinality_kmin: 20
  cardinality_kmax: 35
  solver: ecos
estimators:
  mu: {method: huber, window_days: 252, delta: 1.5}
  sigma: {method: ledoit_wolf, window_days: 252, nonlinear: true}
  costs: {linear_bps: 10, slippage_model: adv20_piecewise}
reporting:
  metrics: [sharpe_hac, sortino, vol, cvar5, maxdd, turnover, costs_bps, te_benchmark, hit_rate]
walkforward:
  train_days: 252
  test_days: 21
  purge_days: 2
  embargo_days: 2
  n_splits: 60
```

## 🧱 Arquitetura Funcional

```
┌───────────────────────┐
│  Data Layer           │ ← ingestão, limpeza, feature store (Parquet)
├───────────────────────┤
│  Estimators           │ ← μ robusto, Σ shrinkage, métricas de risco
├───────────────────────┤
│  Optimizer Core       │ ← QP/SOCP com custos, turnover, cardinalidade
├───────────────────────┤
│  Metaheuristics       │ ← busca de subset, hiperparâmetros, stress
├───────────────────────┤
│  Backtesting Engine   │ ← walk-forward, purging, execução com custos
├───────────────────────┤
│  Reporting            │ ← métricas OOS, gráficos, relatório 10 páginas
└───────────────────────┘
```

## 🧠 Módulos Principais

### `itau_quant.data`
- `loader.py`: ingestão CSV → retornos; salva artefatos em `data/processed/`.
- Próximos passos: calendário de pregões, limpeza de liquidez (`adv_20`, `amihud`).

### `itau_quant.optimization`
- `estimators.py` (WIP): médias Huber, shrinkage Ledoit-Wolf, posterior BL.
- `solvers.py` (WIP): solucionadores QP e mean-CVaR com restrições de grupo e turnover.

### `itau_quant.backtesting`
- `engine.py` (WIP): rebalance mensal, purging/embargo, gatilhos de risco.
- `metrics.py`: slated para métricas pós-custos, tracking error, hit-rate.

### `itau_quant.utils`
- `logging_config.py`: configuração padrão de logging estruturado (debug em desenvolvimento).

## 📂 Layout do Repositório

```
ITAU-Quant/
├── data/
│   ├── raw/              # dumps imutáveis (CSV)
│   └── processed/        # artefatos derivados (Parquet, Feather)
├── notebooks/            # exploração e narrativas
├── reports/              # PDFs finais e anexos
├── src/itau_quant/       # código de produção (pacote)
├── tests/                # suíte Pytest espelhando a árvore de src/
├── configs/              # YAML de universo, otimização, backtests (a criar)
├── PRD.md                # documento de produto detalhado
└── README.md
```

## 🌍 Universo ARARA (resumo)

| Classe de Ativo       | Tickers principais              | Peso máx | Peso por ativo |
|-----------------------|---------------------------------|----------|----------------|
| US Equity Broad       | SPY, QQQ, IWM                  | 35%      | 15%            |
| Developed ex-US       | EFA                            | 20%      | 20%            |
| Emerging Markets      | EEM                            | 15%      | 15%            |
| US Sectors            | XLC … XLU (11 ETFs)            | 35%      | 12%            |
| Factor Tilt           | USMV, MTUM, QUAL, VLUE, SIZE   | 30%      | 12%            |
| Treasuries            | SHY, IEI, IEF, TLT             | 60%      | 25%            |
| Credit                | LQD, HYG, EMB, EMLC            | 40%      | 20%            |
| Real Assets           | VNQ, VNQI, GLD, DBC            | 30%      | 12%            |
| Crypto (spot ETFs)    | IBIT, ETHA                     | 5%       | 3%             |

Critérios de inclusão: ETF ≥ 3 anos, `ADV20 ≥ USD 10mm`, preço ≥ USD 5, sem ETNs
alavancados/inversos. Exclusões temporárias por dados faltantes ou liquidez extrema.
A lista completa será versionada em `configs/universe_arara.yaml`.

## 🧭 Plano Detalhado da Carteira ARARA

### Por que esta carteira existe
- Entregar retorno absoluto consistente com volatilidade anualizada
  inferior a 12% e drawdown controlado para investidores institucionais com horizonte ≥ 3 anos.
- Atuar como núcleo “core plus”: beta diversificado globalmente com sobreposição de fatores
  defensivos e proteção de cauda via renda fixa longa e real assets.
- Ser totalmente transparente, replicável e passível de auditoria por meio deste repositório.

### Objetivos quantitativos
- **Retorno anual alvo:** CDI + 4 p.p. (estimado em termos realistas após custos).
- **Risco máximo:** volatilidade 12% e CVaR(5%) ≤ 8% conforme tabela de guardrails.
- **Correlação:** manter correlação com Ibovespa ≤ 0,40 e com MSCI ACWI ≤ 0,70.
- **Liquidez:** carteira negociável em menos de 2 dias úteis considerando ADV20.

### Estrutura de buckets estratégicos

| Bucket               | Função no portfólio                    | Alocação estratégica | Desvio tático |
|----------------------|-----------------------------------------|----------------------|---------------|
| Núcleo Ações EUA     | Capturar crescimento secular e liquidez | 25%                  | ±10 p.p.      |
| Ações Desenvolvidos  | Diversificar exposição cíclica          | 15%                  | ±7 p.p.       |
| Emergentes           | Beta controlado a crescimento global    | 8%                   | ±5 p.p.       |
| Fatores Smart Beta   | Suavizar volatilidade e drawdown        | 12%                  | ±6 p.p.       |
| Crédito Global       | Carry com controle de risco             | 15%                  | ±7 p.p.       |
| Treasuries           | Defesa contra choques de risco          | 15%                  | ±10 p.p.      |
| Real Assets          | Hedge inflacionário                     | 8%                   | ±5 p.p.       |
| Alternativos Liquid. | Exposição oportunística (ex. cripto)    | 2%                   | 0 a +3 p.p.    |

**Disciplina de alocação.** As bandas são metas por bucket; a soma final do portfólio
fecha em 100%.

### Regras de construção
- Seleção de ativos limitada a ETFs UCITS/US domiciled com custo total < 80 bps.
- Limite mínimo de 20 ativos e máximo de 35 para evitar concentração e garantir
  execução eficiente.
- Restrições de peso por classe replicam a tabela do universo, com somatório
  dos buckets respeitando bandas táticas.
- **Moeda e FX.** Todas as métricas e o alvo são medidos em **BRL** (base CDI).
  **Exposição cambial líquida |≤ 30% vs BRL**. Hedge dinâmico: 30% padrão; **70%** quando
  volatilidade ex-ante > 15% ou drawdown > 10%.
- Proibição de alavancagem explícita; derivativos apenas para hedge quando ativos
  equivalentes não estiverem disponíveis.

### Processo de rebalanceamento
- **Rebalance base:** 1º dia útil de cada mês.
- **Rebalance extraordinário:** ativa quando drawdown > 15% ou volatilidade ex-ante > 15%.
- Utilizar otimização multiobjetivo (max Sharpe vs. penalidade L1) para restringir
  turnover entre 5% e 20% ao mês.
- Custos modelados com 10 bps lineares + slippage não linear em função do ADV20.
- Fluxos de entrada/saída são aplicados pro-rata antes do rebalanceamento.

### Monitoramento e gatilhos de risco
- Acompanhamento diário das métricas: volatilidade, CVaR, drawdown, perda máxima em
  janela de 20 dias, tracking error vs. benchmark MSCI ACWI NR (60%) + Bloomberg Global
  Aggregate (40%), ambos não hedgeados para BRL.
- **Modo defensivo:** reduzir risco em 50% quando drawdown > 15% ou volatilidade ex-ante > 15%.
- **Modo crítico:** reduzir risco em 75% quando drawdown > 20% e volatilidade ex-ante > 18%.
- Stress tests trimestrais: cenários históricos (2008, 2020), choques de curva, desvalorização
  do BRL, queda sincronizada de fatores.
- Relatórios mensais com decomposição de performance por bucket e fator.

### Governança e compliance
- Comitê de investimento se reúne quinzenalmente; decisões registradas em ata.
- Backtesting deve ser atualizado semestralmente com dados mais recentes e
  resultado validado por revisão cruzada.
- Documentar fontes de dados, codesets de limpeza e qualquer override manual em `reports/`.
- Versões de configuração (`configs/*.yaml`) versionadas com convenção semântica e teste unitário.

### Roadmap evolutivo da carteira
- Expandir universo para ETFs temáticos/ESG conforme liquidez permitir.
- Avaliar overlay de opções (Collar) para reduzir perda em cauda após primeira fase de validação.
- Integrar sinal macro proprietário (filtros de ciclo) para ajustar bandas táticas.
- Construir dashboard em `reports/` com métricas ao vivo e logs de decisão.

## 🧭 O que é, exatamente, a nossa carteira

**Missão.** Entregar retorno absoluto com controle estrito de risco: alvo CDI + 4 p.p. a.a.,
volatilidade ≤ 12%, max drawdown ≤ 15% e CVaR(5%) ≤ 8% após custos. Horizonte ≥ 3 anos.
Sem alavancagem. **Exposição cambial líquida |≤ 30% vs BRL** com hedge dinâmico (30% padrão,
70% quando volatilidade ex-ante > 15% ou drawdown > 10%).

**Universo investível.** 40+ ETFs globais líquidos (EUA/UCITS).
Inclusão: histórico ≥ 3 anos, `ADV20 ≥ USD 10 mi`, preço ≥ USD 5, TER competitivo,
sem alavancados/inversos. Exclusão temporária por dados faltantes ou iliquidez.
Universo versionado em `configs/universe_arara.yaml`.

**Alocação estratégica por buckets.**

| Bucket                | Alvo | Banda | Exemplos de tickers        |
|-----------------------|------|-------|----------------------------|
| Núcleo Ações EUA      | 25%  | ±10   | SPY, QQQ, IWM              |
| Desenvolvidos ex-US   | 15%  | ±7    | EFA                        |
| Emergentes            | 8%   | ±5    | EEM                        |
| Fatores (US)          | 12%  | ±6    | USMV, MTUM, QUAL, VLUE, SIZE |
| Crédito Global        | 15%  | ±7    | LQD, HYG, EMB, EMLC        |
| Treasuries (curva)    | 15%  | ±10   | SHY, IEI, IEF, TLT         |
| Real Assets           | 8%   | ±5    | VNQ, VNQI, GLD, DBC        |
| Alternativos líquidos | 2%   | 0 a +3| IBIT, ETHA                 |

**Regras de construção.**
- Cardinalidade entre 20 e 35 ativos para evitar concentração e facilitar execução.
- Limites por ativo e por classe conforme tabela do universo; proibido short.
- Hedge cambial dinâmico: 30% padrão; 70% quando volatilidade ex-ante > 15% ou drawdown > 10%.
- Cripto ≤ 5% do portfólio via ETFs spot, alinhado a governança e liquidez.

**Formulação do otimizador (núcleo).**

```
max_w  μᵀw − λ wᵀΣw − η ‖w − w_{t−1}‖₁ − cᵀ|w − w_{t−1}|

s.a.
1)  1ᵀ w = 1,   0 ≤ w_i ≤ u_i
2)  Buckets:     ℓ_g ≤ Σ_{i∈g} w_i ≤ u_g
3)  Turnover:    ‖w − w_{t−1}‖₁ ≤ τ
4)  Cardinal.:   K_min ≤ Σ_i z_i ≤ K_max,   w_i ≤ U_i z_i,   z_i ∈ {0,1}
5)  Moeda:       |Σ_i FX_i · w_i| ≤ 0.30, com FX_i = exposição USD de i vs BRL (sinal + para USD-long)
```

Alternativa robusta: mean-CVaR com α ∈ [1%, 5%] (LP/SOCP) sob retorno-alvo ou CVaR limitado.

**Estimadores.** `μ`: média robusta (Huber) em janela móvel com opção Black-Litterman
quando houver views. `Σ`: Ledoit-Wolf (versão shrinkage não linear quando `N` alto).
Custos: 10 bps lineares por round-trip + slippage crescente com `ADV20` e tamanho da ordem.

| Componente | Default | Notas |
|------------|---------|-------|
| μ (retorno) | Huber mean, janela 252d, δ = 1.5 | Resistente a outliers extremos |
| Σ (cov.) | Ledoit-Wolf não linear, janela 252d | Estável quando `N` é alto |
| λ | Calibrado para vol ex-ante ≈ 10–12% | Ajustado em YAML de configuração |
| η (penalidade L1) | 0.50 | Mantém turnover no intervalo 5–20% |
| τ (cap de turnover) | 0.20 | Limite duro de giro mensal |
| Custos | 10 bps linear + slippage vs `ADV20` | Aplicado em bps do notional |
| K_min / K_max | 20 / 35 | Cardinalidade desejada |
| Taxa livre (Sharpe) | CDI diário | Correção HAC anualizada |

**Rebalance e execução.**
- Base no 1º dia útil de cada mês.
- **Modo defensivo:** reduzir risco em 50% quando drawdown > 15% ou volatilidade ex-ante > 15%.
- **Modo crítico:** reduzir 75% quando drawdown > 20% e volatilidade ex-ante > 18%.
- Turnover alvo 5–20%, lotes mínimos respeitados e caixa residual tratado pro-rata.

**Validação e métricas.** Walk-forward com purging/embargo. Baselines: 1/N, min-var
(shrinkage), risk-parity. Report: Sharpe (HAC), Sortino, volatilidade, CVaR(5%), max drawdown,
turnover realizado, custos em bps, tracking error, hit-rate, intervalos de confiança por
bootstrap em blocos.

**Transparência e governança.** Comitê quinzenal, atas versionadas, configs em YAML,
artefatos do backtest armazenados em `reports/`. Overrides de risco documentados.

Consulte **PRD.md → Seção “Resumo executivo p/ stakeholders”**
para o texto pronto de comunicação.

## 📎 Carteira ARARA - Explicação Completa para Iniciantes

### O que estamos construindo?
Uma **carteira de investimentos automatizada** que investe globalmente usando ETFs (fundos
negociados em bolsa, como "cestas" de ações ou títulos que você compra de uma vez só).

Imagine um **robô investidor** que todo mês decide quanto colocar em cada investimento, sempre
tentando maximizar retorno e minimizar risco.

---

### 🎯 Nossos Objetivos (em português claro)

| O que queremos         | Meta        | Explicação simples                                            |
|------------------------|-------------|----------------------------------------------------------------|
| Retorno anual          | CDI + 4%    | Ganhar 4% a mais que a taxa básica de juros brasileira         |
| Volatilidade           | ≤ 12% a.a.  | O quanto o valor da carteira "balança" — queremos pouco balanço |
| Drawdown máximo        | ≤ 15%       | Se a carteira valer R$ 100, nunca queremos ver cair abaixo de R$ 85 |
| Sharpe Ratio           | ≥ 0.80      | Medida de eficiência: quanto retorno ganhamos para cada unidade de risco |
| Turnover mensal        | 5–20%       | Quanto da carteira mudamos por mês (menos troca = menos custos) |

---

### 🌍 Onde investimos? (Os 8 "Baldes")

Dividimos o dinheiro em 8 categorias, cada uma com uma função:

| Balde                  | % do Total | Para que serve                        | Exemplo real                               |
|------------------------|------------|---------------------------------------|--------------------------------------------|
| Ações EUA              | 25% ± 10%  | Motor principal de crescimento        | Ex.: ETF que replica o S&P 500              |
| Ações Europa/Japão     | 15% ± 7%   | Diversificação geográfica             | Ex.: ETF com empresas da Europa e Ásia      |
| Emergentes             | 8% ± 5%    | Apostar em países em crescimento      | Ex.: ETF com Brasil, China, Índia           |
| Fatores Smart          | 12% ± 6%   | Ações "espertas" que caem menos       | Ex.: ETFs USMV, QUAL, MTUM                   |
| Crédito                | 15% ± 7%   | Empréstimos que pagam juros           | Ex.: Títulos de empresas e governos         |
| Treasuries             | 15% ± 10%  | Super seguro, proteção em crises      | Ex.: Títulos do governo americano           |
| Ativos Reais           | 8% ± 5%    | Proteção contra inflação              | Ex.: Imóveis listados, ouro, commodities    |
| Cripto                 | 2% ± 3%    | Aposta em tecnologia nova             | Ex.: Bitcoin e Ethereum via ETFs regulados  |

*Nota:* o "±" indica a faixa de flexibilidade. Ex.: Ações EUA pode variar entre 15% e 35% conforme o cenário.

---

### 🤖 Como o "robô" decide?

#### 1. Coleta de dados

```python
# Exemplo simplificado
precos_ontem = [100, 50, 75]
precos_hoje = [102, 49, 76]
retornos = [(h - o) / o for h, o in zip(precos_hoje, precos_ontem)]
# SPY subiu 2%, EEM caiu 2%, etc.
```

#### 2. Estima retorno e risco futuros
- **Retorno esperado (μ)**: quanto esperamos ganhar. Usamos uma **média robusta** que ignora
  dias extremos.
- **Risco/Covariância (Σ)**: como os ativos se movem juntos. Usamos **Ledoit-Wolf**, técnica que
  melhora estimativas quando temos poucos dados.

#### 3. Otimização (a mágica)
O robô resolve este problema matemático:

```
Maximizar: Retorno Esperado - Penalidade de Risco - Custos de Transação

Respeitando:
- Soma dos pesos = 100%
- Limites de cada balde (ex.: cripto ≤ 5%)
- Não mudar mais de 20% por mês (controle de custos)
- Ter entre 20 e 35 ativos (nem muito concentrado, nem muito pulverizado)
```

#### 4. Execução mensal
- Todo **1º dia útil do mês** recalculamos tudo.
- **Modo defensivo:** se a carteira perdeu mais que 15% ou a volatilidade subir acima de 15%,
  cortamos 50% do risco.
- **Modo crítico:** se a perda passar de 20% e a volatilidade subir acima de 18%, cortamos 75%.

---

### 💰 Custos (super importante!)

| Tipo de custo          | Valor típico        | Exemplo                                             |
|------------------------|---------------------|-----------------------------------------------------|
| Taxa do ETF            | 0.03% – 0.80% a.a.  | SPY cobra 0.09% ao ano                              |
| Corretagem             | ~0.10% por operação | Comprar/vender na bolsa                             |
| Slippage               | Variável            | Diferença entre preço esperado e preço executado    |
| Impacto no mercado     | Depende do tamanho  | Ordens grandes movem o preço                        |

**Nosso diferencial:** incluímos custos *dentro* da otimização, não depois.

---

### 📊 Como validamos que funciona?

#### Backtesting (teste no passado)
- Pegamos dados de 2010–2024.
- Simulamos como se estivéssemos operando mês a mês.
- Sem "olhar para o futuro" — evitamos vieses como look-ahead.

#### Comparamos com estratégias simples
1. **1/N:** divide igual entre todos (ingênuo, mas difícil de bater).
2. **Mínima Variância:** foca só em minimizar risco.
3. **Risk Parity:** cada ativo contribui igualmente para o risco.

Se não ganharmos dessas, algo está errado!

#### Métricas que acompanhamos
- **Sharpe Ratio:** retorno por unidade de risco (buscamos > 0.8).
- **Max Drawdown:** maior queda do pico ao vale.
- **CVaR 5%:** perda média nos 5% piores cenários.
- **Hit Rate:** percentual de meses com retorno positivo.
- **Tracking error:** comparação com MSCI ACWI NR (60%) + Bloomberg Global Aggregate (40%),
  ambos sem hedge para BRL.

---

### 🔍 Termos técnicos essenciais

| Termo             | O que significa                          | Por que importa                               |
|-------------------|-------------------------------------------|------------------------------------------------|
| ETF               | Fundo que replica um índice e negocia em bolsa | Diversificação instantânea e baixo custo |
| Volatilidade      | O quanto o preço varia                    | Risco ≈ incerteza ≈ volatilidade              |
| Drawdown          | Queda em relação ao último pico           | Ajuda a medir a dor financeira                |
| Sharpe Ratio      | (Retorno - taxa livre de risco) / volatilidade | Mede eficiência do portfólio           |
| Turnover          | % da carteira que mudamos                 | Muito giro = muitos custos                    |
| Rebalanceamento   | Ajustar pesos periodicamente              | Vender o que subiu, comprar o que caiu         |
| Walk-forward      | Teste rolante no tempo                    | Evita overfitting                             |
| Bootstrap         | Reamostragem estatística                  | Calcula intervalos de confiança               |
| CVaR              | Perda média nas piores situações          | Mede risco de cauda (eventos extremos)        |
| Hedge cambial     | Proteger contra variação do dólar         | Importante para investidor brasileiro         |

---

### ✨ Por que nossa abordagem é diferente?

**Abordagem tradicional:**
1. Otimiza um portfólio "perfeito".
2. Só depois descobre que custa caro executar.
3. Resultado real decepciona.

**Nossa abordagem:**
1. **Custos já entram na otimização** desde o primeiro passo.
2. **Turnover controlado** por design.
3. **Performance realista** após considerar fricções de mercado.

---

### 📝 Resumo para a Anna

Estamos construindo um **sistema automatizado** que:
- Investe globalmente em 8 categorias de ativos.
- Rebalanceia mensalmente com disciplina quantitativa.
- Busca CDI + 4% ao ano com risco controlado.
- Considera custos reais desde o planejamento.
- É 100% transparente e auditável.

**Grande diferencial:** não prometemos retornos impossíveis. Entregamos um sistema robusto,
realista e executável que reconhece e trata todas as fricções do mundo real. É como ter um
**piloto automático sofisticado** para investimentos, que sabe quando acelerar, quando frear e
quanto custa cada manobra.

### ❓ FAQ - Perguntas que a Anna provavelmente fará

**P: Quanto precisamos investir para começar?**
R: Mínimo sugerido USD 100k para diluir custos fixos e sustentar a cardinalidade desejada.

**P: E se o modelo errar?**
R: Acionamos o modo defensivo (DD > 15% ou vol > 15%) e, se necessário, o modo crítico
(DD > 20% e vol > 18%), além de comparar com estratégias simples para detectar desvios.

**P: Quanto tempo leva o rebalanceamento?**
R: Cálculo ~5 minutos; execução: ordens distribuídas em 1–2 dias úteis conforme a liquidez
dos ETFs.

**P: Podemos override manual?**
R: Sim, desde que haja justificativa técnica e registro em ata do comitê de investimento.

## 🔬 Validação e Métricas

- Comparar sempre com baselines: 1/N, Min-Var (shrinkage), Risk-Parity.
- Métricas pós-custos: Sharpe (HAC), Sortino, vol, CVaR 5%, Max DD, turnover, custos em bps,
  tracking error, hit-rate.
- Bootstrap em blocos para intervalos de confiança e análise de estabilidade.

## 🗺️ Roadmap

- [x] Estrutura do pacote `itau_quant` e loader de dados inicial.
- [ ] Estimadores robustos (μ, Σ) com testes unitários.
- [ ] Núcleo convexo (`solvers.py`) com custos/turnover.
- [ ] Meta-heurística para cardinalidade e tuning de hiperparâmetros.
- [ ] Motor de backtesting com walk-forward completo.
- [ ] Pipeline de relatório (PDF ≤ 10 páginas + seção GenAI).

## 📚 Referências Essenciais

- Ledoit & Wolf (2004) — Honey, I Shrunk the Sample Covariance Matrix.
- DeMiguel, Garlappi & Uppal (2009) — Optimal Versus Naive Diversification.
- Kolm, Tütüncü & Fabozzi (2014) — 60 Years of Portfolio Optimization.
- Lopez de Prado (2018) — Advances in Financial Machine Learning (purging/embargo).

## 📝 Licença

Distribuído sob a [licença MIT](LICENSE).
---
*Disciplina na modelagem, ceticismo na validação, convicção na execução.*
