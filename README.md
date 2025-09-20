# Desafio-ITAU-Quant

Este repositório contém a implementação de uma estratégia de alocação de ativos quantitativa, seguindo as melhores práticas de modelagem, backtesting e engenharia de software.

## Plano de Implementação

O projeto será desenvolvido seguindo uma estrutura modular e testável. Abaixo estão detalhadas as etapas e o conteúdo de cada componente.

### 1. Estrutura do Projeto

A organização dos diretórios e arquivos foi planejada para separar responsabilidades, facilitar a manutenção e garantir a reprodutibilidade dos resultados.

```
ITAU-Quant/
├── data/
│   ├── raw/          # Dados brutos, imutáveis (ex: CSVs baixados)
│   └── processed/    # Dados limpos e prontos para uso (ex: parquet, hdf5)
│
├── notebooks/
│   ├── 01-data-exploration.ipynb   # Análise exploratória inicial
│   ├── 02-model-prototyping.ipynb  # Prototipagem rápida dos otimizadores
│   └── 03-results-analysis.ipynb   # Análise e visualização dos resultados OOS
│
├── src/
│   └── itau_quant/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py         # Funções para carregar e pré-processar dados
│       │
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── estimators.py     # estimate_mu_cov(): Huber, Ledoit-Wolf, etc.
│       │   └── solvers.py        # solve_convex(): Modelos CVXPY (MV, CVaR)
│       │
│       ├── backtesting/
│       │   ├── __init__.py
│       │   ├── engine.py         # Lógica do backtest walk-forward (rebalance_backtest)
│       │   └── metrics.py        # Cálculo de métricas OOS (Sharpe, CVaR, Turnover)
│       │
│       └── utils/
│           ├── __init__.py
│           └── logging_config.py # Configuração centralizada de logging
│
├── tests/
│   ├── __init__.py
│   ├── data/
│   │   └── test_loader.py
│   ├── optimization/
│   │   └── test_solvers.py
│   └── backtesting/
│       └── test_engine.py
│
├── reports/
│   └── final_report.md           # Relatório final, conforme regra 40
│
├── .gitignore
├── pyproject.toml                # Config do projeto, dependências (com ruff, black, pytest)
└── README.md                     # Documentação principal
```

### 2. Detalhamento dos Módulos

#### `src/itau_quant/data/loader.py`
- **Responsabilidade**: Carregar dados brutos da pasta `data/raw/`, realizar a limpeza, alinhamento de datas e tratamento de dados faltantes.
- **Funções Principais**:
    - `load_asset_prices()`: Carrega os preços dos ativos.
    - `calculate_returns()`: Calcula os retornos (decimais) a partir dos preços.
    - `preprocess_data()`: Orquestra o pipeline de pré-processamento.
- **Saída**: Retorna um DataFrame de retornos limpos, pronto para ser usado pelos estimadores, e salvo em `data/processed/`.

#### `src/itau_quant/optimization/estimators.py`
- **Responsabilidade**: Estimar os parâmetros de entrada para os modelos de otimização (vetor de retornos esperados `μ` e matriz de covariância `Σ`).
- **Funções Principais**:
    - `estimate_mu_cov()`: Função principal que pode delegar para diferentes estimadores.
    - `robust_mean_estimator()`: Implementa estimador robusto para `μ` (ex: Huber).
    - `ledoit_wolf_covariance()`: Implementa a estimação de `Σ` com shrinkage de Ledoit-Wolf.
- **Entrada**: DataFrame de retornos históricos.
- **Saída**: `μ` (np.array) e `Σ` (np.array).

#### `src/itau_quant/optimization/solvers.py`
- **Responsabilidade**: Implementar as formulações dos problemas de otimização convexa utilizando `cvxpy`.
- **Funções Principais**:
    - `solve_convex()`: Função genérica que delega para as implementações específicas.
    - `mean_variance_optimizer()`: Implementa o otimizador de média-variância quadrático com custos de transação e turnover, conforme a Regra `20`.
    - `cvar_optimizer()`: (Opcional) Implementa a minimização do CVaR.
- **Entrada**: `μ`, `Σ`, pesos anteriores `w_prev`, e parâmetros do problema (λ, η, etc.).
- **Saída**: DataFrame com os pesos ótimos `w_optimal`.

#### `src/itau_quant/backtesting/engine.py`
- **Responsabilidade**: Executar a simulação histórica (backtest) usando a estratégia de rebalanceamento walk-forward.
- **Funções Principais**:
    - `rebalance_backtest()`: Itera sobre o período de teste, chama os estimadores e o solver em cada ponto de rebalanceamento, e calcula o PnL da estratégia, incluindo custos de transação.
- **Lógica**: Em cada data `t`:
    1. Seleciona a janela de dados de treinamento (ex: últimos `N` períodos).
    2. Chama `estimate_mu_cov()` com os dados de treino.
    3. Chama `solve_convex()` para obter os novos pesos `w_t`.
    4. Calcula o turnover `|w_t - w_{t-1}|` e os custos.
    5. Calcula o retorno do portfólio no período `t` e armazena os resultados.

#### `src/itau_quant/backtesting/metrics.py`
- **Responsabilidade**: Calcular as métricas de performance out-of-sample (OOS).
- **Funções Principais**:
    - `calculate_oos_metrics()`: Orquestra o cálculo de todas as métricas.
    - Métricas individuais para: Retorno Anualizado, Volatilidade, Sharpe Ratio (com correção HAC), Sortino, CVaR (5%), Turnover Médio Realizado, Drawdown Máximo.
- **Entrada**: Série temporal de retornos do portfólio, pesos e benchmarks.
- **Saída**: Um dicionário ou DataFrame com todas as métricas calculadas.

### 3. Testes (`tests/`)

- Testes unitários serão criados para cada módulo:
    - `test_loader.py`: Garante que os dados são carregados corretamente (dimensões, ausência de NaNs).
    - `test_solvers.py`: Testa se as restrições são atendidas (soma dos pesos = 1, bounds) e se o solver converge para casos simples.
    - `test_engine.py`: Garante que a lógica do backtest (cálculo de PnL, custos) está correta.

### 4. Configuração e Dependências (`pyproject.toml`)

- O arquivo `pyproject.toml` gerenciará as dependências do projeto (`pandas`, `numpy`, `cvxpy`, `scikit-learn`, `pytest`) e a configuração de ferramentas de qualidade de código como `black` e `ruff`.
