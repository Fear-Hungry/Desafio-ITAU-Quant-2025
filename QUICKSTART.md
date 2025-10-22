# 🚀 QUICKSTART - Como Rodar o Sistema Completo

## ✅ Pré-requisitos

Certifique-se de que o ambiente está configurado:

```bash
# Verificar instalação
poetry --version
python --version  # deve ser >= 3.9

# Instalar dependências
poetry install

# Ativar ambiente (opcional)
poetry shell
```

---

## 🎯 Opção 1: Otimização Simples (Rápido - 30 segundos)

**O que faz:** Otimiza um portfolio com dados reais e gera pesos.

```bash
poetry run python run_portfolio_arara.py
```

**Saída esperada:**
```
✅ Portfolio final:
   • 7-10 ativos ativos
   • Soma dos pesos: 1.000000

📊 Alocação (top 10):
   SPY   : 12.50% ████████████
   TLT   : 15.00% ███████████████
   GLD   : 10.00% ██████████
   ...

📈 Métricas Ex-Ante:
   • Retorno esperado:  +8.50%
   • Volatilidade:      10.20%
   • Sharpe Ratio:      0.83
```

**Arquivos gerados:**
- `results/portfolio_weights_YYYYMMDD_HHMMSS.csv` - Pesos do portfolio
- `results/portfolio_metrics_YYYYMMDD_HHMMSS.csv` - Métricas

---

## 📊 Opção 2: Backtest Walk-Forward (Completo - 2-5 minutos)

**O que faz:** Valida a estratégia com dados históricos (out-of-sample).

```bash
poetry run python run_backtest_walkforward.py
```

**Saída esperada:**
```
📊 [4/4] Calculando métricas de performance...
   📈 Métricas Out-of-Sample:
      • Retorno anualizado:    +9.50%
      • Volatilidade anual:    11.20%
      • Sharpe Ratio:          0.85
      • Max Drawdown:          -12.50%
      • Win Rate:              55.00%

   📊 Comparação com SPY:
      • SPY Retorno anual:     +12.00%
      • Alpha vs SPY:          -2.50%
      • Sharpe Improvement:    +0.15
```

**Arquivos gerados:**
- `results/backtest_returns_YYYYMMDD_HHMMSS.csv` - Série temporal de retornos
- `results/backtest_metrics_YYYYMMDD_HHMMSS.csv` - Métricas de performance

---

## 🎨 Opção 3: Script de Teste Sintético (Debug - 5 segundos)

**O que faz:** Testa o sistema com dados sintéticos (sem internet).

```bash
poetry run python test_portfolio_run.py
```

**Quando usar:** Para validar que tudo está funcionando antes de usar dados reais.

---

## ⚙️ Personalização

### Modificar Universo de Ativos

Edite `run_portfolio_arara.py`:

```python
TICKERS = [
    'SPY', 'QQQ', 'IWM',  # US Equity
    'EFA', 'VGK',         # Intl Equity
    'TLT', 'IEF',         # Fixed Income
    'GLD', 'SLV',         # Commodities
]
```

### Ajustar Parâmetros de Risco

```python
RISK_AVERSION = 3.0      # 2=agressivo, 5=conservador
MAX_POSITION = 0.15      # 15% max por ativo
TURNOVER_PENALTY = 0.10  # penalidade de giro
```

### Escolher Método de Shrinkage

```python
SHRINKAGE_METHOD = 'ledoit_wolf'  # ou 'nonlinear', 'tyler'
```

---

## 📖 Uso Programático

### Exemplo: Otimização Customizada

```python
from itau_quant.data.loader import DataLoader
from itau_quant.estimators.mu import mean_return
from itau_quant.estimators.cov import ledoit_wolf_shrinkage
from itau_quant.optimization.core.mv_qp import solve_mean_variance, MeanVarianceConfig
import pandas as pd

# 1. Carregar dados
import yfinance as yf
tickers = ['SPY', 'TLT', 'GLD']
data = yf.download(tickers, period='2y', auto_adjust=True)
prices = data['Close']
returns = prices.pct_change().dropna()

# 2. Estimar parâmetros
mu = mean_return(returns.tail(252)) * 252
sigma, _ = ledoit_wolf_shrinkage(returns.tail(252))
sigma = sigma * 252

# 3. Otimizar
config = MeanVarianceConfig(
    risk_aversion=3.0,
    turnover_penalty=0.0,
    turnover_cap=None,
    lower_bounds=pd.Series(0.0, index=tickers),
    upper_bounds=pd.Series(0.40, index=tickers),
    previous_weights=pd.Series(0.0, index=tickers),
    cost_vector=None,
    solver="ECOS",
)

result = solve_mean_variance(mu, sigma, config)
print(result.weights)
```

### Exemplo: Black-Litterman com Views

```python
from itau_quant.estimators.bl import black_litterman
import numpy as np

# Market equilibrium (equal-weight como prior)
market_weights = pd.Series(1/3, index=['SPY', 'TLT', 'GLD'])

# Views: "SPY vai superar TLT em 5%"
P = pd.DataFrame([[1, -1, 0]], columns=['SPY', 'TLT', 'GLD'])
Q = pd.Series([0.05])

# Aplicar Black-Litterman
mu_bl, sigma_bl = black_litterman(
    market_weights=market_weights,
    cov=sigma,
    P=P,
    Q=Q,
    tau=0.025,
    risk_aversion=3.0,
)

# Otimizar com posterior
result = solve_mean_variance(mu_bl, sigma_bl, config)
```

### Exemplo: Risk Parity

```python
from itau_quant.optimization.core.risk_parity import risk_parity

weights_rp = risk_parity(
    cov=sigma,
    method='iterative',  # ou 'log_barrier'
    max_iter=100,
)

print(weights_rp)
```

---

## 🧪 Rodar Testes

```bash
# Todos os testes
poetry run pytest

# Testes específicos
poetry run pytest tests/optimization/
poetry run pytest tests/backtesting/

# Com verbose
poetry run pytest -v

# Com coverage (se pytest-cov instalado)
poetry run pytest --cov=src/itau_quant
```

---

## 📁 Estrutura de Saída

```
results/
├── portfolio_weights_20251019_120000.csv
├── portfolio_metrics_20251019_120000.csv
├── backtest_returns_20251019_120000.csv
└── backtest_metrics_20251019_120000.csv
```

### Formato: portfolio_weights_*.csv

```csv
ticker,weight
SPY,0.1250
TLT,0.1500
GLD,0.1000
...
```

### Formato: backtest_returns_*.csv

```csv
date,return
2024-01-02,0.0012
2024-01-03,-0.0005
...
```

---

## 🐛 Troubleshooting

### Erro: "No module named 'yfinance'"

```bash
poetry add yfinance
```

### Erro: "Solver not found"

```bash
poetry add ecos
poetry add osqp
```

### Erro: "Data insuficiente"

Reduza `ESTIMATION_WINDOW` ou aumente o período de download:

```python
START_DATE = END_DATE - timedelta(days=365 * 5)  # aumentar para 5 anos
```

### Performance lenta

Reduza o universo de ativos ou aumente `REBALANCE_FREQ`:

```python
TICKERS = ['SPY', 'TLT', 'GLD']  # apenas 3 ativos
REBALANCE_FREQ = 63  # rebalancear trimestralmente
```

---

## 📊 Benchmarks Esperados

### Otimização Simples
- ⏱️ Tempo: < 1 minuto
- 📈 Sharpe Ex-Ante: 0.5 - 1.2
- 🎯 Ativos ativos: 5-15

### Backtest Walk-Forward (5 anos)
- ⏱️ Tempo: 2-5 minutos
- 📈 Sharpe OOS: 0.4 - 0.9
- 📉 Max Drawdown: -10% a -20%
- ✅ Win Rate: 50-60%

---

## 🎯 Próximos Passos

1. **Validação:** Rodar backtest e comparar com benchmarks
2. **Otimização:** Ajustar parâmetros (risk aversion, turnover)
3. **Comparação:** Testar múltiplas estratégias (MV, RP, HRP, GA)
4. **Produção:** Integrar com sistema de execução real

---

## 💡 Dicas Avançadas

### Usar Genetic Algorithm

```python
from itau_quant.optimization.ga.genetic import run_genetic_algorithm

result = run_genetic_algorithm(
    universe=tickers,
    returns=returns,
    config={
        'population': {'size': 20},
        'generations': 50,
        'mutation_rate': 0.1,
    }
)
```

### Comparar Estratégias

```bash
# Criar script de comparação
poetry run python scripts/compare_strategies.py
```

### Gerar Relatório Completo

```python
from itau_quant.evaluation.report import build_and_export_report

build_and_export_report(
    returns=portfolio_returns,
    benchmark_returns=spy_returns,
    output_file='results/report.html',
)
```

---

## 📞 Suporte

- 📖 Documentação: Ver `CLAUDE.md` e `PRD.md`
- 🧪 Testes: `tests/` contém 230+ testes
- 🐛 Issues: Verificar logs em `poetry run pytest -v`

---

**Versão:** 1.0  
**Última atualização:** 2025-10-19  
**Status:** ✅ Produção
