# Notas Técnicas

Este documento consolida a formulação matemática e as hipóteses principais da
estratégia PRISM-R. Use-o como complemento às docstrings e aos notebooks de
exploração.

## 1. Estimação de retornos esperados (μ)

- **Média Huber**: resolve
  \[
  \hat{\mu} = \arg\min_{\mu} \sum_{t=1}^{T} \rho_c(r_t - \mu)
  \]
  com função de perda quadrática-truncada controlada pelo parâmetro `c`.
- **Shrinkage bayesiano** (opcional nos scripts legados): contrai a média amostral
  para um *prior* conservador, mitigando *overfitting* em universos grandes.
- **Anualização**: os retornos diários são multiplicados por 252 para alinhar a
  escala ao custo de capital anual.

## 2. Estimação de covariância (Σ)

- **Ledoit–Wolf**: estima a covariância shrinkando a matriz amostral `S` para
  uma matriz estruturada `F`, `Σ = (1 - δ) S + δ F`, com `δ` calculado em
  fechado.
- **Checagem de PSD**: rotinas em `itau_quant.estimators.cov` projetam a matriz
  no cone PSD quando necessário, preservando simetria numérica.

## 3. Otimização média-variância

- Problema resolvido em `solve_mean_variance`:
  \[
  \max_w \; w^\top \mu - \lambda w^\top Σ w - η \|w - w_{t-1}\|_1
  \]
  sujeito a `1ᵀ w = 1`, limites caixa `l ≤ w ≤ u` e, quando configurado,
  restrições adicionais (alocação por fator, *risk budgets*, etc.).
- Penalidade de turnover (`η`) e *clipping* opcional (`τ`) permitem calibrar o
  custo de transação implícito.

## 4. Backtest walk-forward

- Usa *purging*/*embargo* para evitar *look-ahead bias* na divisão treino/validação.
- Métricas de risco incluem Sharpe com correção HAC, CVaR a 5% e *max drawdown*.
- O motor registra `weights`, `trades` e `ledger` para auditoria completa.

## 5. Considerações práticas

- **Dados**: todo arquivo intermediário é versionado sob `data/processed` com
  nomes determinísticos a partir de `DataLoader`.
- **Reprodutibilidade**: seeds são controlados via `Settings.random_seed` e
  guardados junto do resumo do backtest.
- **Extensões**: o arcabouço suporta migração para Black-Litterman e restrições
  fatoriais bastando plugar novos estimadores no passo 2.
