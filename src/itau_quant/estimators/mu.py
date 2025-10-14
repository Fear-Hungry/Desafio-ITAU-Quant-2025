"""Expected return estimation blueprint.

Objetivo
--------
Fornecer estimadores de retorno (μ) resistentes a outliers e combináveis com
Black-Litterman/fatores.

Componentes sugeridos
---------------------
- `mean_return(returns, method="simple")`:
    Média simples ou geométrica como baseline.
- `huber_mean(returns, c=1.5)`:
    Média robusta com perda de Huber; devolver valores e pesos efetivos.
- `student_t_mean(returns, nu=5)`:
    Estimativa assumindo distribuição t-Student (máxima verossimilhança).
- `bayesian_shrinkage_mean(returns, prior=None, strength=0.2)`:
    Shrinkage em direção a benchmark (ex.: equal-weight market risk premium).
- `confidence_intervals(returns, method="bootstrap", alpha=0.05)`:
    Intervalos de confiança/credibilidade para μ.
- `blend_with_black_litterman(mu_prior, cov, views=None, **kwargs)`:
    Wrapper que delega para `estimators.bl` quando views são informadas.
- `annualize(mu, periods_per_year)`:
    Converte retornos diários/mensais para base anual.

Requisitos
----------
- Inputs esperados: DataFrame de retornos alinhados; manter labels.
- Suportar masks para tratar ativos com histórico curto.
- Documentar se retornos estão em decimal ou porcentagem.

Testes recomendados
-------------------
- `tests/estimators/test_mu.py` cobrindo:
    * comparação com média simples para dados sem ruído,
    * robustez do Huber à presença de outliers extremos,
    * verificação de shrinkage em direção ao prior,
    * integração com `black_litterman` (views vazias → prior).
"""
