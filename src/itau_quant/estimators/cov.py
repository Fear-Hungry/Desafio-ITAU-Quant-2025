"""Covariance estimation toolbox blueprint.

Objetivo
--------
Fornecer estimadores de covariância robustos/estáveis para alimentar os
otimizadores de portfólio.

Componentes sugeridos
---------------------
- `sample_cov(returns)`:
    Estimativa padrão com validação de shape e limpeza de NaNs.
- `ledoit_wolf_shrinkage(returns)`:
    Implementar shrinkage linear (Ledoit-Wolf) com cálculo de parâmetro
    ótimo fechado.
- `nonlinear_shrinkage(returns)`:
    Utilizar a abordagem de Ledoit & Wolf (2018) ou equivalente para shrinkage
    não linear (espectral).
- `tyler_m_estimator(returns, max_iter=100, tol=1e-6)`:
    Estimador robusto para distribuições elípticas, retornando matriz PSD
    normalizada.
- `student_t_cov(returns, nu)`:
    Covariância sob hipótese t-Student (graus de liberdade ``nu``).
- `project_to_psd(matrix, epsilon=1e-6)`:
    Utilitário para corrigir pequenas violações de semidefinitude.
- `regularize_cov(matrix, method="diag", floor=None)`:
    Ajuda a manter condicionamento controlado (ex.: adicionar floor na diag).

Requisitos adicionais
---------------------
- Todos os métodos devem devolver DataFrame/ndarray coerentes com o índice e
  colunas originais dos retornos.
- Oferecer argumento ``returns`` como DataFrame para manter rótulos.
- Adicionar checagens de estabilidade (condicionamento, determinante).
- Documentar as referências acadêmicas usadas.

Testes recomendados
-------------------
- `tests/estimators/test_cov.py` cobrindo:
    * equivalência com `numpy.cov` em casos lineares simples,
    * semidefinitude garantida (autovalores ≥ 0),
    * robustez a outliers (comparando variação entre métodos),
    * comportamento com colunas altamente correlacionadas/colineares.
"""
