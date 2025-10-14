"""Black-Litterman estimator blueprint.

Objetivo
--------
Fornecer funções para gerar retornos esperados combinando equilíbrio de mercado
com opiniões discretas, seguindo o framework de Black-Litterman.

Componentes sugeridos
---------------------
- `reverse_optimization(weights, cov, risk_aversion)`:
    Derive o vetor de retornos implícitos (π) a partir das alocações de mercado.
- `build_projection_matrix(views)`:
    Converte views estruturadas (dict/dataclass) em matrizes ``P`` e ``Q``.
- `view_uncertainty(views, tau, cov)`:
    Constrói ``Omega`` (matriz de incerteza das views) suportando opções:
    * escala diagonal usando volatilidade do ativo;
    * fator comum (idêntico para todas as views);
    * matriz customizada fornecida pelo usuário.
- `posterior_returns(pi, cov, P, Q, Omega, tau)`:
    Combina prior e views retornando μ_BL e matriz de covariância ajustada.
- `black_litterman(...)`:
    Função principal que orquestra as etapas acima, aplicando verificações
    numéricas (PSD, condicionamento) e normalizando pesos.

Checklist de implementação
--------------------------
- Validar dimensões entre ``P``, ``Q`` e número de ativos.
- Usar fator de escala ``tau`` configurável.
- Permitir prior via pesos de mercado ou vetor já calculado.
- Suportar views absolutas e relativas.
- Explicar claramente no docstring das funções como os inputs devem ser
  estruturados (e.g., `views=[{"type": "absolute", "ticker": "SPY", ...}]`).
- Incluir testes em `tests/estimators/test_bl.py` cobrindo:
    * ausência de views (retorno igual ao prior),
    * views absolutas simples,
    * views relativas com Omega customizado,
    * comportamento quando tau → 0 (prior domina) e tau grande (views dominam).
"""
