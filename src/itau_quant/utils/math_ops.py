"""Blueprint for reusable mathematical operations.

Objetivo
--------
Centralizar rotinas matemáticas comuns utilizadas por otimização, estimadores e
pós-processamento.

Componentes sugeridos
---------------------
- `project_to_simplex(vector, sum_to=1.0)`
    Implementa projeção Euclidiana no simplex (útil para pesos não-negativos).
- `soft_threshold(vector, lam)`
    Operador soft-thresholding para promover sparsidade (|x|-λ)+ sign(x).
- `normalize_vector(vector, norm="l2")`
    Normaliza por norma L1/L2 ou max.
- `weighted_norm(vector, weights, order)`
    Cálculo de norma ponderada (ex.: ∑ w_i |x_i|^p) para penalidades customizadas.
- `clip_with_tolerance(vector, lower, upper, tol=1e-9)`
    Clipping numérico robusto, evitando ultrapassar bounds por erros de floating.
- `stable_inverse(matrix, ridge=1e-8)`
    Inversão regularizada (adiciona ridge) para matrizes quase singulares.
- `expm1_safe(x)`
    Variante que usa `np.expm1` porém lida com arrays/Series/truncamentos.

Considerações
-------------
- Aceitar pandas Series/DataFrames além de numpy arrays, devolvendo mesmo tipo.
- Documentar comportamento para entradas 1D/2D e broadcast.
- Garantir testes de estabilidade numérica (valores extremos).

Testes recomendados
-------------------
- `tests/utils/test_math_ops.py` cobrindo:
    * projeção no simplex mantendo soma=1,
    * soft-threshold com casos positivos/negativos,
    * inversão estável comparada com `np.linalg.inv` em matriz bem condicionada,
    * normalização L1/L2 em vetores com zeros.
"""
