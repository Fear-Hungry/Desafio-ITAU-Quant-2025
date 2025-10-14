"""Blueprint for friendly input validation helpers.

Objetivo
--------
Oferecer funções utilitárias que inspecionam dados de entrada (DataFrames,
numpy arrays) e levantam exceções descritivas antes de alimentar pipelines.

Componentes sugeridos
---------------------
- `assert_no_nans(obj, context="")`
    Garante ausência de valores faltantes e, caso existam, relata índice/colunas.
- `assert_shape(obj, expected_shape=None, min_rows=None, min_cols=None)`
    Verifica dimensões mínimas/esperadas, com mensagens claras.
- `assert_symmetric(matrix, atol=1e-8)`
    Confirma simetria (essencial para matrizes de covariância).
- `assert_psd(matrix, atol=1e-8)`
    Checa semidefinitude positiva (autovalores ≥ -atol) e sugere correções.
- `validate_returns_frame(df)`
    Função especializada: índice ordenado, colunas não vazias, dtype numérico.
- `validate_weights_vector(weights)`
    Assegura soma ≈ 1, sem NaN, e magnitude dentro de limites.

Considerações
-------------
- Integrar com `utils.logging_config` para logs detalhados quando disponível.
- Preferir levantar `ValueError`/`TypeError` com mensagens amigáveis.
- Reutilizar nos módulos `data`, `optimization`, `risk` para garantir consistência.

Testes recomendados
-------------------
- `tests/utils/test_checks.py` cobrindo:
    * disparo de erro em presença de NaNs/simetria quebrada,
    * aceitação de inputs válidos,
    * mensagens contendo contexto informativo (nome da função ou do dataset).
"""
