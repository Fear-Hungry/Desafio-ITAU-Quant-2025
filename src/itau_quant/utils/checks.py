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

import pandas as pd # essencial para manipulação de DataFrames
import numpy as np # essencial para manipulação numérica

# primeira função: verificar se existem valores NaNs (representando dados faltantes)
def assert_no_nans(obj, context=""):
    """ O objeto a ser verificado pode ser uma tabela, coluna ou array.
    Se encontrar NaNs, levanta ValueError com detalhes.
    context: string opcional para identificar a origem (ex: nome da função).
    """
    # verificamos se o objeto é um DataFrame ou Series do pandas:
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if obj.isnull().values.any(): # essa propriedade retorna True se houver NaNs
            # localizar posições dos NaNs
            if isinstance(obj, pd.DataFrame): # DataFrame
                nan_positions = np.argwhere(obj.isnull().values)
                # o nan_positions é uma lista de tuplas (row, col) que indicam onde estão os NaNs
                # em details vamos formatar isso para a mensagem
                details = ", ".join([f"(row {r}, col '{obj.columns[c]}')" for r, c in nan_positions])
            else:  # Series
                nan_positions = obj[obj.isnull()].index.tolist() # índices dos NaNs
                details = ", ".join([f"index {i}" for i in nan_positions]) # formatar para mensagem
            msg = f"Input contains NaNs at positions: {details}."
            if context:
                msg = f"[{context}] " + msg # adiciona contexto se fornecido
            raise ValueError(msg) # levantamos o erro com a mensagem detalhada
    elif isinstance(obj, np.ndarray): # se for um numpy array
        if np.isnan(obj).any():# verifica se há NaNs
            # localizar índices dos NaNs
            nan_indices = np.argwhere(np.isnan(obj))
            details = ", ".join([f"index {tuple(idx)}" for idx in nan_indices])# formatar para mensagem
            msg = f"Entrada: {details}." # mensagem detalhada
            if context: # adiciona contexto se fornecido
                msg = f"[{context}] " + msg 
            raise ValueError(msg) # levantamos o erro com a mensagem detalhada
    else:
        raise TypeError("Não é DataFrame, Series ou ndarray.") # tipo inválido
    

# segunda função: verificar se o shape (dimensões) está conforme esperado
def assert_shape(obj, expected_shape=None, min_rows=None, min_cols=None, context=""):
    """ Verifica se o objeto (DataFrame ou ndarray) tem o shape esperado.
    Pode checar shape exato (expected_shape) ou mínimos (min_rows, min_cols).
    Levanta ValueError com detalhes se não conformar.
    context: string opcional para identificar a origem (ex: nome da função).
    """
    " args:"
    " expected_shape: tupla (rows, cols) esperada exatamente"
    " min_rows: int, número mínimo de linhas esperado"
    " min_cols: int, número mínimo de colunas esperado"
    " context: string opcional para contexto na mensagem de erro"
    if isinstance(obj, pd.DataFrame): # se for DataFrame
        actual_shape = obj.shape # obtém o shape atual
    elif isinstance(obj, np.ndarray): # se for ndarray
        actual_shape = obj.shape # obtém o shape atual
    else:
        raise TypeError("Não é DataFrame ou ndarray.") # tipo inválido

    if expected_shape is not None: # se esperado shape exato foi fornecido
        if actual_shape != expected_shape: # compara com o atual
            msg = f"Esperado shape {expected_shape}, mas recebido {actual_shape}." # mensagem detalhada
            if context: # adiciona contexto se fornecido
                msg = f"[{context}] " + msg
            raise ValueError(msg)

    if min_rows is not None:# se mínimo de linhas foi fornecido
        if actual_shape[0] < min_rows: # compara com o atual
            msg = f"Esperado no mínimo {min_rows} linhas, mas recebido {actual_shape[0]}." # mensagem detalhada
            if context: # adiciona contexto se fornecido
                msg = f"[{context}] " + msg # adiciona contexto se fornecido
            raise ValueError(msg) # levantamos o erro com a mensagem detalhada

    if min_cols is not None: # se mínimo de colunas foi fornecido
        if actual_shape[1] < min_cols: # compara com o atual
            msg = f"Esperado no mínimo {min_cols} colunas, mas recebido {actual_shape[1]}." # mensagem detalhada
            if context: # adiciona contexto se fornecido
                msg = f"[{context}] " + msg # adiciona contexto se fornecido
            raise ValueError(msg) # levantamos o erro com a mensagem detalhada
        
# terceira função: verificar se uma matriz é simétrica
def assert_symmetric(matrix, atol=1e-8):
    """ Verifica se a matriz é simétrica dentro de uma tolerância atol.
    Levanta ValueError se não for simétrica.
    """
    if not isinstance(matrix, np.ndarray): # verifica se é ndarray
        raise TypeError("A entrada deve ser um ndarray.") # tipo inválido
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: # verifica se é quadrada
        raise ValueError("A matriz deve ser quadrada.") # não é quadrada
    if not np.allclose(matrix, matrix.T, atol=atol): # compara com a transposta
        raise ValueError("A matriz não é simétrica dentro da tolerância especificada.") # não é simétrica

# quarta função: verificar se uma matriz é semidefinida positiva
def assert_psd(matrix, atol=1e-8):
    """ Verifica se a matriz é semidefinida positiva (autovalores ≥ -atol).
    Levanta ValueError se não for PSD.
    """
    if not isinstance(matrix, np.ndarray): # verifica se é ndarray
        raise TypeError("A entrada deve ser um ndarray.") # tipo inválido
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: # verifica se é quadrada
        raise ValueError("A matriz deve ser quadrada.") # não é quadrada
    # calcula os autovalores
    eigenvalues = np.linalg.eigvalsh(matrix) # eigvalsh é mais estável para matrizes simétricas
    if np.any(eigenvalues < -atol): # verifica se algum autovalor é menor que -atol
        raise ValueError("A matriz não é semidefinida positiva dentro da tolerância especificada.") # não é PSD
    
# quinta função: validar um DataFrame de retornos
def validate_returns_frame(df):
    """ Valida um DataFrame de retornos financeiros.
    Checa: índice datetime ordenado, colunas não vazias, dtype numérico, sem NaNs.
    Levanta ValueError/TypeError com detalhes se não conformar.
    """
    if not isinstance(df, pd.DataFrame): # verifica se é DataFrame
        raise TypeError("A entrada deve ser um DataFrame.") # tipo inválido
    if df.empty: # verifica se está vazio
        raise ValueError("O DataFrame de retornos está vazio.") # vazio
    if not pd.api.types.is_datetime64_any_dtype(df.index): # verifica se o índice é datetime
        raise TypeError("O índice do DataFrame deve ser do tipo datetime.") # índice inválido
    if not df.index.is_monotonic_increasing: # verifica se o índice está ordenado
        raise ValueError("O índice do DataFrame deve estar ordenado em ordem crescente.") # não está ordenado
    if df.isnull().values.any(): # verifica se há NaNs
        raise ValueError("O DataFrame de retornos contém valores NaN.") # contém NaNs
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes): # verifica se todas as colunas são numéricas
        raise TypeError("Todas as colunas do DataFrame devem ser de tipo numérico.") # colunas não numéricas
    
# sexta função: validar um vetor de pesos
def validate_weights_vector(weights):
    """ Valida um vetor de pesos (numpy array ou pandas Series).
    Checa: soma ≈ 1, sem NaNs, magnitude dentro de limites.
    Levanta ValueError/TypeError com detalhes se não conformar.
    """
    if isinstance(weights, pd.Series): # se for Series
        weights = weights.values # converte para ndarray
    if not isinstance(weights, np.ndarray): # verifica se é ndarray
        raise TypeError("Os pesos devem ser um ndarray ou Series.") # tipo inválido
    if weights.ndim != 1: # verifica se é 1D
        raise ValueError("Os pesos devem ser um vetor unidimensional.") # não é 1D
    if np.isnan(weights).any(): # verifica se há NaNs
        raise ValueError("O vetor de pesos contém valores NaN.") # contém NaNs
    total = np.sum(weights) # soma dos pesos
    if not np.isclose(total, 1.0, atol=1e-6): # verifica se a soma é aproximadamente 1
        raise ValueError(f"A soma dos pesos deve ser 1.0, mas é {total}.") # soma incorreta
    if np.any(np.abs(weights) > 1e6): # verifica se algum peso é muito grande em magnitude
        raise ValueError("Alguns pesos têm magnitude excessivamente alta.") # magnitude alta
