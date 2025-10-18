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
# Importa bibliotecas necessárias
import numpy as np               # NumPy: operações matemáticas e manipulação de arrays
import pandas as pd              # Pandas: suporte a Series e DataFrames
from typing import Union         # Union: permite indicar múltiplos tipos em tipagem

# Define um tipo genérico que pode ser numpy array, pandas Series ou DataFrame
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


# ----------------------------------------------------------------------
# Funções auxiliares para padronizar entrada e saída entre numpy/pandas
# ----------------------------------------------------------------------

def _handle_input_type(data: ArrayLike):
    """Extrai os valores numpy e guarda o tipo/metadados originais."""
    
    # Caso a entrada seja uma Series do pandas
    if isinstance(data, pd.Series):
        # Retorna valores numpy, tipo original (Series), e o índice (index)
        return data.values, pd.Series, data.index, None
    
    # Caso a entrada seja um DataFrame do pandas
    elif isinstance(data, pd.DataFrame):
        # Retorna valores numpy, tipo original (DataFrame), índice e colunas
        return data.values, pd.DataFrame, data.index, data.columns
    
    # Caso seja um array numpy puro
    elif isinstance(data, np.ndarray):
        # Retorna o próprio array e informações vazias de índice/coluna
        return data, np.ndarray, None, None
    
    # Caso não seja nenhum tipo suportado
    else:
        # Lança erro de tipo
        raise TypeError(f"Tipo de entrada não suportado: {type(data)}")


def _reconstruct_output(data: np.ndarray, original_type, index=None, columns=None):
    """Reconstrói o tipo pandas original a partir do resultado numpy."""
    
    # Se o tipo original era uma Series
    if original_type == pd.Series:
        # Reconstrói o objeto Series com o mesmo índice
        return pd.Series(data, index=index)
    
    # Se o tipo original era um DataFrame
    if original_type == pd.DataFrame:
        # Reconstrói o DataFrame com índice e colunas originais
        return pd.DataFrame(data, index=index, columns=columns)
    
    # Caso contrário (numpy array), retorna o resultado como está
    return data


# ----------------------------------------------------------------------
# 1. Projeção no simplex
# ----------------------------------------------------------------------

def project_to_simplex(vector: np.ndarray, sum_to: float = 1.0) -> np.ndarray:
    """
    Projeta um vetor no simplex padrão (valores >=0 e soma = 1).
    """

    # Verifica se o vetor é 1D, pois o algoritmo só funciona nesse caso
    if vector.ndim != 1:
        raise ValueError("A projeção no simplex só é definida para vetores 1D.")
        
    # Obtém o tamanho do vetor (número de elementos)
    n_features = vector.shape[0]
    
    # Ordena os valores do vetor em ordem decrescente
    u = np.sort(vector)[::-1]
    
    # Calcula a soma cumulativa subtraindo o valor alvo da soma
    cssv = np.cumsum(u) - sum_to
    
    # Cria vetor de índices (1, 2, 3, ...)
    ind = np.arange(n_features) + 1
    
    # Verifica quais elementos satisfazem a condição do simplex
    cond = u - cssv / ind > 0
    
    # Define rho como o último índice que satisfaz a condição (ou n_features se nenhum)
    rho = ind[cond][-1] if np.any(cond) else n_features
    
    # Calcula o deslocamento (theta) que garante soma correta
    theta = (cssv[rho - 1]) / rho if rho > 0 else 0
    
    # Calcula o vetor projetado: valores negativos viram 0
    w = np.maximum(vector - theta, 0)
    
    # Retorna o vetor projetado no simplex
    return w


# ----------------------------------------------------------------------
# 2. Soft-thresholding
# ----------------------------------------------------------------------

def soft_threshold(data: ArrayLike, lam: float) -> ArrayLike:
    """
    Aplica o operador de soft-thresholding S(x, λ) = sign(x) * max(|x| - λ, 0)
    """

    # Garante que o parâmetro λ seja não negativo
    if lam < 0:
        raise ValueError("O parâmetro lambda (lam) deve ser não-negativo.")
    
    # Converte entrada para numpy e armazena metadados originais
    values, original_type, index, columns = _handle_input_type(data)
    
    # Aplica a fórmula do soft-thresholding
    result = np.sign(values) * np.maximum(np.abs(values) - lam, 0)
    
    # Reconstrói o resultado no mesmo formato de entrada (numpy/pandas)
    return _reconstruct_output(result, original_type, index, columns)


# ----------------------------------------------------------------------
# 3. Normalização vetorial
# ----------------------------------------------------------------------

def normalize_vector(vector: ArrayLike, norm: str = "l2") -> ArrayLike:
    """
    Normaliza um vetor ou colunas de um DataFrame usando norma L1, L2 ou max.
    """

    # Extrai valores e metadados
    values, original_type, index, columns = _handle_input_type(vector)
    
    # Define o eixo: None se for vetor 1D, 0 se for matriz 2D
    axis = 0 if values.ndim > 1 else None

    # Calcula norma L2
    if norm == "l2":
        norm_val = np.linalg.norm(values, axis=axis, keepdims=True)
    # Calcula norma L1
    elif norm == "l1":
        norm_val = np.sum(np.abs(values), axis=axis, keepdims=True)
    # Calcula norma máxima
    elif norm == "max":
        norm_val = np.max(np.abs(values), axis=axis, keepdims=True)
    # Caso tipo inválido
    else:
        raise ValueError("Norma deve ser 'l1', 'l2', ou 'max'.")

    # Evita divisão por zero substituindo zeros por 1.0
    norm_val[norm_val == 0] = 1.0
    
    # Divide os valores originais pela norma (normalização)
    result = values / norm_val
    
    # Retorna o resultado no mesmo formato do input
    return _reconstruct_output(result, original_type, index, columns)


# ----------------------------------------------------------------------
# 4. Norma ponderada
# ----------------------------------------------------------------------

def weighted_norm(vector: ArrayLike, weights: ArrayLike, order: int = 2) -> float:
    """
    Calcula norma ponderada: (∑ w_i * |x_i|^p)^(1/p)
    """

    # Extrai valores do vetor e dos pesos
    vec_values, _, _, _ = _handle_input_type(vector)
    w_values, _, _, _ = _handle_input_type(weights)

    # Garante que ambos tenham o mesmo formato
    if vec_values.shape != w_values.shape:
        raise ValueError("Vetor e pesos devem ter o mesmo formato.")

    # Calcula |x_i|^p multiplicado pelos pesos
    weighted_abs = w_values * np.power(np.abs(vec_values), order)
    
    # Soma tudo e aplica a raiz (1/p)
    return np.power(np.sum(weighted_abs), 1.0 / order)


# ----------------------------------------------------------------------
# 5. Clipping numérico com tolerância
# ----------------------------------------------------------------------

def clip_with_tolerance(vector: ArrayLike, lower: float, upper: float, tol: float = 1e-9) -> ArrayLike:
    """
    Realiza o clipping numérico de forma robusta, respeitando tolerâncias.
    """

    # Extrai valores e metadados
    values, original_type, index, columns = _handle_input_type(vector)

    # Ajusta valores muito próximos dos limites inferior/superior
    values[np.isclose(values, lower, atol=tol)] = lower
    values[np.isclose(values, upper, atol=tol)] = upper
    
    # Aplica clip padrão (garante que valores fiquem entre lower e upper)
    result = np.clip(values, lower, upper)
    
    # Retorna no mesmo formato do input
    return _reconstruct_output(result, original_type, index, columns)


# ----------------------------------------------------------------------
# 6. Inversão estável de matriz
# ----------------------------------------------------------------------

def stable_inverse(matrix: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """
    Calcula inversa de matriz com regularização (ridge) para estabilidade numérica.
    """

    # Verifica se a entrada é uma matriz quadrada (necessário para inversão)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A entrada deve ser uma matriz quadrada.")
    
    # Obtém a dimensão da matriz (n x n)
    n = matrix.shape[0]
    
    # Cria uma matriz identidade do mesmo tamanho
    identity = np.eye(n)
    
    # Adiciona ridge * identidade para estabilizar a inversão
    stabilized = matrix + ridge * identity
    
    # Calcula a inversa usando álgebra linear
    return np.linalg.inv(stabilized)


# ----------------------------------------------------------------------
# 7. Versão segura de expm1 (exp(x) - 1)
# ----------------------------------------------------------------------

def expm1_safe(data: ArrayLike) -> ArrayLike:
    """
    Wrapper seguro para np.expm1, compatível com pandas e numpy.
    """

    # Converte entrada para numpy e salva metadados
    values, original_type, index, columns = _handle_input_type(data)
    
    # Calcula exp(x) - 1 de forma estável (numérica)
    result = np.expm1(values)
    
    # Reconstrói o mesmo tipo da entrada
    return _reconstruct_output(result, original_type, index, columns)

