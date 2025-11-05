"""Blueprint for deterministic seed management.

Objetivo
--------
Fornecer funções centralizadas para fixar seeds em bibliotecas usadas no projeto,
garantindo reprodutibilidade entre runs.

Componentes sugeridos
---------------------
- `set_global_seeds(seed, *, numpy=True, python=True, pandas=True, cvxpy=True)`
    Configura seeds do módulo `random`, `numpy.random`, `pandas`, e, quando possível,
    de solvers suportados (ex.: SCS, OSQP) chamando `solver_utils.set_solver_seed`.
- `seed_context(seed)`
    Context manager que aplica seeds temporariamente e restaura estado após uso.
- `rng_factory(seed)`
    Retorna `numpy.random.Generator` com política definida (ex.: PCG64).
- `hash_seed_from_config(config)`
    Gera seed determinística a partir de config (ex.: string YAML → hash → int).
- `register_seed_logging(logger, seed)`
    Loga seed atual para auditoria.

Considerações
-------------
- Documentar que algumas bibliotecas (TensorFlow, PyTorch) exigem setups extras.
- Lidar com seeds negativos ou maiores que 2**32-1 (normalizar).
- Usar `numpy.random.default_rng` como default moderno.

Testes recomendados
-------------------
- `tests/utils/test_seed.py` cobrindo:
    * repetibilidade (mesmo seed → outputs iguais),
    * contexto temporário restabelecendo estado anterior,
    * integração com solver_utils (mock) setando seed corretamente.
"""

import hashlib
import json
import logging
import random
from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np

# Constante para normalização da seed
MAX_SEED_VALUE = 2**32


def set_solver_seed(seed: int) -> None:
    """
    Função mock para configurar a seed de solvers de otimização.

    Tenta delegar para ``arara_quant.optimization.core.solver_utils.set_solver_seed``
    caso esteja disponível; caso contrário, apenas informa via log.
    """
    logging.info(f"[SOLVER MOCK] Configurando seed do solver para: {seed}")

    try:
        from arara_quant.optimization.core import solver_utils as core_solver_utils
    except ImportError:
        logging.debug("solver_utils indisponível; pulando configuração do solver.")
        return

    delegate = getattr(core_solver_utils, "set_solver_seed", None)
    if callable(delegate):
        delegate(seed)
    else:
        logging.debug(
            "solver_utils.set_solver_seed não implementado; sem configuração adicional."
        )


def set_global_seeds(
    seed: int,
    *,
    numpy: bool = True,
    python: bool = True,
    pandas: bool = True,
    cvxpy: bool = True,
):
    """
    Configura seeds globais para as principais bibliotecas para garantir reprodutibilidade.

    Atenção: Bibliotecas como TensorFlow, PyTorch e outras que usam GPUs podem
    exigir configurações adicionais e desativação de algoritmos não-determinísticos
    para garantir 100% de reprodutibilidade.

    Args:
        seed (int): O valor da seed a ser usada. Será normalizada para um inteiro
                    positivo de 32 bits.
        numpy (bool): Se True, define a seed para `numpy.random`.
        python (bool): Se True, define a seed para o módulo `random` do Python.
        pandas (bool): Se True, garante que o ambiente para operações estocásticas
                       do pandas (que depende do numpy) seja semeado.
        cvxpy (bool): Se True, tenta configurar a seed para solvers de otimização
                      suportados via `solver_utils`.
    """
    # Normaliza a seed para o intervalo [0, 2**32 - 1]
    seed = abs(seed) % MAX_SEED_VALUE

    if python:
        random.seed(seed)
        logging.debug(f"Seed do módulo 'random' configurada para {seed}.")

    if numpy:
        np.random.seed(seed)
        logging.debug(f"Seed global do 'numpy.random' configurada para {seed}.")

    if pandas:
        # O pandas utiliza o gerador de números aleatórios do NumPy, então
        # configurar a seed do NumPy já é suficiente para operações como df.sample().
        # Esta flag serve mais para clareza e documentação.
        logging.debug("Ambiente do pandas semeado via NumPy.")

    if cvxpy:
        # Delega a configuração específica do solver para um módulo dedicado
        set_solver_seed(seed)
        logging.debug(
            f"Tentativa de configurar seed de solvers via solver_utils com {seed}."
        )


@contextmanager
def seed_context(seed: int):
    """
    Context manager que aplica seeds temporariamente e restaura o estado anterior.

    Ideal para seções de código que precisam ser reprodutíveis isoladamente,
    sem afetar o estado global de aleatoriedade do resto da aplicação.

    Args:
        seed (int): A seed a ser aplicada temporariamente dentro do contexto.
    """
    # Guarda o estado original dos geradores de números aleatórios
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    logging.debug(f"Entrando no contexto com seed {seed}. Estado original salvo.")

    try:
        set_global_seeds(seed)
        yield
    finally:
        # Restaura o estado original ao sair do contexto
        random.setstate(original_python_state)
        np.random.set_state(original_numpy_state)
        logging.debug(
            "Saindo do contexto. Estado original de aleatoriedade restaurado."
        )


def rng_factory(seed: Optional[int] = None) -> np.random.Generator:
    """
    Cria e retorna uma instância do gerador de números aleatórios moderno do NumPy.

    Esta é a abordagem recomendada para código novo, pois cria um gerador isolado
    que não afeta o estado global `np.random`.

    Args:
        seed (Optional[int]): A seed para o gerador. Se None, a inicialização
                              será não-determinística.

    Returns:
        np.random.Generator: Uma instância do gerador de números aleatórios.
    """
    return np.random.default_rng(seed)


def hash_seed_from_config(config: Dict[str, Any]) -> int:
    """
    Gera uma seed determinística a partir de um dicionário de configuração.

    O dicionário é convertido para uma string JSON com chaves ordenadas para
    garantir que a mesma configuração sempre gere o mesmo hash e, consequentemente,
    a mesma seed.

    Args:
        config (Dict[str, Any]): O dicionário de configuração.

    Returns:
        int: Uma seed de 32 bits gerada a partir do hash da configuração.
    """
    # Serializa o dicionário para uma string JSON, com chaves ordenadas para garantir
    # que a representação seja sempre a mesma para a mesma config.
    config_str = json.dumps(config, sort_keys=True)

    # Cria um hash SHA-256 da string codificada em bytes
    hasher = hashlib.sha256(config_str.encode("utf-8"))

    # Converte o digest hexadecimal do hash para um inteiro
    hash_int = int(hasher.hexdigest(), 16)

    # Normaliza o inteiro para o intervalo de seed de 32 bits
    return hash_int % MAX_SEED_VALUE


def register_seed_logging(logger: logging.Logger, seed: int):
    """
    Loga a seed que está sendo usada para fins de auditoria e reprodutibilidade.

    Args:
        logger (logging.Logger): A instância do logger a ser usada.
        seed (int): A seed que está sendo registrada.
    """
    logger.info(f"Execução utilizando a seed: {seed}")
