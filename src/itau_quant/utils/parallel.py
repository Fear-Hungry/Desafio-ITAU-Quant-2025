"""Blueprint for parallel execution helpers.

Objetivo
--------
Abstrair execução paralela de tarefas (GA, backtests, bootstrap) de forma
reprodutível e fácil de integrar.

Componentes sugeridos
---------------------
- `parallel_map(func, iterable, backend="thread", max_workers=None, chunksize=1)`
    Interface genérica semelhante a map que decide entre ThreadPool/ProcessPool/
    joblib conforme backend.
- `batched(iterable, batch_size)`
    Helper para enviar lotes de tarefas aos workers.
- `with_seed_context(seed, worker_id)`
    Garante seeds determinísticas por worker (np/random/randomstate).
- `run_ga_in_parallel(population, evaluator, backend, max_workers)`
    Função especializada para avaliar indivíduos do GA distribuindo workload.
- `parallel_backtest(strategies, engine, backend, max_workers)`
    Executa múltiplos cenários/backtests em paralelo capturando resultados.
- `collect_exceptions(results)`
    Junta exceções levantadas por workers e re-levanta com contexto.

Considerações
-------------
- Suportar fallback para execução sequencial (debug). 
- Incluir timeouts e cancelamento gracioso.
- Garantir que recursos (processos) sejam finalizados corretamente.
- Integrar com logging para medir tempo total por job.

Testes recomendados
-------------------
- `tests/utils/test_parallel.py` cobrindo:
    * execução determinística com seeds fixos,
    * propagação de exceções dos workers,
    * comparação de resultados sequencial vs. paralelo,
    * verificação de que recursos são liberados (sem processos zumbis).
"""
import time
import logging
import random
import itertools
from contextlib import contextmanager
from typing import Callable, Iterable, List, Any, Optional

import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
from joblib import Parallel, delayed

# Configuração básica de logging para medir o tempo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Componentes Principais ---

def parallel_map(
    func: Callable,
    iterable: Iterable,
    backend: str = "process",
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None
) -> List[Any]:
    """
    Interface genérica semelhante a `map` para execução paralela de tarefas.

    Args:
        func (Callable): A função a ser aplicada a cada item do iterável.
        iterable (Iterable): O conjunto de dados a ser processado.
        backend (str): O backend de paralelização. Opções: 'process' (padrão), 
                       'thread', 'joblib', ou 'sequential' para debug.
        max_workers (Optional[int]): Número máximo de workers. Se None, usa o padrão
                                     do backend.
        timeout (Optional[float]): Tempo máximo em segundos para esperar pelo 
                                   resultado de cada tarefa.

    Returns:
        List[Any]: Uma lista com os resultados na mesma ordem do iterável de entrada.
    
    Raises:
        TimeoutError: Se uma tarefa exceder o tempo limite especificado.
        Exception: Propaga a primeira exceção encontrada em um dos workers.
    """
    job_name = getattr(func, '__name__', 'anonymous_job')
    logging.info(f"Iniciando job paralelo '{job_name}' com backend '{backend}'...")
    start_time = time.time()
    
    # Fallback para execução sequencial (ótimo para debugging)
    if backend == 'sequential' or max_workers == 1:
        results = [func(item) for item in iterable]
        end_time = time.time()
        logging.info(f"Job '{job_name}' concluído em {end_time - start_time:.2f}s.")
        return results

    # Mapeamento de backends para executores
    executor_map = {
        "thread": ThreadPoolExecutor,
        "process": ProcessPoolExecutor
    }
    
    if backend in executor_map:
        # Usando concurrent.futures (ThreadPoolExecutor ou ProcessPoolExecutor)
        # Usar submit em vez de map para ter controle fino sobre exceções e timeouts
        with executor_map[backend](max_workers=max_workers) as executor:
            future_to_item = {executor.submit(func, item): item for item in iterable}
            results = {}
            try:
                for future in as_completed(future_to_item, timeout=timeout):
                    item = future_to_item[future]
                    try:
                        results[item] = future.result()
                    except Exception as e:
                        # Captura a exceção e a armazena para análise posterior
                        logging.error(f"Worker para o item '{item}' gerou uma exceção: {e}")
                        results[item] = e # Armazena a exceção no lugar do resultado
            except TimeoutError:
                logging.error(f"Job '{job_name}' excedeu o timeout global de {timeout}s.")
                raise
        
        # Reordena os resultados e coleta exceções
        ordered_results = [results[item] for item in iterable]

    elif backend == 'joblib':
        # Usando Joblib, que é ótimo para tarefas com numpy
        # 'loky' é o backend de processo padrão e mais robusto do joblib
        joblib_backend = 'threading' if backend == 'thread' else 'loky'
        tasks = [delayed(func)(item) for item in iterable]
        ordered_results = Parallel(n_jobs=max_workers, backend=joblib_backend)(tasks)
        
    else:
        raise ValueError(f"Backend '{backend}' não reconhecido. Use 'process', 'thread', 'joblib' ou 'sequential'.")
        
    end_time = time.time()
    logging.info(f"Job '{job_name}' concluído em {end_time - start_time:.2f}s.")
    
    return ordered_results

# --- Helpers e Funções Especializadas ---

def batched(iterable: Iterable, batch_size: int) -> Iterable[tuple]:
    """
    Helper para agrupar um iterável em lotes de tamanho fixo.
    Ex: batched('ABCDEFG', 3) --> ABC DEF G
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, batch_size)):
        yield batch

@contextmanager
def with_seed_context(base_seed: int, worker_id: int):
    """
    Context manager para garantir seeds determinísticas por worker.
    Isso é crucial para reprodutibilidade em simulações estocásticas.
    
    Uso:
    with with_seed_context(base_seed=42, worker_id=i):
        # código que usa np.random ou random
    """
    try:
        # Gera uma seed única e determinística para este worker
        worker_seed = base_seed + worker_id
        
        # Guarda o estado original dos geradores de números aleatórios
        original_np_state = np.random.get_state()
        original_random_state = random.getstate()
        
        # Define as novas seeds
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        yield
    finally:
        # Restaura o estado original para não afetar outras partes do programa
        np.random.set_state(original_np_state)
        random.set_state(original_random_state)

def collect_exceptions(results: List[Any], re_raise: bool = True):
    """
    Coleta exceções de uma lista de resultados de workers e opcionalmente as levanta.

    Args:
        results (List[Any]): Lista de resultados, que pode conter objetos Exception.
        re_raise (bool): Se True, levanta um RuntimeError contendo as exceções.

    Returns:
        tuple[list, list]: Uma tupla contendo (resultados_validos, excecoes).
    """
    exceptions = [res for res in results if isinstance(res, BaseException)]
    valid_results = [res for res in results if not isinstance(res, BaseException)]
    
    if exceptions and re_raise:
        error_messages = "\n".join([f"  - {type(e).__name__}: {e}" for e in exceptions])
        raise RuntimeError(f"{len(exceptions)} worker(s) falharam com as seguintes exceções:\n{error_messages}")
        
    return valid_results, exceptions

# --- Funções de Aplicação Específicas (Exemplos) ---

def run_ga_in_parallel(
    population: List[Any],
    evaluator_func: Callable,
    backend: str = "process",
    max_workers: Optional[int] = None
) -> List[float]:
    """
    Função especializada para avaliar a fitness de uma população de um GA em paralelo.
    """
    logging.info(f"Avaliando {len(population)} indivíduos da população...")
    results_with_errors = parallel_map(evaluator_func, population, backend, max_workers)
    
    # Processa resultados e garante que exceções sejam tratadas
    fitness_scores, exceptions = collect_exceptions(results_with_errors, re_raise=True)
    
    return fitness_scores

def parallel_backtest(
    strategies: List[Any],
    backtest_engine: Callable,
    backend: str = "process",
    max_workers: Optional[int] = None
) -> List[Any]:
    """
    Executa múltiplos backtests de estratégias em paralelo.
    """
    logging.info(f"Executando backtest para {len(strategies)} estratégias...")
    results_with_errors = parallel_map(backtest_engine, strategies, backend, max_workers)
    
    # Processa resultados, levantando um erro se algum backtest falhar
    backtest_results, _ = collect_exceptions(results_with_errors, re_raise=True)

    return backtest_results
