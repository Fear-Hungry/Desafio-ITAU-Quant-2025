"""Blueprint for timing/benchmark utilities.

Objetivo
--------
Criar ferramentas para medir e registrar tempos de execução de trechos críticos.

Componentes sugeridos
---------------------
- `time_block(name, logger=None)`
    Context manager que registra início/fim usando `time.perf_counter()` e envia
    log estruturado (nome, duração, metadados).
- `time_function(logger=None)`
    Decorador que aplica `time_block` a funções, preservando metadados (`functools.wraps`).
- `Timer` class
    Objeto com métodos `start`, `stop`, `elapsed`, reutilizável em loops.
- `benchmark(fn, *args, repeat=3, number=1)`
    Mede tempos repetidos (similar a `timeit`) e retorna estatísticas (mean/std).
- `profile_memory(fn, *args)` (opcional)
    Integra com `tracemalloc` para medir memória durante execução.

Considerações
-------------
- Logs devem usar `utils.logging_config` para manter padrão.
- Incluir opção de coletar métricas (ex.: enviar para Prometheus ou JSON).
- Permitir desativar facilmente (flag global) para não impactar produção.

Testes recomendados
-------------------
- `tests/utils/test_timing.py` cobrindo:
    * context manager registrando duração aproximada conhecida,
    * decorador preservando nome/docstring,
    * benchmark retornando número de execuções correto,
    * (opcional) memory profiler fornecendo dados positivos.
"""

import time
import logging
import functools
import tracemalloc
from contextlib import contextmanager
from typing import Callable, Optional, List, Dict, Any

import numpy as np

# --- Configuração Global ---

# Flag para desativar todos os timers e benchmarks, útil para produção.
TIMING_ENABLED = True

# Em um projeto real, isso seria configurado por um módulo central de logging.
# Para este exemplo, vamos configurar um logger básico.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
default_logger = logging.getLogger("TimingUtils")


# --- Componentes Principais ---

@contextmanager
def time_block(name: str, logger: Optional[logging.Logger] = None, collect_metrics: Optional[Dict] = None):
    """
    Context manager para medir e registrar o tempo de execução de um bloco de código.

    Args:
        name (str): Um nome descritivo para o bloco de código sendo medido.
        logger (Optional[logging.Logger]): Logger a ser usado. Se None, usa o logger padrão.
        collect_metrics (Optional[Dict]): Um dicionário onde a métrica de duração será
                                          adicionada. Ex: {'timing.my_block': 0.123}.
    """
    if not TIMING_ENABLED:
        yield
        return

    log = logger or default_logger
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        log.info(f"Block '{name}' executed in {duration:.4f}s")
        if collect_metrics is not None:
            metric_name = f"timing.{name.replace(' ', '_').lower()}"
            collect_metrics[metric_name] = duration

def time_function(logger: Optional[logging.Logger] = None):
    """
    Decorador que aplica `time_block` a uma função inteira.

    Preserva os metadados da função original, como __name__ e __doc__.

    Args:
        logger (Optional[logging.Logger]): O logger a ser usado. Se None, usa o padrão.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not TIMING_ENABLED:
                return func(*args, **kwargs)
            
            func_name = func.__name__
            with time_block(name=f"Function <{func_name}>", logger=logger):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class Timer:
    """
    Um objeto Timer reutilizável para medir intervalos de tempo, útil dentro de loops.
    """
    def __init__(self):
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None

    def start(self):
        """Inicia ou reinicia o timer."""
        self._start_time = time.perf_counter()
        self._stop_time = None

    def stop(self):
        """Para o timer e registra o tempo final."""
        if self._start_time is None:
            raise RuntimeError("Timer não foi iniciado. Chame start() primeiro.")
        self._stop_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Retorna o tempo decorrido em segundos.
        
        Se o timer estiver parado, retorna o intervalo (stop - start).
        Se o timer estiver rodando, retorna o intervalo desde o start até agora.
        """
        if self._start_time is None:
            return 0.0
        if self._stop_time is None:
            # Timer está rodando
            return time.perf_counter() - self._start_time
        # Timer está parado
        return self._stop_time - self._start_time
        
    def __repr__(self) -> str:
        status = "running" if self._stop_time is None else "stopped"
        return f"<Timer status={status} elapsed={self.elapsed:.4f}s>"

def benchmark(
    fn: Callable,
    *args,
    repeat: int = 3,
    number: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Mede a execução de uma função repetidamente e retorna estatísticas.

    Similar ao `timeit`, mas com uma interface mais simples e saída estatística.

    Args:
        fn (Callable): A função a ser benchmarked.
        *args: Argumentos posicionais para a função.
        repeat (int): Quantas vezes o loop de medição deve ser repetido.
        number (int): Quantas vezes a função deve ser chamada dentro de cada medição.
        **kwargs: Argumentos nomeados para a função.

    Returns:
        Dict[str, Any]: Um dicionário com 'mean', 'std' e a lista de 'runs'.
    """
    if not TIMING_ENABLED:
        return {'mean': 0.0, 'std': 0.0, 'runs': []}
        
    timings = []
    for _ in range(repeat):
        start_time = time.perf_counter()
        for _ in range(number):
            fn(*args, **kwargs)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    
    timings_arr = np.array(timings)
    return {
        'mean': np.mean(timings_arr),
        'std': np.std(timings_arr),
        'runs': timings
    }

def _format_bytes(size: int) -> str:
    """Formata bytes em uma string legível (KiB, MiB, etc.)."""
    if size < 1024:
        return f"{size} bytes"
    for unit in ['KiB', 'MiB', 'GiB']:
        size /= 1024
        if size < 1024:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} TiB"

def profile_memory(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Executa uma função e mede o pico de uso de memória com `tracemalloc`.

    Args:
        fn (Callable): A função a ser perfilada.
        *args: Argumentos posicionais.
        **kwargs: Argumentos nomeados.

    Returns:
        Dict[str, Any]: Dicionário com o resultado da função e o pico de memória.
    """
    tracemalloc.start()
    
    try:
        result = fn(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        
    return {
        'result': result,
        'memory_peak': peak,
        'memory_peak_human': _format_bytes(peak)
    }
