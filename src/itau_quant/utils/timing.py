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
