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
