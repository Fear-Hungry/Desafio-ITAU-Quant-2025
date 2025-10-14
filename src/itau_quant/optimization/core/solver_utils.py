"""Blueprint for CVXPy solver utility helpers.

Objetivo
--------
Fornecer utilitários padronizados para configurar, executar e monitorar solvers
quando resolvemos problemas convexos do portfólio.

Componentes sugeridos
---------------------
- `select_solver(preferred=None, fallback=True)`
    Seleciona solver disponível (ECOS, OSQP, SCS, MOSEK) seguindo prioridade.
- `set_solver_seed(solver, seed)`
    Ajusta seeds quando suportado (ex.: SCS, GUROBI) para reprodutibilidade.
- `solve_problem(problem, solver, solver_kwargs)`
    Wrapper que executa `problem.solve`, captura logs, tempo, convergência.
- `process_status(problem)`
    Converte `problem.status` em enum customizado (`OPTIMAL`, `INFEASIBLE`, etc.).
- `handle_warnings(problem, logger)`
    Emite avisos quando solver converge com warnings (ex.: status ``OPTIMAL_INACCURATE``).
- `warm_start(problem, previous_solution)`
    Lida com warm-start caso disponível.

Considerações
-------------
- Centralizar configuração de tolerâncias (`eps`, `max_iters`, `acceleration`).
- Garantir compatibilidade com logs estruturados (`utils.logging_config`).
- Fornecer caminhos de fallback (tentar outro solver se o primeiro falhar).
- Opcional: medir métricas de performance (tempo, iterações) para telemetry.

Testes recomendados
-------------------
- `tests/optimization/test_solver_utils.py` com mocks de problemas:
    * seleção de solver quando preferido indisponível,
    * propagação adequada de ``solver_kwargs`` para `problem.solve`,
    * conversão de status (ex.: ``cp.settings.OPTIMAL`` → ``Status.OPTIMAL``),
    * fallback automático quando solver retorna erro.
"""
