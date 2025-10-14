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
