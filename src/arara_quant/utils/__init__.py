"""Shared utilities: validations, math ops, parallelism, seeds, timing, typing.

Componentes expostos
--------------------
- `checks` → validação de entradas (NaNs, PSD, shapes).
- `math_ops` → operações matemáticas comuns (simplex, soft-threshold).
- `data_loading` → leitura flexível de CSV/Parquet/Pickle para otimização/backtests.
- `yaml_loader` → parser YAML mínimo com fallback para ambientes sem PyYAML.
- `parallel` → execução paralela/avaliadores GA.
- `seed` → controle determinístico de seeds.
- `timing` → medição de performance/benchmarking.
- `typing` → aliases e Protocols compartilhados.

Importe via ``from arara_quant.utils import ...`` para manter acoplamento baixo.
"""
