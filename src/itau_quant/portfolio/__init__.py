"""High-level portfolio orchestration package.

Componentes expostos
--------------------
- `rebalancer` → pipeline principal de rebalance.
- `rounding` → ajustes discretos de pesos/ordens.
- `scheduler` → agenda regular de rebalanceamentos.
- `triggers` → monitoramento de gatilhos extraordinários.

Consumidores externos devem importar via ``from itau_quant.portfolio import ...``
para manter encapsulamento da lógica interna.
"""
