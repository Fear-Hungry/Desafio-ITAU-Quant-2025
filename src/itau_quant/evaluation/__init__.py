"""Componentes para avaliar estratégias (métricas + visualizações + relatórios).

Exposição sugerida
------------------
- `stats` → funções para desempenho, risco e intervalos de confiança.
- `plots` → geração de tearsheets e gráficos diagnósticos.
- `report` → orquestração final (HTML/PDF) com metadados e visualizações.

Importadores externos devem consumir via ``from itau_quant.evaluation import ...``
para manter encapsulamento das subcamadas.
"""
