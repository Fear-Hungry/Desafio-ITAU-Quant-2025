"""Blueprint for heuristic allocation baselines (HRP, clustering, EW).

Objetivo
--------
Disponibilizar heurísticas rápidas (sem programação convexa) usadas como
baseline/comparativo ou fallback quando solvers falham.

Componentes sugeridos
---------------------
- `equal_weight(universe, constraints=None)`
    Distribuição uniforme ajustada a constraints simples (ex.: bounds).
- `hierarchical_risk_parity(cov, method="single", min_cluster_size=2)`
    Implementação HRP (López de Prado) usando dendrogramas e alocação top-down.
- `inverse_variance_portfolio(cov)`
    Peso ∝ 1/σ² como baseline simples.
- `cluster_then_allocate(cov, n_clusters, method="kmeans")`
    Clustering + equal-weight intra-cluster + risk parity inter-cluster.
- `heuristic_allocation(data, config)`
    Dispatcher que escolhe heurística baseada na config (EW/IVP/HRP/cluster).

Considerações
-------------
- Utilizar utilitários de `data.processing` para garantir índices alinhados.
- Normalizar saídas (soma = 1) e respeitar bounds fornecidos.
- Prever fallback quando covariância é singular (adicionar ridge).
- Permitir retorno de diagnósticos (clusters, distância entre ativos).

Testes recomendados
-------------------
- `tests/optimization/heuristics/test_hrp.py` cobrindo:
    * equal-weight respeitando cardinalidade total,
    * HRP replicando exemplos do paper (com dataset sintético),
    * comparação IVP vs. HRP em matriz diagonal (devem coincidir),
    * clusterização limitando tamanho mínimo.
"""
