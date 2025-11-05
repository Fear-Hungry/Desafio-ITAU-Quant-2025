"""Public API for the optimisation genetic-algorithm components."""

from .crossover import crossover_factory
from .evaluation import evaluate_population
from .genetic import run_genetic_algorithm
from .mutation import mutation_pipeline
from .population import (
    Individual,
    decode_individual,
    diversified_population,
    encode_individual,
    ensure_feasible,
    jaccard_distance,
    random_individual,
    warm_start_population,
)
from .selection import selection_pipeline

__all__ = [
    "Individual",
    "decode_individual",
    "diversified_population",
    "encode_individual",
    "ensure_feasible",
    "jaccard_distance",
    "random_individual",
    "warm_start_population",
    "selection_pipeline",
    "mutation_pipeline",
    "crossover_factory",
    "evaluate_population",
    "run_genetic_algorithm",
]
