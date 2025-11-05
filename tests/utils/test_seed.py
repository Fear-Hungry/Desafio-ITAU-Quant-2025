import random

import numpy as np
from arara_quant.utils import seed as seed_utils


def test_set_global_seeds_invokes_solver_seed(monkeypatch):
    calls = []

    monkeypatch.setattr(
        seed_utils, "set_solver_seed", lambda value: calls.append(value)
    )

    random.seed(0)
    np.random.seed(0)

    seed_utils.set_global_seeds(123, cvxpy=True)

    normalized_seed = 123 % seed_utils.MAX_SEED_VALUE
    assert calls == [normalized_seed]

    expected_random = random.Random(normalized_seed).random()
    expected_numpy = np.random.RandomState(normalized_seed).rand()

    assert random.random() == expected_random
    assert np.random.rand() == expected_numpy
