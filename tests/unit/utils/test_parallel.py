from arara_quant.utils.parallel import parallel_map


def _extract_value(payload: dict) -> int:
    return payload["value"]


def test_parallel_map_handles_unhashable_items():
    items = [{"value": 1}, {"value": 2}, {"value": 1}]

    results = parallel_map(_extract_value, items, backend="thread", max_workers=2)

    assert results == [1, 2, 1]
