from __future__ import annotations

import time

from itau_quant.utils import timing


def test_time_block_records_duration(monkeypatch):
    logs = []

    class DummyLogger:
        def info(self, msg):
            logs.append(msg)

    start = time.perf_counter()

    with timing.time_block("test", logger=DummyLogger()):
        pass

    assert logs and "test" in logs[0]
    assert time.perf_counter() - start >= 0


def test_timer_elapsed():
    timer = timing.Timer()
    timer.start()
    timer.stop()
    assert timer.elapsed >= 0


def test_benchmark_runs_function(monkeypatch):
    count = {"calls": 0}

    def fn():
        count["calls"] += 1

    REPEAT = 2
    NUMBER = 3
    EXPECTED_TOTAL_CALLS = REPEAT * NUMBER

    result = timing.benchmark(fn, repeat=REPEAT, number=NUMBER)
    assert count["calls"] == EXPECTED_TOTAL_CALLS
    assert len(result["runs"]) == REPEAT
