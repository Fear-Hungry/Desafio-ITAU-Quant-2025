from __future__ import annotations

import time

import pytest

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

    result = timing.benchmark(fn, repeat=2, number=3)
    assert count["calls"] == 6
    assert len(result["runs"]) == 2
