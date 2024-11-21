from __future__ import annotations

from typing import TYPE_CHECKING

from lib.benchmarks.ackley import ACKLEY_BENCH

if TYPE_CHECKING:
    from hpoglue import BenchmarkDescription, FunctionalBenchmark

BENCHMARKS: dict[str, BenchmarkDescription | FunctionalBenchmark] = {}

BENCHMARKS[ACKLEY_BENCH.desc.name] = ACKLEY_BENCH

__all__ = [
    "BENCHMARKS",
]