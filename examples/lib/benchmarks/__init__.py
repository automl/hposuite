from __future__ import annotations

from typing import TYPE_CHECKING

from examples.lib.benchmarks.ex_functional_bench import Ex_Functional_Bench

if TYPE_CHECKING:
    from hpoglue.benchmark import BenchmarkDescription

BENCHMARKS: dict[str, BenchmarkDescription] = {}
BENCHMARKS[Ex_Functional_Bench.desc.name] = Ex_Functional_Bench

__all__ = [
    "BENCHMARKS",
]
