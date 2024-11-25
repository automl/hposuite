from __future__ import annotations

from typing import TYPE_CHECKING

from hposuite.benchmarks.mfp_bench import mfpbench_benchmarks
from hposuite.benchmarks.pymoo import pymoo_benchmarks
from hposuite.benchmarks.synthetic import ACKLEY_BENCH

if TYPE_CHECKING:
    from hpoglue.benchmark import BenchmarkDescription

BENCHMARKS: dict[str, BenchmarkDescription] = {}
MF_BENCHMARKS: dict[str, BenchmarkDescription] = {}
NON_MF_BENCHMARKS: dict[str, BenchmarkDescription] = {}
for desc in mfpbench_benchmarks():
    BENCHMARKS[desc.name] = desc
    if desc.fidelities is not None:
        MF_BENCHMARKS[desc.name] = desc
    else:
        NON_MF_BENCHMARKS[desc.name] = desc
for desc in pymoo_benchmarks():
    BENCHMARKS[desc.name] = desc
    if desc.fidelities is not None:
        MF_BENCHMARKS[desc.name] = desc
    else:
        NON_MF_BENCHMARKS[desc.name] = desc

BENCHMARKS[ACKLEY_BENCH.desc.name] = ACKLEY_BENCH

__all__ = [
    "BENCHMARKS",
    "MF_BENCHMARKS",
    "NON_MF_BENCHMARKS",
]
