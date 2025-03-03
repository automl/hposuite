from __future__ import annotations

from typing import TYPE_CHECKING

from hposuite.benchmarks.bbob import bbob_benchmarks
from hposuite.benchmarks.mfp_bench import mfpbench_benchmarks
from hposuite.benchmarks.pymoo import pymoo_benchmarks
from hposuite.benchmarks.synthetic import ACKLEY_BENCH, BRANIN_BENCH

if TYPE_CHECKING:
    from hpoglue import BenchmarkDescription

BENCHMARKS: dict[str, BenchmarkDescription] = {}
MF_BENCHMARKS: dict[str, BenchmarkDescription] = {}
NON_MF_BENCHMARKS: dict[str, BenchmarkDescription] = {}

BENCHMARKS[ACKLEY_BENCH.desc.name] = ACKLEY_BENCH
BENCHMARKS[BRANIN_BENCH.desc.name] = BRANIN_BENCH

for desc in mfpbench_benchmarks():
    BENCHMARKS[desc.name] = desc
    if desc.fidelities is not None:
        MF_BENCHMARKS[desc.name] = desc
    else:
        NON_MF_BENCHMARKS[desc.name] = desc

for desc in pymoo_benchmarks():
    BENCHMARKS[desc.name] = desc
    NON_MF_BENCHMARKS[desc.name] = desc

for desc in bbob_benchmarks():
    BENCHMARKS[desc.name] = desc
    NON_MF_BENCHMARKS[desc.name] = desc


__all__ = [
    "BENCHMARKS",
    "MF_BENCHMARKS",
    "NON_MF_BENCHMARKS",
]
