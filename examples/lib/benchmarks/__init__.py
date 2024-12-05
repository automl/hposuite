from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lib.benchmarks.ex_functional_bench import Ex_Functional_Bench
from lib.benchmarks.ex_surrogate_bench import ex_surrogate_bench
from lib.benchmarks.ex_synthetic_mf import ex_synthetic_mf_bench

if TYPE_CHECKING:
    from hpoglue.benchmark import BenchmarkDescription


BENCHMARKS: dict[str, BenchmarkDescription] = {}
BENCHMARKS[Ex_Functional_Bench.desc.name] = Ex_Functional_Bench

ex_sg_bench = ex_surrogate_bench(datadir=Path.cwd().absolute().parent / "data")
BENCHMARKS[ex_sg_bench.name] = ex_sg_bench

for desc in ex_synthetic_mf_bench():
    BENCHMARKS[desc.name] = desc

__all__ = [
    "BENCHMARKS",
]
