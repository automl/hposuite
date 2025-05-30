from __future__ import annotations

import importlib
import logging
import types
from collections.abc import Generator, Iterator

from hpoglue import BenchmarkDescription, FunctionalBenchmark

from hposuite.exceptions import OptBenchNotInstalledError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

modules = [
    ("lib.benchmarks.ex_functional_bench", "Ex_Functional_Bench"),
    ("lib.benchmarks.ex_surrogate_bench", "ex_surrogate_bench"),
    ("lib.benchmarks.ex_synthetic_mf", "ex_synthetic_mf_bench"),
]

imported_benches = []

for module_name, *attrs in modules:
    try:
        module = importlib.import_module(module_name)
        for attr in attrs:
            bench = getattr(module, attr)
            if isinstance(bench, types.FunctionType):
                bench = bench()
            match bench:
                case FunctionalBenchmark() | BenchmarkDescription():
                    imported_benches.append(bench)
                case Iterator() | Generator():
                    for b in bench:
                        match b:
                            case FunctionalBenchmark() | BenchmarkDescription():
                                imported_benches.append(b)
                            case _:
                                raise ValueError(f"Unexpected benchmark type: {type(b)}")
                case _:
                    raise ValueError(f"Unexpected benchmark type: {type(bench)}")
    except ImportError as e:
        logger.warning(OptBenchNotInstalledError(module_name, e.msg))


BENCHMARKS: dict[str, BenchmarkDescription] = {}
MF_BENCHMARKS: dict[str, BenchmarkDescription] = {}
NON_MF_BENCHMARKS: dict[str, BenchmarkDescription] = {}


for bench in imported_benches:
    BENCHMARKS[bench.name] = bench
    _bench = bench
    if isinstance(_bench, FunctionalBenchmark):
        _bench = _bench.desc

    if _bench.fidelities is not None:
        MF_BENCHMARKS[_bench.name] = _bench
    else:
        NON_MF_BENCHMARKS[_bench.name] = _bench

__all__ = [
    "BENCHMARKS",
    "MF_BENCHMARKS",
    "NON_MF_BENCHMARKS",
]

