from __future__ import annotations

from hpoglue import BenchmarkDescription, Optimizer

from hposuite.benchmarks import BENCHMARKS
from hposuite.optimizers import OPTIMIZERS


def register(*things: type[Optimizer] | BenchmarkDescription) -> None:
    """Register a benchmark or optimizer."""
    for thing in things:
        if isinstance(thing, BenchmarkDescription):
            if thing.name in BENCHMARKS:
                raise ValueError(f"Duplicate benchmark name: {thing.name}")
            BENCHMARKS[thing.name] = thing
        elif issubclass(thing, Optimizer):
            if thing.name in OPTIMIZERS:
                raise ValueError(f"Duplicate optimizer name: {thing.name}")
            OPTIMIZERS[thing.name] = thing
        else:
            raise TypeError(f"Cannot register {thing}!")
