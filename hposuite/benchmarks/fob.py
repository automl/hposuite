from __future__ import annotations

import logging
from collections.abc import Iterator

from hpoglue import BenchmarkDescription, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity

logger = logging.getLogger(__name__)

try:
    import fob
except ImportError as e:
    logger.warning("Please install the required package for Fast Optimizer Benchmark (FOB): fob")
    fob = None

def fob_benchmarks(datadir=None) -> Iterator[BenchmarkDescription]:
    """Yield BenchmarkDescription objects for each FOB benchmark."""
    if fob is None:
        return

    # FOB provides a list of benchmark names
    for name in fob.list_benchmarks():
        desc = BenchmarkDescription(
            name=name,
            fob_benchmark=True,  # Custom flag if you want
            create_benchmark=lambda name=name: fob.get_benchmark(name),
            fidelities=None,  # Set if FOB supports fidelities
            objectives=None,  # Set if FOB supports multi-objective
            datadir=datadir,
        )
        yield desc