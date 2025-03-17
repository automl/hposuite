from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from hpoglue import BenchmarkDescription, Measure, Result, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity

from hposuite.utils import is_package_installed

if TYPE_CHECKING:
    import mfpbench
    from hpoglue import Query


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _get_surrogate_benchmark(
    description: BenchmarkDescription,
    *,
    benchmark_name: str,
) -> SurrogateBenchmark:
    import mfpbench
    bench = mfpbench.get(benchmark_name)
    query_function = partial(_mfpbench_surrogate_query_function, benchmark=bench)
    return SurrogateBenchmark(
        desc=description,
        benchmark=bench,
        config_space=bench.space,
        query=query_function,
    )


def _mfpbench_surrogate_query_function(query: Query, benchmark: mfpbench.Benchmark) -> Result:
    if query.fidelity is not None:
        assert isinstance(query.fidelity, tuple)
        _, fid_value = query.fidelity
    else:
        fid_value = None
    return Result(
        query=query,
        values=benchmark.query(
            query.config.values,
            at=fid_value,
        ).as_dict(),
        fidelity=query.fidelity,
    )


def mfh() -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the MF-Hartmann Benchmarks.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each combination of correlation and dimensions in the MFH Benchmarks.
    """
    import mfpbench
    env = Env(
        name="py310-mfpbench-1.10-mfh",
        python_version="3.10",
        requirements=("mf-prior-bench>=1.10.0",),
        post_install=(),
    )
    for req in env.requirements:
        if not is_package_installed(req):
            logger.warning(f"Please install the required package for pd1: {req}", stacklevel=2)
            return
    for correlation in ("bad", "good", "moderate", "terrible"):
        for dims in (3, 6):
            name = f"mfh{dims}_{correlation}"
            _min = -3.32237 if dims == 3 else -3.86278  # noqa: PLR2004
            yield BenchmarkDescription(
                name=name,
                config_space=mfpbench.get(name).space,
                load=partial(_get_surrogate_benchmark, benchmark_name=name),
                costs={
                    "fid_cost": Measure.cost((0.05, 1), minimize=True),
                },
                fidelities={
                    "z": RangeFidelity.from_tuple((1, 100, 1), supports_continuation=True),
                },
                metrics={
                    "value": Measure.metric((_min, np.inf), minimize=True),
                },
                has_conditionals=False,
                is_tabular=False,
                env=env,
                mem_req_mb = 1024,
            )

def ex_synthetic_mf_bench():
    yield from mfh()
