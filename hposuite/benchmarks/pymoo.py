"""Interfacing benchmarks from the Pymoo library."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from hpoglue.benchmark import BenchmarkDescription, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.measure import Measure
from hpoglue.result import Result

if TYPE_CHECKING:
    import pymoo
    from hpoglue.query import Query

def get_pymoo_space(
    pymoo_prob: pymoo.core.problem.Problem
) -> ConfigurationSpace:
    """Get ConfigSpace from pymoo problem."""
    n_var = pymoo_prob.n_var
    xl, xu = pymoo_prob.xl, pymoo_prob.xu
    hps = [Float(name=f"x{i}", bounds=[xl[i], xu[i]]) for i in range(n_var)]
    space = ConfigurationSpace()
    space.add(hps)
    return space


def _pymoo_query_function(
    query: Query,
    benchmark: pymoo.core.problem.Problem,
) -> Result:
    assert query.fidelity is None
    config_vals = np.array(list(query.config.values.values()))
    values = benchmark.evaluate(config_vals)[0]    #TODO: Handle this for multiobjective
    values = {"value": values}
    return Result(
        query=query,
        values=values,
        fidelity=None,
    )


def _get_pymoo_problems(
    description: BenchmarkDescription,
    *,
    function_name: str,
    **kwargs: Any,
)-> SurrogateBenchmark:

    import pymoo
    import pymoo.problems

    bench = pymoo.problems.get_problem(function_name, **kwargs)
    query_function = partial(_pymoo_query_function, benchmark=bench)
    return SurrogateBenchmark(
        desc=description,
        benchmark=bench,
        config_space=get_pymoo_space(bench),
        query=query_function,
    )


_pymoo_so = [
    "ackley",
    "griewank",
    "himmelblau",
    "rastrigin",
    "rosenbrock",
    "schwefel",
    "sphere",
    "zakharov",
]


def pymoo_problems() -> Iterator[BenchmarkDescription]:
    env = Env(
        name="py310-pymoo-0.6.1.3",
        requirements=("pymoo==0.6.1.3"),
        post_install=None
    )
    for prob_name in _pymoo_so:
        yield BenchmarkDescription(
            name=f"pymoo-{prob_name}",
            env=env,
            load = partial(_get_pymoo_problems, function_name=prob_name),
            has_conditionals=False,
            metrics={
                    "value": Measure.metric((-np.inf, np.inf), minimize=True),
                },
            is_tabular=False,
            mem_req_mb=1024,
        )

def pymoo_benchmarks(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    yield from pymoo_problems()
