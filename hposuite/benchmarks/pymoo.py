"""Interfacing benchmarks from the Pymoo library."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from hpoglue import BenchmarkDescription, Measure, Result, SurrogateBenchmark
from hpoglue.env import Env

if TYPE_CHECKING:
    import pymoo
    from hpoglue import Query

def get_pymoo_space(
    pymoo_prob: pymoo.core.problem.Problem
) -> ConfigurationSpace:
    """Get ConfigSpace from pymoo problem."""
    n_var = pymoo_prob.n_var
    xl, xu = pymoo_prob.xl, pymoo_prob.xu
    return ConfigurationSpace(
        {f"x{i}": (xl[i], xu[i]) for i in range(n_var)}
    )


def _pymoo_query_function(
    query: Query,
    benchmark: pymoo.core.problem.Problem,
) -> Result:
    assert query.fidelity is None
    config_vals = np.array(list(query.config.values.values()))
    values = benchmark.evaluate(config_vals).tolist()
    if len(values) > 1:
        values = {f"value{i}": val for i, val in enumerate(values, start=1)}
    else:
        values = {"value": values[0]}
    return Result(
        query=query,
        values=values,
        fidelity=None,
    )


def _get_pymoo_problems(
    function_name: str,
    **kwargs: Any,
)-> pymoo.core.problem.Problem:

    import pymoo.problems

    match function_name:
        case "omnitest":
            from pymoo.problems.multi.omnitest import OmniTest
            return OmniTest()
        case "sympart":
            from pymoo.problems.multi.sympart import SYMPART
            return SYMPART()
        case "sympart_rotated":
            from pymoo.problems.multi.sympart import SYMPARTRotated
            return SYMPARTRotated()
        case _:
            return pymoo.problems.get_problem(function_name, **kwargs)


def _pymoo_surrogate_bench(
    desc: BenchmarkDescription,
    pymoo_prob: pymoo.core.problem.Problem,
) -> SurrogateBenchmark:
    query_function = partial(_pymoo_query_function, benchmark=pymoo_prob)
    return SurrogateBenchmark(
        desc=desc,
        config_space=get_pymoo_space(pymoo_prob),
        benchmark=pymoo_prob,
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

def pymoo_so_problems() -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for single-objective problems using the pymoo library."""
    env = Env(
        name="py310-pymoo-0.6.1.3",
        python_version="3.10",
        requirements=("pymoo==0.6.1.3"),
        post_install=None
    )
    for prob_name in _pymoo_so:
        pymoo_prob = _get_pymoo_problems(prob_name)
        yield BenchmarkDescription(
            name=f"pymoo-{prob_name}",
            config_space=get_pymoo_space(pymoo_prob),
            load = partial(_pymoo_surrogate_bench, pymoo_prob=pymoo_prob),
            has_conditionals=False,
            metrics={
                    "value": Measure.metric((-np.inf, np.inf), minimize=True),
                },
            is_tabular=False,
            env=env,
            mem_req_mb=1024,
        )

_pymoo_mo = [
    "kursawe",
    "zdt1",
    "zdt2",
    "zdt3",
    "zdt4",
    "zdt5",
    "zdt6",
    "omnitest",
    "sympart",
    "sympart_rotated",
]

def pymoo_mo_problems() -> Iterator[BenchmarkDescription]:
    """Generator function that yields benchmark descriptions for multi-objective problems
    using the pymoo library.
    """
    env = Env(
        name="py310-pymoo-0.6.1.3",
        requirements=("pymoo==0.6.1.3"),
        post_install=None
    )
    for prob_name in _pymoo_mo:
        pymoo_prob = _get_pymoo_problems(prob_name)
        yield BenchmarkDescription(
            name=f"pymoo-{prob_name}",
            config_space=get_pymoo_space(pymoo_prob),
            load = partial(_pymoo_surrogate_bench,  pymoo_prob=pymoo_prob),
            has_conditionals=False,
            metrics={
                    "value1": Measure.metric((-np.inf, np.inf), minimize=True),
                    "value2": Measure.metric((-np.inf, np.inf), minimize=True),
                },
            is_tabular=False,
            env=env,
            mem_req_mb=1024,
        )


_pymoo_many_obj = [
    "dtlz1",
    "dtlz2",
    "dtlz3",
    "dtlz4",
    "dtlz5",
    "dtlz6",
    "dtlz7",
]


def pymoo_many_obj_problems() -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for many-objective problems using the PyMOO library."""
    env = Env(
        name="py310-pymoo-0.6.1.3",
        requirements=("pymoo==0.6.1.3"),
        post_install=None
    )
    for prob_name in _pymoo_many_obj:
        pymoo_prob = _get_pymoo_problems(prob_name)
        yield BenchmarkDescription(
            name=f"pymoo-{prob_name}",
            config_space=get_pymoo_space(pymoo_prob),

            load = partial(_pymoo_surrogate_bench,  pymoo_prob=pymoo_prob),
            has_conditionals=False,
            # PyMOO DTLZ problems have 3 objectives by default, can be changed by setting n_obj
            metrics={
                    "value1": Measure.metric((-np.inf, np.inf), minimize=True),
                    "value2": Measure.metric((-np.inf, np.inf), minimize=True),
                    "value3": Measure.metric((-np.inf, np.inf), minimize=True),
                },
            is_tabular=False,
            env=env,
            mem_req_mb=1024,
        )


def pymoo_benchmarks() -> Iterator[BenchmarkDescription]:
    """Generate benchmark descriptions for various pymoo problems.

    This function yields benchmark descriptions for single-objective,
    multi-objective, and many-objective problems from the pymoo library.
    """
    yield from pymoo_so_problems()
    yield from pymoo_mo_problems()
    yield from pymoo_many_obj_problems()
