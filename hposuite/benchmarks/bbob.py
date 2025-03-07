from __future__ import annotations

from collections.abc import Callable
from functools import partial
from itertools import product
from typing import TYPE_CHECKING

import ioh
import numpy as np
from ConfigSpace import ConfigurationSpace
from hpoglue import BenchmarkDescription, Measure, Query, Result, SurrogateBenchmark
from hpoglue.env import Env

if TYPE_CHECKING:
    from ioh.iohcpp.problem import IntegerSingleObjective, RealSingleObjective  # type: ignore

bbob_functions_dict = {
    #Separable

    "f1": "Sphere",
    "f2": "Ellipsoid separable",
    "f3": "Rastrigin separable",
    "f4": "Skew Rastrigin-Bueche",
    "f5": "Linear slope",

    # Low or moderate conditioning
    "f6": "Attractive sector",
    "f7": "Step-ellipsoid",
    "f8": "Rosenbrock original",
    "f9": "Rosenbrock rotated",

    # High conditioning and unimodal
    "f10": "Ellipsoid",
    "f11": "Discus",
    "f12": "Bent cigar",
    "f13": "Sharp ridge",
    "f14": "Sum of different powers",

    # Multi-modal with adequate global structure
    "f15": "Rastrigin",
    "f16": "Weierstrass",
    "f17": "Schaffer F7, condition 10",
    "f18": "Schaffer F7, condition 1000",
    "f19": "Griewank-Rosenbrock F8F2",

    # Multi-modal with weak global structure
    "f20": "Schwefel x*sin(x)",
    "f21": "Gallagher 101 peaks",
    "f22": "Gallagher 21 peaks",
    "f23": "Katsuura",
    "f24": "Lunacek bi-Rastrigin"
}


bbob_dims = (2, 3, 5, 10, 20, 40)
bbob_instances = (0, 1, 2)


def _get_bbob_space(
    bbob_function: RealSingleObjective | IntegerSingleObjective
) -> ConfigurationSpace:
    lower_bounds = bbob_function.bounds.lb
    upper_bounds = bbob_function.bounds.ub
    return ConfigurationSpace({
            f"x{i}": (lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))
        }
    )


def _get_bbob_fn(
    func: tuple[str, int, int]
) -> RealSingleObjective | IntegerSingleObjective:
    f_id, dim, ins = func
    f_id = int(f_id.split("f")[-1])
    return ioh.get_problem(
        fid=f_id,
        dimension=dim,
        instance=ins,
        problem_class=ioh.ProblemClass.BBOB
    )


def _bbob_query_fn(
    query: Query,
    bbob_function: RealSingleObjective | IntegerSingleObjective
) -> Result:
    config = query.config.values
    assert query.fidelity is None, "Fidelity is not supported for BBOB Benchmark Suite."
    out = bbob_function(list(config.values()))

    return Result(
        query=query,
        fidelity=None,
        values={
            "value": out
        }
    )


def _bbob_surrogate_bench(
    desc: BenchmarkDescription,
    bbob_function: Callable
) -> SurrogateBenchmark:
    query_function = partial(_bbob_query_fn, bbob_function=bbob_function)
    return SurrogateBenchmark(
        desc=desc,
        config_space=desc.config_space,
        benchmark=bbob_function,
        query=query_function,
    )


def bbob_desc() -> BenchmarkDescription:
    """Generates benchmark descriptions for the BBOB (Black-Box Optimization Benchmarking) suite.

    This function iterates over all combinations of function IDs, dimensions, and instances
    defined in `bbob_functions_dict`, `bbob_dims`, and `bbob_instances` respectively. For each
    combination, it creates a `BenchmarkDescription` object with the corresponding configuration
    space, benchmark function, and environment settings.
    """
    env = Env(
        name="py310-bbob-ioh-0.3.14",
        python_version="3.10",
        requirements=("ioh>=0.3.14"),
        post_install=None
    )
    for f_id, dim, ins in product(bbob_functions_dict.keys(), bbob_dims, bbob_instances):
        bbob_function = _get_bbob_fn((f_id, int(dim), int(ins)))
        yield BenchmarkDescription(
            name=f"bbob-{f_id}-{dim}-{ins}",
            config_space=_get_bbob_space(bbob_function=bbob_function),
            load=partial(_bbob_surrogate_bench, bbob_function=bbob_function),
            metrics={
                "value": Measure.metric((-np.inf, np.inf), minimize=True),
            },
            test_metrics=None,
            costs=None,
            fidelities=None,
            has_conditionals=False,
            is_tabular=False,
            env=env,
            mem_req_mb=1024,
        )


def bbob_benchmarks():
    """Generator function that yields benchmark descriptions from the bbob_desc function."""
    yield from bbob_desc()