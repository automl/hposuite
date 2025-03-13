from __future__ import annotations

import warnings
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from hpoglue import BenchmarkDescription, Config, Measure, TabularBenchmark

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
    function_name: str,
    datadir: str | Path | None = None,
) -> list[Config]:
    """Returns a list of all possible configurations for the BBOB function's Tabular Benchmark."""
    table = _get_bbob_table(function_name, datadir)
    config_keys = [k for k in table.columns if "x" in k]
    return TabularBenchmark.get_tabular_config_space(table, config_keys)


def _get_bbob_table(
    function_name: str,
    datadir: str | Path | None = None,
) -> pd.DataFrame:
    return pd.read_parquet(datadir / f"{function_name}.parquet")


def _get_bbob_tabular_benchmark(
    description: BenchmarkDescription,
    *,
    function_name: str,
    datadir: str | Path | None = None,
) -> TabularBenchmark:
    """Creates a TabularBenchmark object for the BBOB Tabular Benchmark Suite."""
    bench = _get_bbob_table(function_name, datadir)
    config_keys = [k for k in bench.columns if "x" in k]
    return TabularBenchmark(
        desc=description,
        table=bench,
        id_key="config_id",
        config_keys=config_keys,
    )


def bbob_tabular_desc(datadir: str | Path | None = None) -> BenchmarkDescription:
    """Generates benchmark descriptions for the Synthetic BBOB Tabular Benchmark suite.

    This function iterates over all combinations of function IDs, dimensions, and instances
    defined in `bbob_functions_dict`, `bbob_dims`, and `bbob_instances` respectively. For each
    combination, it creates a `BenchmarkDescription` object with the corresponding configuration
    space, benchmark function, and environment settings.
    """
    for f_id, dim, ins in product(bbob_functions_dict.keys(), bbob_dims, bbob_instances):
        function_name = f"{f_id}-{dim}-{ins}"
        try:
            space = _get_bbob_space(function_name, datadir)
        except FileNotFoundError:
            warnings.warn(  # noqa: B028
                f"Could not find BBOB Tabular Benchmark data for {function_name}. Skipping. "
                f"Run `python -m hposuite.benchmarks.create_tabular "
                f"--benchmark bbob-{function_name} -suite bbob_tabular --task {function_name}`"
                "to create the benchmark data."
            )
            continue
        yield BenchmarkDescription(
            name=f"bbob_tabular-{function_name}",
            config_space=space,
            load=partial(
                _get_bbob_tabular_benchmark,
                function_name=function_name,
                datadir=datadir
            ),
            metrics={
                "value": Measure.metric((-np.inf, np.inf), minimize=True),
            },
            test_metrics=None,
            costs=None,
            fidelities=None,
            has_conditionals=False,
            is_tabular=True,
            mem_req_mb=1024,
        )


def bbob_tabular_benchmarks(datadir: str | Path | None = None) :
    """A generator that yields all BBOB Tabular benchmarks."""
    if isinstance(datadir, str):
        datadir = Path(datadir).resolve()

    if datadir is None:
        datadir = Path("data", "bbob_tabular").resolve()

    yield from bbob_tabular_desc(datadir=datadir)