from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from hpoglue import BenchmarkDescription, Config, Measure, TabularBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity

from hposuite.constants import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_mfh_space(
    function_name: str,
    datadir: Path,
) -> list[Config]:
    """Returns a list of all possible configurations for the MF Hartmann's Tabular Benchmark."""
    table = _get_mfh_table(function_name, datadir)
    config_keys = [k for k in table.columns if "X" in k]
    return TabularBenchmark.get_tabular_config_space(table, config_keys)


def _get_mfh_table(
    function_name: str,
    datadir: Path,
) -> pd.DataFrame:
    return pd.read_parquet(datadir / f"{function_name}.parquet")


def _get_mfh_tabular_benchmark(
    description: BenchmarkDescription,
    *,
    function_name: str,
    datadir: Path,
) -> TabularBenchmark:
    """Creates a TabularBenchmark object for the MF-Hartmann Tabular Benchmark Suite."""
    try:
        bench = _get_mfh_table(function_name, datadir)
    except FileNotFoundError as e:
        logger.error(
            f"Could not find mfh Tabular Benchmark data for {function_name}. Skipping. "
            f"Run `python -m hposuite.benchmarks.create_tabular --benchmark {function_name} "
            f"-suite mfh_tabular --task {function_name}` to create the benchmark data."
        )
        raise e
    config_keys = [k for k in bench.columns if "x" in k]
    return TabularBenchmark(
        desc=description,
        table=bench,
        id_key="config_id",
        config_keys=config_keys,
    )


def mfh_tabular_desc(datadir: Path) -> BenchmarkDescription:
    """Generates benchmark descriptions for the Synthetic MF-Hartmann Tabular Benchmark suite.

    This function iterates over all combinations of function IDs, dimensions, and instances
    defined in `mfh_functions_dict`, `mfh_dims`, and `mfh_instances` respectively. For each
    combination, it creates a `BenchmarkDescription` object with the corresponding configuration
    space, benchmark function, and environment settings.
    """
    for correlation in ("bad", "good", "moderate", "terrible"):
        for dims in (3, 6):
            name = f"mfh{dims}_{correlation}"
            _min = -3.32237 if dims == 3 else -3.86278  # noqa: PLR2004
            try:
                space = _get_mfh_space(name, datadir)
            except FileNotFoundError:
                continue
            yield BenchmarkDescription(
                name=f"mfh_tabular-{name}",
                config_space=space,
                load=partial(
                    _get_mfh_tabular_benchmark,
                    function_name=name,
                    datadir=datadir
                ),
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
                env=Env.empty(),
                mem_req_mb = 1024,
            )


def mfh_tabular_benchmarks(datadir: str | Path | None = None) :
    """A generator that yields all mfh Tabular benchmarks."""
    if isinstance(datadir, str):
        datadir = Path(datadir).resolve() / "mfh_tabular"
    elif datadir is None:
        datadir = DATA_DIR / "mfh_tabular"

    yield from mfh_tabular_desc(datadir=datadir)