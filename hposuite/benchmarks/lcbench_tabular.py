from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import numpy as np
from hpoglue import BenchmarkDescription, Measure, TabularBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity

from hposuite.utils import is_package_installed

mfp_logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



def _lcbench_tabular(
    description: BenchmarkDescription,
    *,
    task_id: str,
    datadir: Path | str | None = None,
    remove_constants: bool = True,
) -> TabularBenchmark:
    import mfpbench

    if isinstance(datadir, str):
        datadir = Path(datadir).absolute().resolve()

    if datadir is None:
        datadir = Path("data", "lcbench-tabular").absolute().resolve()

    bench = mfpbench.LCBenchTabularBenchmark(
        task_id=task_id,
        datadir=datadir,
        remove_constants=remove_constants,
    )
    return TabularBenchmark(
        desc=description,
        table=bench.table,
        id_key="id",  # Key in the table to uniquely identify configs
        config_keys=bench.config_keys,  # Keys in the table that correspond to configs
    )


def lcbench_tabular(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the LCBench tabular Benchmark.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each task in the LCBench tabular Benchmark.
    """
    task_ids = (
        "adult",
        "airlines",
        "albert",
        "Amazon_employee_access",
        "APSFailure",
        "Australian",
        "bank-marketing",
        "blood-transfusion-service-center",
        "car",
        "christine",
        "cnae-9",
        "connect-4",
        "covertype",
        "credit-g",
        "dionis",
        "fabert",
        "Fashion-MNIST",
        "helena",
        "higgs",
        "jannis",
        "jasmine",
        "jungle_chess_2pcs_raw_endgame_complete",
        "kc1",
        "KDDCup09_appetency",
        "kr-vs-kp",
        "mfeat-factors",
        "MiniBooNE",
        "nomao",
        "numerai28.6",
        "phoneme",
        "segment",
        "shuttle",
        "sylvine",
        "vehicle",
        "volkert",
    )
    if datadir is not None and "lcbench-tabular" in os.listdir(datadir):
        datadir = datadir / "lcbench-tabular"
    import mfpbench
    env = Env(
        name="py310-mfpbench-1.9-lcbench-tabular",
        python_version="3.10",
        requirements=(
            "mf-prior-bench>=1.9.0",
            "pandas>2",
            "pyarrow"
        ),
        post_install=_download_data_cmd("lcbench-tabular", datadir=datadir),
    )
    for req in env.requirements:
        if not is_package_installed(req):
            mfp_logger.warning(
                f"Please install the required package for lcbench_tabular: {req}",
                stacklevel=2
            )
            return
    for task_id in task_ids:
        yield BenchmarkDescription(
            name=f"lcbench_tabular-{task_id}",
            config_space=mfpbench.get(
                "lcbench_tabular",
                task_id=task_id,
                datadir=datadir,
            ).space,
            load=partial(_lcbench_tabular, task_id=task_id, datadir=datadir),
            is_tabular=True,
            fidelities={
                "epoch": RangeFidelity.from_tuple((1, 51, 1), supports_continuation=True),
            },
            costs={
                "time": Measure.cost((0, np.inf), minimize=True),
            },
            metrics={
                "val_accuracy": Measure.metric((0.0, 100.0), minimize=False),
                "val_cross_entropy": Measure.metric((0, np.inf), minimize=True),
                "val_balanced_accuracy": Measure.metric((0, 100), minimize=False),
            },
            test_metrics={
                "test_accuracy": Measure.metric((0.0, 100.0), minimize=False),
                "test_balanced_accuracy": Measure.test_metric((0, 100), minimize=False),
                "test_cross_entropy": Measure.test_metric(bounds=(0, np.inf), minimize=True),
            },
            env=env,
            mem_req_mb=4096,
        )


def _download_data_cmd(key: str, datadir: Path | None = None) -> tuple[str, ...]:
    install_cmd = f"python -m mfpbench download --benchmark {key}"
    if datadir is not None:
        install_cmd += f" --data-dir {datadir.resolve()}"
    return tuple(install_cmd.split(" "))


def lcbench_tabular_benchmarks():
    """Generator function that yields benchmark descriptions from the lcbench_tabular benchmark."""
    yield from lcbench_tabular()