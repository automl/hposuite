from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from hpoglue import BenchmarkDescription, Measure, Result, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity

from hposuite.utils import is_package_installed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if TYPE_CHECKING:
    import mfpbench
    from hpoglue import Query


pd1_benchmarks = (
    "cifar100_wideresnet_2048",
    "imagenet_resnet_512",
    "lm1b_transformer_2048",
    "translatewmt_xformer_64",
)


def _get_surrogate_benchmark(
    description: BenchmarkDescription,
    *,
    benchmark_name: str,
    datadir: Path | str | None = None,
    **kwargs: Any,
) -> SurrogateBenchmark:
    import mfpbench

    if datadir is not None:
        datadir = Path(datadir).absolute().resolve()
        if benchmark_name in pd1_benchmarks:
            if "pd1" in os.listdir(datadir):
                datadir = datadir / "pd1"
            else:
                raise ValueError(
                    f"Could not find pd1-{benchmark_name} Benchmark data in {datadir}. "
                    "Download the benchmark data using the command: \n"
                    f'python -m mfpbench download --benchmark "pd1" --data-dir {datadir}'
                )
        kwargs["datadir"] = datadir
    bench = mfpbench.get(benchmark_name, **kwargs)
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


def _download_data_cmd(key: str, datadir: Path | None = None) -> tuple[str, ...]:
    install_cmd = f"python -m mfpbench download --benchmark {key}"
    if datadir is not None:
        install_cmd += f" --data-dir {datadir.resolve()}"
    return tuple(install_cmd.split(" "))


def ex_surrogate_bench(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the PD1 Benchmarks.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each PD1 benchmark.
    """
    if datadir is None:
        datadir = Path(__file__).parent.parent.parent.parent.resolve() / "data"
    import mfpbench
    env = Env(
        name="py310-mfpbench-1.10-pd1",
        python_version="3.10",
        requirements=("mf-prior-bench[pd1]>=1.10.0",),
        post_install=_download_data_cmd("pd1", datadir=datadir),
    )
    for req in env.requirements:
        if not is_package_installed(req):
            logger.warning(f"Please install the required package for pd1: {req}", stacklevel=2)
            return
    yield BenchmarkDescription(
        name="Ex_Surrogate_Bench",
        config_space=mfpbench.pd1.benchmarks.PD1cifar100_wideresnet_2048._create_space(),
        load=partial(
            _get_surrogate_benchmark, benchmark_name="cifar100_wideresnet_2048", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics={"test_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 199, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )



