"""Script to interface with the MF Prior Bench library."""
# TODO(eddiebergman): Right now it's not clear how to set defaults for multi-objective.
# Do we want to prioritize obj and cost (i.e. accuracy and time) or would we rather two
# objectives (i.e. accuracy and cross entropy)?
# Second, for a benchmark to be useful, it should provide a reference point from which to compute
# hypervolume. For bounded costs this is fine but we can't do so for something like time.
# For tabular datasets, we could manually look for the worst time value
# TODO(eddiebergman): Have not included any of the conditional benchmarks for the moment
# as it seems to crash
# > "nb301": NB301Benchmark,
# > "rbv2_super": RBV2SuperBenchmark,
# > "rbv2_aknn": RBV2aknnBenchmark,
# > "rbv2_glmnet": RBV2glmnetBenchmark,
# > "rbv2_ranger": RBV2rangerBenchmark,
# > "rbv2_rpart": RBV2rpartBenchmark,
# > "rbv2_svm": RBV2svmBenchmark,
# > "rbv2_xgboost": RBV2xgboostBenchmark,
# > "iaml_glmnet": IAMLglmnetBenchmark,
# > "iaml_ranger": IAMLrangerBenchmark,
# > "iaml_rpart": IAMLrpartBenchmark,
# > "iaml_super": IAMLSuperBenchmark,
# > "iaml_xgboost": IAMLxgboostBenchmark,

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from hpoglue import BenchmarkDescription, Measure, Result, SurrogateBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity

from hposuite.utils import is_package_installed

mfp_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import mfpbench
    from hpoglue import Query


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
            mfp_logger.warning(f"Please install the required package for mfh: {req}", stacklevel=2)
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


def pd1(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the PD1 Benchmarks.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each PD1 benchmark.
    """
    import mfpbench
    env = Env(
        name="py310-mfpbench-1.10-pd1",
        python_version="3.10",
        requirements=("mf-prior-bench[pd1]>=1.10.0",),
        post_install=_download_data_cmd("pd1", datadir=datadir),
    )
    for req in env.requirements:
        if not is_package_installed(req):
            mfp_logger.warning(f"Please install the required package for pd1: {req}", stacklevel=2)
            return
    yield BenchmarkDescription(
        name="pd1-cifar100-wide_resnet-2048",
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
    yield BenchmarkDescription(
        name="pd1-imagenet-resnet-512",
        config_space=mfpbench.pd1.benchmarks.PD1imagenet_resnet_512._create_space(),
        load=partial(
            _get_surrogate_benchmark, benchmark_name="imagenet_resnet_512", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics={"test_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 99, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
    yield BenchmarkDescription(
        name="pd1-lm1b-transformer-2048",
        config_space=mfpbench.pd1.benchmarks.PD1lm1b_transformer_2048._create_space(),
        load=partial(
            _get_surrogate_benchmark, benchmark_name="lm1b_transformer_2048", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics={"test_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 74, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=24576,
    )
    yield BenchmarkDescription(
        name="pd1-translate_wmt-xformer_translate-64",
        config_space=mfpbench.pd1.benchmarks.PD1translatewmt_xformer_64._create_space(),
        load=partial(
            _get_surrogate_benchmark, benchmark_name="translatewmt_xformer_64", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics={"test_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 19, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=24576,
    )


def _download_data_cmd(key: str, datadir: Path | None = None) -> tuple[str, ...]:
    install_cmd = f"python -m mfpbench download --benchmark {key}"
    if datadir is not None:
        install_cmd += f" --data-dir {datadir.resolve()}"
    return tuple(install_cmd.split(" "))


def mfpbench_benchmarks(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for various MF-Prior-Bench.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each benchmark.
    """
    if datadir is None:
        datadir=Path(__file__).parent.parent.parent.absolute() / "data"
    yield from mfh()
    yield from pd1(datadir)
