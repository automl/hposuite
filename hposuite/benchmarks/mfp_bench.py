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

import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from hpoglue.benchmark import BenchmarkDescription, SurrogateBenchmark, TabularBenchmark
from hpoglue.env import Env
from hpoglue.fidelity import RangeFidelity
from hpoglue.measure import Measure
from hpoglue.result import Result

if TYPE_CHECKING:
    import mfpbench

    from hpoglue.query import Query


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


_lcbench_task_ids = (
    "3945",
    "7593",
    "34539",
    "126025",
    "126026",
    "126029",
    "146212",
    "167104",
    "167149",
    "167152",
    "167161",
    "167168",
    "167181",
    "167184",
    "167185",
    "167190",
    "167200",
    "167201",
    "168329",
    "168330",
    "168331",
    "168335",
    "168868",
    "168908",
    "168910",
    "189354",
    "189862",
    "189865",
    "189866",
    "189873",
    "189905",
    "189906",
    "189908",
    "189909",
)


def _download_data_cmd(key: str, datadir: Path | None = None) -> tuple[str, ...]:
    install_cmd = f"python -m mfpbench download --benchmark {key}"
    if datadir is not None:
        install_cmd += f" --data-dir {datadir.resolve()}"
    return tuple(install_cmd.split(" "))


def lcbench_surrogate(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the LCBench surrogate Benchmark.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each task in the LCBench surrogate Benchmark.
    """
    if datadir is not None and "yahpo" in os.listdir(datadir):
        datadir = datadir / "yahpo"
    import mfpbench
    env = Env(
        name="py310-mfpbench-1.9-yahpo",
        requirements=("mf-prior-bench[yahpo]==1.9.0",),
        post_install=_download_data_cmd("yahpo", datadir=datadir),
    )
    for task_id in _lcbench_task_ids:
        yield BenchmarkDescription(
            name=f"yahpo-lcbench-{task_id}",
            config_space=mfpbench.get("lcbench", task_id=task_id).space,
            load=partial(_lcbench_tabular, task_id=task_id, datadir=datadir),
            metrics={
                "val_accuracy": Measure.metric((0.0, 100.0), minimize=False),
                "val_cross_entropy": Measure.metric((0, np.inf), minimize=True),
                "val_balanced_accuracy": Measure.metric((0, 100), minimize=False),
            },
            test_metrics={
                "test_balanced_accuracy": Measure.test_metric((0, 100), minimize=False),
                "test_cross_entropy": Measure.test_metric(bounds=(0, np.inf), minimize=True),
            },
            costs={
                "time": Measure.cost((0, np.inf), minimize=True),
            },
            fidelities={
                "epoch": RangeFidelity.from_tuple((1, 52, 1), supports_continuation=True),
            },
            env=env,
            mem_req_mb=4096,
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
        requirements=("mf-prior-bench[tabular]==1.9.0",),
        post_install=_download_data_cmd("lcbench-tabular", datadir=datadir),
    )
    for task_id in task_ids:
        yield BenchmarkDescription(
            name=f"lcbench_tabular-{task_id}",
            config_space=mfpbench.get("lcbench_tabular", task_id=task_id).space,
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


def mfh(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
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
        name="py310-mfpbench-1.9-mfh",
        python_version="3.10",
        requirements=("mf-prior-bench==1.9.0",),
        post_install=(),
    )
    for correlation in ("bad", "good", "moderate", "terrible"):
        for dims in (3, 6):
            name = f"mfh{dims}_{correlation}"
            _min = -3.32237 if dims == 3 else -3.86278  # noqa: PLR2004
            yield BenchmarkDescription(
                name=name,
                config_space=mfpbench.get(name).space,
                load=partial(_get_surrogate_benchmark, benchmark_name=name, datadir=datadir),
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


def jahs(datadir: Path | None = None) -> Iterator[BenchmarkDescription]:
    """Generates benchmark descriptions for the JAHSBench Benchmark.

    Args:
        datadir (Path | None): The directory where the data is stored.
        If None, the default directory is used.

    Yields:
        Iterator[BenchmarkDescription]: An iterator over BenchmarkDescription objects
        for each task in JAHSBench.
    """
    if datadir is not None and "jahs" in os.listdir(datadir):
        datadir = datadir / "jahs"
    import mfpbench
    task_ids = ("CIFAR10", "ColorectalHistology", "FashionMNIST")
    env = Env(
        name="py310-mfpbench-1.9-jahs",
        python_version="3.10",
        requirements=("mf-prior-bench[jahs-bench]==1.9.0",),
        post_install=_download_data_cmd("jahs", datadir=datadir),
    )
    for task_id in task_ids:
        name = f"jahs-{task_id}"
        yield BenchmarkDescription(
            name=name,
            config_space=mfpbench.get("jahs", task_id=task_id).space,
            load=partial(
                _get_surrogate_benchmark,
                benchmark_name="jahs",
                task_id=task_id,
                datadir=datadir,
            ),
            metrics={
                "valid_acc": Measure.metric((0.0, 100.0), minimize=False),
            },
            test_metrics={
                "test_acc": Measure.test_metric((0.0, 100.0), minimize=False),
            },
            fidelities={
                "epoch": RangeFidelity.from_tuple((1, 200, 1), supports_continuation=True),
            },
            costs={
                "runtime": Measure.cost((0, np.inf), minimize=True),
            },
            has_conditionals=False,
            is_tabular=False,
            env=env,
            mem_req_mb=12288,
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
    if datadir is not None and "pd1" in os.listdir(datadir):
        datadir = datadir / "pd1"
    import mfpbench
    env = Env(
        name="py310-mfpbench-1.9-pd1",
        python_version="3.10",
        requirements=("mf-prior-bench[pd1]==1.9.0",),
        post_install=_download_data_cmd("pd1", datadir=datadir),
    )
    yield BenchmarkDescription(
        name="pd1-cifar100-wide_resnet-2048",
        config_space=mfpbench.get("cifar100_wideresnet_2048", datadir=datadir).space,
        load=partial(
            _get_surrogate_benchmark, benchmark_name="cifar100_wideresnet_2048", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics=None,
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 199, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
    yield BenchmarkDescription(
        name="pd1-imagenet-resnet-512",
        config_space=mfpbench.get("imagenet_resnet_512", datadir=datadir).space,
        load=partial(
            _get_surrogate_benchmark, benchmark_name="imagenet_resnet_512", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics=None,
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 99, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=12288,
    )
    yield BenchmarkDescription(
        name="pd1-lm1b-transformer-2048",
        config_space=mfpbench.get("lm1b_transformer_2048", datadir=datadir).space,
        load=partial(
            _get_surrogate_benchmark, benchmark_name="lm1b_transformer_2048", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics=None,
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 74, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=24576,
    )
    yield BenchmarkDescription(
        name="pd1-translate_wmt-xformer_translate-64",
        config_space=mfpbench.get("translatewmt_xformer_64", datadir=datadir).space,
        load=partial(
            _get_surrogate_benchmark, benchmark_name="translatewmt_xformer_64", datadir=datadir
        ),
        metrics={"valid_error_rate": Measure.metric(bounds=(0, 1), minimize=True)},
        test_metrics=None,
        costs={"train_cost": Measure.cost(bounds=(0, np.inf), minimize=True)},
        fidelities={"epoch": RangeFidelity.from_tuple((1, 19, 1), supports_continuation=True)},
        is_tabular=False,
        has_conditionals=False,
        env=env,
        mem_req_mb=24576,
    )


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
    # yield from lcbench_surrogate(datadir)
    # yield from lcbench_tabular(datadir)
    yield from mfh(datadir)
    # yield from jahs(datadir)
    yield from pd1(datadir)
