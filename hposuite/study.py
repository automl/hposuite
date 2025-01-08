from __future__ import annotations

import hashlib
import logging
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import yaml
from hpoglue import BenchmarkDescription, FunctionalBenchmark, Optimizer, Problem
from hpoglue.env import (
    GLUE_PYPI,
    get_current_installed_hpoglue_version,
)

from hposuite.benchmarks import BENCHMARKS
from hposuite.constants import DEFAULT_STUDY_DIR
from hposuite.optimizers import OPTIMIZERS
from hposuite.run import Run

if TYPE_CHECKING:
    from hpoglue.budget import BudgetType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

smac_logger = logging.getLogger("smac")
smac_logger.setLevel(logging.ERROR)



OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]

BenchWith_Objs_Fids: TypeAlias = tuple[BenchmarkDescription, Mapping[str, Any]]

GLOBAL_SEED = 42


@dataclass(kw_only=True)
class Study:
    """Represents a Suite study for hyperparameter optimization."""

    name: str

    output_dir: Path | None = None

    study_yaml_path: Path = field(init=False)

    optimizers: list[OptWithHps] | list[type[Optimizer]] = field(init=False)

    benchmarks: list[BenchWith_Objs_Fids] | list[BenchmarkDescription] = field(init=False)

    experiments: list[Run]

    seeds: Iterable[int] | int | None = None

    num_seeds: int = 1

    budget: int = 50

    group_by: Literal["opt", "bench", "opt_bench", "seed", "mem"] | None = None

    continuations: bool = True

    def __post_init__(self):

        self.optimizers = []
        self.benchmarks = []
        seeds = set()

        for run in self.experiments:
            opt_keys = [opt[0].name for opt in self.optimizers if self.optimizers]
            if not opt_keys or run.optimizer.name not in opt_keys:
                self.optimizers.append(
                    (
                        run.optimizer,
                        run.optimizer_hyperparameters
                    )
                )
            bench_keys = [bench[0].name for bench in self.benchmarks if self.benchmarks]
            if not bench_keys or run.benchmark.name not in bench_keys:
                self.benchmarks.append(
                    (
                        run.benchmark,
                        {
                            "objectives": run.problem.get_objectives(),
                            "fidelities": run.problem.get_fidelities()
                        }
                    )
                )
            seeds.add(run.seed)

        if self.seeds is None:
            self.seeds = list(seeds)
        self.num_seeds = len(self.seeds)

        name_parts: list[str] = []
        name_parts.append(";".join([f"{opt[0].name}{opt[-1]}" for opt in self.optimizers]))
        name_parts.append(";".join([f"{bench[0].name}{bench[-1]}" for bench in self.benchmarks]))
        name_parts.append(f"seeds={self.seeds}")
        name_parts.append(f"budget={self.budget}")

        if self.name is None:
            self.name = hashlib.sha256((".".join(name_parts)).encode()).hexdigest()

        self.output_dir = self.output_dir / self.name
        self.study_yaml_path = self.output_dir / "study_config.yaml"
        self.write_yaml()
        logger.info(f"Created study at {self.output_dir}")


        if len(self.experiments) > 1:
            self._dump_runs(
                group_by=self.group_by,
                exp_dir=self.output_dir,
            )


    def _update_study(
        self,
        *,
        new_seeds: Iterable[int],
    ):
        """Update the study with new seeds."""
        more_experiments = Study.generate(
            optimizers=self.optimizers,
            benchmarks=self.benchmarks,
            expdir=self.output_dir,
            budget=self.budget,
            seeds=new_seeds,
            continuations=self.continuations,
        )

        self.seeds.extend(new_seeds)
        self.num_seeds += len(new_seeds)
        self.experiments.extend(more_experiments)
        self.write_yaml()



    def to_dict(self) -> dict[str, Any]:
        """Convert the study to a dictionary."""
        optimizers = []
        benchmarks = []
        continuations = 0
        for run in self.experiments:
            run._set_paths(self.output_dir)
            run.write_yaml()
            opt_keys = [opt["name"] for opt in optimizers if optimizers]
            if not opt_keys or run.optimizer.name not in opt_keys:
                optimizers.append(
                    {
                        "name": run.optimizer.name,
                        "hyperparameters": run.optimizer_hyperparameters or {},
                    }
                )
            bench_keys = [bench["name"] for bench in benchmarks if benchmarks]
            if not bench_keys or run.benchmark.name not in bench_keys:
                benchmarks.append(
                    {
                        "name": run.benchmark.name,
                        "objectives": run.problem.get_objectives(),
                        "fidelities": run.problem.get_fidelities(),
                        "costs": run.problem.get_costs(),
                    }
                )
            continuations += run.problem.continuations

        continuations = continuations > 0

        return {
            "name": self.name,
            "output_dir": str(self.output_dir),
            "optimizers": optimizers,
            "benchmarks": benchmarks,
            "seeds": self.seeds,
            "num_seeds": self.num_seeds,
            "budget": self.budget,
            "continuations": continuations,
        }


    def write_yaml(self) -> None:
        """Write the study config to a YAML file."""
        self.study_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with self.study_yaml_path.open("w") as file:
            yaml.dump(self.to_dict(), file, sort_keys=False)


    @classmethod
    def generate_seeds(
        cls,
        num_seeds: int,
        offset: int = 0, # To offset number of seeds
    ):
        """Generate a set of seeds using a Global Seed."""
        cls._rng = np.random.default_rng(GLOBAL_SEED)
        _num_seeds = num_seeds + offset
        _seeds = cls._rng.integers(0, 2 ** 32, size=_num_seeds)
        return _seeds[offset:].tolist()


    @classmethod
    def generate(  # noqa: C901, PLR0912, PLR0915
        cls,
        optimizers: (
            type[Optimizer]
            | OptWithHps
            | list[type[Optimizer]]
            | list[OptWithHps | type[Optimizer]]
        ),
        benchmarks: (
            BenchmarkDescription
            | BenchWith_Objs_Fids
            | list[BenchmarkDescription]
            | list[BenchWith_Objs_Fids | BenchmarkDescription]
        ),
        *,
        budget: BudgetType | int,
        seeds: Iterable[int] | None = None,
        num_seeds: int = 1,
        costs: int = 0,
        multi_objective_generation: Literal["mix_metric_cost", "metric_only"] = "mix_metric_cost",
        on_error: Literal["warn", "raise", "ignore"] = "warn",
        continuations: bool = True,
    ) -> list[Run]:
        """Generate a set of problems for the given optimizer and benchmark.

        If there is some incompatibility between the optimizer, the benchmark and the requested
        amount of objectives, fidelities or costs, a ValueError will be raised.

        Args:
            optimizers: The optimizer class to generate problems for.
                Can provide a single optimizer or a list of optimizers.
                If you wish to provide hyperparameters for the optimizer, provide a tuple with the
                optimizer.

            benchmarks: The benchmark to generate problems for.
                Can provide a single benchmark or a list of benchmarks.

            expdir: Which directory to store experiment results into.

            budget: The budget to use for the problems. Budget defaults to a n_trials budget
                where when multifidelty is enabled, fractional budget can be used and 1 is
                equivalent a full fidelity trial.

            seeds: The seed or seeds to use for the problems.

            num_seeds: The number of seeds to generate. Only used if seeds is None.

            costs: The number of costs to generate problems for.

            multi_objective_generation: The method to generate multiple objectives.

            on_error: The method to handle errors.
                * "warn": Log a warning and continue.
                * "raise": Raise an error.
                * "ignore": Ignore the error and continue.

            continuations: Whether to use continuations for the run.

        Returns:
            A list of Run objects.
        """
        # Generate seeds
        match seeds:
            case None:
                seeds = cls.generate_seeds(num_seeds)
            case Iterable():
                pass
            case int():
                seeds = [seeds]

        _optimizers: list[OptWithHps]
        match optimizers:
            case tuple():
                _opt, hps = optimizers
                _optimizers = [(_opt, hps)]
            case list():
                _optimizers = [o if isinstance(o, tuple) else (o, {}) for o in optimizers]
            case type():
                _optimizers = [(optimizers, {})]
            case _:
                raise TypeError(
                    "Expected Optimizer or list[Optimizer] or tuple[Optimizer, dict] or "
                    f"list[tuple[Optimizer, dict]], got {type(optimizers)}"
                )

        _benchmarks: list[BenchWith_Objs_Fids]
        match benchmarks:
            case tuple():
                _bench, objsfids = benchmarks
                _benchmarks = [(_bench, objsfids)]
            case list():
                _benchmarks = [b if isinstance(b, tuple) else (b, {}) for b in benchmarks]
            case BenchmarkDescription():
                _benchmarks = [(benchmarks, {})]
            case _:
                raise TypeError(
                    "Expected BenchmarkDescription or list[BenchmarkDescription] or "
                    "tuple[BenchmarkDescription, dict] or list[tuple[BenchmarkDescription, dict]],"
                    f" got {type(benchmarks)}"
                )

        _problems: list[Problem] = []
        for (opt, hps), (bench, objs_fids) in product(_optimizers, _benchmarks):
            try:

                objectives: int | str | list[str]
                fidelities: int | str | list[str] | None

                objectives = objs_fids.get("objectives", 1)
                fidelities = objs_fids.get("fidelities", None)
                if fidelities is None and bench.fidelities:
                    match opt.support.fidelities[0]:
                        case "single":
                            fidelities = 1
                        case "many":
                            fidelities = len(bench.fidelities)
                        case None:
                            fidelities = None
                        case _:
                            raise ValueError("Invalid fidelity support")


                _problem = Problem.problem(
                    optimizer=opt,
                    optimizer_hyperparameters=hps,
                    benchmark=bench,
                    objectives=objectives,
                    budget=budget,
                    fidelities=fidelities,
                    costs=costs,
                    multi_objective_generation=multi_objective_generation,
                    continuations=continuations,
                )
                _problems.append(_problem)
            except ValueError as e:
                match on_error:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case "warn":
                        warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
                        continue

        _runs_per_problem: list[Run] = []
        for _problem, _seed in product(_problems, seeds):
            try:
                _runs_per_problem.append(
                    Run(
                        problem=_problem,
                        seed=_seed,
                    )
                )
            except ValueError as e:
                match on_error:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case "warn":
                        warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
                        continue

        logger.info(f"Generated {len(_runs_per_problem)} runs")

        return _runs_per_problem


    def create_env(             #TODO: This is not called for now. Fix this.
        self,
        *,
        how: Literal["venv", "conda"] = "venv",
        hpoglue: Literal["current_version"] | str,
    ) -> None:
        """Set up the isolation for the experiment."""
        if hpoglue == "current_version":
            raise NotImplementedError("Not implemented yet.")

        match hpoglue:
            case "current_version":
                _version = get_current_installed_hpoglue_version()
                req = f"{GLUE_PYPI}=={_version}"
            case str():
                req = hpoglue
            case _:
                raise ValueError(f"Invalid value for `hpoglue`: {hpoglue}")

        requirements = [req, *self.env.requirements]

        if self.env_path.exists():
            logger.info(f"Environment already exists: {self.env.identifier}")
            return

        logger.info(f"Installing deps: {self.env.identifier}")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        with self.venv_requirements_file.open("w") as f:
            f.write("\n".join(requirements))

        # if self.env_path.exists(): TODO: Does this need to be after install step?
        #     return

        self.env_path.parent.mkdir(parents=True, exist_ok=True)

        env_dict = self.env.to_dict()
        env_dict.update({"env_path": str(self.env_path), "hpoglue_source": req})

        logger.info(f"Installing env: {self.env.identifier}")
        match how:
            case "venv":
                logger.info(f"Creating environment {self.env.identifier} at {self.env_path}")
                self.venv.create(
                    path=self.env_path,
                    python_version=self.env.python_version,
                    requirements_file=self.venv_requirements_file,
                    exists_ok=False,
                )
                if self.env.post_install:
                    logger.info(f"Running post install for {self.env.identifier}")
                    with self.post_install_steps.open("w") as f:
                        f.write("\n".join(self.env.post_install))
                    self.venv.run(self.env.post_install)
            case "conda":
                raise NotImplementedError("Conda not implemented yet.")
            case _:
                raise ValueError(f"Invalid value for `how`: {how}")

    def _group_by(
        self,
        group_by: Literal["opt", "bench", "opt_bench", "seed", "mem"] | None,
    ) -> Mapping[str, list[Run]]:
        """Group the runs by the specified group."""
        if group_by is None:
            return {"all": self.experiments}

        _grouped_runs = {}
        for run in self.experiments:
            key = ""
            match group_by:
                case "opt":
                    key = run.optimizer.name
                case "bench":
                    key = run.benchmark.name
                case "opt_bench":
                    key = f"{run.optimizer.name}_{run.benchmark.name}"
                case "seed":
                    key = str(run.seed)
                case "mem":
                    key = f"{run.mem_req_mb}mb"
                case _:
                    raise ValueError(f"Invalid group_by: {group_by}")

            if key not in _grouped_runs:
                _grouped_runs[key] = []
            _grouped_runs[key].append(run)

        return _grouped_runs

    def _dump_runs(
        self,
        group_by: Literal["opt", "bench", "opt_bench", "seed", "mem"] | None,
        exp_dir: Path,
    ) -> None:
        """Dump the grouped runs into separate files."""
        grouped_runs = self._group_by(group_by)
        for key, runs in grouped_runs.items():
            with (exp_dir / f"dump_{key}.txt").open("w") as f:
                for run in runs:
                    f.write(
                        f"python -m hpoglue"
                        f" --optimizers {run.optimizer.name}"
                        f" --benchmarks {run.benchmark.name}"
                        f" --seeds {run.seed}"
                        f" --budget {run.problem.budget.total}"
                    )
                    f.write("\n")
            logger.info(f"Dumped experiments to {exp_dir / f'dump_{key}.txt'}")


    def optimize(  # noqa: C901, PLR0912
        self,
        *,
        exec_type: Literal["sequential", "parallel"] = "sequential",
        add_seeds: Iterable[int] | int | None = None,
        add_num_seeds: int | None = None,
        overwrite: bool = False,
        continuations: bool = True,
    ) -> None:
        """Execute multiple atomic runs using a list of Optimizers and a list of Benchmarks.

        Args:
            exec_type: The type of execution to use.
            Supported types are "sequential", "parallel" and "dump".

            add_seeds: The seed or seeds to add to the study.

            add_num_seeds: The number of seeds to generate and add to the study.

            NOTE: Only one of `add_seeds` and `add_num_seeds` can be provided.

            group_by: The grouping to use for the runs dump.
            Supported types are "opt", "bench", "opt_bench", "seed", and "mem"
            Only used when `exec_type` is "dump" for multiple runs.

            overwrite: Whether to overwrite existing results.

            continuations: Whether to calculate continuations cost.
                Note: Only works for Multi-fidelity Optimizers.

            on_error: The method to handle errors.
                    * "warn": Log a warning and continue.
                    * "raise": Raise an error.
                    * "ignore": Ignore the error and continue.

        """
        if add_seeds is not None and add_num_seeds is not None:
            logger.warning(
                "Cannot provide both `add_seeds` and `add_num_seeds`!"
                "Using only `add_seeds` and ignoring `add_num_seeds`"
            )
            add_num_seeds = None
        if isinstance(add_seeds, int):
            add_seeds = [add_seeds]

        _seeds: list[int] = []

        match add_seeds, add_num_seeds:
            case None, None:
                pass
            case Iterable(), None:
                seed = list(set(add_seeds))
                for seed in add_seeds:
                    if seed not in self.seeds+_seeds:
                        _seeds.append(seed)
            case None, int():
                _num_seeds = add_num_seeds
                offset = 0
                while _num_seeds > 0:
                    new_seeds = [
                        s for s in Study.generate_seeds(_num_seeds, offset=offset)
                        if s not in self.seeds
                    ]
                    _seeds.extend(new_seeds)
                    _num_seeds -= len(_seeds)
                    offset += _num_seeds + len(new_seeds)
            case _:
                raise ValueError(
                    "Invalid combination of types for `add_seeds` and `add_num_seeds`"
                    "Expected (Iterable[int] | int | None, int | None),"
                    f"got ({type(add_seeds)}, {type(add_num_seeds)})"
                )


        if _seeds:
            self._update_study(new_seeds=_seeds)

        if overwrite:
            logger.info("Overwrite flag is set to True. Existing results will be overwritten!")

        if (len(self.experiments) > 1):
            match exec_type:
                case "sequential":
                    logger.info(f"Running {len(self.experiments)} experiments sequentially")
                    for i, run in enumerate(self.experiments, start=1):
                        # run.create_env(hpoglue=f"-e {Path.cwd()}")
                        logger.info(f"Running experiment {i}/{len(self.experiments)}")
                        run.run(
                            continuations=continuations,
                            overwrite=overwrite,
                            progress_bar=False,
                        )
                case "parallel":
                    raise NotImplementedError("Parallel execution not implemented yet!")
                case _:
                    raise ValueError(f"Invalid exceution type: {exec_type}")
        else:
            run = self.experiments[0]
            # run.create_env(hpoglue=f"-e {Path.cwd()}")
            logger.info("Running single experiment")
            run.run(
                continuations=continuations,
                overwrite=overwrite,
                progress_bar=False,
            )


def create_study(  # noqa: C901, PLR0912, PLR0915
    *,
    name: str | None = None,
    output_dir: str| Path | None = None,
    optimizers: (
        str
        | tuple[str, Mapping[str, Any]]
        | type[Optimizer]
        | OptWithHps
        | list[tuple[str, Mapping[str, Any]]]
        | list[str]
        | list[OptWithHps]
        | list[type[Optimizer]]
    ),
    benchmarks: (
        str
        | BenchmarkDescription
        | FunctionalBenchmark
        | tuple[str, Mapping[str, Any]]
        | BenchWith_Objs_Fids # tuple[BenchmarkDescription, Mapping[str, Any]]
        | tuple[FunctionalBenchmark, Mapping[str, Any]]
        | list[str]
        | list[BenchmarkDescription]
        | list[FunctionalBenchmark]
        | list[tuple[str, Mapping[str, Any]]]
        | list[BenchWith_Objs_Fids]   # list[tuple[BenchmarkDescription, Mapping[str, Any]]]
        | list[tuple[FunctionalBenchmark, Mapping[str, Any]]]
    ),
    seeds: Iterable[int] | int | None = None,
    num_seeds: int = 1,
    budget: int = 50,
    group_by: Literal["opt", "bench", "opt_bench", "seed", "mem"] | None = None,
    on_error: Literal["warn", "raise", "ignore"] = "warn",
    continuations: bool = True,
) -> Study:
    """Create a Study object.

    Args:
        name: The name of the study.

        output_dir: The main output directory where the hposuite studies are saved.

        optimizers: The list of optimizers to use.
                    Usage: [
                        (
                            optimizer: str | type[Optimizer],
                            {
                                "hp_name": hp_value
                            }
                        )
                    ]

        benchmarks: The list of benchmarks to use.
                    Usage: [
                        (
                            benchmark: str | BenchmarkDescription | FunctionalBenchmark,
                            {
                                "objectives": [list of objectives] | number of objectives,
                                "fidelities": [list of fidelities] | number of fidelities | None,
                            }
                        )
                    ]

        seeds: The seed or seeds to use for the experiment.

        num_seeds: The number of seeds to generate.

        budget: The budget for the experiment.

        group_by: The grouping to use for the runs dump.

        on_error: The method to handle errors while generating runs for the study.

        continuations: Whether to calculate continuations cost.

    Returns:
        A Study object.
    """
    match output_dir:
        case None:
            output_dir = DEFAULT_STUDY_DIR
        case str():
            output_dir = Path(output_dir)
        case Path():
            pass
        case _:
            raise TypeError(f"Invalid type for output_dir: {type(output_dir)}")

    assert optimizers, "At least one optimizer must be provided!"
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    assert benchmarks, "At least one benchmark must be provided!"
    if not isinstance(benchmarks, list):
        benchmarks = [benchmarks]

    _optimizers: list[OptWithHps] = []
    match optimizers[0]:
        case str():
            for optimizer in optimizers:
                assert optimizer in OPTIMIZERS, f"Optimizer must be one of {OPTIMIZERS.keys()}"
                _optimizers.append(OPTIMIZERS[optimizer])
        case tuple():
            for optimizer, opt_hps in optimizers:
                assert optimizer in OPTIMIZERS, f"Optimizer must be one of {OPTIMIZERS.keys()}"
                _optimizers.append((OPTIMIZERS[optimizer], opt_hps))
        case type():
            _optimizers = optimizers
        case _:
            raise TypeError(f"Unknown Optimizer type {type(optimizers[0])}")

    _benchmarks: list[BenchWith_Objs_Fids] = []
    match benchmarks[0]:
        case str():
            for benchmark in benchmarks:
                assert benchmark in BENCHMARKS, f"Benchmark must be one of {BENCHMARKS.keys()}"
                if not isinstance(BENCHMARKS[benchmark], FunctionalBenchmark):
                    _benchmarks.append(BENCHMARKS[benchmark])
                else:
                    _benchmarks.append(BENCHMARKS[benchmark].description)
        case BenchmarkDescription():
            for benchmark in benchmarks:
                _benchmarks.append(benchmark)
        case FunctionalBenchmark():
            for benchmark in benchmarks:
                _benchmarks.append(benchmark.description)
        case tuple():
            for benchmark, bench_hps in benchmarks:
                match benchmark:
                    case str():
                        assert benchmark in BENCHMARKS, "Benchmark must be one of"
                        f"{BENCHMARKS.keys()}"
                        if not isinstance(BENCHMARKS[benchmark], FunctionalBenchmark):
                            _benchmarks.append((BENCHMARKS[benchmark], bench_hps))
                        else:
                            _benchmarks.append((BENCHMARKS[benchmark].description, bench_hps))
                    case BenchmarkDescription():
                        _benchmarks.append((benchmark, bench_hps))
                    case FunctionalBenchmark():
                        _benchmarks.append((benchmark.description, bench_hps))
                    case _:
                        raise TypeError(f"Unknown Benchmark type {type(benchmark)}")
        case _:
            raise TypeError(f"Unknown Benchmark type {type(benchmarks[0])}")

    if isinstance(seeds, Iterable):
        seeds = list(set(seeds))

    experiments = Study.generate(
        optimizers=_optimizers,
        benchmarks=_benchmarks,
        budget=budget,
        seeds=seeds,
        num_seeds=num_seeds,
        on_error=on_error,
        continuations=continuations
    )

    return Study(
        name=name,
        output_dir=output_dir,
        experiments=experiments,
        seeds=seeds,
        num_seeds=num_seeds,
        budget=budget,
        group_by=group_by,
        continuations=continuations,
    )