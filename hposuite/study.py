from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Mapping
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from hpoglue.benchmark import BenchmarkDescription
from hpoglue.constants import DEFAULT_RELATIVE_EXP_DIR
from hpoglue.env import (
    GLUE_PYPI,
    get_current_installed_hpoglue_version,
)
from hpoglue.optimizer import Optimizer
from hpoglue.problem import Problem

from hposuite.benchmarks import BENCHMARKS
from hposuite.optimizers import OPTIMIZERS
from hposuite.run import Run

if TYPE_CHECKING:
    from hpoglue.budget import BudgetType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

smac_logger = logging.getLogger("smac")
smac_logger.setLevel(logging.ERROR)


OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]

GLOBAL_SEED = 42



class Study:
    """Represents a Suite study for hyperparameter optimization."""
    def __init__(
        self,
        name: str,
        output_dir: Path | None = None,
    ):
        """Initialize a Study object with a name and a results directory.

        Args:
            name: The name of the study.
            output_dir: The directory to store the experiment results.
        """
        self.name = name
        if output_dir is None:
            output_dir = Path.cwd().absolute().parent / "hpo-glue-output"
        self.output_dir = output_dir

    @classmethod
    def generate_seeds(
        cls,
        num_seeds: int,
    ):
        """Generate a set of seeds using a Global Seed."""
        cls._rng = np.random.default_rng(GLOBAL_SEED)
        return cls._rng.integers(0, 2 ** 30 - 1, size=num_seeds)


    @classmethod
    def generate(  # noqa: C901, PLR0912, PLR0913, PLR0915
        cls,
        optimizers: (
            type[Optimizer]
            | OptWithHps
            | list[type[Optimizer]]
            | list[OptWithHps | type[Optimizer]]
        ),
        benchmarks: BenchmarkDescription | Iterable[BenchmarkDescription],
        *,
        expdir: Path | str = DEFAULT_RELATIVE_EXP_DIR,
        budget: BudgetType | int,
        seeds: Iterable[int] | None = None,
        num_seeds: int = 1,
        fidelities: int | None = None,
        objectives: int = 1,
        costs: int = 0,
        multi_objective_generation: Literal["mix_metric_cost", "metric_only"] = "mix_metric_cost",
        on_error: Literal["warn", "raise", "ignore"] = "warn",
        continuations: bool = False,
        precision: int | None = None
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
            fidelities: The number of fidelities to generate problems for.
            objectives: The number of objectives to generate problems for.
            costs: The number of costs to generate problems for.
            multi_objective_generation: The method to generate multiple objectives.
            on_error: The method to handle errors.

                * "warn": Log a warning and continue.
                * "raise": Raise an error.
                * "ignore": Ignore the error and continue.
            continuations: Whether to use continuations for the run.
            precision: The precision to use for the HP configs.
        """
        # Generate seeds
        match seeds:
            case None:
                seeds = cls.generate_seeds(num_seeds).tolist()
            case Iterable():
                pass
            case int():
                seeds = [seeds]

        _benchmarks: list[BenchmarkDescription] = []
        match benchmarks:
            case BenchmarkDescription():
                _benchmarks = [benchmarks]
            case Iterable():
                _benchmarks = list(benchmarks)
            case _:
                raise TypeError(
                    "Expected BenchmarkDescription or Iterable[BenchmarkDescription],"
                    f" got {type(benchmarks)}"
                )

        _optimizers: list[OptWithHps]
        match optimizers:
            case tuple():
                _opt, hps = optimizers
                _optimizers = [(_opt, hps)]
            case list():
                _optimizers = [o if isinstance(o, tuple) else (o, {}) for o in optimizers]
            case _:
                _optimizers = [(optimizers, {})]

        _problems: list[Problem] = []
        for (opt, hps), bench in product(_optimizers, _benchmarks):
            try:
                if fidelities is None:
                    match opt.support.fidelities[0]:
                        case "single":
                            fidelities = 1
                        case "many":
                            fidelities = len(bench.fidelities) if bench.fidelities else None
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
                    precision=precision
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
                if "single" not in _problem.optimizer.support.fidelities:
                    continuations = False
                _runs_per_problem.append(
                    Run(
                        problem=_problem,
                        seed=_seed,
                        expdir=Path(expdir),
                        continuations=continuations
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
        *,
        overwrite: bool = False,
        precision: int | None = None,
    ) -> None:
        """Dump the grouped runs into separate files."""
        grouped_runs = self._group_by(group_by)
        for key, runs in grouped_runs.items():
            with (exp_dir / f"dump_{key}.txt").open("w") as f:
                for run in runs:
                    f.write(
                        f"python -m hpoglue"
                        # f"--exp_name {self.name}"
                        f" --optimizers {run.optimizer.name}"
                        f" --benchmarks {run.benchmark.name}"
                        f" --seeds {run.seed}"
                        f" --budget {run.problem.budget.total}"
                    )
                    if overwrite:
                        f.write(" --overwrite")
                    if run.continuations:
                        f.write(" --continuations")
                    if precision:
                        f.write(f" --precision {precision}")
                    f.write("\n")
            logger.info(f"Dumped experiments to {exp_dir / f'dump_{key}.txt'}")


    def optimize(  # noqa: C901, PLR0912, PLR0913
        self,
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
        benchmarks: list[str],
        *,
        seeds: Iterable[int] | int | None = None,
        num_seeds: int = 1,
        budget: int = 50,
        n_objectives: int = 1,
        n_fidelities: int | None = None,
        precision: int | None = None,
        exec_type: Literal["sequential", "parallel", "dump"] = "sequential",
        group_by: Literal["opt", "bench", "opt_bench", "seed", "mem"] | None = None,
        overwrite: bool = False,
        continuations: bool = False,
        on_error: Literal["warn", "raise", "ignore"] = "warn",
    ) -> None:
        """Execute multiple atomic runs using a list of Optimizers and a list of Benchmarks.

        Args:
            optimizers: The list of optimizers to use.

            benchmarks: The list of benchmarks to use.

            seeds: The seed or seeds to use for the experiment.

            num_seeds: The number of seeds to generate.

            budget: The budget for the experiment.

            n_objectives: The number of objectives to use.

            n_fidelities: The number of fidelities to use.

            precision: The precision of the optimization run(s).

            exec_type: The type of execution to use.
            Supported types are "sequential", "parallel" and "dump".

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
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        if not isinstance(benchmarks, list):
            benchmarks = [benchmarks]

        _optimizers = []
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

        _benchmarks = []
        for benchmark in benchmarks:
            assert benchmark in BENCHMARKS, f"Benchmark must be one of {BENCHMARKS.keys()}"
            _benchmarks.append(BENCHMARKS[benchmark])

        exp_dir = self.output_dir

        self.experiments = Study.generate(
            optimizers=_optimizers,
            benchmarks=_benchmarks,
            expdir=exp_dir,
            budget=budget,
            seeds=seeds,
            num_seeds=num_seeds,
            fidelities=n_fidelities,
            objectives=n_objectives,
            on_error=on_error,
            precision=precision,
            continuations=continuations
        )
        for run in self.experiments:
            run.write_yaml()

        if (len(self.experiments) > 1):
            match exec_type:
                case "sequential":
                    logger.info("Running experiments sequentially")
                    for run in self.experiments:
                        # run.create_env(hpoglue=f"-e {Path.cwd()}")
                        run.run(
                            overwrite=overwrite,
                            progress_bar=False,
                        )
                case "parallel":
                    raise NotImplementedError("Parallel execution not implemented yet!")
                case "dump":
                    logger.info("Dumping experiments")
                    self._dump_runs(
                        group_by=group_by,
                        exp_dir=exp_dir,
                        overwrite=overwrite,
                        precision=precision,
                    )
                case _:
                    raise ValueError(f"Invalid exceution type: {exec_type}")
        else:
            run = self.experiments[0]
            # run.create_env(hpoglue=f"-e {Path.cwd()}")
            run.run(
                overwrite=overwrite,
                progress_bar=False,
            )


def create_study(
        output_dir: Path | None = None,
        name: str | None = None,
    ) -> Study:
    """Create a Study object."""
    if output_dir is None:
        output_dir = Path.cwd().absolute().parent / "hpo-glue-output"
    """Create a Study object."""
    if name is None:
        name = f"glue_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return Study(name, output_dir)