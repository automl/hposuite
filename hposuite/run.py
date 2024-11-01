from __future__ import annotations

import logging
import shutil
import subprocess
import traceback
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd
import yaml
from hpoglue.benchmark import BenchmarkDescription
from hpoglue.budget import CostBudget, TrialBudget
from hpoglue.config import Config
from hpoglue.constants import DEFAULT_RELATIVE_EXP_DIR
from hpoglue.dataframe_utils import reduce_dtypes
from hpoglue.env import (
    GLUE_PYPI,
    Env,
    Venv,
    get_current_installed_hpoglue_version,
)
from hpoglue.optimizer import Optimizer
from hpoglue.problem import Problem
from hpoglue.query import Query
from hpoglue.result import Result

if TYPE_CHECKING:
    from hpoglue.budget import BudgetType


OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]

T = TypeVar("T", bound=Hashable)


logger = logging.getLogger(__name__)

GLOBAL_SEED = 42


def _try_delete_if_exists(path: Path) -> None:
    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        except Exception as e:
            logger.exception(e)
            logger.error(f"Error deleting {path}: {e}")


@dataclass
class Run:
    """A run of a benchmark."""

    problem: Problem
    """The problem that was run."""

    seed: int
    """The seed used for the run."""

    optimizer: type[Optimizer] = field(init=False)
    """The optimizer to use for the Problem"""

    expdir: Path = field(default=DEFAULT_RELATIVE_EXP_DIR)
    """Default directory to use for experiments."""

    optimizer_hyperparameters: Mapping[str, Any] = field(default_factory=dict, init=False)
    """The hyperparameters to use for the optimizer"""

    benchmark: BenchmarkDescription = field(init=False)
    """The benchmark that the Problem was run on."""

    env: Env = field(init=False)
    """The environment to setup the optimizer in for `isolated` mode."""

    working_dir: Path = field(init=False)
    """The working directory for the run."""

    complete_flag: Path = field(init=False)
    """The flag to indicate the run is complete."""

    error_file: Path = field(init=False)
    """The file to store the error traceback if the run crashed."""

    running_flag: Path = field(init=False)
    """The flag to indicate the run is currently running."""

    queue_flag: Path = field(init=False)
    """The flag to indicate the run is queued."""

    df_path: Path = field(init=False)
    """The path to the dataframe for the run."""

    venv_requirements_file: Path = field(init=False)
    """The path to the requirements file for the run."""

    post_install_steps: Path = field(init=False)
    """The path to the post install steps file for the run."""

    run_yaml_path: Path = field(init=False)
    """The path to the run yaml file for the run."""

    env_description_file: Path = field(init=False)
    """The path to the environment description file for the run."""

    env_path: Path = field(init=False)
    """The path to the environment for the run."""

    continuations: bool = field(default=False)
    """Whether to use continuations for the run."""

    mem_req_mb: int = field(init=False)
    """The memory requirement for the run in mb.
    Calculated as the sum of the memory requirements of the optimizer and the benchmark.
    """

    def __post_init__(self) -> None:
        self.benchmark = self.problem.benchmark
        self.optimizer = self.problem.optimizer
        self.optimizer_hyperparameters = self.problem.optimizer_hyperparameters
        name_parts: list[str] = [
            self.problem.name,
            f"seed={self.seed}",
        ]
        if len(self.optimizer_hyperparameters) > 0:
            name_parts.append(
                ",".join(f"{k}={v}" for k, v in self.optimizer_hyperparameters.items())
            )
        self.name = ".".join(name_parts)
        self.optimizer.support.check_opt_support(who=self.optimizer.name, problem=self.problem)

        match self.benchmark.env, self.optimizer.env:
            case (None, None):
                self.env = Env.empty()
            case (None, Env()):
                self.env = self.optimizer.env
            case (Env(), None):
                self.env = self.benchmark.env
            case (Env(), Env()):
                self.env = Env.merge(self.benchmark.env, self.optimizer.env)
            case _:
                raise ValueError("Invalid combination of benchmark and optimizer environments")

        self.working_dir = self.expdir.absolute().resolve() / self.name
        self.complete_flag = self.working_dir / "complete.flag"
        self.error_file = self.working_dir / "error.txt"
        self.running_flag = self.working_dir / "running.flag"
        self.df_path = self.expdir / "dfs" / f"{self.name}.parquet"
        self.venv_requirements_file = self.working_dir / "venv_requirements.txt"
        self.queue_flag = self.working_dir / "queue.flag"
        self.requirements_ran_with_file = self.working_dir / "requirements_ran_with.txt"
        self.env_description_file = self.working_dir / "env.yaml"
        self.post_install_steps = self.working_dir / "venv_post_install.sh"
        self.run_yaml_path = self.working_dir / "run_config.yaml"
        self.env_path = self.expdir / "envs" / self.env.identifier
        self.mem_req_mb = self.problem.mem_req_mb + self.optimizer.mem_req_mb

    @property
    def venv(self) -> Venv:
        return Venv(self.env_path)

    @property
    def conda(self) -> Venv:
        raise NotImplementedError("Conda not implemented yet.")

    def run(
        self,
        *,
        on_error: Literal["raise", "continue"] = "raise",
        overwrite: Run.State | str | Sequence[Run.State | str] | bool = False,
        progress_bar: bool = True,
    ) -> Report:
        """Run the Run.

        Args:
            on_error: How to handle errors. In any case, the error will be written
                into the [`working_dir`][hpoglue.run.Run.working_dir]
                of the problem.

                * If "raise", raise an error.
                * If "continue", log the error and continue.

            overwrite: What to overwrite.

                * If a single value, overwrites problem in that state,
                * If a list of states, overwrites any problem in one of those
                 states.
                * If `True`, overwrite problems in all states.
                * If `False`, don't overwrite any problems.

            progress_bar: Whether to show a progress bar.
        """
        from hpoglue._run import _run

        if on_error not in ("raise", "continue"):
            raise ValueError(f"Invalid value for `on_error`: {on_error}")

        overwrites = Run.State.collect(overwrite)

        state = self.state()
        if state in overwrites:
            logger.info(f"Overwriting {self.name} in `{state=}` at {self.working_dir}.")
            self.set_state(Run.State.PENDING)

        if self.df_path.exists():
            logger.info(f"Loading results for {self.name} from {self.working_dir}")
            return Run.Report.from_df(
                df=pd.read_parquet(self.df_path),
                run=self,
            )

        """ TODO
        if self.working_dir.exists():
            raise RuntimeError(
                "The optimizer ran before but no dataframe of results was found at "
                f"{self.df_path}. Set `overwrite=[{state}]` to rerun problems in this state"
            )
        """
        self.set_state(Run.State.PENDING)
        _hist: list[Result] = []
        try:
            self.set_state(self.State.RUNNING)
            _, _hist = _run(
                problem=self.problem,
                seed=self.seed,
                run_name=self.name,
                on_error=on_error,
                progress_bar=progress_bar,
                continuations=self.continuations,
            )
        except Exception as e:
            self.set_state(Run.State.CRASHED, err_tb=(e, traceback.format_exc()))
            logger.exception(e)
            logger.error(f"Error in Run {self.name}: {e}")
        logger.info(f"COMPLETED running {self.name}")
        logger.info(f"Saving {self.name} at {self.working_dir}")
        logger.info(f"Results dumped at {self.df_path.absolute()}")
        return self.post_process(history=_hist)


    def post_process(
        self,
        *,
        history: list[Result],
    ) -> Report:
        """Post process the run.

        Args:
            history: The history of the run.

        Returns:
            The Report of the run.
        """
        report = Run.Report(
            run=self,
            results=history,
        )
        self.set_state(Run.State.COMPLETE, df=report.df())
        return report


    def create_env(
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

        logger.info(f"Installing deps: {self.env.identifier}")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        with self.venv_requirements_file.open("w") as f:
            f.write("\n".join(requirements))

        if self.env_path.exists():
            return

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "seed": self.seed,
            "optimizer": self.optimizer.name,
            "optimizer_hyperparameters": self.optimizer_hyperparameters,
            "expdir": str(self.expdir),
        }

    @classmethod
    def from_yaml(cls, path: Path) -> Run:
        with path.open("r") as file:
            return Run.from_dict(yaml.safe_load(file))

    def write_yaml(self) -> None:
        self.run_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_yaml_path.open("w") as file:
            yaml.dump(self.to_dict(), file, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        from hposuite.optimizers import OPTIMIZERS

        if data["optimizer"] not in OPTIMIZERS:
            raise ValueError(
                f"Optimizer {data['benchmark']} not found in optimizers!"
                " Please make sure your optimizer is registed in `OPTIMIZERS`"
                " before loading/parsing."
            )

        return Run(
            problem=Problem.from_dict(data["problem"]),
            seed=data["seed"],
            optimizer=OPTIMIZERS[data["optimizer"]],
            optimizer_hyperparameters=data["optimizer_hyperparameters"],
            expdir=Path(data["expdir"]),
        )

    def state(self) -> Run.State:
        """Return the state of the run.

        Args:
            run: The run to get the state for.
        """
        if self.complete_flag.exists():
            return Run.State.COMPLETE

        if self.error_file.exists():
            return Run.State.CRASHED

        if self.running_flag.exists():
            return Run.State.RUNNING

        if self.queue_flag.exists():
            return Run.State.QUEUED

        return Run.State.PENDING

    def set_state(
        self,
        state: Run.State,
        *,
        df: pd.DataFrame | None = None,
        err_tb: tuple[Exception, str] | None = None,
    ) -> None:
        """Set the run to a certain state.

        Args:
            state: The state to set the problem to.
            df: Optional dataframe to save if setting to [`Run.State.COMPLETE`].
            err_tb: Optional error traceback to save if setting to [`Run.State.CRASHED`].
        """
        _flags = (self.complete_flag, self.error_file, self.running_flag, self.queue_flag)
        match state:
            case Run.State.PENDING:
                for _file in (*_flags, self.df_path, self.requirements_ran_with_file):
                    _try_delete_if_exists(_file)

                with self.run_yaml_path.open("w") as f:
                    yaml.dump(self.to_dict(), f, sort_keys=False)

            case Run.State.QUEUED:
                for _file in (*_flags, self.df_path, self.requirements_ran_with_file):
                    _try_delete_if_exists(_file)

                self.queue_flag.touch()

            case Run.State.RUNNING:
                for _file in (*_flags, self.df_path, self.requirements_ran_with_file):
                    _try_delete_if_exists(_file)

                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.df_path.parent.mkdir(parents=True, exist_ok=True)

                # lines = subprocess.run(
                #     [self.venv.pip, "freeze"],  # noqa: S603
                #     check=True,
                #     capture_output=True,
                #     text=True,
                # )
                # with self.requirements_ran_with_file.open("w") as f:
                #     f.write(lines.stdout)

                self.running_flag.touch()

            case Run.State.CRASHED:
                for _file in (*_flags, self.df_path):
                    _try_delete_if_exists(_file)

                with self.error_file.open("w") as f:
                    if err_tb is None:
                        f.write("None")
                    else:
                        exc, tb = err_tb
                        f.write(f"{tb}\n{exc}")

            case Run.State.COMPLETE:
                for _file in (*_flags, self.df_path):
                    _try_delete_if_exists(_file)

                self.complete_flag.touch()

                if df is not None:
                    df.to_parquet(self.df_path)
            case _:
                raise ValueError(f"Unknown state {state}")


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
        fidelities: int = 0,
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

    class State(str, Enum):
        """The state of a problem."""

        PENDING = "PENDING"
        QUEUED = "QUEUED"
        RUNNING = "RUNNING"
        CRASHED = "CRASHED"
        COMPLETE = "COMPLETE"

        @classmethod
        def collect(
            cls,
            state: str | Run.State | bool | Iterable[Run.State | str],
        ) -> list[Run.State]:
            """Collect state requested."""
            match state:
                case True:
                    return list(cls)
                case False:
                    return []
                case Run.State():
                    return [state]
                case str():
                    return [cls(state)]
                case _:
                    return [cls(s) if isinstance(s, str) else s for s in state]


    @dataclass
    class Report:
        """The report of a Run."""

        run: Run
        results: list[Result]

        problem: Problem = field(init=False)

        def __post_init__(self) -> None:
            self.problem = self.run.problem

        def df(  # noqa: C901, PLR0912, PLR0915
            self,
            *,
            incumbent_trajectory: bool = False,
        ) -> pd.DataFrame:
            """Return the history as a pandas DataFrame.

            Args:
                incumbent_trajectory: Whether to only include the incumbents trajectory.

            Returns:
                The history as a pandas DataFrame.
            """
            problem = self.problem

            def _encode_result(_r: Result) -> dict[str, Any]:
                _rparts: dict[str, Any] = {
                    "config.id": _r.config.config_id,
                    "query.id": _r.query.query_id,
                    "result.budget_cost": _r.budget_cost,
                    "result.budget_used_total": _r.budget_used_total,
                }
                match _r.query.fidelity:
                    case None:
                        _rparts["query.fidelity.count"] = 0
                    case (name, val):
                        _rparts["query.fidelity.count"] = 1
                        _rparts["query.fidelity.1.name"] = name
                        _rparts["query.fidelity.1.value"] = val
                    case Mapping():
                        _rparts["query.fidelity.count"] = len(_r.query.fidelity)
                        for i, (k, v) in enumerate(_r.query.fidelity.items(), start=1):
                            _rparts[f"query.fidelity.{i}.name"] = k
                            _rparts[f"query.fidelity.{i}.value"] = v

                match problem.objective:
                    case (_name, _measure):
                        _rparts["result.objective.1.value"] = _r.values[_name]
                    case Mapping():
                        for i, name in enumerate(problem.objective, start=1):
                            _rparts[f"result.cost.{i}.value"] = _r.values[name]

                match problem.fidelity:
                    case None:
                        pass
                    case (name, _):
                        assert isinstance(_r.fidelity, tuple)
                        _rparts["result.fidelity.1.value"] = _r.fidelity[1]
                    case Mapping():
                        assert isinstance(_r.fidelity, Mapping)
                        for i, name in enumerate(problem.fidelity, start=1):
                            _rparts[f"result.fidelity.{i}.value"] = _r.fidelity[name]

                match problem.cost:
                    case None:
                        pass
                    case (name, _):
                        _rparts["result.cost.1.value"] = _r.values[name]
                    case Mapping():
                        for i, name in enumerate(problem.cost, start=1):
                            _rparts[f"result.fidelity.{i}.value"] = _r.values[name]

                _rparts["result.continuations_cost.1"] = _r.continuations_cost

                return _rparts

            parts = {}
            parts["run.name"] = self.run.name
            parts["problem.name"] = problem.name

            match problem.objective:
                case (name, measure):
                    parts["problem.objective.count"] = 1
                    parts["problem.objective.1.name"] = name
                    parts["problem.objective.1.minimize"] = measure.minimize
                    parts["problem.objective.1.min"] = measure.bounds[0]
                    parts["problem.objective.1.max"] = measure.bounds[1]
                case Mapping():
                    list(problem.objective)
                    parts["problem.objective.count"] = len(problem.objective)
                    for i, (k, v) in enumerate(problem.objective.items(), start=1):
                        parts[f"problem.objective.{i}.name"] = k
                        parts[f"problem.objective.{i}.minimize"] = v.minimize
                        parts[f"problem.objective.{i}.min"] = v.bounds[0]
                        parts[f"problem.objective.{i}.max"] = v.bounds[1]
                case _:
                    raise TypeError("Objective must be a tuple (name, measure) or a mapping")

            match problem.fidelity:
                case None:
                    parts["problem.fidelity.count"] = 0
                case (name, fid):
                    parts["problem.fidelity.count"] = 1
                    parts["problem.fidelity.1.name"] = name
                    parts["problem.fidelity.1.min"] = fid.min
                    parts["problem.fidelity.1.max"] = fid.max
                case Mapping():
                    list(problem.fidelity)
                    parts["problem.fidelity.count"] = len(problem.fidelity)
                    for i, (k, v) in enumerate(problem.fidelity.items(), start=1):
                        parts[f"problem.fidelity.{i}.name"] = k
                        parts[f"problem.fidelity.{i}.min"] = v.min
                        parts[f"problem.fidelity.{i}.max"] = v.max
                case _:
                    raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

            match problem.cost:
                case None:
                    parts["problem.cost.count"] = 0
                case (name, cost):
                    parts["problem.cost.count"] = 1
                    parts["problem.cost.1.name"] = name
                    parts["problem.cost.1.minimize"] = cost.minimize
                    parts["problem.cost.1.min"] = cost.bounds[0]
                    parts["problem.cost.1.max"] = cost.bounds[1]
                case Mapping():
                    list(problem.cost)
                    parts["problem.cost.count"] = len(problem.cost)
                    for i, (k, v) in enumerate(problem.cost.items(), start=1):
                        parts[f"problem.cost.{i}.name"] = k
                        parts[f"problem.cost.{i}.minimize"] = v.minimize
                        parts[f"problem.cost.{i}.min"] = v.bounds[0]
                        parts[f"problem.cost.{i}.max"] = v.bounds[1]

            _df = pd.DataFrame.from_records([_encode_result(r) for r in self.results])
            _df = _df.sort_values("result.budget_used_total", ascending=True)
            for k, v in parts.items():
                _df[k] = v

            _df["problem.benchmark"] = problem.benchmark.name
            match problem.budget:
                case TrialBudget(total):
                    _df["problem.budget.kind"] = "TrialBudget"
                    _df["problem.budget.total"] = total
                case CostBudget(total):
                    _df["problem.budget.kind"] = "CostBudget"
                    _df["problem.budget.total"] = total
                case _:
                    raise NotImplementedError(f"Unknown budget type {problem.budget}")

            _df["run.seed"] = self.run.seed
            _df["run.opt.name"] = self.run.optimizer.name

            if len(self.run.optimizer_hyperparameters) > 0:
                for k, v in self.run.optimizer_hyperparameters.items():
                    _df[f"run.opt.hp.{k}"] = v

                _df["run.opt.hp_str"] = ",".join(
                    f"{k}={v}" for k, v in self.run.optimizer_hyperparameters.items()
                )
            else:
                _df["run.opt.hp_str"] = "default"

            _df = _df.sort_values("result.budget_used_total", ascending=True)

            _df = reduce_dtypes(
                _df,
                reduce_int=True,
                reduce_float=True,
                categories=True,
                categories_exclude=("config.id", "query.id"),
            )

            if incumbent_trajectory:
                if not isinstance(self.problem.objective, tuple):
                    raise ValueError(
                        "Incumbent trajectory only supported for single objective."
                        f" Problem {self.problem.name} has {len(self.problem.objective)} objectives"
                        f" for run {self.run.name}"
                    )

                if self.problem.objective[1].minimize:
                    _df["_tmp_"] = _df["result.objective.1.value"].cummin()
                else:
                    _df["_tmp_"] = _df["result.objective.1.value"].cummax()

                _df = _df.drop_duplicates(subset="_tmp_", keep="first").drop(columns="_tmp_")  # type: ignore

            match self.problem.objective:
                case (_, measure):
                    _low, _high = measure.bounds
                    if not np.isinf(_low) and not np.isinf(_high):
                        _df["result.objective.1.normalized_value"] = (
                            _df["result.objective.1.value"] - _low
                        ) / (_high - _low)
                case Mapping():
                    for i, (_, measure) in enumerate(self.problem.objective.items(), start=1):
                        _low, _high = measure.bounds
                        if not np.isinf(_low) and not np.isinf(_high):
                            _df[f"result.objective.{i}.normalized_value"] = (
                                _df[f"result.objective.{i}.value"] - _low
                            ) / (_high - _low)
                case _:
                    raise TypeError("Objective must be a tuple (name, measure) or a mapping")

            return _df

        @classmethod
        def from_df(cls, df: pd.DataFrame, run: Run) -> Run.Report:  # noqa: C901, PLR0915
            """Load a GLUEReport from a pandas DataFrame.

            Args:
                df: The dataframe to load from. Will subselect rows
                    that match the problem name.
                run: The run definition.
            """
            problem = run.problem

            def _row_to_result(series: pd.Series) -> Result:
                _row = series.to_dict()
                _result_values: dict[str, Any] = {}
                match problem.objective:
                    case (name, _):
                        assert int(_row["problem.objective.count"]) == 1
                        assert str(_row["problem.objective.1.name"]) == name
                        _result_values[name] = _row["result.objective.1.value"]
                    case Mapping():
                        assert int(_row["problem.objective.count"]) == len(problem.objective)
                        for i, k in enumerate(problem.objective, start=1):
                            assert str(_row[f"problem.objective.{i}.name"]) == k
                            _result_values[k] = _row[f"result.objective.{i}.value"]
                    case _:
                        raise TypeError("Objective must be a tuple (name, measure) or a mapping")

                match problem.fidelity:
                    case None:
                        assert int(_row["problem.fidelity.count"]) == 0
                        _result_fidelity = None
                    case (name, fid):
                        assert int(_row["problem.fidelity.count"]) == 1
                        assert str(_row["problem.fidelity.1.name"]) == name
                        _result_fidelity = (name, fid.kind(_row["result.fidelity.1.value"]))
                    case Mapping():
                        assert int(_row["problem.fidelity.count"]) == len(problem.fidelity)
                        _result_fidelity = {}
                        for i, (name, fid) in enumerate(problem.fidelity.items(), start=1):
                            assert str(_row[f"problem.fidelity.{i}.name"]) == name
                            _result_fidelity[name] = fid.kind(_row[f"result.fidelity.{i}.value"])
                    case _:
                        raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

                match problem.cost:
                    case None:
                        assert int(_row["problem.cost.count"]) == 0
                    case (name, _):
                        assert int(_row["problem.cost.count"]) == 1
                        assert str(_row["problem.cost.1.name"]) == name
                        _result_values[name] = _row["result.cost.1.value"]
                    case Mapping():
                        assert int(_row["problem.cost.count"]) == len(problem.cost)
                        for i, (name, _) in enumerate(problem.cost.items(), start=1):
                            assert str(_row[f"problem.cost.{i}.name"]) == name
                            _result_values[name] = _row[f"result.cost.{i}.value"]
                    case _:
                        raise TypeError("Must be a tuple (name, fidelitiy) or a mapping")

                _query_f_count = int(_row["query.fidelity.count"])
                _query_fidelity: None | tuple[str, int | float] | dict[str, int | float]
                match _query_f_count:
                    case 0:
                        _query_fidelity = None
                    case 1:
                        _name = str(_row["query.fidelity.1.name"])
                        assert run.benchmark.fidelities is not None
                        _fid = run.benchmark.fidelities[_name]
                        _val = _fid.kind(_row["query.fidelity.1.value"])
                        _query_fidelity = (_name, _val)
                    case _:
                        _query_fidelity = {}
                        for i in range(1, _query_f_count + 1):
                            _name = str(_row[f"query.fidelity.{i}.name"])
                            assert run.benchmark.fidelities is not None
                            _fid = run.benchmark.fidelities[_name]
                            _query_fidelity[_name] = _fid.kind(_row[f"query.fidelity.{i}.value"])

                return Result(
                    query=Query(
                        config=Config(config_id=str(_row["config.id"]), values=None),
                        optimizer_info=None,
                        request_trajectory=False,
                        fidelity=_query_fidelity,
                    ),
                    budget_cost=float(_row["result.budget_cost"]),
                    budget_used_total=float(_row["result.budget_used_total"]),
                    values=_result_values,
                    fidelity=_result_fidelity,
                    trajectory=None,
                )

            this_run = df["run.name"] == run.name
            run_columns = [c for c in df.columns if c.startswith("run.")]
            problem_columns = [c for c in df.columns if c.startswith("problem.")]
            dup_rows = df[this_run].drop_duplicates(subset=run_columns + problem_columns)
            if len(dup_rows) > 1:
                raise ValueError(
                    f"Multiple run rows found for the provided df for run '{run.name}'"
                    f"\n{dup_rows}"
                )

            df = df[this_run].sort_values("result.budget_used_total")  # noqa: PD901
            return cls(
                run=run,
                results=[_row_to_result(row) for _, row in df[this_run].iterrows()],
            )

        def save(self, path: Path) -> None:
            """Save the report to a path."""
            self.df().to_parquet(path, index=False)

        @classmethod
        def from_path(cls, path: Path, problem: Run) -> Run.Report:
            """Load a report from a path."""
            df = pd.read_parquet(path)  # noqa: PD901
            return cls.from_df(df, run=problem)
