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
from hpoglue.utils import configpriors_to_dict, dict_to_configpriors

from hposuite.benchmarks import BENCHMARKS
from hposuite.constants import DEFAULT_STUDY_DIR
from hposuite.optimizers import OPTIMIZERS
from hposuite.run import Run
from hposuite.utils import HPOSUITE_EDITABLE

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
    """A class to represent a hposuite Study consisting of multiple Runs."""

    name: str | None = None
    """The name of the study."""

    output_dir: str | Path = DEFAULT_STUDY_DIR
    """The outer directory where all Studies are stored"""

    study_dir: Path = field(init=False)
    """The directory for storing the Study results.
        study_dir = output_dir/study_name
    """

    study_yaml_path: Path = field(init=False)
    """The path to the study configuration YAML file."""

    optimizers: list[OptWithHps] = field(init=False)
    """The list of optimizers used in the study."""

    benchmarks: list[BenchWith_Objs_Fids] = field(init=False)
    """The benchmarks used in the study."""

    experiments: list[Run]
    """The list of experiments in the study."""

    seeds: Iterable[int] | int | None = None
    """The seeds used in the study."""

    num_seeds: int = 1
    """The number of seeds used in the study."""

    budget: int = 50
    """The budget for the study."""

    group_by: Literal["opt", "bench", "opt_bench", "seed", "mem"] | None = None
    """The grouping method to dump run commands for the study."""

    continuations: bool = field(init=False)
    """Whether to use continuations for the study."""

    def __post_init__(self):  # noqa: C901

        self.optimizers = []
        self.benchmarks = []
        seeds = set()

        opt_keys = []
        bench_keys = {}
        continuations = 0
        for run in self.experiments:
            continuations += run.problem.continuations
            opt_name = run.name.split("benchmark")[0]
            if opt_name not in opt_keys:
                self.optimizers.append(
                    (
                        run.optimizer,
                        run.optimizer_hyperparameters or {},
                    )
                )
                opt_keys.append(opt_name)

            # TODO: This doesn't uniquely identify runs with priors (Problem level issue).

            _store_val = {
                "objectives": run.problem.get_objectives(),
                "fidelities": run.problem.get_fidelities(),
                "costs": run.problem.get_costs(),
                "priors": run.problem.priors,
            }

            bench = bench_keys.setdefault(run.benchmark.name, (run.benchmark, []))

            if _store_val not in bench[1]:
                for variant in bench[1]:
                    if variant["objectives"] == _store_val["objectives"]:
                        for key in ["fidelities", "costs", "priors"]:
                            variant[key] = variant[key] or _store_val[key]
                        break
                else:
                    bench[1].append(_store_val)

            seeds.add(run.seed)

        for _, v in bench_keys.items():
            [self.benchmarks.append(
                (
                    v[0],
                    var,
                )
            ) for var in v[1]]

        self.continuations = continuations > 0

        if self.seeds is None:
            self.seeds = list(seeds)
        self.num_seeds = len(self.seeds) if isinstance(self.seeds, Iterable) else 1

        name_parts: list[str] = []
        name_parts.append(";".join([f"{opt[0].name}{opt[-1]}" for opt in self.optimizers]))
        name_parts.append(";".join([f"{bench[0].name}{bench[-1]}" for bench in self.benchmarks]))
        name_parts.append(f"seeds={self.seeds}")
        name_parts.append(f"budget={self.budget}")

        if self.name is None:
            self.name = hashlib.sha256((".".join(name_parts)).encode()).hexdigest()

        self.study_dir = self.output_dir / self.name
        self.study_yaml_path = self.study_dir / "study_config.yaml"
        self.write_yaml()
        logger.info(f"Created study at {self.study_dir.absolute()}")


        if len(self.experiments) > 1:
            self._dump_runs(
                group_by=self.group_by,
                exp_dir=self.study_dir,
            )


    def _update_study(
        self,
        *,
        new_seeds: Iterable[int],
    ) -> None:
        """Update the study with new seeds."""
        more_experiments = Study.generate(
            optimizers=self.optimizers,
            benchmarks=self.benchmarks,
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
        for run in self.experiments:
            run._set_paths(self.study_dir)
            run.write_yaml()

        _optimizers = [{"name": opt[0].name, "hyperparameters": opt[1]} for opt in self.optimizers]
        _benchmarks = []
        for bench in self.benchmarks:
            _priors = None
            if bench[1]["priors"]:
                _priors = configpriors_to_dict(bench[1]["priors"])
                _priors = list(_priors)
            _benchmarks.append(
                {
                    "name": bench[0].name,
                    "objectives": bench[1]["objectives"],
                    "fidelities": bench[1]["fidelities"],
                    "costs": bench[1]["costs"],
                    "priors": _priors,
                }
            )

        return {
            "study_name": self.name,
            "output_dir": str(self.output_dir.absolute()),
            "optimizers": _optimizers,
            "benchmarks": _benchmarks,
            "seeds": self.seeds,
            "num_seeds": self.num_seeds,
            "budget": self.budget,
            "continuations": self.continuations,
        }


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Study:  # noqa: C901, PLR0912
        """Create a Study object from a dictionary."""
        # TODO: Add documentation for usage
        # TODO: Test for all allowed types in Study.create_study()
        _optimizers = []
        for opt in data["optimizers"]:
            match opt:
                case Mapping():
                    assert "name" in opt, f"Optimizer name not found in {opt}"
                    if len(opt) == 1 or not opt.get("hyperparameters"):
                        _optimizers.append((opt["name"], {}))
                    else:
                        _optimizers.append(tuple(opt.values()))
                case str():
                    _optimizers.append((opt, {}))
                case tuple():
                    assert len(opt) <= 2, "Each Optimizer must only have a name and hyperparameters"  # noqa: PLR2004
                    assert isinstance(opt[0], str), "Expected str for optimizer name"
                    if len(opt) == 1:
                        _optimizers.append((opt[0], {}))
                    else:
                        assert isinstance(opt[1], Mapping), (
                            "Expected Mapping for Optimizer hyperparameters"
                        )
                        _optimizers.append(opt)
                case _:
                    raise ValueError(
                        f"Invalid type for optimizer: {type(opt)}. "
                        "Expected Mapping, str or tuple"
                    )

        _benchmarks = []
        for bench in data["benchmarks"]:
            match bench:
                case Mapping():
                    assert "name" in bench, f"Benchmark name not found in {bench}"
                    assert "objectives" in bench, f"Benchmark objectives not found in {bench}"
                    _benchmarks.append(
                        (
                            bench["name"],
                            {
                                "objectives": bench["objectives"],
                                "fidelities": bench.get("fidelities"),
                                "costs": bench.get("costs"),
                                "priors": bench.get("priors"),
                            }
                        )
                    )
                case str():
                    _benchmarks.append(bench)
                case tuple():
                    assert len(bench) == 2, "Each Benchmark must only have a name and a Mapping"  # noqa: PLR2004
                    assert isinstance(bench[0], str), "Expected str for benchmark name"
                    assert isinstance(bench[1], Mapping), (
                        "Expected Mapping for Benchmark objectives and fidelities"
                    )
                    _benchmarks.append(bench)
                case _:
                    raise ValueError(
                        f"Invalid type for benchmark: {type(bench)}. "
                        "Expected Mapping, str or tuple"
                    )


        return create_study(
            name=data.get("study_name"),
            output_dir=data.get("output_dir"),
            optimizers=_optimizers,
            benchmarks=_benchmarks,
            seeds=data.get("seeds"),
            num_seeds=data.get("num_seeds", 1),
            budget=data.get("budget", 50),
            group_by=data.get("group_by"),
            on_error=data.get("on_error", "warn"),
        )


    @classmethod
    def from_yaml(cls, yaml_file: Path | str) -> Study:
        """Create a Study instance from a YAML file.

        Args:
            yaml_file: The path to the YAML file containing the study data.

        Returns:
            An instance of the Study class populated with data from the YAML file.
        """
        if isinstance(yaml_file, str):
            yaml_file = Path(yaml_file)
        with yaml_file.open("r") as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)


    def write_yaml(self) -> None:
        """Write the study config to a YAML file."""
        self.study_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with self.study_yaml_path.open("w") as file:
            yaml.dump(self.to_dict(), file, sort_keys=False)


    @classmethod
    def from_problems(  # noqa: C901, PLR0912, PLR0915
        cls,
        problems: Problem | list[Problem],
        seeds: Iterable[int] | None = None,
        num_seeds: int = 1,
        name: str | None = None,
        output_dir: Path | str | None = DEFAULT_STUDY_DIR,
        group_by: Literal["opt", "bench", "opt_bench", "seed", "mem"] | None = None,
        on_error: Literal["warn", "raise", "ignore"] = "warn"
    ) -> Study:
        """Create a Study object from a list of Problems.

        Args:
            name: The name of the study.

            output_dir: The main output directory where the hposuite studies are saved.

            problems: The list of problems to generate runs for.

            seeds: The seed or seeds to use for the experiment.

            num_seeds: The number of seeds to generate.

            group_by: The grouping to use for the runs dump.

            on_error: The method to handle errors while generating runs for the study.

        Returns:
            A Study object.
        """
        if not isinstance(problems, list):
            problems = [problems]


        match output_dir:
            case None:
                output_dir = DEFAULT_STUDY_DIR
            case str():
                output_dir = Path(output_dir)
            case Path():
                pass
            case _:
                raise TypeError(f"Invalid type for output_dir: {type(output_dir)}")


        if not seeds and not (num_seeds and num_seeds > 0):
            match on_error:
                case "raise":
                    raise ValueError("At least one seed or num_seeds must be provided")
                case "warn" | "ignore":
                    warnings.warn(
                        "At least one seed or num_seeds must be provided"
                        "Continuing with num_seeds=1",
                        stacklevel=2
                    )
                    num_seeds = 1
                case _:
                    raise TypeError(
                        f"Invalid value for on_error: {on_error}"
                    )

        # Generate seeds
        match seeds:
            case None:
                seeds = cls.generate_seeds(num_seeds)
            case Iterable():
                seeds = list(set(seeds))
            case int():
                seeds = [seeds]


        _budget = problems[0].budget.total
        _problems: list[Problem] = []
        for _problem in problems:
            try:
                if not isinstance(_problem, Problem):
                    raise TypeError(
                        f"Expected Problem or list[Problem], got {type(_problem)}"
                    )
                if _problem.budget.total != _budget:
                    raise ValueError("All problems must have the same budget")
                _problems.append(_problem)
            except Exception as e:
                match on_error:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case "warn":
                        warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
                        continue


        #Generate runs from problems
        _runs_per_problem: Mapping[str, Run] = {}
        for _problem, _seed in product(_problems, seeds):
            try:
                _run = Run(
                        problem=_problem,
                        seed=_seed,
                    )
                if _run.name not in _runs_per_problem:
                    _runs_per_problem[_run.name] = _run
            except ValueError as e:
                match on_error:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case "warn":
                        warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
                        continue
        _runs_per_problem = list(_runs_per_problem.values())

        logger.info(f"Generated {len(_runs_per_problem)} runs")

        return cls(
            name=name,
            output_dir=output_dir,
            experiments=_runs_per_problem,
            seeds=seeds,
            num_seeds=num_seeds,
            budget=_budget,
            group_by=group_by
        )


    @classmethod
    def generate_seeds(
        cls,
        num_seeds: int,
        offset: int = 0, # To offset number of seeds
    ) -> list[int]:
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

            multi_objective_generation: The method to generate multiple objectives.

            on_error: The method to handle errors.
                * "warn": Log a warning and continue.
                * "raise": Raise an error.
                * "ignore": Ignore the error and continue.

            continuations: Whether to use continuations for the run.

        Returns:
            A list of Run objects.

        Note:
            `Study.generate()` only generates the Run objects.
            It does not set absolute paths or write yaml files.
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
                costs: int | str | list[str] | None

                objectives = objs_fids.get("objectives", 1)
                if isinstance(objectives, list) and len(objectives) == 1:
                    objectives = objectives[0]

                fidelities = objs_fids.get("fidelities", None)
                if isinstance(fidelities, list) and len(fidelities) == 1:
                    fidelities = fidelities[0]

                costs = objs_fids.get("costs", None)
                if isinstance(costs, list) and len(costs) == 1:
                    costs = costs[0]

                priors = objs_fids.get("priors")

                match fidelities, bench.fidelities:
                    case None, None:
                        fidelities = None
                    case None, _:
                        match opt.support.fidelities[0]:
                            case "single":
                                fidelities = 1
                            case "many":
                                fidelities = len(bench.fidelities)
                            case None:
                                fidelities = None
                            case _:
                                raise ValueError("Invalid fidelity support")
                    case str() | int() | list(), _:
                        match opt.support.fidelities[0]:
                            case str():
                                pass
                            case None:
                                fidelities = None
                            case _:
                                raise ValueError("Invalid fidelity support")
                    case _:
                        raise ValueError(
                            f"Invalid fidelity type: {type(fidelities)}. "
                            f"Expected None, str, int or list"
                        )

                if priors:
                    priors = dict_to_configpriors(priors)


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
                    priors=priors,
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


        # NOTE: REDUNDANCY CHECK: _runs_per_problem is a dict now to avoid duplicate runs esp.
        # for eg. in case a BO Opt being used in combination with a
        # MF Opt on a benchmark with multiple fidelities -> explicitly adding different fidelities
        # in multiple benchmark instances would create redundant problems if BO Opts are present.
        _runs_per_problem: Mapping[str, Run] = {}
        for _problem, _seed in product(_problems, seeds):
            try:
                _run = Run(
                        problem=_problem,
                        seed=_seed,
                    )
                if _run.name not in _runs_per_problem:
                    _runs_per_problem[_run.name] = _run
            except ValueError as e:
                match on_error:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case "warn":
                        warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
                        continue
        _runs_per_problem = list(_runs_per_problem.values())

        logger.info(f"Generated {len(_runs_per_problem)} runs")

        return _runs_per_problem


    def _group_by(  # noqa: C901
        self,
        group_by: Literal["opt", "bench", "opt_bench", "seed", "mem"] | None,
    ) -> Mapping[str, list[Run]]:
        """Group the runs by the specified group."""
        _grouped_runs = {}
        filtered_runs = {}

        # To avoid duplicate run dumps because we do not support having
        # objectives, fidelities and priors in the hpoglue CLI run command
        for run in self.experiments:
            _run_hash = ".".join(
                [
                    run.optimizer.name,
                    run.benchmark.name,
                    str(run.seed),
                    f"{run.mem_req_mb}mb",
                ]
            )
            if _run_hash not in filtered_runs:
                filtered_runs[_run_hash] = run

        if group_by is None:
            return {"all": list(filtered_runs.values())}

        for run in filtered_runs.values():
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
            logger.info(f"Dumped experiments to {(exp_dir / f'dump_{key}.txt').absolute()}")


    def optimize(  # noqa: C901, PLR0912
        self,
        *,
        exec_type: Literal["sequential", "parallel"] = "sequential",
        add_seeds: Iterable[int] | int | None = None,
        add_num_seeds: int | None = None,
        overwrite: bool = False,
        continuations: bool = True,
        auto_env_handling: bool = False,
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

            auto_env_handling: Whether to automatically create and use isolated run environments.

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
                add_seeds = list(set(add_seeds))
                for seed in add_seeds:
                    if seed not in self.seeds:
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
                        if auto_env_handling:
                            run.create_env(hposuite=f"-e {HPOSUITE_EDITABLE}")
                        logger.info(f"Running experiment {i}/{len(self.experiments)}")
                        run.run(
                            continuations=continuations,
                            overwrite=overwrite,
                            progress_bar=False
                        )
                case "parallel":
                    raise NotImplementedError("Parallel execution not implemented yet!")
                case _:
                    raise ValueError(f"Invalid exceution type: {exec_type}")
        else:
            run = self.experiments[0]
            if auto_env_handling:
                run.create_env(hposuite=f"-e {Path.cwd()}")
            logger.info("Running single experiment")
            run.run(
                continuations=continuations,
                overwrite=overwrite,
                progress_bar=False,
            )
        logger.info(f"Completed study with {len(self.experiments)} runs")


def create_study(  # noqa: C901, PLR0912, PLR0915
    *,
    name: str | None = None,
    output_dir: str| Path | None = DEFAULT_STUDY_DIR,
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
                                "costs": [list of costs] | number of costs | None,
                                "priors": {
                                    "objective": Prior
                                }
                            }
                        )
                    ]

        seeds: The seed or seeds to use for the experiment.

        num_seeds: The number of seeds to generate.

        budget: The budget for the experiment.

        group_by: The grouping to use for the runs dump.

        on_error: The method to handle errors while generating runs for the study.

    Returns:
        A Study object.
    """
    # TODO: Add redundancy checks for repeated optimizers and benchmarks with same configs
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


    if not seeds and not (num_seeds and num_seeds > 0):
        match on_error:
            case "raise":
                raise ValueError("At least one seed or num_seeds must be provided")
            case "warn" | "ignore":
                warnings.warn(
                    "At least one seed or num_seeds must be provided"
                    "Continuing with num_seeds=1",
                    stacklevel=2
                )
                num_seeds = 1
            case _:
                raise TypeError(
                    f"Invalid value for on_error: {on_error}"
                )


    _optimizers: list[OptWithHps] = []
    for optimizer in optimizers:
        match optimizer:
            case str():
                assert optimizer in OPTIMIZERS, (
                    f"Optimizer must be one of {OPTIMIZERS.keys()}\n"
                    f"Found {optimizer}"
                )
                _optimizers.append(OPTIMIZERS[optimizer])
            case tuple():
                opt, hps = optimizer
                match opt:
                    case str():
                        assert opt in OPTIMIZERS, (
                            f"Optimizer must be one of {OPTIMIZERS.keys()}\n"
                            f"Found {opt}"
                        )
                        _optimizers.append((OPTIMIZERS[opt], hps))
                    case type():
                        _optimizers.append((opt, hps))
                    case _:
                        raise TypeError(f"Unknown Optimizer type {type(opt)}")
            case type():
                _optimizers.append(optimizer)
            case _:
                raise TypeError(f"Unknown Optimizer type {type(optimizers)}")


    _benchmarks: list[BenchWith_Objs_Fids] = []
    for benchmark in benchmarks:
        match benchmark:
            case str():
                assert benchmark in BENCHMARKS, (
                    f"Benchmark must be one of {BENCHMARKS.keys()}\n"
                    f"Found {benchmark}"
                )
                if not isinstance(BENCHMARKS[benchmark], FunctionalBenchmark):
                    _benchmarks.append(BENCHMARKS[benchmark])
                else:
                    _benchmarks.append(BENCHMARKS[benchmark].desc)
            case BenchmarkDescription():
                _benchmarks.append(benchmark)
            case FunctionalBenchmark():
                _benchmarks.append(benchmark.desc)
            case tuple():
                bench, bench_hps = benchmark
                match bench:
                    case str():
                        assert bench in BENCHMARKS, (
                            f"Benchmark must be one of {BENCHMARKS.keys()}\n"
                            f"Found {bench}"
                        )
                        if not isinstance(BENCHMARKS[bench], FunctionalBenchmark):
                            _benchmarks.append((BENCHMARKS[bench], bench_hps))
                        else:
                            _benchmarks.append((BENCHMARKS[bench].desc, bench_hps))
                    case BenchmarkDescription():
                        _benchmarks.append((bench, bench_hps))
                    case FunctionalBenchmark():
                        _benchmarks.append((bench.desc, bench_hps))
                    case _:
                        raise TypeError(f"Unknown Benchmark type {type(benchmark)}")
            case _:
                raise TypeError(f"Unknown Benchmark type {type(benchmarks)}")

    if isinstance(seeds, Iterable):
        seeds = list(set(seeds))

    experiments = Study.generate(
        optimizers=_optimizers,
        benchmarks=_benchmarks,
        budget=budget,
        seeds=seeds,
        num_seeds=num_seeds,
        on_error=on_error,
    )

    return Study(
        name=name,
        output_dir=output_dir,
        experiments=experiments,
        seeds=seeds,
        num_seeds=num_seeds,
        budget=budget,
        group_by=group_by,
    )