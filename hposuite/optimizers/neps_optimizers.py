from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import neps
import numpy as np
from hpoglue import Config, Optimizer, Problem, Query, Result
from hpoglue.env import Env
from neps import AskAndTell, algorithms
from neps.space.parsing import convert_configspace

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NepsOptimizer(Optimizer):
    """Base class for Neps Optimizers."""
    name = "NepsOptimizer"

    mem_req_mb = 1024

    def __init__(
        self,
        problem: Problem,
        space: neps.SearchSpace,
        optimizer: AskAndTell,
        seed: int,
        working_directory: str | Path,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        self.problem = problem
        self.space = space

        if not isinstance(working_directory, Path):
            working_directory = Path(working_directory)
        self.seed = seed
        self.working_dir = working_directory

        self.optimizer = optimizer
        self.trial_counter = 0


    def ask(self) -> Query:
        """Ask the optimizer for a new trial."""
        import copy
        trial = self.optimizer.ask() # TODO: Figure out fidelity
        fidelity = None
        _config = copy.deepcopy(trial.config)
        match self.problem.fidelities:
            case None:
                pass
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsOptimizer.")
            case (fid_name, fidelity):
                # query with max fidelity for MF optimizers
                _fid_value = _config.pop(fid_name)
                fidelity = (fid_name, _fid_value)
            case _:
                raise TypeError(
                    "Fidelity must be a tuple or a Mapping. \n"
                    f"Got {type(self.problem.fidelities)}."
                )
        self.trial_counter += 1
        return Query(
            config = Config(config_id=self.trial_counter, values=_config),
            fidelity=fidelity,
            optimizer_info=trial
        )

    @abstractmethod
    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""


class NepsBO(NepsOptimizer):
    """Bayesian Optimization in Neps."""

    name = "NepsBO"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("single",),
        cost_awareness=(None,),
        tabular=False,
    )

    env = Env(
        name="Neps-0.12.2",
        python_version="3.10",
        requirements=("neural-pipeline-search==0.12.2",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        opt = algorithms.PredefinedOptimizers["bayesian_optimization"](
            space = space
        )
        optimizer = AskAndTell(opt)
        import torch
        torch.manual_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer=optimizer,
            seed=seed,
            working_directory=working_directory,
        )

    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        match self.problem.objectives:
            case (name, obj):
                cost = obj.as_minimize(result.values[name])
            case Mapping():
                raise ValueError("NepsBO only supports single-objective problems.")
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(self.problem.objectives)}."
                )
        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=cost
        )


class NepsRW(NepsOptimizer):
    """Random Weighted Scalarization of objectives using Bayesian Optimization in Neps."""

    name = "NepsRW"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
    )

    env = Env(
        name="Neps-0.12.2",
        python_version="3.10",
        requirements=("neural-pipeline-search==0.12.2",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        scalarization_weights: Literal["equal", "random"] | Mapping[str, float] = "random",
        searcher: str = "bayesian_optimization",
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        self.searcher = searcher
        space = convert_configspace(problem.config_space)

        opt = algorithms.PredefinedOptimizers[self.searcher](
            space = space
        )
        optimizer = AskAndTell(opt)
        import torch
        torch.manual_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer=optimizer,
            seed=seed,
            working_directory=working_directory,
        )

        self.objectives = self.problem.get_objectives()

        self._rng = np.random.default_rng(seed=self.seed)
        match scalarization_weights:
            case Mapping():
                self.scalarization_weights = scalarization_weights
            case "equal":
                self.scalarization_weights = {
                    obj: 1.0/len(self.objectives) for obj in self.objectives
                }
            case "random":
                weights = self._rng.uniform(size=len(self.objectives))
                self.scalarization_weights = {
                    obj: weight/sum(weights) for obj, weight in zip(self.objectives, weights)  # noqa: B905
                }
            case _:
                raise ValueError(
                    f"Invalid scalarization_weights: {scalarization_weights}. "
                    "Expected 'equal', 'random', or a Mapping."
                )


    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        costs = {
            key: obj.as_minimize(result.values[key])
            for key, obj in self.problem.objectives.items()
        }
        scalarized_objective = sum(
            self.scalarization_weights[obj] * costs[obj] for obj in self.objectives
        )
        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=scalarized_objective
        )


class NepsHyperbandRW(NepsOptimizer):
    """Random Weighted Scalarization of objectives using Hyperband for budget allocation in Neps."""

    name = "NepsHyperbandRW"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("many"),
        cost_awareness=(None,),
        tabular=False,
    )

    env = Env(
        name="Neps-0.12.2",
        python_version="3.10",
        requirements=("neural-pipeline-search==0.12.2",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        scalarization_weights: Literal["equal", "random"] | Mapping[str, float] = "random",
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the optimizer."""
        space = convert_configspace(problem.config_space)

        _fid = None
        min_fidelity: int | float
        max_fidelity: int | float
        match problem.fidelities:
            case None:
                raise ValueError("NepsHyperbandRW requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsHyperbandRW.")
            case (fid_name, fidelity):
                _fid = fidelity
                min_fidelity = fidelity.min
                max_fidelity = fidelity.max
                match _fid.kind:
                    case _ if _fid.kind is int:
                        space.fidelities = {
                            f"{fid_name}": neps.Integer(
                                lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                            )
                        }
                    case _ if _fid.kind is float:
                        space.fidelities = {
                            f"{fid_name}": neps.Float(
                                lower=min_fidelity, upper=max_fidelity, is_fidelity=True
                            )
                        }
                    case _:
                        raise TypeError(
                            f"Invalid fidelity type: {type(_fid.kind).__name__}. "
                            "Expected int or float."
                        )
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")


        opt = algorithms.PredefinedOptimizers["hyperband"](
            space = space
        )
        optimizer = AskAndTell(opt)
        import torch
        torch.manual_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer=optimizer,
            seed=seed,
            working_directory=working_directory,
        )

        self.objectives = self.problem.get_objectives()
        self._rng = np.random.default_rng(seed=self.seed)
        match scalarization_weights:
            case Mapping():
                self.scalarization_weights = scalarization_weights
            case "equal":
                self.scalarization_weights = {
                    obj: 1.0/len(self.objectives) for obj in self.objectives
                }
            case "random":
                weights = self._rng.uniform(size=len(self.objectives))
                self.scalarization_weights = {
                    obj: weight/sum(weights) for obj, weight in zip(self.objectives, weights)  # noqa: B905
                }
            case _:
                raise ValueError(
                    f"Invalid scalarization_weights: {scalarization_weights}. "
                    "Expected 'equal', 'random', or a Mapping."
                )


    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        costs = {
            key: obj.as_minimize(result.values[key])
            for key, obj in self.problem.objectives.items()
        }
        scalarized_objective = sum(
            self.scalarization_weights[obj] * costs[obj] for obj in self.objectives
        )
        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=scalarized_objective
        )