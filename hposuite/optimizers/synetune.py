from __future__ import annotations

# https://syne-tune.readthedocs.io/en/latest/examples.html#ask-tell-interface
# https://syne-tune.readthedocs.io/en/latest/examples.html#ask-tell-interface-for-hyperband
import datetime
from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import ConfigSpace as CS  # noqa: N817

from hpoglue.env import Env
from hpoglue.config import Config
from hpoglue.optimizer import Optimizer
from hpoglue.problem import Problem
from hpoglue.query import Query

if TYPE_CHECKING:
    from syne_tune.config_space import (
        Domain,
    )
    from syne_tune.optimizer.scheduler import TrialScheduler

    from hpoglue.result import Result


class SyneTuneOptimizer(Optimizer):
    """Base class for SyneTune Optimizers."""

    name = "SyneTune_base"

    env = Env(
        name="syne_tune-0.13.0",
        python_version="3.10.12",
        requirements=("syne_tune==0.13.0",),
    )

    @abstractmethod
    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        optimizer: TrialScheduler,
    ):
        """Create a SyneTune Optimizer instance for a given problem statement."""
        working_directory.mkdir(parents=True, exist_ok=True)
        self.problem = problem
        self.optimizer = optimizer
        self._counter = 0

    def ask(self) -> Query:
        """Get a configuration from the optimizer."""
        from syne_tune.backend.trial_status import Trial

        self._counter += 1
        trial_suggestion = self.optimizer.suggest(self._counter)
        assert trial_suggestion is not None
        assert trial_suggestion.config is not None
        name = str(self._counter)
        trial = Trial(
            trial_id=self._counter,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )

        fidelity = None
        if isinstance(self.problem.fidelity, tuple):
            fidelity = trial_suggestion.config.pop(self.problem.fidelity[0])

        # TODO: How to get the fidelity??
        return Query(
            config=Config(config_id=name, values=trial.config),
            fidelity=fidelity,
            optimizer_info=trial,
        )

    def tell(self, result: Result) -> None:
        """Update the SyneTune Optimizer with the result of a query."""
        match self.problem.objective:
            case Mapping():
                results_obj_dict = {
                    key: result.values[key]
                    for key in result.values
                    if key in self.problem.objective
                }
            case (metric_name, _):
                results_obj_dict = {metric_name: result.values[metric_name]}
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        self.optimizer.on_trial_complete(
            trial=result.query.optimizer_info,  # type: ignore
            result=results_obj_dict,
        )


class SyneTuneBO(SyneTuneOptimizer):
    """SyneTune Bayesian Optimization."""

    name = "SyneTune_BO"
    support = Problem.Support(
        fidelities=(None,),
        objectives=("single",),
        cost_awareness=(None,),
        tabular=False,
    )

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        config_space: CS.ConfigurationSpace | list[Config],
        **kwargs: Any,
    ):
        """Create a SyneTune Bayesian Optimization instance for a given problem statement.

        Args:
            problem: The problem statement.
            seed: The random seed.
            working_directory: The working directory to store the results.
            config_space: The configuration space.
            **kwargs: Additional arguments for the BayesianOptimization.
        """
        from syne_tune.optimizer.baselines import BayesianOptimization

        match problem.fidelity:
            case None:
                pass
            case tuple():
                clsname = self.__class__.__name__
                # raise ValueError(f"{clsname} does not multi-fidelity spaces")
                pass
            case Mapping():
                clsname = self.__class__.__name__
                raise NotImplementedError(f"{clsname} does not support many-fidelity")
            case _:
                raise TypeError("fidelity_space must be a list, dict or None")

        match problem.cost:
            case None:
                pass
            case tuple() | Mapping():
                clsname = self.__class__.__name__
                raise ValueError(f"{clsname} does not support cost-awareness")
            case _:
                raise TypeError("cost_awareness must be a list, dict or None")

        metric_name: str
        mode: Literal["min", "max"]
        match problem.objective:
            case (name, metric):
                metric_name = name
                mode = "min" if metric.minimize else "max"
            case Mapping():
                raise NotImplementedError(
                    "# TODO: Multiobjective not yet implemented for SyneTuneBO!"
                )
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        synetune_cs: dict[str, Domain]
        match config_space:
            case CS.ConfigurationSpace():
                synetune_cs = configspace_to_synetune_configspace(config_space)
            case list():
                raise ValueError("SyneTuneBO does not support tabular benchmarks")
            case _:
                raise TypeError("config_space must be of type ConfigSpace.ConfigurationSpace")

        if isinstance(problem.fidelity, tuple):
            synetune_cs[problem.fidelity[0]] = problem.fidelity[1]  # TODO: Check this

        super().__init__(
            problem=problem,
            seed=seed,
            working_directory=working_directory,
            optimizer=BayesianOptimization(
                config_space=synetune_cs,
                metric=metric_name,
                mode=mode,
                random_seed=seed,
                **kwargs,
            ),
        )


class SyneTuneBOHB(SyneTuneOptimizer):
    """SyneTune BOHB."""

    name = "SyneTune_BOHB"
    support = Problem.Support(
        fidelities=(None, "single"),
        objectives=("single",),
        cost_awareness=(None,),
        tabular=False,
    )

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        config_space: CS.ConfigurationSpace | list[Config],
        **kwargs: Any,
    ):
        """Create a SyneTune BOHB instance for a given problem statement.

        Args:
            problem: The problem statement.
            seed: The random seed.
            working_directory: The working directory to store the results.
            config_space: The configuration space.
            **kwargs: Additional arguments for the BayesianOptimization.
        """
        from syne_tune.optimizer.baselines import BOHB

        match problem.fidelity:
            case None:
                min_fidelity = None
                max_fidelity = None
            case (_, fidelity):
                min_fidelity = fidelity.min
                max_fidelity = fidelity.max
            case Mapping():
                clsname = self.__class__.__name__
                raise NotImplementedError(f"{clsname} does not support many-fidelity")
            case _:
                raise TypeError("fidelity_space must be a list, dict or None")

        match problem.cost:
            case None:
                pass
            case tuple() | Mapping():
                clsname = self.__class__.__name__
                raise ValueError(f"{clsname} does not support cost-awareness")
            case _:
                raise TypeError("cost_awareness must be a list, dict or None")

        metric_name: str
        mode: Literal["min", "max"]
        match problem.objective:
            case (name, metric):
                metric_name = name
                mode = "min" if metric.minimize else "max"
            case Mapping():
                raise NotImplementedError(
                    "# TODO: Multiobjective not yet implemented for SyneTuneBO!"
                )
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        synetune_cs: dict[str, Domain]
        match config_space:
            case CS.ConfigurationSpace():
                synetune_cs = configspace_to_synetune_configspace(config_space)
            case list():
                raise ValueError("SyneTuneBO does not support tabular benchmarks")
            case _:
                raise TypeError("config_space must be of type ConfigSpace.ConfigurationSpace")
            
        if isinstance(problem.fidelity, tuple):
            synetune_cs[problem.fidelity[0]] = problem.fidelity[1]  # TODO: Check this

        super().__init__(
            problem=problem,
            seed=seed,
            working_directory=working_directory,
            optimizer=BOHB(
            config_space=self.synetune_cs,
            mode = mode,
            metric = metric_name,
            random_seed = seed,
            type="stopping",
            max_t = problem.fidelity[1].max,
            resource_attr = problem.fidelity[0],
            grace_period=1,
            eta=3,
            )
        )


def configspace_to_synetune_configspace(
    config_space: CS.ConfigurationSpace,
) -> dict[str, Domain | Any]:
    """Convert ConfigSpace to SyneTune config_space."""
    from syne_tune.config_space import (
        choice,
        lograndint,
        loguniform,
        ordinal,
        randint,
        uniform,
    )

    if any(config_space.get_conditions()):
        raise NotImplementedError("ConfigSpace with conditions not supported")

    if any(config_space.get_forbiddens()):
        raise NotImplementedError("ConfigSpace with forbiddens not supported")

    synetune_cs: dict[str, Domain | Any] = {}
    for hp in config_space.get_hyperparameters():
        match hp:
            case CS.OrdinalHyperparameter():
                synetune_cs[hp.name] = ordinal(hp.sequence)
            case CS.CategoricalHyperparameter() if hp.weights is not None:
                raise NotImplementedError("CategoricalHyperparameter with weights not supported")
            case CS.CategoricalHyperparameter():
                synetune_cs[hp.name] = choice(hp.choices)
            case CS.UniformIntegerHyperparameter() if hp.log:
                synetune_cs[hp.name] = lograndint(hp.lower, hp.upper)
            case CS.UniformIntegerHyperparameter():
                synetune_cs[hp.name] = randint(hp.lower, hp.upper)
            case CS.UniformFloatHyperparameter() if hp.log:
                synetune_cs[hp.name] = loguniform(hp.lower, hp.upper)
            case CS.UniformFloatHyperparameter():
                synetune_cs[hp.name] = uniform(hp.lower, hp.upper)
            case CS.Constant():
                synetune_cs[hp.name] = hp.value
            case _:
                raise ValueError(f"Hyperparameter {hp.name} of type {type(hp)} is not supported")

    return synetune_cs
