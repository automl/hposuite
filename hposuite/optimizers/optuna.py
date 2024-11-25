from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import Any, override

import ConfigSpace as CS  # noqa: N817
import optuna
from hpoglue.config import Config
from hpoglue.optimizer import Optimizer
from hpoglue.problem import Problem
from hpoglue.query import Query

if TYPE_CHECKING:
    from hpoglue.result import Result
    from optuna.distributions import BaseDistribution

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OptunaOptimizer(Optimizer):
    """The Optuna Optimizer.

    # TODO: Document me
    """

    name = "Optuna"
    support = Problem.Support(
        fidelities=(None,),  # TODO: Implement fidelity support
        objectives=("single", "many"),
        cost_awareness=(None,),
        tabular=False,
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,
        config_space: list[Config] | CS.ConfigurationSpace,
        **kwargs: Any,
    ) -> None:
        """Create an Optuna Optimizer instance for a given problem statement."""
        import optuna
        from optuna.samplers import NSGAIISampler, TPESampler

        self._distributions: dict[str, BaseDistribution]
        match config_space:
            case CS.ConfigurationSpace():
                self._distributions = _configspace_to_optuna_distributions(config_space)
            case list():
                raise NotImplementedError("# TODO: Tabular not yet implemented for Optuna!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        self.optimizer: optuna.study.Study
        match problem.objectives:
            case (_, objective):
                self.optimizer = optuna.create_study(
                    sampler=TPESampler(seed=seed, **kwargs),
                    storage=None,
                    pruner=None,  # TODO(eddiebergman): Figure out how to use this for MF
                    study_name=f"{problem.name}-{seed}",
                    load_if_exists=False,
                    direction="minimize" if objective.minimize else "maximize",
                )
            case Mapping():
                self.optimizer = optuna.create_study(
                    sampler=NSGAIISampler(seed=seed, **kwargs),
                    storage=None,
                    pruner=None,  # TODO(eddiebergman): Figure out how to use this for MF
                    study_name=f"{problem.name}-{seed}",
                    load_if_exists=False,
                    directions=[
                        "minimize" if obj.minimize else "maximize"
                        for obj in problem.objectives.values()
                    ],
                )
            case _:
                raise ValueError("Objective must be a string or a list of strings!")

        self.problem = problem
        self.working_directory = working_directory
        self._trial_lookup: dict[str, optuna.trial.Trial] = {}

    @override
    def ask(self) -> Query:
        match self.problem.fidelities:
            case None:
                trial = self.optimizer.ask(self._distributions)
                name = f"trial_{trial.number}"
                return Query(
                    config=Config(config_id=name, values=trial.params),
                    fidelity=None,
                    optimizer_info=trial,
                )
            case tuple():
                # TODO(eddiebergman): Not sure if just using
                # trial.number is enough in MF setting
                raise NotImplementedError("# TODO: Fidelity-aware not yet implemented for Optuna!")
            case Mapping():
                raise NotImplementedError("# TODO: Fidelity-aware not yet implemented for Optuna!")
            case _:
                raise TypeError("Fidelity must be None or a tuple!")

    @override
    def tell(self, result: Result) -> None:
        match self.problem.objectives:
            case (name, _):
                _values = result.values[name]
            case Mapping():
                _values = [result.values[key] for key in self.problem.objectives]
            case _:
                raise TypeError("Objective must be a string or a list of strings!")

        match self.problem.costs:
            case None:
                pass
            case tuple():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Optuna!")
            case Mapping():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Optuna!")
            case _:
                raise TypeError("Cost must be None or a mapping!")

        assert isinstance(result.query.optimizer_info, optuna.trial.Trial)
        self.optimizer.tell(
            trial=result.query.optimizer_info,
            values=_values,
            state=optuna.trial.TrialState.COMPLETE,
            skip_if_finished=False,
        )


def _configspace_to_optuna_distributions(
    config_space: CS.ConfigurationSpace,
) -> dict[str, BaseDistribution]:
    from optuna.distributions import (
        CategoricalDistribution as Cat,
        FloatDistribution as Float,
        IntDistribution as Int,
    )

    if len(config_space.get_conditions()) > 0:
        raise NotImplementedError("Conditions are not yet supported!")

    if len(config_space.get_forbiddens()) > 0:
        raise NotImplementedError("Forbiddens are not yet supported!")

    optuna_space: dict[str, BaseDistribution] = {}
    for hp in config_space.get_hyperparameters():
        match hp:
            case CS.UniformIntegerHyperparameter():
                optuna_space[hp.name] = Int(hp.lower, hp.upper, log=hp.log)
            case CS.UniformFloatHyperparameter():
                optuna_space[hp.name] = Float(hp.lower, hp.upper, log=hp.log)
            case CS.CategoricalHyperparameter():
                if hp.weights is not None:
                    raise NotImplementedError("Weights on categoricals are not yet supported!")
                optuna_space[hp.name] = Cat(hp.choices)
            case CS.Constant():
                optuna_space[hp.name] = Cat([hp.value])
            case CS.OrdinalHyperparameter():
                raise NotImplementedError("Ordinal hyperparameters are not yet supported!")
            case _:
                raise ValueError("Unrecognized type of hyperparameter in ConfigSpace!")

    return optuna_space
