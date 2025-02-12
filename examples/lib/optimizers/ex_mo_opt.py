from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import Any

import ConfigSpace as CS  # noqa: N817
import optuna
from hpoglue import Config, Optimizer, Problem, Query
from hpoglue.env import Env

if TYPE_CHECKING:
    from hpoglue import Result
    from optuna.distributions import BaseDistribution

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Ex_MO_Opt(Optimizer):
    """Example of a Multi-objective Optimizer using Optuna NSGA II Sampler."""

    name = "Ex_MO_Opt"

    env = Env(
        name="optuna-4.0.0",
        python_version="3.10",
        requirements=("optuna==4.0.0",),
    )


    support = Problem.Support(
        fidelities=(None,),
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
        **kwargs: Any,
    ) -> None:
        """Create an Optuna Optimizer instance for a given problem statement."""
        import optuna
        from optuna.samplers import NSGAIISampler

        self._distributions: dict[str, BaseDistribution]
        config_space = problem.config_space
        match config_space:
            case CS.ConfigurationSpace():
                self._distributions = _configspace_to_optuna_distributions(config_space)
            case list():
                raise NotImplementedError("# TODO: Tabular not yet implemented for Optuna!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        self.optimizer: optuna.study.Study
        assert isinstance(problem.objectives, Mapping), (
            "Objectives must be a Mapping for this example Multi-objective optimizer!"
        )
        self.optimizer = optuna.create_study(
            sampler=NSGAIISampler(seed=seed, **kwargs),
            storage=None,
            pruner=None,
            study_name=f"{problem.name}-{seed}",
            load_if_exists=False,
            directions=[
                "minimize" if obj.minimize else "maximize"
                for obj in problem.objectives.values()
            ],
        )

        self.problem = problem
        self.working_directory = working_directory
        self._trial_lookup: dict[str, optuna.trial.Trial] = {}


    def ask(self) -> Query:
        """Ask the optimizer for a new configuration to evaluate."""
        assert self.problem.fidelities is None, "Fidelities are not supported for this optimizer!"
        trial = self.optimizer.ask(self._distributions)
        name = f"trial_{trial.number}"
        return Query(
            config=Config(config_id=name, values=trial.params),
            fidelity=None,
            optimizer_info=trial,
        )


    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a query."""
        _values = [result.values[key] for key in self.problem.objectives]

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

    if len(config_space.conditions) > 0:
        raise NotImplementedError("Conditions are not yet supported!")

    if len(config_space.forbidden_clauses) > 0:
        raise NotImplementedError("Forbiddens are not yet supported!")

    optuna_space: dict[str, BaseDistribution] = {}
    for hp in list(config_space.values()):
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
