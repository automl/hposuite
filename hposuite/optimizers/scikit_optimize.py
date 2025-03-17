from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import ConfigSpace as CS  # noqa: N817
import skopt
from hpoglue import Config, Optimizer, Problem, Query
from hpoglue.env import Env

if TYPE_CHECKING:
    from hpoglue import Result
    from skopt.space.space import Space

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


base_estimators = ["GP", "RF", "ET", "GBRT"]
acq_funcs = ["LCB", "EI", "PI", "gp_hedge"]
acq_optimizers = ["sampling", "lbfgs", "auto"]

class SkoptOptimizer(Optimizer):
    """The Scikit_Optimize Optimizer."""

    name = "Scikit_Optimize"

    env = Env(
        name="scikit_optimize-0.10.2",
        python_version="3.10",
        requirements=("scikit-optimize==0.10.2",),
    )

    support = Problem.Support(
        fidelities=(None,),  # NOTE: Skopt does not support multi-fidelity optimization
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
        base_estimator: str = "GP",
        acq_func: str = "gp_hedge",
        acq_optimizer: str = "auto",
    ) -> None:
        """Create an Skopt Optimizer instance for a given problem statement."""
        self.config_space = problem.config_space
        self._space: list[Space]
        match self.config_space:
            case CS.ConfigurationSpace():
                self._space = _configspace_to_skopt_space(self.config_space)
            case list():
                raise ValueError("SciKit-Optimize does not support list-type config spaces!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        assert base_estimator in base_estimators, f"base_estimator must be one of {base_estimators}"

        assert acq_func in acq_funcs, f"acq_func must be one of {acq_funcs}"

        assert acq_optimizer in acq_optimizers, f"acq_optimizer must be one of {acq_optimizers}"

        self.optimizer: skopt.optimizer.Optimizer
        self.optimizer = skopt.optimizer.Optimizer(
            dimensions=self._space,
            base_estimator=base_estimator,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            random_state=seed,
            n_initial_points=5
        )

        self.problem = problem
        self.working_directory = working_directory
        self.trial_counter = 0


    def ask(self) -> Query:
        match self.problem.fidelities:
            case None:
                config = self.optimizer.ask()
                config_values = {
                    hp.name: value
                    for hp, value in zip(
                        self.config_space.get_hyperparameters(),
                        config,
                        strict=False
                    )
                }
                assert list(config_values.keys()) == \
                    list(self.config_space.get_hyperparameter_names())
                assert list(config_values.keys()) == [hp.name for hp in self._space]
                name = f"trial_{self.trial_counter}"
                self.trial_counter += 1
                return Query(
                    config=Config(config_id=name, values=config_values),
                    fidelity=None,
                    optimizer_info=None,
                )
            case tuple():
                raise ValueError("Multi-fidelity optimization not supported by Scikit_Optimize!")
            case Mapping():
                raise ValueError("Many-fidelity optimization not supported by Scikit_Optimize!")
            case _:
                raise TypeError("Fidelity must be None, a tuple or a Mapping!")


    def tell(self, result: Result) -> None:
        match self.problem.objectives:
            case (name, _):
                _values = result.values[name]
            case Mapping():
                raise ValueError("Multiobjective not supported by Scikit_Optimize!")
            case _:
                raise TypeError("Objective must be a string or a list of strings!")

        match self.problem.costs:
            case None:
                pass
            case tuple():
                raise ValueError("Cost-aware optimization not supported by Scikit_Optimize!")
            case Mapping():
                raise ValueError("Cost-aware optimization not supported by Scikit_Optimize!")
            case _:
                raise TypeError("Cost must be None or a mapping!")

        _ = self.optimizer.tell(
            list(result.query.config.values.values()),
            _values
        )


def _configspace_to_skopt_space(  # noqa: C901
    config_space: CS.ConfigurationSpace,
) -> dict[str, Space]:
    import numpy as np
    from skopt.space.space import Categorical, Integer, Real

    if len(config_space.conditions) > 0:
        raise NotImplementedError("Conditions are not yet supported!")

    if len(config_space.forbidden_clauses) > 0:
        raise NotImplementedError("Forbiddens are not yet supported!")

    skopt_space: list[float] = []
    for hp in list(config_space.values()):
        match hp:
            case CS.UniformIntegerHyperparameter() if hp.log:
                skopt_space.append(Integer(hp.lower, hp.upper, name=hp.name, log=hp.log))
            case CS.UniformIntegerHyperparameter():
                skopt_space.append(Integer(hp.lower, hp.upper, name=hp.name))
            case CS.UniformFloatHyperparameter() if hp.log:
                skopt_space.append(Real(hp.lower, hp.upper, name=hp.name, log=hp.log))
            case CS.UniformFloatHyperparameter():
                skopt_space.append(Real(hp.lower, hp.upper, name=hp.name))
            case CS.CategoricalHyperparameter() if hp.weights is not None:
                weights = np.asarray(hp.weights) / np.sum(hp.weights)
                skopt_space.append(Categorical(hp.choices, name=hp.name, prior=weights))
            case CS.CategoricalHyperparameter():
                skopt_space.append(Categorical(hp.choices, name=hp.name))
            case CS.Constant():
                skopt_space.append(Categorical([hp.value], name=hp.name))
            case CS.OrdinalHyperparameter():
                skopt_space.append(Categorical(list(hp.sequence), name=hp.name))
            case _:
                raise ValueError(
                    f"Unrecognized type of hyperparameter in ConfigSpace: {hp.__class__.__name__}!"
                )

    return skopt_space

