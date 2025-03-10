from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import Any

import ConfigSpace as CS  # noqa: N817
import nevergrad as ng
import numpy as np
from hpoglue import Config, Optimizer, Problem, Query
from hpoglue.env import Env
from nevergrad.parametrization import parameter

if TYPE_CHECKING:
    from hpoglue import Result
    from nevergrad.optimization.base import (
        ConfiguredOptimizer as ConfNGOptimizer,
        Optimizer as NGOptimizer,
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

opt_list = sorted(ng.optimizers.registry.keys())
ext_opts = {
    "Hyperopt": ng.optimization.optimizerlib.NGOpt13,
    "CMA-ES": ng.optimization.optimizerlib.ParametrizedCMA,
    "bayes_opt": ng.optimization.optimizerlib.ParametrizedBO,
    "DE": ng.families.DifferentialEvolution,
    "EvolutionStrategy": ng.families.EvolutionStrategy,
}


class NevergradOptimizer(Optimizer):
    """The Nevergrad Optimizer."""

    name = "Nevergrad"

    env = Env(
        name="nevergrad-1.0.5",
        python_version="3.10",
        requirements=("nevergrad==1.0.5",),
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
        optimizer_name: str = "NGOpt",
    ) -> None:
        """Create a Nevergrad Optimizer instance for a given Problem."""
        self._parametrization: dict[str, parameter.Parameter]
        config_space = problem.config_space
        match config_space:
            case CS.ConfigurationSpace():
                self._parametrization = _configspace_to_nevergrad_space(config_space)
            case list():
                raise NotImplementedError("# TODO: Tabular not yet implemented for Nevergrad!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        self.seed = seed
        self.problem = problem
        self.working_directory = working_directory

        if optimizer_name not in list(ext_opts.keys()) + opt_list:
            raise ValueError(f"Unknown optimizer: {optimizer_name}!")

        self.optimizer: NGOptimizer | ConfNGOptimizer
        match problem.objectives:
            case tuple() | Mapping():
                if optimizer_name in ext_opts:
                    if optimizer_name == "Hyperopt":
                        ng_opt = ext_opts["Hyperopt"]
                    else:
                        ng_opt = ext_opts[optimizer_name]()
                    self.optimizer = ng_opt(
                        parametrization=self._parametrization,
                        budget=self.problem.budget.total, # TODO: Check this
                        num_workers=1,
                    )
                else:
                    self.optimizer = ng.optimizers.registry[optimizer_name](
                        parametrization=self._parametrization,
                        budget=self.problem.budget.total, # TODO: Check this
                        num_workers=1,
                    )
                self.optimizer.parametrization.random_state = np.random.RandomState(seed)
            case _:
                raise ValueError("Objective must be a string or a list of strings!")

        logger.info(f"Initialized Nevergrad Optimizer with {self.optimizer.name}!")

        self.history: dict[str, tuple[ng.p.Parameter, float | list[float] | None]] = {}
        self.counter = 0


    def ask(self) -> Query:
        match self.problem.fidelities:
            case None:
                config: parameter.Parameter = self.optimizer.ask()
                name = f"{self.counter}_{config.value}_{self.seed}"
                self.history[name] = (config, None)
                self.counter += 1
                return Query(
                    config=Config(
                        config_id=name,
                        values=config.value),
                        # allow_inactive_with_values=True,
                    fidelity=None,
                    optimizer_info=config,
                )
            case tuple():
                # TODO(eddiebergman): Not sure if just using
                # trial.number is enough in MF setting
                raise NotImplementedError("# TODO: Fidelity-aware not available in Nevergrad!")
            case Mapping():
                raise NotImplementedError("# TODO: Fidelity-aware not available in Nevergad!")
            case _:
                raise TypeError("Fidelity must be None or a tuple!")


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
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Nevergrad!")
            case Mapping():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for Nevergrad!")
            case _:
                raise TypeError("Cost must be None or a mapping!")

        assert isinstance(result.query.optimizer_info, parameter.Parameter)
        self.optimizer.tell(
            result.query.optimizer_info,
            _values,
        )


def _configspace_to_nevergrad_space(  # noqa: C901, PLR0912
    config_space: CS.ConfigurationSpace,
) -> dict[str, ng.p.Instrumentation]:

    if len(config_space.conditions) > 0:
        raise NotImplementedError("Conditions are not yet supported!")

    if len(config_space.forbidden_clauses) > 0:
        raise NotImplementedError("Forbiddens are not yet supported!")

    ng_space = ng.p.Dict()
    for hp in list(config_space.values()):
        match hp:
            case CS.UniformIntegerHyperparameter():
                if hp.log:
                    ng_space[hp.name] = ng.p.Log(lower=hp.lower, upper=hp.upper).set_integer_casting()
                else:
                    ng_space[hp.name] = ng.p.Scalar(lower=hp.lower, upper=hp.upper).set_integer_casting()
            case CS.UniformFloatHyperparameter():
                if hp.log:
                    ng_space[hp.name] = ng.p.Log(lower=hp.lower, upper=hp.upper)
                else:
                    ng_space[hp.name] = ng.p.Scalar(lower=hp.lower, upper=hp.upper)
            case CS.CategoricalHyperparameter():
                if hp.weights is not None:
                    raise NotImplementedError("Weights on categoricals are not yet supported!")
                ng_space[hp.name] = ng.p.Choice(hp.choices)
            case CS.Constant():
                ng_space[hp.name] = ng.p.Choice([hp.value])
            case CS.OrdinalHyperparameter():
                ng_space[hp.name] = ng.p.TransitionChoice(hp.sequence)
            case _:
                raise ValueError("Unrecognized type of hyperparameter in ConfigSpace!")

    return ng_space
