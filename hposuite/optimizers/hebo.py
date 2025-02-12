from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ConfigSpace import ConfigurationSpace
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hpoglue import Config, Optimizer, Problem, Query
from hpoglue.env import Env

if TYPE_CHECKING:
    from hpoglue import Result

class HEBOOptimizer(Optimizer):
    """The HEBO Optimizer."""

    name = "HEBO"

    env = Env(
        name="hebo-0.3.6",
        python_version="3.10",
        requirements=("hebo==0.3.6",),
    )

    support = Problem.Support(
        fidelities=(None,),
        objectives=("single", "many"),
        cost_awareness=(None,),
        tabular=False,
        priors=False,
    )

    mem_req_mb = 1024


    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: Path,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        """Initialize the HEBO optimizer."""
        self.problem = problem
        self.config_space: ConfigurationSpace = problem.config_space
        self._space: DesignSpace
        match self.config_space:
            case ConfigurationSpace():
                self._space = _configspace_to_hebo_designspace(self.config_space)
            case list():
                raise ValueError("HEBO does not support list-type config spaces!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        self.optimizer: HEBO = HEBO(
            space=self._space,
            scramble_seed=seed,
            **kwargs
        )
        self._trial_counter = 0


    def ask(self) -> Query:
        """Asks the optimizer to suggest a new configuration to evaluate.

        Returns:
            Query: A new query containing the suggested configuration, fidelity, and
                    other relevant optimizer information.
        """
        match self.problem.fidelities:
            case None:
                suggestion = self.optimizer.suggest(1)
                hyp = suggestion.iloc[0].to_dict()

                from ConfigSpace.hyperparameters import OrdinalHyperparameter
                for k in hyp:
                    hp_type = self._space.paras[k]
                    if hp_type.is_numeric and hp_type.is_discrete and not np.isnan(hyp[k]):
                        hyp[k] = int(hyp[k])
                        # Now we need to check if it is an ordinal hp
                        hp_k = self.config_space[k]
                        if isinstance(hp_k, OrdinalHyperparameter):
                            hyp[k] = hp_k.sequence[hyp[k]]
                return Query(
                    config=Config(config_id=self._trial_counter, values=hyp),
                    fidelity=None,
                    optimizer_info=suggestion,
                )
            case tuple():
                raise ValueError("HEBO does not support multi-fidelity optimization!")
            case Mapping():
                raise ValueError("HEBO does not support many-fidelity optimization!")
            case _:
                raise TypeError("Fidelity must be None, a tuple or a Mapping!")


    def tell(self, result: Result) -> None:
        """Updates the optimizer with the result of an evaluation.

        Parameters:
        result: The result of an evaluation containing the objective values
                and other relevant information.
        """
        match self.problem.objectives:
            case (name, _):
                _values = np.asarray([result.values[name]])
            case Mapping():
                raise ValueError("# TODO: Multiobjective not yet implemented for HEBO")
            case _:
                raise TypeError("Objective must be a string or a list of strings!")

        match self.problem.costs:
            case None:
                pass
            case tuple():
                raise ValueError("Cost-aware optimization not supported by HEBO!")
            case Mapping():
                raise ValueError("Cost-aware optimization not supported by HEBO!")
            case _:
                raise TypeError("Cost must be None or a Mapping!")

        self.optimizer.observe(
            result.query.optimizer_info,
            _values,
        )


def _configspace_to_hebo_designspace(config_space: ConfigurationSpace) -> DesignSpace:  # noqa: C901, PLR0912
    """Convert ConfigSpace to HEBO design space."""
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        Constant,
        FloatHyperparameter,
        IntegerHyperparameter,
        OrdinalHyperparameter,
    )
    if len(config_space.conditions) > 0:
        raise NotImplementedError("Conditions are not yet supported!")

    if len(config_space.forbidden_clauses) > 0:
        raise NotImplementedError("Forbiddens are not yet supported!")

    hebo_space: list[dict[str, Any]] = []

    for hp in list(config_space.values()):
        if isinstance(hp, IntegerHyperparameter):
            if hp.log:
                hebo_space.append(
                    {"name": hp.name, "type": "pow_int", "lb": hp.lower, "ub": hp.upper}
                )
            else:
                hebo_space.append({"name": hp.name, "type": "int", "lb": hp.lower, "ub": hp.upper})
        elif isinstance(hp, FloatHyperparameter):
            if hp.log:
                hebo_space.append({"name": hp.name, "type": "pow", "lb": hp.lower, "ub": hp.upper})
            else:
                hebo_space.append({"name": hp.name, "type": "num", "lb": hp.lower, "ub": hp.upper})
        elif isinstance(hp, CategoricalHyperparameter):
            hebo_space.append({"name": hp.name, "type": "cat", "categories": hp.choices})
        elif isinstance(hp, OrdinalHyperparameter):
            hebo_space.append({
                "name": hp.name,
                "type": "step_int",
                "lb": 0,
                "ub": len(hp.sequence),
                "step": 1,
            })
        elif isinstance(hp, Constant):
            hebo_space.append({"name": hp.name, "type": "cat", "categories": [hp.value]})
        else:
            raise ValueError(f"Unknown hyperparameter type: {hp.__class__.__name__}")
    return DesignSpace().parse(hebo_space)