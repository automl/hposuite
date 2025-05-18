from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import neps
import numpy as np
from hpoglue import Config, Optimizer, Problem, Query, Result
from hpoglue.env import Env
from neps import AskAndTell, algorithms

from hposuite.utils import set_priors_as_defaults

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from hpoglue.fidelity import Fidelity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(seed: int) -> None:
    """Set the seed for the optimizer."""
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


class NepsOptimizer(Optimizer):
    """Base class for Neps Optimizers."""
    name = "NepsOptimizer"


    def __init__(
        self,
        *,
        problem: Problem,
        space: neps.SearchSpace,
        optimizer: str,
        seed: int,
        working_directory: str | Path,
        fidelities: tuple[str, Fidelity] | None = None,
        random_weighted_opt: bool = False,
        scalarization_weights: Literal["equal", "random"] | Mapping[str, float] = "random",
        **kwargs: Any,
    ) -> None:
        """Initialize the optimizer."""
        self.problem = problem
        self.space = space

        match fidelities:
            case None:
                pass
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
                raise TypeError("Fidelity must be a tuple or None.")


        if not isinstance(working_directory, Path):
            working_directory = Path(working_directory)
        self.seed = seed
        self.working_dir = working_directory

        self.optimizer = AskAndTell(
            algorithms.PredefinedOptimizers[optimizer](
                space,
                **kwargs,
            )
        )
        self.trial_counter = 0

        self.objectives = self.problem.get_objectives()
        self.random_weighted_opt = random_weighted_opt
        self.scalarization_weights = None

        if self.random_weighted_opt:
            assert len(self.objectives) > 1, (
                "Random weighted optimization is only supported for multi-objective problems."
            )
            match scalarization_weights:
                case Mapping():
                    self.scalarization_weights = scalarization_weights
                case "equal":
                    self.scalarization_weights = (
                        dict.fromkeys(self.objectives, 1.0 / len(self.objectives))
                    )
                case "random":
                    weights = np.random.uniform(size=len(self.objectives))  # noqa: NPY002
                    self.scalarization_weights = dict(zip(self.objectives, weights, strict=True))
                case _:
                    raise ValueError(
                        f"Invalid scalarization_weights: {scalarization_weights}. "
                        "Expected 'equal', 'random', or a Mapping."
                    )


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
                raise NotImplementedError("Many-fidelity not yet implemented for NePS.")
            case (fid_name, _):
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


    def tell(self, result: Result) -> None:
        """Tell the optimizer about the result of a trial."""
        match self.problem.objectives:
            case (name, metric):
                _values = metric.as_minimize(result.values[name])
            case Mapping():
                _values = {
                    key: obj.as_minimize(result.values[key])
                    for key, obj in self.problem.objectives.items()
                }
                if self.random_weighted_opt:
                    _values = sum(
                        self.scalarization_weights[obj] * _values[obj] for obj in self.objectives
                    )
                else:
                    _values = list(_values.values())
            case _:
                raise TypeError(
                    "Objective must be a tuple or a Mapping! "
                    f"Got {type(self.problem.objectives)}."
                )

        match self.problem.costs:
            case None:
                pass
            case tuple():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for NePS!")
            case Mapping():
                raise NotImplementedError("# TODO: Cost-aware not yet implemented for NePS!")
            case _:
                raise TypeError("Cost must be None or a mapping!")

        self.optimizer.tell(
            trial=result.query.optimizer_info,
            result=_values,
        )


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
        name="Neps-0.13.0",
        python_version="3.10",
        requirements=("neural-pipeline-search>=0.13.0",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        initial_design_size: int | Literal["ndim"] = "ndim",
        use_priors: bool = False,
    ) -> None:
        """Initialize the optimizer."""
        match problem.fidelities:
            case None:
                pass
            case Mapping() | tuple():
                raise ValueError("NepsBO does not support fidelities.")
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        match problem.objectives:
            case tuple():
                pass
            case Mapping():
                raise ValueError("NepsBO only supports single-objective problems.")
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(problem.objectives)}."
                )

        space = configspace_to_pipeline_space(problem.config_space)
        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            seed=seed,
            working_directory=working_directory,
            optimizer="bayesian_optimization",
            initial_design_size=initial_design_size,
            use_priors=use_priors,
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
        name="Neps-0.13.0",
        python_version="3.10",
        requirements=("neural-pipeline-search>=0.13.0",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        scalarization_weights: Literal["equal", "random"] | Mapping[str, float] = "random",
        initial_design_size: int | Literal["ndim"] = "ndim",
        use_priors: bool = False,
    ) -> None:
        """Initialize the optimizer."""
        space = configspace_to_pipeline_space(problem.config_space)


        match problem.fidelities:
            case None:
                pass
            case Mapping() | tuple():
                raise ValueError("NepsRW does not support fidelities.")
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        match problem.objectives:
            case tuple():
                raise ValueError("NepsRW only supports multi-objective problems.")
            case Mapping():
                pass
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(problem.objectives)}."
                )

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer="bayesian_optimization",
            seed=seed,
            working_directory=working_directory,
            random_weighted_opt=True,
            scalarization_weights=scalarization_weights,
            initial_design_size=initial_design_size,
            use_priors=use_priors,
        )


class NepsSuccessiveHalving(NepsOptimizer):
    """Neps Successive Halving."""

    name = "NepsSuccessiveHalving"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("single"),
        cost_awareness=(None,),
        tabular=False,
    )

    env = Env(
        name="Neps-0.13.0",
        python_version="3.10",
        requirements=("neural-pipeline-search>=0.13.0",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        eta: int = 3,
        sampler: Literal["uniform", "prior"] = "uniform",
    ) -> None:
        """Initialize the optimizer."""
        space = configspace_to_pipeline_space(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsSuccessiveHalving requires a fidelity.")
            case Mapping():
                raise NotImplementedError(
                    "Many-fidelity not yet implemented for NepsSuccessiveHalving."
                )
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        match problem.objectives:
            case tuple():
                pass
            case Mapping():
                raise ValueError("NepsSuccessiveHalving only supports single-objective problems.")
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(problem.objectives)}."
                )

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            seed=seed,
            working_directory=working_directory,
            optimizer="successive_halving",
            fidelities=_fid,
            eta=eta,
            sampler=sampler,
        )


class NepsHyperband(NepsOptimizer):
    """NepsHyperband."""

    name = "NepsHyperband"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("single"),
        cost_awareness=(None,),
        tabular=False,
    )

    env = Env(
        name="Neps-0.13.0",
        python_version="3.10",
        requirements=("neural-pipeline-search>=0.13.0",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        eta: int = 3,
        sampler: Literal["uniform", "prior"] = "uniform",
    ) -> None:
        """Initialize the optimizer."""
        space = configspace_to_pipeline_space(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsHyperband requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsHyperband.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        match problem.objectives:
            case tuple():
                pass
            case Mapping():
                raise ValueError("NepsHyperband only supports single-objective problems.")
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(problem.objectives)}."
                )

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            seed=seed,
            working_directory=working_directory,
            optimizer="hyperband",
            fidelities=_fid,
            eta=eta,
            sampler=sampler,
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
        name="Neps-0.13.0",
        python_version="3.10",
        requirements=("neural-pipeline-search>=0.13.0",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        scalarization_weights: Literal["equal", "random"] | Mapping[str, float] = "random",
        eta: int = 3,
        sampler: Literal["uniform", "prior"] = "uniform",
    ) -> None:
        """Initialize the optimizer."""
        space = configspace_to_pipeline_space(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsHyperbandRW requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsHyperbandRW.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        match problem.objectives:
            case tuple():
                raise ValueError("NepsHyperbandRW only supports multi-objective problems.")
            case Mapping():
                pass
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(problem.objectives)}."
                )

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            optimizer="hyperband",
            seed=seed,
            working_directory=working_directory,
            fidelities=_fid,
            random_weighted_opt=True,
            scalarization_weights=scalarization_weights,
            eta=eta,
            sampler=sampler,
        )



class NepsASHA(NepsOptimizer):
    """NepsASHA."""

    name = "NepsASHA"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("single"),
        cost_awareness=(None,),
        tabular=False,
    )

    env = Env(
        name="Neps-0.13.0",
        python_version="3.10",
        requirements=("neural-pipeline-search>=0.13.0",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        eta: int = 3,
        sampler: Literal["uniform", "prior"] = "uniform",
    ) -> None:
        """Initialize the optimizer."""
        space = configspace_to_pipeline_space(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsASHA requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsASHA.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        match problem.objectives:
            case tuple():
                pass
            case Mapping():
                raise ValueError("NepsASHA only supports single-objective problems.")
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(problem.objectives)}."
                )

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            seed=seed,
            working_directory=working_directory,
            eta=eta,
            optimizer="asha",
            fidelities=_fid,
            sampler=sampler,
        )


class NepsAsyncHB(NepsOptimizer):
    """Neps Async Hyperband."""

    name = "NepsAsyncHB"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("single"),
        cost_awareness=(None,),
        tabular=False,
    )

    env = Env(
        name="Neps-0.13.0",
        python_version="3.10",
        requirements=("neural-pipeline-search>=0.13.0",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        eta: int = 3,
    ) -> None:
        """Initialize the optimizer."""
        space = configspace_to_pipeline_space(problem.config_space)

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsAsyncHB requires a fidelity.")
            case Mapping():
                raise NotImplementedError("Many-fidelity not yet implemented for NepsAsyncHB.")
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        match problem.objectives:
            case tuple():
                pass
            case Mapping():
                raise ValueError("NepsAsyncHB only supports single-objective problems.")
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(problem.objectives)}."
                )

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            seed=seed,
            working_directory=working_directory,
            optimizer="async_hb",
            fidelities=_fid,
            eta=eta,
        )


class NepsPriorband(NepsOptimizer):
    """NepsPriorband."""

    name = "NepsPriorband"

    support = Problem.Support(
        fidelities=("single",),
        objectives=("single"),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
    )

    env = Env(
        name="Neps-0.13.0",
        python_version="3.10",
        requirements=("neural-pipeline-search>=0.13.0",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        eta: int = 3,
        sample_prior_first: bool | Literal["highest_fidelity"] = False,
        base: Literal["successive_halving", "hyperband", "asha", "async_hb"] = "hyperband",
    ) -> None:
        """Initialize the optimizer."""
        assert len(problem.priors[1]) == 1, (
            "NepsPriorband only supports single-objective priors. "
        )
        config_space = set_priors_as_defaults(
            config_space=problem.config_space,
            priors=next(iter(problem.priors[1].values())),
            distribution="normal",
        )
        space = configspace_to_pipeline_space(
            config_space,
            use_priors=True,
        )

        _fid = None
        match problem.fidelities:
            case None:
                raise ValueError("NepsPriorband requires a fidelity.")
            case Mapping():
                raise NotImplementedError(
                    "Many-fidelity not yet implemented for NepsPriorband."
                )
            case (fid_name, fidelity):
                _fid = (fid_name, fidelity)
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        match problem.objectives:
            case tuple():
                pass
            case Mapping():
                raise ValueError("NepsPriorband only supports single-objective problems.")
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(problem.objectives)}."
                )

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            seed=seed,
            working_directory=working_directory,
            optimizer="priorband",
            fidelities=_fid,
            eta=eta,
            base=base,
            sample_prior_first=sample_prior_first,
        )


class NepsPiBO(NepsOptimizer):
    """Neps PiBO - Bayesian Optimization with User Beliefs."""

    name = "NepsPiBO"

    support = Problem.Support(
        fidelities=(None,),
        objectives=("single",),
        cost_awareness=(None,),
        tabular=False,
        priors=True,
    )

    env = Env(
        name="Neps-0.13.0",
        python_version="3.10",
        requirements=("neural-pipeline-search>=0.13.0",)
    )

    mem_req_mb = 1024

    def __init__(
        self,
        *,
        problem: Problem,
        seed: int,
        working_directory: str | Path,
        initial_design_size: int | Literal["ndim"] = "ndim",
        sample_prior_first: bool = False,
    ) -> None:
        """Initialize the optimizer."""
        assert len(problem.priors[1]) == 1, (
            "NepsPiBO only supports single-objective priors. "
        )
        config_space = set_priors_as_defaults(
            config_space=problem.config_space,
            priors=next(iter(problem.priors[1].values())),
            distribution="normal",
        )
        space = configspace_to_pipeline_space(
            config_space,
            use_priors=True,
        )

        match problem.fidelities:
            case None:
                pass
            case Mapping() | tuple():
                raise ValueError("NepsPiBO does not support fidelities.")
            case _:
                raise TypeError("Fidelity must be a tuple or a Mapping.")

        match problem.objectives:
            case tuple():
                pass
            case Mapping():
                raise ValueError("NepsPiBO only supports single-objective problems.")
            case _:
                raise TypeError(
                    "Objectives must be a tuple or a Mapping. \n"
                    f"Got {type(problem.objectives)}."
                )

        set_seed(seed)

        super().__init__(
            problem=problem,
            space=space,
            seed=seed,
            working_directory=working_directory,
            optimizer="pibo",
            initial_design_size=initial_design_size,
            sample_prior_first=sample_prior_first,
        )


def configspace_to_pipeline_space(  # noqa: C901
    config_space: ConfigurationSpace,
    *,
    use_priors: bool = False,
    std: float = 0.25,
) -> neps.SearchSpace:
    """Convert a ConfigurationSpace to a Neps SearchSpace.

    Args:
        config_space: The ConfigurationSpace to convert.
        use_priors: Whether to use priors for the hyperparameters.
            NePS only supports Normally distributed priors, therefore
            if the hyperparameter is not Normally distributed and `use_priors` is True,
            the default value of the hyperparameter is used as the mean to create a
            Normally distributed prior.
            If `use_priors` is False and the hyperparameter is Normally distributed,
            `use_priors` is ignored and it is converted to a NePS hyperparameter with
            its mean as the prior.
        std: The standard deviation of the prior. Only used if `use_priors` is True and/or
            the hyperparameter is Normally distributed.
            Translated to the confidence of the prior in NePS as 1 - std.
    """
    from ConfigSpace.hyperparameters import (
        BetaFloatHyperparameter,
        BetaIntegerHyperparameter,
        CategoricalHyperparameter,
        Constant,
        NormalFloatHyperparameter,
        NormalIntegerHyperparameter,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )
    from neps.space.parameters import Parameter
    space: dict[str, Parameter | Constant] = {}
    std_to_confidences: dict[float, str] = {
        0.25: "high",
        0.5: "medium",
        0.75: "low",
    }
    if any(config_space.conditions) or any(config_space.forbidden_clauses):
        raise NotImplementedError(
            "The ConfigurationSpace has conditions or forbidden clauses, "
            "which are not supported by neps."
        )
    for hp in list(config_space.values()):
        match hp:
            case NormalFloatHyperparameter():
                if not use_priors:
                    warnings.warn(
                        "use priors=False but hyperparameter is a NormalFloatHyperparameter, "
                        "which defaults to a Prior in NePS. Ignoring use_priors.",
                        stacklevel=2,
                    )
                space[hp.name] = neps.Float(
                    lower=hp.lower,
                    upper=hp.upper,
                    log=hp.log,
                    prior=hp.mu,
                    prior_confidence=std_to_confidences[hp.sigma],
                )
            case UniformFloatHyperparameter() | BetaFloatHyperparameter():
                space[hp.name] = neps.Float(
                    lower=hp.lower,
                    upper=hp.upper,
                    log=hp.log,
                    prior=hp.default_value if use_priors else None,
                    prior_confidence=std_to_confidences[std]
                )
            case NormalIntegerHyperparameter():
                if not use_priors:
                    warnings.warn(
                        "use priors=False but hyperparameter is a NormalIntegerHyperparameter, "
                        "which defaults to a Prior in NePS. Ignoring use_priors.",
                        stacklevel=2,
                    )
                space[hp.name] = neps.Integer(
                    lower=hp.lower,
                    upper=hp.upper,
                    log=hp.log,
                    prior=hp.mu,
                    prior_confidence=std_to_confidences[hp.sigma],
                )
            case UniformIntegerHyperparameter() | BetaIntegerHyperparameter():
                space[hp.name] = neps.Integer(
                    lower=hp.lower,
                    upper=hp.upper,
                    log=hp.log,
                    prior=hp.default_value if use_priors else None,
                    prior_confidence=std_to_confidences[std],
                )
            case CategoricalHyperparameter():
                assert hp.weights is None, (
                    "Weights on categoricals are not yet supported!"
                )
                space[hp.name] = neps.Categorical(
                    choices=hp.choices,
                    prior=hp.default_value if use_priors else None,
                    prior_confidence=std_to_confidences[std]
                )
            case OrdinalHyperparameter():
                space[hp.name] = neps.Categorical(
                    choices=hp.sequence,
                    prior=hp.default_value if use_priors else None,
                    prior_confidence=std_to_confidences[std]
                )
            case Constant():
                space[hp.name] = neps.Constant(
                    value=hp.value,
                )
    return neps.SearchSpace(space)
