from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from ConfigSpace import ConfigurationSpace
from hpoglue import Config, Optimizer, Problem, Query
from hpoglue.budget import CostBudget, TrialBudget
from hpoglue.env import Env
from smac import BlackBoxFacade, Scenario
from smac.runhistory import StatusType, TrialValue

if TYPE_CHECKING:
    from hpoglue import Result
    from smac.runhistory import TrialInfo


def _dummy_target_function(*args: Any, budget: int | float, seed: int) -> NoReturn:  # noqa: ARG001
    raise RuntimeError("This should never be called!")


class Ex_Blackbox_Opt(Optimizer):
    name = "Ex_Blackbox_Opt"

    env = Env(
        name="SMAC-2.1",
        python_version="3.10",
        requirements=("smac==2.1",)
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
        xi: float = 0.0,
    ):
        """Create a SMAC BO Optimizer instance for a given problem.

        Args:
            problem: The problem to optimize.
            seed: Random seed for the optimizer.
            working_directory: Working directory to store SMAC run.
            xi: Exploration-exploitation trade-off parameter. Defaults to 0.0.
        """
        config_space = problem.config_space
        match config_space:
            case ConfigurationSpace():
                pass
            case list():
                raise ValueError("SMAC does not support tabular benchmarks!")
            case _:
                raise TypeError("Config space must be a list or a ConfigurationSpace!")

        match problem.fidelities:
            case None:
                pass
            case tuple() | Mapping():
                raise ValueError("SMAC BO does not support multi-fidelity benchmarks!")
            case _:
                raise TypeError("Fidelity must be a string or a list of strings!")

        match problem.objectives:
            case Mapping():
                metric_names = list(problem.objectives.keys())
            case (metric_name, _):
                metric_names = metric_name
            case _:
                raise TypeError("Objective must be a tuple of (name, metric) or a mapping")

        working_directory.mkdir(parents=True, exist_ok=True)

        match problem.budget:
            case TrialBudget():
                budget = problem.budget.total
            case CostBudget():
                raise ValueError("SMAC BO does not support cost-aware benchmarks!")
            case _:
                raise TypeError("Budget must be a TrialBudget or a CostBudget!")

        scenario = Scenario(
            configspace=config_space,
            deterministic=True,
            objectives=metric_names,
            n_trials=budget,
            seed=seed,
            output_directory=working_directory / "smac-output",
            min_budget=None,
            max_budget=None,
        )
        self.problem = problem
        self.working_directory = working_directory
        self.config_space = config_space
        self._trial_lookup: dict[Hashable, TrialInfo] = {}
        self._seed = seed

        scenario = Scenario(
            configspace=self.config_space,
            deterministic=True,
            objectives=metric_names,
            n_trials=budget,
            seed=seed,
            output_directory=self.working_directory / "smac-output",
        )
        self.optimizer = BlackBoxFacade(
            scenario=scenario,
            logging_level=False,
            target_function=_dummy_target_function,
            intensifier=BlackBoxFacade.get_intensifier(scenario),
            acquisition_function=BlackBoxFacade.get_acquisition_function(scenario, xi=xi),
            overwrite=True,
        )

    def ask(self) -> Query:
        """Ask SMAC for a new config to evaluate."""
        smac_info = self.optimizer.ask()
        assert smac_info.instance is None, "We don't do instance benchmarks!"

        config = smac_info.config
        raw_config = dict(config)
        config_id = str(self.optimizer.intensifier.runhistory.config_ids[config])

        return Query(
            config=Config(config_id=config_id, values=raw_config),
            fidelity=None,
            optimizer_info=smac_info,
        )

    def tell(self, result: Result) -> None:
        """Tell SMAC the result of the query."""
        match self.problem.objectives:
            case Mapping():
                cost = [
                    obj.as_minimize(result.values[key])
                    for key, obj in self.problem.objectives.items()
                ]
            case (key, obj):
                cost = obj.as_minimize(result.values[key])
            case _:
                raise TypeError("Objective must be a string or a list of strings")

        self.optimizer.tell(
            result.query.optimizer_info,  # type: ignore
            TrialValue(
                cost=cost,
                time=0.0,
                starttime=0.0,
                endtime=0.0,
                status=StatusType.SUCCESS,
                additional_info={},
            ),
            save=True,
        )
