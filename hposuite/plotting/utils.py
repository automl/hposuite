from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.ERROR)

DataContainer: TypeAlias = np.ndarray | (pd.DataFrame | pd.Series)
D = TypeVar("D", bound=DataContainer)

ROOT_DIR = Path(__file__).absolute().resolve().parent.parent.parent

SEED_COL = "run.seed"
PROBLEM_COL = "problem.name"
OPTIMIZER_COL = "run.opt.name"
HP_COL = "run.opt.hp_str"
SINGLE_OBJ_NAME = "problem.objective.1.name"
SINGLE_OBJ_COL = "result.objective.1.value"
SINGLE_OBJ_MINIMIZE_COL = "problem.objective.1.minimize"
SECOND_OBJ_NAME = "problem.objective.2.name"
SECOND_OBJ_COL = "result.objective.2.value"
SECOND_OBJ_MINIMIZE_COL = "problem.objective.2.minimize"
BUDGET_USED_COL = "result.budget_used_total"
BUDGET_TOTAL_COL = "problem.budget.total"
FIDELITY_COL = "result.fidelity.1.value"
FIDELITY_MIN_COL = "problem.fidelity.1.min"
FIDELITY_MAX_COL = "problem.fidelity.1.max"
CONTINUATIONS_COL = "result.continuations_cost.1"


def plot_results(  # noqa: PLR0915
    *,
    report: dict[str, Any],
    budget_type: str,
    budget: int,
    objective: str,
    minimize: bool,
    save_dir: Path,
    benchmarks_name: str,
) -> None:
    """Plot the results for the optimizers on the given benchmark."""
    marker_list = [
        "o",
        "X",
        "^",
        "H",
        ">",
        "^",
        "p",
        "P",
        "*",
        "h",
        "<",
        "s",
        "x",
        "+",
        "D",
        "d",
        "|",
        "_",
    ]
    markers = cycle(marker_list)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore
    colors_mean = cycle(colors)
    optimizers = list(report.keys())
    plt.figure(figsize=(20, 10))
    optim_res_dict = {}
    continuations = False
    for instance in optimizers:
        logger.info(f"Plotting {instance}")
        optim_res_dict[instance] = {}
        seed_cost_dict = {}
        for seed in report[instance]:
            results = report[instance][seed]["results"]
            cost_list = results[SINGLE_OBJ_COL].values.astype(np.float64)

            # Automatically set budget to TrialBudget if Non-Multifidelity Optimizers are used
            if (
                np.all(results[FIDELITY_COL].to_numpy() == results[FIDELITY_COL].iloc[0])
                or
                results[FIDELITY_COL].iloc[0] is None
            ):
                budget_type = "TrialBudget"

            # Automatically set budget to FidelityBudget if Multifidelity Optimizers are used
            if results[FIDELITY_COL].iloc[0] is not None:
                budget_type = "FidelityBudget"
            match budget_type:
                case "FidelityBudget":
                    budget_list = results[FIDELITY_COL].values.astype(np.float64)
                    budget_list = np.cumsum(budget_list)
                    budget = budget_list[-1]
                    budget_type = "FidelityBudget"
                case "TrialBudget":
                    budget_list = results[BUDGET_USED_COL].values.astype(np.float64)
                case _:
                    raise NotImplementedError(f"Budget type {budget_type} not implemented")

            if not pd.isna(results[CONTINUATIONS_COL].iloc[0]):
                continuations = True
                continuations_list = results[CONTINUATIONS_COL].values.astype(np.float64)
                continuations_list = np.cumsum(continuations_list)

            seed_cost_dict[seed] = pd.Series(cost_list, index=budget_list)
            if continuations:
                seed_cont_dict = pd.Series(cost_list, index=continuations_list)

        seed_cost_df = pd.DataFrame(seed_cost_dict)
        seed_cost_df = seed_cost_df.ffill(axis=0)
        seed_cost_df = seed_cost_df.dropna(axis=0)
        means = pd.Series(seed_cost_df.mean(axis=1), name=f"means_{instance}")
        std = pd.Series(seed_cost_df.std(axis=1), name=f"std_{instance}")
        optim_res_dict[instance]["means"] = means
        optim_res_dict[instance]["std"] = std
        means = means.cummin() if minimize else means.cummax()
        means = means.drop_duplicates()
        std = std.loc[means.index]
        means[budget] = means.iloc[-1]
        std[budget] = std.iloc[-1]
        col_next = next(colors_mean)

        plt.step(
            means.index,
            means,
            where="post",
            label=instance,
            marker=next(markers),
            markersize=10,
            markerfacecolor="#ffffff",
            markeredgecolor=col_next,
            markeredgewidth=2,
            color=col_next,
            linewidth=3,
        )
        plt.fill_between(
            means.index,
            means - std,
            means + std,
            alpha=0.2,
            step="post",
            color=col_next,
            edgecolor=col_next,
            linewidth=2,
        )

        #For plotting continuations
        if continuations:
            seed_cont_df = pd.DataFrame(seed_cont_dict)
            seed_cont_df = seed_cont_df.ffill(axis=0)
            seed_cont_df = seed_cont_df.dropna(axis=0)
            means_cont = pd.Series(seed_cont_df.mean(axis=1), name=f"means_{instance}")
            std_cont = pd.Series(seed_cont_df.std(axis=1), name=f"std_{instance}")
            optim_res_dict[instance]["cont_means"] = means_cont
            optim_res_dict[instance]["cont_std"] = std_cont
            means_cont = means_cont.cummin() if minimize else means_cont.cummax()
            means_cont = means_cont.drop_duplicates()
            std_cont = std_cont.loc[means_cont.index]
            col_next = next(colors_mean)

            plt.step(
                means_cont.index,
                means_cont,
                where="post",
                label=f"{instance}_w_continuations",
                marker=next(markers),
                markersize=10,
                markerfacecolor="#ffffff",
                markeredgecolor=col_next,
                markeredgewidth=2,
                color=col_next,
                linewidth=3,
            )
            plt.fill_between(
                means_cont.index,
                means_cont - std_cont,
                means_cont + std_cont,
                alpha=0.2,
                step="post",
                color=col_next,
                edgecolor=col_next,
                linewidth=2,
            )
    plt.xlabel(f"{budget_type}")
    plt.ylabel(f"{objective}")
    plt.title(f"Performance of Optimizers on {benchmarks_name}")
    plt.xscale("log")
    if len(optimizers) == 1:
        plt.title(f"Performance of {optimizers[0]} on {benchmarks_name}")
    plt.legend()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{benchmarks_name}_performance.png")


def agg_data(exp_dir: str | Path) -> None:
    """Aggregate the data from the run directory for plotting."""
    df_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    budget_type: str | None = None
    budget: int | None = None
    objective: str | None = None
    minimize = True
    benchmarks_in_dir = [
        (f.name.split(".")[0].split("benchmark=")[-1])
        for f in exp_dir.iterdir() if f.is_dir() and "benchmark=" in f.name]
    benchmarks_in_dir = list(set(benchmarks_in_dir))
    logger.info(f"Found benchmarks: {benchmarks_in_dir}")
    for benchmark in benchmarks_in_dir:
        for file in exp_dir.rglob("*.parquet"):
            if benchmark not in file.name:
                continue
            res_df = pd.read_parquet(file)

            instance = res_df[OPTIMIZER_COL].iloc[0]
            if res_df[HP_COL].iloc[0] is not None:
                instance = f"{instance}_{res_df[HP_COL].iloc[0]}"

            objective = res_df[SINGLE_OBJ_NAME].iloc[0]
            budget_type = "TrialBudget"
            budget = res_df[BUDGET_TOTAL_COL].iloc[0]
            minimize = res_df[SINGLE_OBJ_MINIMIZE_COL].iloc[0]
            seed = res_df[SEED_COL].iloc[0]
            res_df = res_df[
                [
                    SINGLE_OBJ_COL,
                    BUDGET_USED_COL,
                    FIDELITY_COL,
                    CONTINUATIONS_COL,
                    FIDELITY_MIN_COL,
                    FIDELITY_MAX_COL
                ]
            ]
            df_agg[instance][int(seed)] = {"results": res_df}
            assert budget_type is not None
            assert budget is not None
            assert objective is not None
        plot_results(
            report=df_agg,
            budget_type=budget_type,
            budget=budget,
            objective=objective,
            minimize=minimize,
            save_dir=exp_dir / "plots",
            benchmarks_name=benchmark,
        )


def scale(
    unit_xs: int | float | np.number | np.ndarray | pd.Series,
    to: tuple[int | float | np.number, int | float | np.number],
) -> float | np.number | np.ndarray | pd.Series:
    """Scale values from unit range to a new range.

    >>> scale(np.array([0.0, 0.5, 1.0]), to=(0, 10))
    array([ 0.,  5., 10.])

    Parameters
    ----------
    unit_xs:
        The values to scale

    to:
        The new range

    Returns:
    -------
        The scaled values
    """
    return unit_xs * (to[1] - to[0]) + to[0]  # type: ignore


def normalize(
    x: int | float | np.number | np.ndarray | pd.Series,
    *,
    bounds: tuple[int | float | np.number, int | float | np.number],
) -> float | np.number | np.ndarray | pd.Series:
    """Normalize values to the unit range.

    >>> normalize(np.array([0.0, 5.0, 10.0]), bounds=(0, 10))
    array([0. , 0.5, 1. ])

    Parameters
    ----------
    x:
        The values to normalize

    bounds:
        The bounds of the range

    Returns:
    -------
        The normalized values
    """
    if bounds == (0, 1):
        return x

    return (x - bounds[0]) / (bounds[1] - bounds[0])  # type: ignore


def rescale(
    x: int | float | np.number | np.ndarray | pd.Series,
    frm: tuple[int | float | np.number, int | float | np.number],
    to: tuple[int | float | np.number, int | float | np.number],
) -> float | np.ndarray | pd.Series:
    """Rescale values from one range to another.

    >>> rescale(np.array([0, 10, 20]), frm=(0, 100), to=(0, 10))
    array([0, 1, 2])

    Parameters
    ----------
    x:
        The values to rescale

    frm:
        The original range

    to:
        The new range

    Returns:
    -------
        The rescaled values
    """
    if frm != to:
        normed = normalize(x, bounds=frm)
        scaled = scale(unit_xs=normed, to=to)
    else:
        scaled = x

    match scaled:
        case int() | float() | np.number():
            return float(scaled)
        case np.ndarray() | pd.Series():
            return scaled.astype(np.float64)
        case _:
            raise ValueError(f"Unsupported type {type(x)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting Incumbents after GLUE Experiments")

    parser.add_argument(
        "--root_dir", type=Path, help="Location of the root directory", default=Path("./")
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        help="Location of the results directory",
        default="../hpo-glue-output"
    )

    parser.add_argument(
        "--exp_dir", type=str, help="Location of the Experiment directory", default=None
    )

    parser.add_argument("--save_dir", type=str, help="Directory to save the plots", default="plots")

    args = parser.parse_args()

    if args.results_dir is None:
        raise ValueError("Results directory not specified")

    agg_data(args.results_dir)
