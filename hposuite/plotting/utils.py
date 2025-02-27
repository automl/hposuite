from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from itertools import cycle
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
FIDELITY_NAME_COL = "problem.fidelity.1.name"
FIDELITY_MIN_COL = "problem.fidelity.1.min"
FIDELITY_MAX_COL = "problem.fidelity.1.max"
CONTINUATIONS_COL = "result.continuations_cost.1"


def plot_results(  # noqa: C901, PLR0912, PLR0915
    *,
    report: dict[str, Any],
    budget_type: str,
    budget: int,
    objective: str,
    fidelity: str | None,
    cost: str | None,
    minimize: bool,
    save_dir: Path,
    benchmarks_name: str,
    show: bool = False,
    figsize: tuple[int, int] = (20, 10),
    logscale: bool = False,
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
    plt.figure(figsize=figsize)
    optim_res_dict = {}
    continuations = False
    for instance in optimizers:
        logger.info(f"Plotting {instance}")
        optim_res_dict[instance] = {}
        seed_cost_dict = {}
        seed_cont_dict = {}
        for seed in report[instance]:
            results = report[instance][seed]["results"]
            cost_list = results[SINGLE_OBJ_COL].values.astype(np.float64)

            if FIDELITY_COL in results.columns:

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

            if (
                CONTINUATIONS_COL in results.columns
                and
                not pd.isna(results[CONTINUATIONS_COL].iloc[0])
            ):
                continuations = True
                continuations_list = results[CONTINUATIONS_COL].values.astype(np.float64)
                continuations_list = np.cumsum(continuations_list)

            seed_cost_dict[seed] = pd.Series(cost_list, index=budget_list)
            if continuations:
                seed_cont_dict[seed] = pd.Series(cost_list, index=continuations_list)

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
    plot_suffix = f"{benchmarks_name}.{objective=}.{fidelity=}.{cost=}"
    plt.title(f"Plot for optimizers on {plot_suffix}")
    if logscale:
        plt.xscale("log")
    if len(optimizers) == 1:
        plt.title(f"Performance of {optimizers[0]} on {plot_suffix}")
    plt.legend()
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizers = ",".join(optimizers)

    save_path = save_dir / f"{optimizers}.{plot_suffix}.png"
    plt.savefig(save_path)
    logger.info(f"Saved plot to {save_path.absolute()}")
    plt.show()


def agg_data(  # noqa: C901, PLR0912, PLR0915
    study_dir: Path,
    save_dir: Path,
    figsize: tuple[int, int] = (20, 10),
    benchmark_spec: str | list[str] | None = None,
    optimizer_spec: str | list[str] | None = None,
    *,
    logscale: bool = False,
) -> None:
    """Aggregate the data from the run directory for plotting."""
    # df_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    budget_type: str | None = None
    budget: int | None = None
    objective: str | None = None
    minimize = True

    match benchmark_spec:
        case None:
            benchmarks_in_dir = [
                (f.name.split("benchmark=")[-1].split(".")[0])
                for f in study_dir.iterdir() if f.is_dir() and "benchmark=" in f.name]
            benchmarks_in_dir = list(set(benchmarks_in_dir))
            logger.info(f"Found benchmarks: {benchmarks_in_dir}")
        case str():
            benchmarks_in_dir = [benchmark_spec]
        case list():
            benchmarks_in_dir = benchmark_spec
        case _:
            raise ValueError(f"Unsupported type for benchmark_spec: {type(benchmark_spec)}")

    match optimizer_spec:
        case None:
            optimizers_in_dir = None
        case str():
            optimizers_in_dir = [optimizer_spec]
        case list():
            optimizers_in_dir = optimizer_spec
        case _:
            raise ValueError(f"Unsupported type for optimizer_spec: {type(optimizer_spec)}")

    benchmarks_dict: Mapping[str, Mapping[tuple[str, str, str], pd.DataFrame]] = {}

    for benchmark in benchmarks_in_dir:
        for file in study_dir.rglob("*.parquet"):
            if benchmark not in file.name:
                continue
            if (
                optimizers_in_dir is not None
                and not any(spec in file.name for spec in optimizers_in_dir)
            ):
                continue
            _df = pd.read_parquet(file)

            with (file.parent / "run_config.yaml").open("r") as f:
                run_config = yaml.safe_load(f)
            objectives = run_config["problem"]["objectives"]
            if not isinstance(objectives, str) and len(objectives) > 1:
                raise NotImplementedError("Plotting not yet implemented for multi-objective runs.")
            fidelities = run_config["problem"]["fidelities"]
            if fidelities and not isinstance(fidelities, str) and len(fidelities) > 1:
                raise NotImplementedError("Plotting not yet implemented for many-fidelity runs.")
            costs = run_config["problem"]["costs"]
            if costs:
                raise NotImplementedError("Cost-aware optimization not yet implemented in hposuite.")
            seed = int(run_config["seed"])

            benchmark_name = file.name.split("benchmark=")[-1].split(".")[0]
            all_plots_dict = benchmarks_dict.setdefault(benchmark_name, {})
            conf_tuple = (objectives, fidelities, costs)
            if conf_tuple not in all_plots_dict:
                all_plots_dict[conf_tuple] = [_df]
            else:
                all_plots_dict[conf_tuple].append(_df)


    for benchmark, conf_dict in benchmarks_dict.items():
        for conf_tuple, seed_dfs in conf_dict.items():
            df_agg = {}
            objective = conf_tuple[0]
            fidelity = conf_tuple[1]
            cost = conf_tuple[2]
            for _df in seed_dfs:
                if _df.empty:
                    continue

                instance = _df[OPTIMIZER_COL].iloc[0]
                if _df[HP_COL].iloc[0] is not None:
                    instance = f"{instance}_{_df[HP_COL].iloc[0]}"
                budget_type = "TrialBudget"
                budget = _df[BUDGET_TOTAL_COL].iloc[0]
                minimize = _df[SINGLE_OBJ_MINIMIZE_COL].iloc[0]
                seed = _df[SEED_COL].iloc[0]
                res_df = _df[
                    [
                        SINGLE_OBJ_COL,
                        BUDGET_USED_COL,
                    ]
                ]
                if FIDELITY_COL in _df.columns:
                    res_df = pd.concat(
                        [
                            res_df,
                            _df[FIDELITY_COL],
                            _df[FIDELITY_MIN_COL],
                            _df[FIDELITY_MAX_COL],
                        ],
                        axis=1,
                    )
                if CONTINUATIONS_COL in _df.columns:
                    res_df = pd.concat(
                        [
                            res_df,
                            _df[CONTINUATIONS_COL],
                        ],
                        axis=1,
                    )
                if instance not in df_agg:
                    df_agg[instance] = {}
                if int(seed) not in df_agg[instance]:
                    df_agg[instance][int(seed)] = {"results": res_df}
                assert budget_type is not None
                assert budget is not None
                assert objective is not None
            plot_results(
                report=df_agg,
                budget_type=budget_type,
                budget=budget,
                objective=objective,
                fidelity=fidelity,
                cost=cost,
                minimize=minimize,
                save_dir=save_dir,
                benchmarks_name=benchmark,
                figsize=figsize,
                logscale=logscale,
            )
            df_agg.clear()


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
        "--benchmark_spec", "-bs",
        nargs="+",
        type=str,
        help="Specification of the benchmark to plot. \n"
        " (e.g., 'benchmark=pd1-cifar100-wide_resnet-2048', \n"
        " 'benchmark=pd1-cifar100-wide_resnet-2048.objective=valid_error_rate.fidelity=epochs', \n"
        " 'benchmark=pd1-imagenet-resnet-512 benchmark=pd1-cifar100-wide_resnet-2048') \n"
    )
    parser.add_argument(
        "--optimizer_spec", "-os",
        type=str,
        nargs="+",
        help="Specification of the optimizer to plot \n"
        " (e.g., 'optimizer=DEHB', \n"
        " 'optimizer=DEHB.eta=3', \n"
        " 'optimizer=DEHB optimizer=SMAC_Hyperband.eta=3') \n"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Location of the main directory where all studies are stored",
        default=Path.cwd().absolute().parent / "hposuite-output"
    )
    parser.add_argument(
        "--study_dir",
        type=str,
        help="Name of the study directory from where to plot the results",
        default=None
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save the plots",
        default="plots"
    )
    parser.add_argument(
        "--figsize", "-fs",
        type=int,
        nargs="+",
        default=(20, 10),
        help="Size of the figure to plot",
    )
    parser.add_argument(
        "--logscale", "-ls",
        action="store_true",
        help="Use log scale for the x-axis",
    )
    args = parser.parse_args()

    study_dir = args.output_dir / args.study_dir
    save_dir = study_dir / args.save_dir
    figsize = tuple(map(int, args.figsize))

    agg_data(
        study_dir=study_dir,
        save_dir=save_dir,
        figsize=figsize,
        logscale=args.logscale,
        benchmark_spec=args.benchmark_spec,
        optimizer_spec=args.optimizer_spec,
    )
