from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from itertools import cycle
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar

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
OPTIMIZER_COL = "optimizer.name"
BENCHMARK_COL = "benchmark.name"
HP_COL = "optimizer.hp_str"
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
BENCHMARK_COUNT_FIDS = "benchmark.fidelity.count"
BENCHMARK_FIDELITY_NAME = "benchmark.fidelity.1.name"
BENCHMARK_FIDELITY_COL = "benchmark.fidelity.1.value"
BENCHMARK_FIDELITY_MIN_COL = "benchmark.fidelity.1.min"
BENCHMARK_FIDELITY_MAX_COL = "benchmark.fidelity.1.max"
CONTINUATIONS_COL = "result.continuations_cost.1"


def plot_results(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    report: dict[str, Any],
    objective: str,
    fidelity: str | None,
    cost: str | None,
    to_minimize: bool,
    save_dir: Path,
    benchmarks_name: str,
    regret_bound: float | None = None,  # noqa: ARG001
    figsize: tuple[int, int] = (20, 10),
    logscale: bool = False,
    error_bars: Literal["std", "sem"] = "std",
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
            cost_list: pd.Series = results[SINGLE_OBJ_COL].values.astype(np.float64)

            budget_type = "TrialBudget" if fidelity is None else "FidelityBudget"
            match budget_type:
                case "FidelityBudget":
                    if FIDELITY_COL in results.columns:
                        budget_list = results[FIDELITY_COL].values.astype(np.float64)
                        budget_list = np.cumsum(budget_list)
                        budget_type = "FidelityBudget"
                    else:
                        budget_list = np.cumsum(
                            results[BENCHMARK_FIDELITY_MAX_COL].values.astype(np.float64)
                        )
                case "TrialBudget":
                    budget_list = results[BUDGET_USED_COL].values.astype(np.float64)
                case _:
                    raise NotImplementedError(f"Budget type {budget_type} not implemented")
            budget = budget_list[-1]

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
        match error_bars:
            case "std":
                error = pd.Series(seed_cost_df.std(axis=1), name=f"std_{instance}")
            case "sem":
                error = pd.Series(seed_cost_df.sem(axis=1), name=f"sem_{instance}")
            case _:
                raise ValueError(f"Unsupported error bars type {error_bars}")
        optim_res_dict[instance]["means"] = means
        optim_res_dict[instance]["error"] = error
        means = means.cummin() if to_minimize else means.cummax()
        means = means.drop_duplicates()
        error = error.loc[means.index]
        means[budget] = means.iloc[-1]
        error[budget] = error.iloc[-1]
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
            means - error,
            means + error,
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
            means_cont = means_cont.cummin() if to_minimize else means_cont.cummax()
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
    plot_suffix = (
        f"{benchmarks_name}, {objective=}, \n{fidelity=}, {cost=}, "
        f"{budget_type}={budget}, {to_minimize=}, {error_bars=}"
    )
    plt.title(f"Plot for optimizers on {plot_suffix}")
    if logscale:
        plt.xscale("log")
    if len(optimizers) == 1:
        plt.title(f"Performance of {optimizers[0]} on {plot_suffix}")
    plt.legend()
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizers = ",".join(optimizers)

    plot_suffix = plot_suffix.replace("\n", "")

    save_path = save_dir / f"{optimizers}.{plot_suffix}.png"
    plt.savefig(save_path)
    logger.info(f"Saved plot to {save_path.absolute()}")


def agg_data(  # noqa: C901, PLR0912, PLR0915
    study_dir: Path,
    save_dir: Path,
    figsize: tuple[int, int] = (20, 10),
    *,
    benchmark_spec: str | list[str] | None = None,
    optimizer_spec: str | list[str] | None = None,
    error_bars: Literal["std", "sem"] = "std",
    logscale: bool = False,
    budget_type: Literal["TrialBudget", "FidelityBudget", None] = None,
) -> None:
    """Aggregate the data from the run directory for plotting."""
    objective: str | None = None
    minimize = True

    with (study_dir / "study_config.yaml").open("r") as f:
        study_config = yaml.safe_load(f)

    all_benches = [(bench.pop("name"), bench) for bench in study_config["benchmarks"]]

    match benchmark_spec:
        case None:
            benchmarks_in_dir = [
                (f.name.split("benchmark=")[-1].split(".")[0])
                for f in study_dir.iterdir() if f.is_dir() and "benchmark=" in f.name]
            benchmarks_in_dir = list(set(benchmarks_in_dir))
            logger.info(f"Found benchmarks: {benchmarks_in_dir}")
        case str():
            benchmarks_in_dir = [benchmark_spec]
            logger.info(f"Benchmarks specified: {benchmarks_in_dir}")
        case list():
            benchmarks_in_dir = benchmark_spec
            logger.info(f"Benchmarks specified: {benchmarks_in_dir}")
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

    benchmarks_dict: Mapping[str, Mapping[tuple[str, str, str], list[pd.DataFrame]]] = {}

    for benchmark in benchmarks_in_dir:
        logger.info(f"Processing benchmark: {benchmark}")
        for file in study_dir.rglob("*.parquet"):
            if benchmark not in file.name:
                continue
            if (
                optimizers_in_dir is not None
                and not any(spec in file.name for spec in optimizers_in_dir)
            ):
                continue
            _df = pd.read_parquet(file)

            benchmark_name = file.name.split("benchmark=")[-1].split(".")[0]

            with (file.parent / "run_config.yaml").open("r") as f:
                run_config = yaml.safe_load(f)
            objectives = run_config["problem"]["objectives"]
            if not isinstance(objectives, str) and len(objectives) > 1:
                raise NotImplementedError("Plotting not yet implemented for multi-objective runs.")
            fidelities = run_config["problem"]["fidelities"]
            if fidelities and not isinstance(fidelities, str) and len(fidelities) > 1:
                raise NotImplementedError("Plotting not yet implemented for many-fidelity runs.")

            # Add default benchmark fidelity to a blackbox Optimizer to compare it
            # alongside MF optimizers if the latter exist in the study
            bench_num_fids = _df[BENCHMARK_COUNT_FIDS].iloc[0]
            if fidelities is None and budget_type != "TrialBudget" and bench_num_fids >= 1:
                fid = next(
                    bench[1]["fidelities"]
                    for bench in all_benches
                    if bench[0] == benchmark_name
                )
                if fid == _df[BENCHMARK_FIDELITY_NAME].iloc[0]:
                # Study config is saved in such a way that if Blackbox Optimizers
                # are used along with MF optimizers on MF benchmarks, the "fidelities"
                # key in the benchmark instance in the study config is set to the fidelity
                # being used by the MF optimizers. In that case, there is no benchmark
                # instance with fidelity as None. In case of multiple fidelities being used
                # for the same benchmark, separate benchmark instances are created
                # for each fidelity.
                # If only Blackbox Optimizers are used in the study, there is only one
                # benchmark instance with fidelity as None.
                # When a problem with a Blackbox Optimizer is used on a MF benchmark,
                # each config is queried at the highest available 'first' fidelity in the
                # benchmark. Hence, we only set `fidelities` to `fid` if the benchmark instance
                # is the one with the default fidelity, else it would be incorrect.
                    fidelities = fid

            costs = run_config["problem"]["costs"]
            if costs:
                raise NotImplementedError(
                    "Cost-aware optimization not yet implemented in hposuite."
                )
            seed = int(run_config["seed"])
            all_plots_dict = benchmarks_dict.setdefault(benchmark, {})
            conf_tuple = (objectives, fidelities, costs)
            if conf_tuple not in all_plots_dict:
                all_plots_dict[conf_tuple] = [_df]
            else:
                all_plots_dict[conf_tuple].append(_df)


    for benchmark, conf_dict in benchmarks_dict.items():
        for conf_tuple, _all_dfs in conf_dict.items():
            df_agg = {}
            objective = conf_tuple[0]
            fidelity = conf_tuple[1]
            cost = conf_tuple[2]
            for _df in _all_dfs:
                if _df.empty:
                    continue
                instance = _df[OPTIMIZER_COL].iloc[0]
                if _df[HP_COL].iloc[0] is not None:
                    instance = f"{instance}_{_df[HP_COL].iloc[0]}"
                minimize = _df[SINGLE_OBJ_MINIMIZE_COL].iloc[0]
                seed = _df[SEED_COL].iloc[0]
                if instance not in df_agg:
                    df_agg[instance] = {}
                if int(seed) not in df_agg[instance]:
                    df_agg[instance][int(seed)] = {"results": _df}
                assert objective is not None
                benchmark_name = _df[BENCHMARK_COL].iloc[0]
            plot_results(
                report=df_agg,
                objective=objective,
                fidelity=fidelity,
                cost=cost,
                to_minimize=minimize,
                save_dir=save_dir,
                benchmarks_name=benchmark.split("benchmark=")[-1].split(".")[0],
                figsize=figsize,
                logscale=logscale,
                error_bars=error_bars,
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
        "--benchmark_spec", "-benches",
        nargs="+",
        type=str,
        help="Specification of the benchmark to plot. "
        " (e.g., spec: `benchmark=pd1-cifar100-wide_resnet-2048`, "
        " spec: `benchmark=pd1-cifar100-wide_resnet-2048.objective=valid_error_rate.fidelity=epochs`, " # noqa: E501
        " spec: `benchmark=pd1-imagenet-resnet-512 benchmark=pd1-cifar100-wide_resnet-2048`)"
    )
    parser.add_argument(
        "--optimizer_spec", "-opts",
        type=str,
        nargs="+",
        help="Specification of the optimizer to plot - "
        " (e.g., spec: `optimizer=DEHB`, "
        " spec: `optimizer=DEHB.eta=3`, "
        " spec: `optimizer=DEHB optimizer=SMAC_Hyperband.eta=3`) "
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
    parser.add_argument(
        "--error_bars", "-eb",
        type=str,
        choices=["std", "sem"],
        default="std",
        help="Type of error bars to plot - "
        "std: Standard deviation, "
        "sem: Standard error of the mean"
    )
    parser.add_argument(
        "--budget_type", "-bt",
        type=str,
        choices=["TrialBudget", "FidelityBudget", None],
        default=None,
        help="Type of budget to plot. "
        "If the study contains a mix of Blackbox and MF opts, "
        "Blackbox opts are only plotted using TrialBudget separately. "
        "MF opts are still plotted using FidelityBudget."
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
        error_bars=args.error_bars,
        budget_type=args.budget_type,
    )
