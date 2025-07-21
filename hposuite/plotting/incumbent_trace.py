from __future__ import annotations

import argparse
import logging
import warnings
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
CONTINUATIONS_BUDGET_USED_COL = "result.continuations_budget_used_total"


def calc_eqv_full_evals(
    results: pd.Series,
    budget_total: float,
) -> pd.Series:
    """Calculate equivalent full evaluations for fractional costs."""
    evals = np.arange(1, budget_total + 1)
    _df = results.reset_index()
    _df.columns = ["index", "performance"]
    bins = pd.cut(_df["index"], bins=[-np.inf, *evals], right=False, labels=evals)
    _df["group"] = bins
    group_min = _df.groupby("group")["performance"].min()
    result = group_min.reindex(evals)
    return pd.Series(result.values, index=evals)


def plot_results(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    report: dict[str, Any],
    objective: str,
    cost: str | None,
    to_minimize: bool,
    save_dir: Path,
    benchmarks_name: str,
    fidelity: str | None = None,
    priors: str | None = None,
    budget_type: Literal["TrialBudget", "FidelityBudget", None] = None,
    regret_bound: float | None = None,  # noqa: ARG001
    figsize: tuple[int, int] = (20, 10),
    logscale: bool = False,
    error_bars: Literal["std", "sem"] = "std",
    plot_file_name: str | None = None,
    plot_continuations_only: bool = False,
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
    # Sort the instances dict
    report = dict(sorted(report.items()))
    optimizers = list(report.keys())
    plt.figure(figsize=figsize)
    optim_res_dict = {}
    max_budget = 0
    total_study_budget = 0
    # Note (soham): max_budget and total_study_budget are different when MF optimizers are used.
    # max_budget is the maximum budget used by any optimizer, while total_study_budget is the
    # total budget allocated for the study. max_budget would be greater than total_study_budget
    # when MF optimizers are used with continuations.

    for instance in optimizers:
        continuations = False
        logger.info(f"Plotting {instance}")
        optim_res_dict[instance] = {}
        seed_cost_dict = {}
        seed_cont_dict = {}
        is_fid_opt = False
        for seed in report[instance]:
            results = report[instance][seed]["results"]
            total_study_budget = results[BUDGET_TOTAL_COL].iloc[0]
            cost_list: pd.Series = results[SINGLE_OBJ_COL].values.astype(np.float64)

            if (
                CONTINUATIONS_COL in results.columns
                and
                not pd.isna(results[CONTINUATIONS_COL].iloc[0])
            ):
                continuations = True

            is_fid_opt = FIDELITY_COL in results.columns
            if budget_type is None:
                budget_type = "FidelityBudget" if fidelity is not None else "TrialBudget"
            match budget_type:
                case "FidelityBudget":
                    if is_fid_opt:
                        budget_list = results[FIDELITY_COL].values.astype(np.float64)
                        budget_list = np.cumsum(budget_list)
                        if continuations:
                            continuations_list = (
                                results[CONTINUATIONS_COL]
                                .values.astype(np.float64)
                            )
                            continuations_list = np.cumsum(continuations_list)
                    else:
                        budget_list = np.cumsum(
                            results[BENCHMARK_FIDELITY_MAX_COL].values.astype(np.float64)
                        )
                    budget = budget_list[-1]
                case "TrialBudget":
                    budget = total_study_budget
                    budget_list = results[BUDGET_USED_COL].values.astype(np.float64)
                    if continuations:
                        continuations_list = (
                            results[CONTINUATIONS_BUDGET_USED_COL]
                            .values.astype(np.float64)
                        )

                case _:
                    raise NotImplementedError(f"Budget type {budget_type} not implemented")

            max_budget = max(max_budget, budget)

            seed_cost_dict[seed] = pd.Series(cost_list, index=budget_list)
            if is_fid_opt and budget_type == "TrialBudget":
                seed_cost_dict[seed] = calc_eqv_full_evals(
                    seed_cost_dict[seed],
                    budget
                )
            if continuations:
                seed_cont_dict[seed] = pd.Series(cost_list, index=continuations_list)
                import math
                if budget_type == "TrialBudget":
                    seed_cont_dict[seed] = calc_eqv_full_evals(
                        seed_cont_dict[seed],
                        math.ceil(continuations_list[-1]),
                    )
                if results[BUDGET_USED_COL].iloc[-1] > results[BUDGET_TOTAL_COL].iloc[0]:
                    logger.warning(
                        "This Optimizer was run until Continuations budget was exhausted. "
                        "Plot of the Optimizer's incumbent without continuations would exceed"
                        f" the total budget {total_study_budget} in the x axis for "
                        "FidelityBudget. For TrialBudget, the plot is clipped to the "
                        "total budget.",
                        stacklevel=2,
                    )

        plot_wo_continuations = True

        if plot_continuations_only and continuations:
            plot_wo_continuations = False
            logger.info(f"Plotting continuations for {instance}")

        if plot_wo_continuations:
            seed_cost_df = pd.DataFrame(seed_cost_dict)
            seed_cost_df = seed_cost_df.ffill(axis=0)
            seed_cost_df = seed_cost_df.dropna(axis=0)
            means = pd.Series(seed_cost_df.mean(axis=1), name=f"means_{instance}")
            budget = means.index[-1]
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

            plt.step(
                means.index,
                means,
                where="post",
                label=instance,
                marker=next(markers),
                markersize=5,
                markerfacecolor="#ffffff",
                markeredgecolor=None,
                markeredgewidth=1,
                linewidth=1,
            )
            plt.fill_between(
                means.index,
                means - error,
                means + error,
                alpha=0.1,
                step="post",
                edgecolor=None,
                linewidth=1,
            )

        #For plotting continuations
        if continuations:
            seed_cont_df = pd.DataFrame(seed_cont_dict)
            seed_cont_df = seed_cont_df.ffill(axis=0)
            seed_cont_df = seed_cont_df.dropna(axis=0)
            means_cont = pd.Series(seed_cont_df.mean(axis=1), name=f"means_{instance}")
            cont_budget = means_cont.index[-1]
            error_cont = pd.Series(seed_cont_df.std(axis=1), name=f"std_{instance}")
            optim_res_dict[instance]["cont_means"] = means_cont
            optim_res_dict[instance]["cont_std"] = error_cont
            means_cont = means_cont.cummin() if to_minimize else means_cont.cummax()
            means_cont = means_cont.drop_duplicates()
            error_cont = error_cont.loc[means_cont.index]
            means_cont[cont_budget] = means_cont.iloc[-1]
            error_cont[cont_budget] = error_cont.iloc[-1]
            plt.step(
                means_cont.index,
                means_cont,
                where="post",
                label=f"{instance}_w_continuations",
                marker=next(markers),
                markersize=5,
                markerfacecolor="#ffffff",
                markeredgecolor=None,
                markeredgewidth=1,
                linewidth=1,
            )
            plt.fill_between(
                means_cont.index,
                means_cont - error_cont,
                means_cont + error_cont,
                alpha=0.1,
                step="post",
                edgecolor=None,
                linewidth=1,
            )

    xlabel = "Cumulative Fidelity" if budget_type == "FidelityBudget" else "Full Evaluations"
    plt.xlabel(xlabel)
    plt.ylabel(f"{objective}")
    plot_suffix = (
        f"{benchmarks_name}, {objective=}, \n{fidelity=}, {cost=}, "
        f"{budget_type}={max_budget}, {to_minimize=}, {error_bars=}, {priors=}"
    )
    plt.title(f"Plot for optimizers on {plot_suffix}")
    if logscale:
        plt.xscale("log")
    if len(optimizers) == 1:
        plt.title(f"Performance of {optimizers[0]} on {plot_suffix}")
    plt.legend()
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizers = ",".join(optimizers)

    plot_suffix = plot_suffix.replace("\n", "").replace(", ", ".").replace("'", "")

    save_path = save_dir / f"{plot_suffix}.png"
    if plot_file_name:
        save_path = save_dir / f"{plot_file_name}.png"
        if save_path.exists():
            logger.warning(f"{save_path} already exists. Overwriting.")
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
    plot_file_name: str | None = None,
    plot_continuations_only: bool = False,
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
            priors = run_config["problem"]["priors"]
            if priors is not None:
                assert len(priors[1]) == 1, (
                    "Multi-objective or multiple priors are not supported in hposuite yet. "
                    "Please use one single-objective prior."
                )
                priors = priors[0]

            # Add default benchmark fidelity to a blackbox Optimizer to compare it
            # alongside MF optimizers if the latter exist in the study
            bench_num_fids = _df[BENCHMARK_COUNT_FIDS].iloc[0]
            if fidelities is None and bench_num_fids >= 1:
                fid = next(
                    bench[1]["fidelities"]
                    for bench in all_benches
                    if bench[0] == benchmark_name
                )
                if fid == _df[BENCHMARK_FIDELITY_NAME].iloc[0]:
                # Note (soham): Study config is saved in such a way that if Blackbox Optimizers
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
            conf_tuple = (objectives, fidelities, costs, priors)
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
            priors = conf_tuple[3]
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
                budget_type=budget_type,
                cost=cost,
                priors=priors,
                to_minimize=minimize,
                save_dir=save_dir,
                benchmarks_name=benchmark.split("benchmark=")[-1].split(".")[0],
                figsize=figsize,
                logscale=logscale,
                error_bars=error_bars,
                plot_file_name=plot_file_name,
                plot_continuations_only=plot_continuations_only,
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
        "--figsize", "-fig",
        type=int,
        nargs="+",
        default=(20, 10),
        help="Size of the figure to plot",
    )
    parser.add_argument(
        "--logscale", "-log",
        action="store_true",
        help="Use log scale for the x-axis",
    )
    parser.add_argument(
        "--error_bars", "-err",
        type=str,
        choices=["std", "sem"],
        default="sem",
        help="Type of error bars to plot - "
        "std: Standard deviation, "
        "sem: Standard error of the mean"
    )
    parser.add_argument(
        "--budget_type", "-b",
        type=str,
        choices=["TrialBudget", "FidelityBudget", None],
        default=None,
        help="Type of budget to plot. "
        "If the study contains a mix of Blackbox and MF opts, "
        "Blackbox opts are only plotted using TrialBudget separately. "
        "MF opts are still plotted using FidelityBudget."
    )
    parser.add_argument(
        "--plot_file_name", "-name",
        type=str,
        help="Name of the plot file to save",
        default=None
    )
    parser.add_argument(
        "--plot_continuations_only", "-cont",
        action="store_true",
        help="Only plot continuations budget"
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
        plot_file_name=args.plot_file_name,
        plot_continuations_only=args.plot_continuations_only,
    )
