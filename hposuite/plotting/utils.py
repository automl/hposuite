from __future__ import annotations

# ruff: noqa: PD901
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

SEED_COL = "run.seed"
PROBLEM_COL = "problem.name"
OPTIMIZER_COL = "optimizer.name"
HP_COL = "run.opt.hp_str"
SINGLE_OBJ_COL = "result.objective.1.value"
SECOND_OBJ_COL = "result.objective.2.value"
BUDGET_USED_COL = "result.budget_used_total"
BUDGET_TOTAL_COL = "problem.budget.total"


def _plot_single_objective_incumbent_trace_for_problem_and_optimizer(
    df: pd.DataFrame,
    *,
    ycol: str,
    ax: plt.Axes,
    to_minimize: bool,
    regret_bound: float,
    marker: str,
    hue: str,
    label: str,
) -> None:
    assert len(df) > 0
    assert PROBLEM_COL in df.columns

    problem_names = df[PROBLEM_COL]
    assert (problem_names[0] == df[PROBLEM_COL]).all()
    optimizer_names = df[OPTIMIZER_COL]
    assert (optimizer_names[0] == df[OPTIMIZER_COL]).all()
    optimizer_hps = df[HP_COL]
    assert (optimizer_hps[0] == df[HP_COL]).all()
    assert ycol in df.columns

    match to_minimize, regret_bound:
        case True, None:
            assert df[ycol].is_monotonic_decreasing
            df["y"] = df[ycol]
        case True, _:
            assert df[ycol].is_monotonic_decreasing
            df["y"] = df[ycol] - regret_bound
        case False, None:
            assert df[ycol].is_monotonic_increasing
            df["y"] = df[ycol]
        case False, _:
            assert df[ycol].is_monotonic_increasing
            df["y"] = regret_bound - df[ycol]
        case _:
            raise ValueError("Invalid combination of `to_minimize` and `regret_bound`")

    if SEED_COL in df.columns:
        full_frame = pd.DataFrame()
        for seed, seed_df in df.groupby(SEED_COL):
            full_frame[seed] = seed_df[[BUDGET_USED_COL, "y"]].set_index(BUDGET_USED_COL)

        _data = full_frame.sort_index().ffill().dropna()
        _data = _data.agg(["mean", "std"], axis=1)

        _xs = _data.index.astype(np.float32).to_numpy()
        _means = _data["mean"].to_numpy()  # type: ignore
        _stds = _data["std"].to_numpy()  # type: ignore
    else:
        _xs = df[BUDGET_USED_COL].astype(np.float32).to_numpy()
        _means = df["y"].to_numpy()
        _stds = None

    ax.plot(  # type: ignore
        _xs,
        _means,
        drawstyle="steps-post",
        label=label,
        linestyle="solid",  # type: ignore
        markevery=10,
        marker=marker,
        linewidth=3,
    )

    if _stds is not None:
        ax.fill_between(
            _xs,
            _means - _stds,
            _means + _stds,
            alpha=0.2,
            color=hue,
            edgecolor=hue,
            linewidth=2,
            step="post",
        )
