from __future__ import annotations

from itertools import cycle

import matplotlib.pyplot as plt
from more_itertools import take

MARKERS = [
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


def categorical_colors(n: int) -> list[str]:
    """Get a categorical color palette."""
    if n <= 10:  # noqa: PLR2004
        return take(n, iter(plt.get_cmap("tab10").colors))  # type: ignore

    return take(n, cycle(plt.get_cmap("tab20").colors))  # type: ignore


def distinct_markers(n: int) -> list[str]:
    """Get a distinct set of markers."""
    c = cycle(MARKERS)
    return take(n, c)
