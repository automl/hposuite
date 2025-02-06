from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace
from hpoglue import Config, FunctionalBenchmark, Measure, Result

if TYPE_CHECKING:

    from hpoglue import Query


def ackley_fn(x: np.ndarray) -> float:
    """Compute the Ackley function.

    The Ackley function is a widely used benchmark function for testing optimization algorithms.
    It is defined as follows:

        f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c * x_i))) + a + exp(1)

    where:
        - x is a numpy array of input values.
        - n is the number of dimensions (variables), which is set to 2 in this implementation.
        - a, b, and c are constants with typical values a=20, b=0.2, and c=2*pi.

    Parameters:
    x (np.ndarray): Input array of shape (n_var,).

    Returns:
    float: The computed value of the Ackley function.
    """
    n_var=2
    a=20
    b=1/5
    c=2 * np.pi
    part1 = -1. * a * np.exp(-1. * b * np.sqrt((1. / n_var) * np.sum(x * x)))
    part2 = -1. * np.exp((1. / n_var) * np.sum(np.cos(c * x)))

    return part1 + part2 + a + np.exp(1)


def wrapped_ackley(query: Query) -> Result:  # noqa: D103

    y = ackley_fn(
        np.array(query.config.to_tuple())
    )

    return Result(
        query=query,
        fidelity=None,
        values={"y": y},
    )


ACKLEY_BENCH = FunctionalBenchmark(
    name="ackley",
    config_space=ConfigurationSpace(
        {
            f"x{i}": (-32.768, 32.768) for i in range(2)
        }
    ),
    metrics={"y": Measure.metric((0.0, np.inf), minimize=True)},
    query=wrapped_ackley,
    predefined_points={
        "min": (
            Config(
                config_id="min",
                description="This point yields a global optimum of y:0.0",
                values={"x0": 0.0, "x1": 0.0}
            )
        )
    }
)


def branin_fn(x: np.ndarray) -> float:
    """Compute the value of the Branin function.

    The Branin function is a commonly used test function for optimization algorithms.
    It is defined as:

        f(x) = a * (x2 - b * x1^2 + c * x1 - r)^2 + s * (1 - t) * cos(x1) + s

    where:
        b = 5.1 / (4.0 * pi^2)
        c = 5.0 / pi
        t = 1.0 / (8.0 * pi)

    Args:
        x (np.ndarray): A 2-dimensional input array where x[0] is x1 and x[1] is x2.

    Returns:
        float: The computed value of the Branin function.
    """
    x1 = x[0]
    x2 = x[1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def wrapped_branin(query: Query) -> Result:  # noqa: D103

    y = branin_fn(
        np.array(query.config.to_tuple())
    )

    return Result(
        query=query,
        fidelity=None,
        values={"value": y},
    )


BRANIN_BENCH = FunctionalBenchmark(
    name="branin",
    config_space=ConfigurationSpace(
        {
            f"x{i}": (-32.768, 32.768) for i in range(2)
        }
    ),
    metrics={
            "value": Measure.metric((0.397887, np.inf), minimize=True),
        },
    query=wrapped_branin,
    predefined_points={
        "min": (
            Config(
                config_id="min",
                description="This point yields a global optimum of y:0.39787",
                values={"x0": -np.pi, "x1": 12.275},
            )
        ),
        "min2": (
            Config(
                config_id="min2",
                description="This point yields a global optimum of y:0.39787",
                values={"x0": np.pi, "x1": 2.275},
            )
        ),
        "min3": (
            Config(
                config_id="min3",
                description="This point yields a global optimum of y:0.39787",
                values={"x0": 9.42478, "x1": 2.475},
            )
        )
    }
)