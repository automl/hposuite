from __future__ import annotations

from typing import Any

from hpoglue import Problem


class GlueWrapperFunctions:
    """A collection of wrapper functions around certain hpoglue methods."""

    @staticmethod
    def problem_from_dict(data: dict[str, Any]) -> Problem:
        """Convert a dictionary to a Problem instance."""
        from hposuite.benchmarks import BENCHMARKS
        from hposuite.optimizers import OPTIMIZERS

        return Problem.from_dict(
            data=data,
            benchmarks_dict=BENCHMARKS,
            optimizers_dict=OPTIMIZERS,
        )

