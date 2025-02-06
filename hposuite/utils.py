from __future__ import annotations

import os
import sys
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

class HiddenPrints:  # noqa: D101
    def __enter__(self):
        self._original_stdout = sys.stdout
        from pathlib import Path
        sys.stdout = Path(os.devnull).open("w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout