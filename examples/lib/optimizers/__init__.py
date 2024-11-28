from __future__ import annotations

from typing import TYPE_CHECKING

from lib.optimizers.ex_multifidelity_opt import Ex_Multifidelity_Opt
from lib.optimizers.random_search import RandomSearch

if TYPE_CHECKING:
    from hpoglue import Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    RandomSearch.name: RandomSearch,
    Ex_Multifidelity_Opt.name: Ex_Multifidelity_Opt,
}

__all__ = [
    "OPTIMIZERS",
]
