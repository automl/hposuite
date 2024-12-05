from __future__ import annotations

from typing import TYPE_CHECKING

from lib.optimizers.random_search import RandomSearch
from lib.optimizers.ex_multifidelity_opt import Ex_Multifidelity_Opt
from lib.optimizers.ex_blackbox_opt import Ex_Blackbox_Opt

if TYPE_CHECKING:
    from hpoglue import Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    RandomSearch.name: RandomSearch,
    Ex_Multifidelity_Opt.name: Ex_Multifidelity_Opt,
    Ex_Blackbox_Opt.name: Ex_Blackbox_Opt,
}

__all__ = [
    "OPTIMIZERS",
]
