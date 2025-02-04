from __future__ import annotations

from typing import TYPE_CHECKING

from lib.optimizers.ex_blackbox_opt import Ex_Blackbox_Opt
from lib.optimizers.ex_mo_opt import Ex_MO_Opt
from lib.optimizers.ex_multifidelity_opt import Ex_Multifidelity_Opt
from lib.optimizers.random_search import RandomSearch, RandomSearchWithPriors

if TYPE_CHECKING:
    from hpoglue import Optimizer

OPTIMIZERS: dict[str, type[Optimizer]] = {
    RandomSearch.name: RandomSearch,
    RandomSearchWithPriors.name: RandomSearchWithPriors,
    Ex_MO_Opt.name: Ex_MO_Opt,
    Ex_Multifidelity_Opt.name: Ex_Multifidelity_Opt,
    Ex_Blackbox_Opt.name: Ex_Blackbox_Opt,
}

__all__ = [
    "OPTIMIZERS",
]
