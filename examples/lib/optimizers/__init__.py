from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from hposuite.exceptions import OptBenchNotInstalledError

if TYPE_CHECKING:
    from hpoglue import Optimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

modules = [
    ("lib.optimizers.ex_blackbox_opt", "Ex_Blackbox_Opt"),
    ("lib.optimizers.ex_mo_opt", "Ex_MO_Opt"),
    ("lib.optimizers.ex_multifidelity_opt", "Ex_Multifidelity_Opt"),
    ("lib.optimizers.random_search", "RandomSearch", "RandomSearchWithPriors"),
]

imported_opt_cls = []

for module_name, *attrs in modules:
    try:
        module = importlib.import_module(module_name)
        for attr in attrs:
            opt = getattr(module, attr)
            imported_opt_cls.append(opt)
    except ImportError as e:
        logger.warning(OptBenchNotInstalledError(module_name, e.msg))



OPTIMIZERS: dict[str, type[Optimizer]] = {opt.name: opt for opt in imported_opt_cls}

MF_OPTIMIZERS: dict[str, type[Optimizer]] = {}
BB_OPTIMIZERS: dict[str, type[Optimizer]] = {}
MO_OPTIMIZERS: dict[str, type[Optimizer]] = {}
SO_OPTIMIZERS: dict[str, type[Optimizer]] = {}

for name, opt in OPTIMIZERS.items():
    if "single" in opt.support.fidelities:
        MF_OPTIMIZERS[name] = opt
    else:
        BB_OPTIMIZERS[name] = opt
    if "many" in opt.support.objectives:
        MO_OPTIMIZERS[name] = opt
    if "single" in opt.support.objectives:
        SO_OPTIMIZERS[name] = opt
__all__ = [
    "OPTIMIZERS",
    "MF_OPTIMIZERS",
    "BB_OPTIMIZERS",
    "MO_OPTIMIZERS",
    "SO_OPTIMIZERS",
]

