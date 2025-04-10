from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import site
import subprocess
import sys
import warnings
from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Any, TypeAlias

from hpoglue import BenchmarkDescription, FunctionalBenchmark, Optimizer, Problem
from hpoglue.utils import dict_to_configpriors
from packaging import version

OptWithHps: TypeAlias = tuple[type[Optimizer], Mapping[str, Any]]

BenchWith_Objs_Fids: TypeAlias = tuple[BenchmarkDescription, Mapping[str, Any]]

GLOBAL_SEED = 42

logger = logging.getLogger(__name__)


HPOSUITR_REPO = "github.com/automl/hposuite.git"
HPOSUITE_EDITABLE = Path(__file__).parent.parent
HPOSUITE_PYPI = "hposuite"
HPOSUITE_GIT_SSH_INSTALL = "git+ssh://git@github.com/automl/hposuite.git"

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



def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed in the current virtual environment."""
    # Extract version constraint if present
    version_constraints = ["==", ">=", "<=", ">", "<"]
    version_check = None
    for constraint in version_constraints:
        if constraint in package_name:
            package_name, version_spec = package_name.split(constraint)
            version_check = (constraint, version_spec)
            break

    # Normalize package name (replace hyphens with underscores)
    package_name = package_name.replace("-", "_")

    # Remove dependencies from the package name
    package_name = package_name.split("[")[0]

    # Get the site-packages directory of the current virtual environment
    venv_site_packages = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    venv_prefix = sys.prefix  # Virtual environment root

    # Check if the package is installed in the virtual environment
    for site_package_path in venv_site_packages:
        package_path = Path(site_package_path) / package_name

        # Check if the package exists in the site-packages directory
        if package_path.exists() and venv_prefix in str(package_path):
            installed_version = importlib.metadata.version(package_name)
            if version_check:
                constraint, required_version = version_check
                return _check_package_version(installed_version, required_version, constraint)
            return True

        # Check if package is installed as different name (e.g., .dist-info or .egg-info)
        dist_info_pattern = f"{package_name}*"
        dist_info_paths = list(Path(site_package_path).glob(dist_info_pattern))
        if dist_info_paths:
            dist_info_name = dist_info_paths[0].name.replace(".dist-info", "") \
                .replace(".egg-info", "")
            installed_version = dist_info_name.split("-")[-1]
            if version_check:
                constraint, required_version = version_check
                return _check_package_version(installed_version, required_version, constraint)
            return True

    return False

def _check_package_version(
    installed_version: str,
    required_version: str,
    check_key: str,
):
    """Check if the installed package version satisfies the required version."""
    installed_version = version.parse(installed_version)
    required_version = version.parse(required_version)
    match check_key:
        case "==":
            return installed_version == required_version
        case ">=":
            return installed_version >= required_version
        case "<=":
            return installed_version <= required_version
        case ">":
            return installed_version > required_version
        case "<":
            return installed_version < required_version


def get_current_installed_hposuite_version() -> str:
    """Retrieve the currently installed version of hposuite."""
    cmd = ["pip", "show", "hposuite"]
    logger.debug(cmd)
    output = subprocess.run(  # noqa: S603
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = output.stdout.strip().splitlines()
    for line in lines:
        if "Version: " in line:
            return line.split(": ")[1]

    raise RuntimeError(f"Could not find hposuite version in {lines}.")


def get_current_ConfigSpace_version() -> version.Version:  # noqa: N802
    """Retrieve the currently installed version of ConfigSpace."""
    pkg_dict = {dist.name: dist.version for dist in importlib.metadata.distributions()}
    if "ConfigSpace" in pkg_dict:
        return version.parse(pkg_dict["ConfigSpace"])
    raise RuntimeError("ConfigSpace is not installed.")


def compare_installed_CS_version_vs_required(required_version: str) -> str:  # noqa: N802
    """Compare the installed ConfigSpace version with the required version."""
    installed_version = get_current_ConfigSpace_version()
    required_version = version.parse(required_version)
    if installed_version < required_version:
        return "<"
    if installed_version == required_version:
        return "=="
    return ">"


def get_compatible(  # noqa: C901, PLR0912, PLR0915
    *,
    optimizers: (
        str
        | tuple[str, Mapping[str, Any]]
        | type[Optimizer]
        | OptWithHps # tuple[type[Optimizer], Mapping[str, Any]]
        | list[tuple[str, Mapping[str, Any]]]
        | list[str]
        | list[OptWithHps] # list[tuple[type[Optimizer], Mapping[str, Any]]]
        | list[type[Optimizer]]
    ),
    benchmarks: (
        str
        | BenchmarkDescription
        | FunctionalBenchmark
        | tuple[str, Mapping[str, Any]]
        | BenchWith_Objs_Fids # tuple[BenchmarkDescription, Mapping[str, Any]]
        | tuple[FunctionalBenchmark, Mapping[str, Any]]
        | list[str]
        | list[BenchmarkDescription]
        | list[FunctionalBenchmark]
        | list[tuple[str, Mapping[str, Any]]]
        | list[BenchWith_Objs_Fids]   # list[tuple[BenchmarkDescription, Mapping[str, Any]]]
        | list[tuple[FunctionalBenchmark, Mapping[str, Any]]]
    ),
) -> None:
    """Get compatible optimizers and benchmark pairs from lists of optimizers and benchmarks.

    Args:
        optimizers: List of optimizers to check.
                    "all" to check all optimizers.

        benchmarks: List of benchmarks to check.
                    "all" to check all benchmarks.
    """
    from hposuite.benchmarks import BENCHMARKS
    from hposuite.optimizers import OPTIMIZERS
    from hposuite.run import Run

    assert optimizers, "At least one optimizer must be provided!"
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    assert benchmarks, "At least one benchmark must be provided!"
    if not isinstance(benchmarks, list):
        benchmarks = [benchmarks]


    _optimizers: list[OptWithHps] = []
    for optimizer in optimizers:
        match optimizer:
            case "all":
                _optimizers = [(opt, {}) for opt in OPTIMIZERS.values()]
                break
            case str():
                assert optimizer in OPTIMIZERS, (
                    f"Optimizer must be one of {OPTIMIZERS.keys()}\n"
                    f"Found {optimizer}"
                )
                _optimizers.append((OPTIMIZERS[optimizer], {}))
            case tuple():
                opt, hps = optimizer
                match opt:
                    case str():
                        assert opt in OPTIMIZERS, (
                            f"Optimizer must be one of {OPTIMIZERS.keys()}\n"
                            f"Found {opt}"
                        )
                        _optimizers.append((OPTIMIZERS[opt], hps))
                    case type():
                        _optimizers.append((opt, hps))
                    case _:
                        raise TypeError(f"Unknown Optimizer type {type(opt)}")
            case type():
                _optimizers.append(optimizer)
            case _:
                raise TypeError(f"Unknown Optimizer type {type(optimizers)}")


    _benchmarks: list[BenchWith_Objs_Fids] = []
    for benchmark in benchmarks:
        match benchmark:
            case "all":
                _benchmarks = [
                    (
                        bench if not isinstance(bench, FunctionalBenchmark) else bench.desc,
                        {}
                    )
                    for bench in BENCHMARKS.values()
                ]
                break
            case str():
                assert benchmark in BENCHMARKS, (
                    f"Benchmark must be one of {BENCHMARKS.keys()}\n"
                    f"Found {benchmark}"
                )
                if not isinstance(BENCHMARKS[benchmark], FunctionalBenchmark):
                    _benchmarks.append((BENCHMARKS[benchmark], {}))
                else:
                    _benchmarks.append((BENCHMARKS[benchmark].desc, {}))
            case BenchmarkDescription():
                _benchmarks.append((benchmark, {}))
            case FunctionalBenchmark():
                _benchmarks.append((benchmark.desc, {}))
            case tuple():
                bench, bench_hps = benchmark
                match bench:
                    case str():
                        assert bench in BENCHMARKS, (
                            f"Benchmark must be one of {BENCHMARKS.keys()}\n"
                            f"Found {bench}"
                        )
                        if not isinstance(BENCHMARKS[bench], FunctionalBenchmark):
                            _benchmarks.append((BENCHMARKS[bench], bench_hps))
                        else:
                            _benchmarks.append((BENCHMARKS[bench].desc, bench_hps))
                    case BenchmarkDescription():
                        _benchmarks.append((bench, bench_hps))
                    case FunctionalBenchmark():
                        _benchmarks.append((bench.desc, bench_hps))
                    case _:
                        raise TypeError(f"Unknown Benchmark type {type(benchmark)}")
            case _:
                raise TypeError(f"Unknown Benchmark type {type(benchmarks)}")


    _problems: list[Problem] = []
    for (opt, hps), (bench, objs_fids) in product(_optimizers, _benchmarks):
        try:

            objectives: int | str | list[str]
            fidelities: int | str | list[str] | None
            costs: int | str | list[str] | None

            objectives = objs_fids.get("objectives", 1)
            if isinstance(objectives, list) and len(objectives) == 1:
                objectives = objectives[0]

            fidelities = objs_fids.get("fidelities", None)
            if isinstance(fidelities, list) and len(fidelities) == 1:
                fidelities = fidelities[0]

            costs = objs_fids.get("costs", None)
            if isinstance(costs, list) and len(costs) == 1:
                costs = costs[0]

            priors = objs_fids.get("priors")

            _compatible_pairs = []
            _incompatible_pairs = []

            match fidelities, bench.fidelities:
                case None, None:
                    fidelities = None
                case None, _:
                    match opt.support.fidelities[0]:
                        case "single":
                            fidelities = 1
                        case "many":
                            fidelities = len(bench.fidelities)
                        case None:
                            fidelities = None
                        case _:
                            raise ValueError("Invalid fidelity support")
                case str() | int() | list(), _:
                    match opt.support.fidelities[0]:
                        case str():
                            pass
                        case None:
                            fidelities = None
                        case _:
                            raise ValueError("Invalid fidelity support")
                case _:
                    raise ValueError(
                        f"Invalid fidelity type: {type(fidelities)}. "
                        f"Expected None, str, int or list"
                    )

            if priors:
                priors = dict_to_configpriors(priors)


            _problem = Problem.problem(
                optimizer=opt,
                optimizer_hyperparameters=hps,
                benchmark=bench,
                objectives=objectives,
                budget=10,
                fidelities=fidelities,
                costs=costs,
                priors=priors,
            )
            _problems.append(_problem)
        except ValueError as e:
            warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
            continue


    # NOTE: REDUNDANCY CHECK: _runs_per_problem is a dict now to avoid duplicate runs esp.
    # for eg. in case a BO Opt being used in combination with a
    # MF Opt on a benchmark with multiple fidelities -> explicitly adding different fidelities
    # in multiple benchmark instances would create redundant problems if BO Opts are present.
    _runs_per_problem: Mapping[str, Run] = {}
    for _problem in _problems:
        try:
            _run = Run(
                    problem=_problem,
                    seed=1,
                )
            if _run.name not in _runs_per_problem:
                _runs_per_problem[_run.name] = _run
                _compatible_pairs.append(
                    (
                        _problem.optimizer.name,
                        _problem.benchmark.name,
                    )
                )
        except ValueError as e:
            warnings.warn(f"{e}\nTo ignore this, set `on_error='ignore'`", stacklevel=2)
            continue


    match _compatible_pairs, len(_optimizers), len(_benchmarks):
        case [], _, _:
            logger.warning(
                "No compatible pairs found. "
            )
        case _, 1, _:
            opt = _optimizers[0][0].name
            benches = [
                bench for _, bench in _compatible_pairs
            ]
            print(  # noqa: T201
                f"Compatible Benchmarks for Optimizer `{opt}`:\n"
                f"{benches}"
            )
        case _, _, 1:
            bench = _benchmarks[0][0].name
            opts = [
                opt for opt, _ in _compatible_pairs
            ]
            print(  # noqa: T201
                f"Compatible Optimizers for Benchmark `{bench}`:\n"
                f"{opts}"
            )
        case _:
            print(  # noqa: T201
                "Compatible Optimizer-Benchmark pairs:\n"
                f"{_compatible_pairs}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get compatible optimizers and benchmarks.")
    subparsers = parser.add_subparsers(dest="command")
    get_compatible_parser = subparsers.add_parser(
        "get_compatible",
        help="Get compatible optimizers and benchmarks.",
    )
    get_compatible_parser.add_argument(
        "--optimizers", "-opts",
        type=str,
        nargs="+",
        help="List of optimizers to check. Use 'all' to check all optimizers.",
    )
    get_compatible_parser.add_argument(
        "--benchmarks", "-benches",
        type=str,
        nargs="+",
        help="List of benchmarks to check. Use 'all' to check all benchmarks.",
    )
    args = parser.parse_args()

    get_compatible(optimizers=args.optimizers, benchmarks=args.benchmarks)