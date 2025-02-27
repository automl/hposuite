from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import site
import subprocess
import sys
from pathlib import Path
from typing import Any

from hpoglue import Problem
from packaging import version

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