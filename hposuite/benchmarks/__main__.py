from __future__ import annotations

import argparse
import logging

from hposuite.benchmarks import BENCHMARKS, MF_BENCHMARKS, NON_MF_BENCHMARKS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main(
    list: str
):
    match list:
        case "all":
            logger.info(BENCHMARKS.keys())
        case "mf":
            logger.info(MF_BENCHMARKS.keys())
        case "nonmf":
            logger.info(NON_MF_BENCHMARKS.keys())
        # TODO: Implement the following cases
        case "mo":
            raise NotImplementedError("No Multi-objective benchmark definitions exist yet")
        case "so":
            raise NotImplementedError("No Single-objective benchmark definitions exist yet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        required=True,
        choices=["all", "mf", "nonmf", "mo", "so"],
        help="List all Benchmarks or filter by type"
        "\nall: list all Benchmarks"
        "\nmf: list multi-fidelity Benchmarks"
        "\nnonmf: list non-multi-fidelity Benchmarks"
        "\nmo: list multi-objective Benchmarks"
        "\nso: list single-objective Benchmarks",
    )
    args = parser.parse_args()
    main(args.list)
