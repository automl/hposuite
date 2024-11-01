from __future__ import annotations

import argparse
import logging

from hposuite.optimizers import (
    BB_OPTIMIZERS,
    MF_OPTIMIZERS,
    MO_OPTIMIZERS,
    OPTIMIZERS,
    SO_OPTIMIZERS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main(
    list: str
):
    match list:
        case "all":
            logger.info(OPTIMIZERS.keys())
        case "mf":
            logger.info(MF_OPTIMIZERS.keys())
        case "nonmf":
            logger.info(BB_OPTIMIZERS.keys())
        case "mo":
            logger.info(MO_OPTIMIZERS.keys())
        case "so":
            logger.info(SO_OPTIMIZERS.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        required=True,
        choices=["all", "mf", "nonmf", "mo", "so"],
        help="List all Optimizers or filter by type"
        "\nall: list all Optimizers"
        "\nmf: list multi-fidelity Optimizers"
        "\nnonmf: list non-multi-fidelity Optimizers"
        "\nmo: list multi-objective Optimizers"
        "\nso: list single-objective Optimizers",
    )
    args = parser.parse_args()
    main(args.list)
