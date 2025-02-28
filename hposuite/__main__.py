from __future__ import annotations

import argparse
from pathlib import Path

from hposuite.study import Study, create_study


def _study_from_yaml_config(yaml_config: Path) -> Study:
    return Study.from_yaml(yaml_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study_name", "-sn",
        type=str,
        help="Study name",
    )
    parser.add_argument(
        "--output_dir", "-d",
        type=Path,
        help="Results directory",
    )
    parser.add_argument(
        "--study_config", "-cfg",
        type=Path,
        help="Absolute path to the Study configuration file",
    )
    parser.add_argument(
        "--optimizers", "-o",
        nargs="+",
        type=str,
        help="Optimizer to use",
    )
    parser.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        type=str,
        help="Benchmark to use",
    )
    parser.add_argument(
        "--seeds", "-s",
        nargs="+",
        type=int,
        default=None,
        help="Seed(s) to use",
    )
    parser.add_argument(
        "--num_seeds", "-n",
        type=int,
        default=1,
        help="Number of seeds to be generated. "
        "Only used if seeds is not provided",
    )
    parser.add_argument(
        "--budget", "-bgt",
        type=int,
        default=50,
        help="Budget to use",
    )
    parser.add_argument(
        "--overwrite", "-ow",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--continuations", "-c",
        action="store_false",
        help="Use continuations",
    )
    parser.add_argument(
        "--exec_type", "-x",
        type=str,
        default="sequential",
        choices=["sequential", "parallel"],
        help="Execution type",
    )
    parser.add_argument(
        "--group_by", "-g",
        type=str,
        default=None,
        choices=["opt", "bench", "opt_bench", "seed", "mem"],
        help="Runs dump group by\n"
        "Only used if exec_type is dump"
    )
    parser.add_argument(
        "--on_error", "-oe",
        type=str,
        default="warn",
        choices=["warn", "raise", "ignore"],
        help="Action to take on error",
    )
    parser.add_argument(
        "--auto_env_handling", "-ae",
        action="store_true",
        help="Automatically create and use isolated run environments",
    )
    args = parser.parse_args()

    if args.study_config:
        study = _study_from_yaml_config(args.study_config)
    else:
        study = create_study(
            output_dir=args.output_dir,
            name=args.study_name,
            optimizers=args.optimizers,
            benchmarks=args.benchmarks,
            seeds=args.seeds,
            num_seeds=args.num_seeds,
            budget=args.budget,
            group_by=args.group_by,
            on_error=args.on_error,
        )
    study.optimize(
        continuations=args.continuations,
        overwrite=args.overwrite,
        exec_type=args.exec_type,
        auto_env_handling=args.auto_env_handling,
    )

