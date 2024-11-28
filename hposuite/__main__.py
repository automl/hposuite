from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from hposuite.study import create_study


def glue_study(  # noqa: D103, PLR0913
    optimizers: str,
    benchmarks: str,
    *,
    seeds: list,
    num_seeds: int,
    budget: int,
    exp_name: str,
    output_dir: Path,
    exec_type: str,
    group_by: str,
    overwrite: bool,
    continuations: bool,
    on_error: str,
):
    study = create_study(
        output_dir=output_dir,
        name=exp_name,
        optimizers=optimizers,
        benchmarks=benchmarks,
        seeds=seeds,
        num_seeds=num_seeds,
        budget=budget,
        group_by=group_by,
        on_error=on_error,
    )
    study.optimize(
        continuations=continuations,
        overwrite=overwrite,
        exec_type=exec_type,
    )

def _get_from_yaml_config(config_path: Path) -> dict:
    with config_path.open() as file:
        return yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", "-e",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--output_dir", "-od",
        type=Path,
        help="Results directory",
    )
    parser.add_argument(
        "--exp_config", "-ec",
        type=Path,
        help="Absolute path to the experiment configuration file",
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
        action="store_true",
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
    args = parser.parse_args()

    if args.exp_config:
        config = _get_from_yaml_config(args.exp_config)
        glue_study(
            optimizers=config["optimizers"],
            benchmarks=config["benchmarks"],
            seeds=config.get("seeds"),
            num_seeds=config.get("num_seeds", 1),
            budget=config.get("budget", 50),
            exp_name=config.get("exp_name"),
            output_dir=config.get("output_dir"),
            overwrite=config.get("overwrite", False),
            continuations=config.get("continuations", False),
            exec_type=config.get("exec_type", "sequential"),
            group_by=config.get("group_by"),
            on_error=config.get("on_error", "warn"),
        )
    else:
        glue_study(
            optimizers=args.optimizers,
            benchmarks=args.benchmarks,
            seeds=args.seeds,
            num_seeds=args.num_seeds,
            budget=args.budget,
            exp_name=args.exp_name,
            output_dir = args.output_dir,
            overwrite=args.overwrite,
            continuations=args.continuations,
            exec_type=args.exec_type,
            group_by=args.group_by,
            on_error=args.on_error,
        )

