from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from hpoglue import BenchmarkDescription, Config, Query
from hpoglue.fidelity import ContinuousFidelity, ListFidelity, RangeFidelity
from scipy.stats import qmc

from hposuite.benchmarks import BENCHMARKS
from hposuite.constants import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _create_sobol_grid(n_samples, dim, seed=0) -> np.ndarray:
    sobol = qmc.Sobol(d=dim, seed=seed)
    return sobol.random(n_samples)


def _scale_to_bounds(data: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return qmc.scale(data, l_bounds=bounds[0], u_bounds=bounds[1])

def _save_tabular(
    df: pd.DataFrame,
    save_dir: Path,
    name: str,
) -> None:
    """Saves the tabular data to a parquet file."""
    save_dir = save_dir / f"{name}.parquet"
    df.to_parquet(save_dir)
    logger.info(f"Saved tabular data to {save_dir.resolve()}.")


def create_tabular(
    n_samples: int,
    benchmark_desc: BenchmarkDescription,
    tabular_suite_name: str,
    task: str,
    seed: int = 0,
    data_dir: Path = DATA_DIR,
    fidelity: str | None = None,
) -> None:
    """Creates a Tabular Benchmark from the given Synthetic/Functional Benchmark."""
    space = benchmark_desc.config_space
    bench = benchmark_desc.load(benchmark_desc)
    dim = len(space.values())

    fidelity_space: tuple[str, list[int | float]] | None = None

    if fidelity:
        assert fidelity in benchmark_desc.fidelities, (
            f"Fidelity {fidelity} not found in benchmark fidelities."
        )
        match benchmark_desc.fidelities[fidelity]:
            case ContinuousFidelity():
                sampled_fids = _create_sobol_grid(n_samples, 1, seed)
                fidelity_space = (
                    fidelity,
                    _scale_to_bounds(
                        sampled_fids,
                        np.array([
                            [
                                benchmark_desc.fidelities[fidelity].min,
                                benchmark_desc.fidelities[fidelity].max
                            ]
                        ])
                    )
                )
            case ListFidelity():
                fidelity_space = (fidelity, benchmark_desc.fidelities[fidelity].values)
            case RangeFidelity():
                _min, _max, _step = (
                    benchmark_desc.fidelities[fidelity].min,
                    benchmark_desc.fidelities[fidelity].max,
                    benchmark_desc.fidelities[fidelity].stepsize
                )
                fidelity_space = (
                    fidelity,
                    np.arange(_min, _max + _step, _step)
                )
            case _:
                raise TypeError(
                    f"Unsupported fidelity type: {type(benchmark_desc.fidelities[fidelity])}"
                )

    logger.info(f"Generating {n_samples} configs using Sobol sequence for {benchmark_desc.name}.")
    sampled_configs = _create_sobol_grid(n_samples, dim, seed)
    config_dict = {}
    bounds = np.array(
        [
            np.array([hp.lower, hp.upper])
            for hp in space.values()
        ]
    ).T
    assert bounds.shape == (2, dim), "Bounds shape mismatch."
    assert bounds.shape[0] == 2, "First dimension of bounds must be 2." # noqa: PLR2004
    assert bounds.shape[1] == dim, "Dimension mismatch."
    scaled_configs = _scale_to_bounds(sampled_configs, bounds)

    logger.info(f"Querying given benchmark for {n_samples} configs.")
    for config_id, config in enumerate(scaled_configs, start=1):
        configs= {
            hp.name: config[i]
            for i, hp in enumerate(space.values())
        }
        if fidelity:
            for fid in fidelity_space[1]:
                query = Query(
                    config=Config(config_id=config_id, values=configs),
                    fidelity=(fidelity, fid)
                )
                results = bench.query(query).values
                config_dict[(config_id, fid)] = {
                    **configs,
                    **results
                }
            continue

        query = Query(
            config=Config(config_id=config_id, values=configs),
            fidelity=None
        )
        results = bench.query(query).values
        config_dict[config_id] = {
            **configs,
            **results
        }

    logger.info(f"Saving tabular data for {benchmark_desc.name}.")
    _df = pd.DataFrame.from_dict(config_dict, orient="index")
    if fidelity:
        _df.index.names = ["config_id", fidelity]
    else:
        _df.index.name = "config_id"
    print(_df.head())  # noqa: T201
    save_dir = data_dir / tabular_suite_name
    save_dir.mkdir(parents=True, exist_ok=True)
    _save_tabular(_df, save_dir, f"{task}")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--n_samples", "-n",
        type=int,
        default=2000,
        help="Number of configs to generate."
    )
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        required=True,
        help="Benchmark to create tabular benchmark for."
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=0,
        help="Seed for reproducibility."
    )
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default=DATA_DIR,
        help="Directory to save the tabular data."
    )
    parser.add_argument(
        "--tabular_suite_name", "-suite",
        type=str,
        required=True,
        help="Name of the tabular benchmark suite. Eg: bbob_tabular"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="Name of the tabular benchmark. Eg: bbob function name: f1-2-0"
    )
    parser.add_argument(
        "--fidelity", "-f",
        type=str,
        default=None,
        help="Fidelity key to include in the tabular benchmark."
    )
    args = parser.parse_args()

    benchmark = BENCHMARKS[args.benchmark]

    if not isinstance(benchmark, BenchmarkDescription):
        benchmark = benchmark.desc

    if not isinstance(args.data_dir, Path):
        args.data_dir = Path(args.data_dir)

    create_tabular(
        n_samples=args.n_samples,
        benchmark_desc=benchmark,
        tabular_suite_name=args.tabular_suite_name,
        task=args.task,
        data_dir=args.data_dir,
        seed=args.seed,
        fidelity=args.fidelity
    )

