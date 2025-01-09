# hposuite
A lightweight framework for benchmarking HPO algorithms

## Minimal Example to run hposuite

```python
from hposuite import create_study

study = create_study(
    name="hposuite_demo",
    output_dir="./hposuite-output",
    optimizers=[...],
    benchmarks=[...],
    num_seeds=5,
    budget=100,
)

study.optimize()
```

> [!TIP]
> * See below for example of [Running multiple Optimizers on multiple Benchmarks](#Simple-example-to-run-multiple-Optimizers-on-multiple-benchmarks)
> * Check this example [notebook](examples/hposuite_demo.ipynb) for more examples
> * Check out [hpoglue](https://github.com/automl/hpoglue) for core HPO API for interfacing an Optimizer and Benchmark

## Installation

### Create a Virtual Environment using Venv
```bash
python -m venv hposuite_env
source hposuite_env/bin/activate
```
### Installing from PyPI

```bash
pip install hposuite # Current not functional
```

> [!TIP]
> * `pip install hposuite["notebook"]` - For usage in a notebook
> * `pip install hposuite["all]` - To install hposuite with all available optimizers and benchmarks
> * `pip install hposuite["all_opts]` - To install hposuite with all available optimizers only
> * `pip install hposuite["all_benchmarks]` - To install hposuite with all available benchmarks only

### Installation from source

```bash
git clone https://github.com/automl/hposuite.git
cd hposuite

pip install -e . # -e for editable install
```


### Simple example to run multiple Optimizers on multiple benchmarks

```python
from hposuite.benchmarks import BENCHMARKS
from hposuite.optimizers import OPTIMIZERS

from hposuite import create_study

study = create_study(
    name="smachb_dehb_mfh3good_pd1",
    output_dir="./hposuite-output",
    optimizers=[
        OPTIMIZERS["SMAC_Hyperband"],
        OPTIMIZERS["DEHB_Optimizer"]
    ],
    benchmarks=[
        BENCHMARKS["mfh3_good"],
        BENCHMARKS["pd1-imagenet-resnet-512"]
    ],
    num_seeds=5,
    budget=100,
)

study.optimize()

```


### Results and Plotting

#### Results

Results are saved in the `Run` subdirectories within the main `Study` directory as parquet files. \
The `Study` directory and the individual `Run` directory paths are logged when running `Study.optimize()`

#### Plotting

```bash
python -m hposuite.plotting.utils --save_dir <Absolute_path_of_main_directory_to_save_Studies> --study_dir <study_directory_hash>
```