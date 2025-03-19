## Detailed Benchmarks Table

| Benchmark       | Task (exact string in hposuite)              | Type       | Fidelities    | Main Metrics                    | Cost Metrics    |
|-----------------|----------------------------------------------|------------|--------------|----------------------------------|-----------------|
| Ackley          | -                                            | Synthetic  | -            | `"value"`                        | -               |
| Branin          | -                                            | Synthetic  | -            | `"value"`                        | -               |
| MF-Hartmann     | `"mfh3_good"`                                | Synthetic  | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann     | `"mfh3_bad"`                                 | Synthetic  | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann     | `"mfh3_moderate"`                            | Synthetic  | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann     | `"mfh3_terrible"`                            | Synthetic  | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann     | `"mfh6_good"`                                | Synthetic  | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann     | `"mfh6_bad"`                                 | Synthetic  | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann     | `"mfh6_moderate"`                            | Synthetic  | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann     | `"mfh6_terrible"`                            | Synthetic  | `"z"`        | `"value"`                        | `"fid_cost"`    |
| PD1             | `"pd1-cifar100-wide_resnet-2048"`            | Surrogate  | `"epoch"`    | `"valid_error_rate"`             | `"train_cost"`  |
| PD1             | `"pd1-imagenet-resnet-512"`                  | Surrogate  | `"epoch"`    | `"valid_error_rate"`             | `"train_cost"`  |
| PD1             | `"pd1-lm1b-transformer-2048"`                | Surrogate  | `"epoch"`    | `"valid_error_rate"`             | `"train_cost"`  |
| PD1             | `"pd1-translate_wmt-xformer_translate-64"`   | Surrogate  | `"epoch"`    | `"valid_error_rate"`             | `"train_cost"`  |
| LCBench-Tabular | Task IDs ([listed below](#lcbench-tabular-openml-task-ids)) | Tabular    | `"epoch"`    | `"val_accuracy"`, `"val_cross_entropy"`, `"val_balanced_accuracy"` | `"time"` |
| Pymoo           | Single Objective ([listed below](#pymoo-single-objective))  | Synthetic  | -       | `"value"`                    | -           |
| Pymoo           | Multi-Objective ([listed below](#pymoo-multi-objective))   | Synthetic  | -       | `"value1"`, `"value2"`            | -           |
| Pymoo           | Many-Objective ([listed below](#pymoo-many-objective))    | Synthetic  | -       | `"value1"`, `"value2"`, `"value3"`     | -           |
| BBOB            | 24 single objective, noiseless functions in 6 dimensions and 3 instances ([listed below](#bbob-functions)) | Synthetic | -  | `"value"`       | -        |
| BBOB Tabular    | [See here](#bbob-tabular-benchmark)                    | Tabular    | -            | `"value"`                        | -               |
| MF-Hartmann Tabular    | `"mfh_tabular-mfh3_good"`             | Tabular    | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann Tabular    | `"mfh_tabular-mfh3_bad"`              | Tabular    | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann Tabular    | `"mfh_tabular-mfh3_moderate"`         | Tabular    | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann Tabular    | `"mfh_tabular-mfh3_terrible"`         | Tabular    | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann Tabular    | `"mfh_tabular-mfh6_good"`             | Tabular    | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann Tabular    | `"mfh_tabular-mfh6_bad"`              | Tabular    | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann Tabular    | `"mfh_tabular-mfh6_moderate"`         | Tabular    | `"z"`        | `"value"`                        | `"fid_cost"`    |
| MF-Hartmann Tabular    | `"mfh_tabular-mfh6_terrible"`         | Tabular    | `"z"`        | `"value"`                        | `"fid_cost"`    |


> [!TIP]
> * Get all available Benchmarks using the following code snippet:
> ```python 
> from hposuite.benchmarks import BENCHMARKS
> print(BENCHMARKS.keys())
> ```



> [!TIP]
> * MF-Hartmann Tabular and BBOB Tabular benchmarks are synthetic tabular benchmarks, i.e, they are generated from the respective synthetic benchmark suites.
> * To use MF-Hartmann Tabular and BBOB Tabular benchmarks, first generate them as shown [here](#generating-bbob-tabular-and-mf-hartmann-tabular-benchmarks)

-----------------------------------------------------------------------------------

### Download and setup Benchmarks

#### LCBench Tabular

```bash
python -m hposuite.benchmarks.lcbench_tabular \
    setup \
    --datadir data  # Optional
```

#### PD1

```bash
python -m mfpbench \
    download \
    --benchmark pd1 \
    --data-dir data  # Optional
```


### LCBench Tabular OpenML task IDs:

> [!TIP]
> * get the corresponding benchmark in `hposuite` as `lcbench_tabular-{task_id}`

- `"adult"`  
- `"airlines"`  
- `"albert"`  
- `"Amazon_employee_access"`  
- `"APSFailure"`  
- `"Australian"`  
- `"bank-marketing"`  
- `"blood-transfusion-service-center"`  
- `"car"`  
- `"christine"`  
- `"cnae-9"`  
- `"connect-4"`  
- `"covertype"`  
- `"credit-g"`  
- `"dionis"`  
- `"fabert"`  
- `"Fashion-MNIST"`  
- `"helena"`  
- `"higgs"`  
- `"jannis"`  
- `"jasmine"`  
- `"jungle_chess_2pcs_raw_endgame_complete"`  
- `"kc1"`  
- `"KDDCup09_appetency"`  
- `"kr-vs-kp"`  
- `"mfeat-factors"`  
- `"MiniBooNE"`  
- `"nomao"`  
- `"numerai28.6"`  
- `"phoneme"`  
- `"segment"`  
- `"shuttle"`  
- `"sylvine"`  
- `"vehicle"`  
- `"volkert"`  

----------------------------------------------------------------

### Pymoo Problems

Learn more about Pymoo Test Problems [here](https://pymoo.org/problems/test_problems.html)

#### Pymoo Single-Objective  

- `"ackley"`  
- `"griewank"`  
- `"himmelblau"`  
- `"rastrigin"`  
- `"rosenbrock"`  
- `"schwefel"`  
- `"sphere"`  
- `"zakharov"`  

#### Pymoo Multi-Objective  

- `"kursawe"`  
- `"zdt1"`  
- `"zdt2"`  
- `"zdt3"`  
- `"zdt4"`  
- `"zdt5"`  
- `"zdt6"`  
- `"omnitest"`  
- `"sympart"`  
- `"sympart_rotated"`  

#### Pymoo Many-Objective  

- `"dtlz1"`  
- `"dtlz2"`  
- `"dtlz3"`  
- `"dtlz4"`  
- `"dtlz5"`  
- `"dtlz6"`  
- `"dtlz7"`  


-----------------------------------------------------------------------


### BBOB Functions

Learn more about BBOB functions [here](https://numbbo.github.io/coco/testsuites/bbob)

> [!TIP]
> * In `hposuite` BBOB functions are available in dimensions: (2, 3, 5, 10, 20, 40) and instances: (0, 1, 2)
> * get the corresponding benchmark in `hposuite` as `"bbob-{function_id}-{dimension}-{instance}"`


| Function ID    | Function Name                  | Type                                         |
|----------------|--------------------------------|----------------------------------------------|
| `"f1"`         | Sphere                         | Separable                                    |
| `"f2"`         | Ellipsoidal                    | Separable                                    |
| `"f3"`         | Rastrigin                      | Separable                                    |
| `"f4"`         | Bueche-Rastrigin               | Separable                                    |
| `"f5"`         | Linear Slope                   | Separable                                    |
| `"f6"`         | Attractive Sector              | Low or moderate conditioning                 |
| `"f7"`         | Step Ellipsoidal               | Low or moderate conditioning                 |
| `"f8"`         | Rosenbrock Rotated             | Low or moderate conditioning                 |
| `"f9"`         | Rosenbrock                     | Low or moderate conditioning                 |
| `"f10"`        | Ellipsoidal Rotated            | High conditioning and unimodal               |
| `"f11"`        | Discus                         | High conditioning and unimodal               |
| `"f12"`        | Bent Cigar                     | High conditioning and unimodal               |
| `"f13"`        | Sharp Ridge                    | High conditioning and unimodal               |
| `"f14"`        | Different Powers               | High conditioning and unimodal               |
| `"f15"`        | Rastrigin Rotated              | Multi-modal with adequate global structure   |
| `"f16"`        | Weierstrass                    | Multi-modal with adequate global structure   |
| `"f17"`        | Schaffers F7                   | Multi-modal with adequate global structure   |
| `"f18"`        | Schaffers F7 Condition 10      | Multi-modal with adequate global structure   |
| `"f19"`        | Griewank-Rosenbrock            | Multi-modal with adequate global structure   |
| `"f20"`        | Schwefel                       | Multi-modal with weak global structure       |
| `"f21"`        | Gallagher 101 Peaks            | Multi-modal with weak global structure       |
| `"f22"`        | Gallagher 21 Peaks             | Multi-modal with weak global structure       |
| `"f23"`        | Katsuura                       | Multi-modal with weak global structure       |
| `"f24"`        | Lunacek Bi-Rastrigin           | Multi-modal with weak global structure       |


-------------------------------------------------------------------------------------------------------


### BBOB-Tabular Benchmark

To get the corresponding `bbob_tabular` benchmark in `hposuite`, use: `"bbob_tabular-{function_id}-{dimension}-{instance}"`,
where `function_id`, `dimension` and `instance` are the same as in [bbob](#bbob-functions).


-------------------------------------------------------------------------------------------------------


### Generating BBOB Tabular and MF-Hartmann Tabular benchmarks

#### BBOB Tabular Generation

* `ioh>=0.3.14` must be installed
* Run the following command to generate the tabular benchmark for the `bbob-f1-2-0` synthetic benchmark:
    ```bash
    python -m hposuite.benchmarks.create_tabular \
        --benchmark bbob-f1-2-0 \   # full bbob function name in hposuite
        --tabular_suite_name bbob_tabular \     # name of the tabular suite
        --task f1-2-0 \     # Name of the task in the bbob_tabular benchmark suite
        --n_samples 2000     # Number of configurations to generate
    ```

#### MF-Hartmann Tabular Generation

* `mf-prior-bench>=1.10.0` must be installed
* Run the following command to generate the tabular benchmark for the `mfh3_good` synthetic benchmark:
    ```bash
    python -m hposuite.benchmarks.create_tabular \
        --benchmark mfh3_good \   # full mfh function name in hposuite
        --tabular_suite_name mfh_tabular \     # name of the tabular suite
        --task mfh3_good \     # Name of the task in the mfh_tabular benchmark suite
        --n_samples 2000     # Number of configurations to generate
    ```

