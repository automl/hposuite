## Nevergrad README

[Nevergrad Reference](https://github.com/facebookresearch/nevergrad)

| Optimizer      | Optimizer Name in `hposuite`                                                 | Blackbox | Multi-Fidelity (MF) | Multi-Objective (MO) | MO-MF | Priors | Hyperparameters  | Tabular Benchmarks | Continuations   |
|----------------|---------------------------|----------|---------------------|----------------------|-------|--------|-----------------------------------------------|-------------------|-----------------|
| all             | default: `"NGOpt"`. Others [see below](#Nevergrad-optimizers-choice)            | ✓        |                     | ✓                    |       |        | optimizer choice [see below](#Nevergrad-optimizers-choice) |                   |                 |



### Nevergrad optimizers choice

[Nevergrad Optimizers API reference](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#optimizers) <br>
[Nevergrad optimizerlib GitHub](https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/optimization/optimizerlib.py)

Optimizer-Type          | Name                      | Reference            |
------------------------|---------------------------|----------------------|
Configurable Optimizers | check reference           | [Configurable Optimizers](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#configurable-optimizers)
Other Optimizers        | check reference           | [Optimizer](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#optimizers)
Named Optimizer         | `HyperOpt`                  | [Hyperopt](https://github.com/hyperopt/hyperopt)
Named Optimizer         | `CMA-ES `                   | [pycma](https://github.com/CMA-ES/pycma)
Named Optimizer         | `bayes_opt`                 | [bayesian-optimization](https://github.com/bayesian-optimization/BayesianOptimization)
Named Optimizer         | `DE`                        | [Differential Evolution - Nevergrad Optimizers API Reference](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution)
Named Optimizer         | `EvolutionStrategy`         | [Evolution Strategy - Nevergrad Optimizers API Reference](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution)


In general, all optimizers other than the `Named Optimizers` mentioned above, that are accessible through the `hposuite` interface of `Nevergrad` can be obtained by running the following code snippet:

``` python
import nevergrad as ng
print(sorted(ng.optimizers.registry.keys()))
```
