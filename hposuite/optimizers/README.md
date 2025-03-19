## Detailed Overview of Available Optimizers  

| Package                                        | Optimizer                   | Optimizer Name in `hposuite`    | Blackbox | Fidelities Supported | Fidelity Space | Multi-Objective (MO) | Continuations | Expert Priors | Hyperparameters                               | Tabular Benchmarks |
|------------------------------------------------|-----------------------------|---------------------------|----------|--------------------|----------------------|----------------------|--------------|--------------|-----------------------------------------------|-------------------|
| -                                              | RandomSearch                | `"RandomSearch"`          | ✓        | `{0}`              |                   | ✓                    |              |              |                                           | ✓                 |
| -                                              | RandomSearch with priors    | `"RandomSearchWithPriors"` | ✓        | `{0}`              |                   | ✓                    |              | ✓            |                                           |                   |
| [SMAC](https://github.com/automl/SMAC3)        | BlackBoxFacade              | `"SMAC_BO"`               | ✓        | `{0}`              |                   | ✓                    |              |              | `"xi"` ([learn more](https://automl.github.io/SMAC3/main/api/smac.acquisition.function.expected_improvement.html#smac.acquisition.function.expected_improvement.EI))    |                   |
| [SMAC](https://github.com/automl/SMAC3)        | HyperbandFacade             | `"SMAC_Hyperband"`        |          | `{1}`              | Discrete         |                      | ✓            |              | `"eta"` ([learn more](https://automl.github.io/SMAC3/main/api/smac.facade.hyperband_facade.html#smac.facade.hyperband_facade.HyperbandFacade.get_intensifier))                         |                   |
| [SMAC](https://github.com/automl/SMAC3)        | MultiFidelityFacade (BOHB)  | `"SMAC_BOHB"`             |          | `{1}`              | Discrete         |                      | ✓            |              | `"eta"` ([learn more](https://automl.github.io/SMAC3/main/api/smac.facade.multi_fidelity_facade.html#smac.facade.multi_fidelity_facade.MultiFidelityFacade.get_intensifier))     |                   |
| [DEHB](https://github.com/automl/DEHB)         | DEHB                        | `"DEHB"`                  |          | `{1}`              | Discrete       |                      | ✓            |              | `"eta"` ([learn more](https://automl.github.io/DEHB/latest/getting_started/dehb_hps/#dehb-hyperparameters))                           |                   |
| [HEBO](https://github.com/huawei-noah/HEBO)    | HEBO                        | `"HEBO"`                  | ✓        | `{0}`              |                   |                      |              |              |                                           |                   |
| [Nevergrad](https://github.com/facebookresearch/nevergrad) | all  | `"Nevergrad"`         | ✓  | `{0}`              |                   | ✓                    |              |              | Optimizer choice ([see below](#Nevergrad-optimizers-choice)). Default: `"NGOpt"`.  |                   |
| [Optuna](https://github.com/optuna/optuna)     | TPE                         | `"Optuna"` (TPE is automatically selected for single-objective problems) | ✓  | `{0}`              |                   |                      |              |              |                                           |                   |
| [Optuna](https://github.com/optuna/optuna)     | NSGA2                       | `"Optuna"` (NSGA2 is automatically selected for multi-objective problems) |    | `{0}`              |                   | ✓                    |              |              |                                           |                   |
| [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize) | all  | `"Scikit_Optimize"`      | ✓  | `{0}`              |                   |                      |              |              | `"acq_func"`, `"base_estimator"`, `"acq_optimizer"`  [see here for details](#Scikit-Optimize-hyperparameters) |                   |



> [!TIP]
> * Get all available Optimizers using the following code snippet:
> ```python 
> from hposuite.optimizers import OPTIMIZERS
> print(OPTIMIZERS.keys())
> ```

-----------------------------------------------------
### Nevergrad optimizers choice

[Nevergrad Optimizers API reference](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#optimizers) <br>
[Nevergrad optimizerlib GitHub](https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/optimization/optimizerlib.py)

Optimizer-Type          | Name                      | Reference            |
------------------------|---------------------------|----------------------|
Configurable Optimizers | check reference           | [Configurable Optimizers](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#configurable-optimizers)
Other Optimizers        | check reference           | [Optimizer](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#optimizers)
Named Optimizer         | `"Hyperopt"`                  | [Hyperopt](https://github.com/hyperopt/hyperopt)
Named Optimizer         | `"CMA-ES"`                   | [pycma](https://github.com/CMA-ES/pycma)
Named Optimizer         | `"bayes_opt"`                 | [bayesian-optimization](https://github.com/bayesian-optimization/BayesianOptimization)
Named Optimizer         | `"DE"`                        | [Differential Evolution - Nevergrad Optimizers API Reference](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution)
Named Optimizer         | `"EvolutionStrategy"`         | [Evolution Strategy - Nevergrad Optimizers API Reference](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution)


In general, all optimizers other than the `Named Optimizers` mentioned above, that are accessible through the `hposuite` interface of `Nevergrad` can be obtained by running the following code snippet:

``` python
import nevergrad as ng
print(sorted(ng.optimizers.registry.keys()))
```


-------------------------------------------------------------



### Scikit-Optimize hyperparameters

Check out [this page](https://scikit-optimize.github.io/stable/modules/generated/skopt.optimizer.Optimizer.html#skopt.optimizer.Optimizer) for the entire Scikit-Optimize Optimizer Documentation

#### Base Estimators (`"base_estimator"`)

`"GP"`: Gaussian Processes (default) <br>
`"RF"`: Random Forest <br>
`"ET"`: Extra Trees <br>
`"GBRT"`: Gradient Boosted Regression Trees


#### Acquisition Functions (`"acq_func"`)


Acquisition Function | Details             |
---------------------|---------------------|
`"LCB"`                | lower confidence bound. |
`"EI"`                 | for negative expected improvement. |
`"PI"`                 | for negative probability of improvement. |
`"gp_hedge"` (default) | Probabilistically choose one of the above three acquisition functions at every iteration. <br> - The gains `g_i` are initialized to zero. <br> - At every iteration: <br> &nbsp;&nbsp; • Each acquisition function is optimised independently to propose an candidate point `X_i`. <br> &nbsp;&nbsp; • Out of all these candidate points, the next point `X_best` is chosen by `softmax(η * g_i)`. <br> &nbsp;&nbsp; • After fitting the surrogate model with (X_best, y_best), the gains are updated such that  `g_i = -μ(M_i)` |


#### Methods to minimize the Acquisition Function (`"acq_optimizer"`) 

Acquisition optimizer | Details |
----------------------|---------|
`"auto"` (default)      | `"acq_optimizer"` is configured on the basis of the base_estimator and the space searched over. If the space is Categorical or if the estimator provided based on tree-models then this is set to be `"sampling"` |
`"sampling"`            | `"acq_func"` is optimized by computing `"acq_func"` at `n_points` randomly sampled points. |
`"lbfgs"`               | the `"acq_func"` is optimized by: <br> &nbsp;&nbsp; • Sampling `n_restarts_optimizer` points randomly. <br> &nbsp;&nbsp; • `lbfgs` is run for 20 iterations with these points as initial points to find local minima. <br> &nbsp;&nbsp; • The optimal of these local minima is used to update the prior.