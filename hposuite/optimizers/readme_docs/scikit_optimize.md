## Scikit-Optimize README

[Scikit-Optimize Reference](https://github.com/scikit-optimize/scikit-optimize)

| Optimizer                       | Optimizer Name in `hposuite` | Blackbox | Multi-Fidelity (MF) | Multi-Objective (MO) | MO-MF | Priors | Hyperparameters                               | Tabular Benchmarks | Continuations   |
|---------------------------------|---------------------------|----------|---------------------|----------------------|-------|--------|-----------------------------------------------|-------------------|-----------------|
| all  | `"Scikit_Optimize"`      | ✓        |                     |                      |       |        | `"acq_func"`, `"base_estimator"`, `"acq_optimizer"`  [see here for details](#Scikit-Optimize-hyperparameters) |                   |                 |


### Scikit-Optimize hyperparameters

Check out [this page](https://scikit-optimize.github.io/stable/modules/generated/skopt.optimizer.Optimizer.html#skopt.optimizer.Optimizer) for the entire Scikit-Optimize Optimizer Documentation

#### Base Estimators (`base_estimator`)

`GP`: Gaussian Processes (default) <br>
`RF`: Random Forest <br>
`ET`: Extra Trees <br>
`GBRT`: Gradient Boosted Regression Trees


#### Acquisition Functions (`acq_func`)


Acquisition Function | Details             |
---------------------|---------------------|
`LCB`                | lower confidence bound. |
`EI`                 | for negative expected improvement. |
`PI`                 | for negative probability of improvement. |
`gp_hedge` (default) | Probabilistically choose one of the above three acquisition functions at every iteration. <br> - The gains `g_i` are initialized to zero. <br> - At every iteration: <br> &nbsp;&nbsp; • Each acquisition function is optimised independently to propose an candidate point `X_i`. <br> &nbsp;&nbsp; • Out of all these candidate points, the next point `X_best` is chosen by `softmax(η * g_i)`. <br> &nbsp;&nbsp; • After fitting the surrogate model with (X_best, y_best), the gains are updated such that  `g_i = -μ(M_i)` |


#### Methods to minimize the Acquisition Function (`acq_optimizer`) 

Acquisition optimizer | Details |
----------------------|---------|
`auto` (default)      | `acq_optimizer` is configured on the basis of the base_estimator and the space searched over. If the space is Categorical or if the estimator provided based on tree-models then this is set to be `sampling` |
`sampling`            | `acq_func` is optimized by computing `acq_func` at `n_points` randomly sampled points. |
`lbfgs`               | the `acq_func` is optimized by: <br> &nbsp;&nbsp; • Sampling `n_restarts_optimizer` points randomly. <br> &nbsp;&nbsp; • `lbfgs` is run for 20 iterations with these points as initial points to find local minima. <br> &nbsp;&nbsp; • The optimal of these local minima is used to update the prior.