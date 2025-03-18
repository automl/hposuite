## Optuna README

[Optuna Reference](https://github.com/optuna/optuna)

| Optimizer      | Optimizer Name in `hposuite`                                                 | Blackbox | Multi-Fidelity (MF) | Multi-Objective (MO) | MO-MF | Priors | Hyperparameters  | Tabular Benchmarks | Continuations   |
|----------------|---------------------------|----------|---------------------|----------------------|-------|--------|-----------------------------------------------|-------------------|-----------------|
| TPE            | `"Optuna"` (TPE is automatically selected for single-objective problems)     | ✓        |                     |                      |       |        |                                           |                   |               |
| NSGA2          | `"Optuna"` (NSGA2 is automatically selected for multi-objective problems)    |          |                     | ✓                    |       |        |                                           |                   |                     |