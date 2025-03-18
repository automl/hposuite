## Detailed SMAC Optimizers table

[SMAC Reference](https://github.com/automl/SMAC3)

| Optimizer                       | Optimizer Name in `hposuite` | Blackbox | Multi-Fidelity (MF) | Multi-Objective (MO) | MO-MF | Priors | Hyperparameters                               | Tabular Benchmarks | Continuations   |
|---------------------------------|---------------------------|----------|---------------------|----------------------|-------|--------|-----------------------------------------------|-------------------|-----------------|
| BlackBoxFacade                  | `"SMAC_BO"`                  | ✓        |                     | ✓                     |       |        | `xi`                                          |                   |                  |
| HyperbandFacade                 | `"SMAC_Hyperband"`           |          |  ✓                  |                      |       |        | `eta`                                         |                   | ✓                |
| MultiFidelityFacade (BOHB)      | `"SMAC_BOHB"`                |          | ✓                   |                      |       |        | `eta`                                         |                   | ✓                |