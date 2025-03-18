## RandomSearch Optimizers README

[RandomSearch API Reference](https://github.com/automl/hposuite/blob/main/hposuite/optimizers/random_search.py)

| Optimizer            | Optimizer Name in `hposuite` | Blackbox | Multi-Fidelity (MF) | Multi-Objective (MO) | MO-MF | Priors | Hyperparameters                    | Tabular Benchmarks | Continuations   |
|---------------------------------|---------------------------|----------|---------------------|----------------------|-------|--------|-----------------------------------------------|-------------------|-----------------|
| RandomSearch                | `"RandomSearch"`          | ✓        |                     | ✓                    |       |        |                                           | ✓                 |               |
| RandomSearch with priors    | `"RandomSearchWithPriors"` | ✓        |                     | ✓                    |       | ✓      |                                           |                   |               |