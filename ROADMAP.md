# Current version: 0.1.0

# Version 0.1.0

Initial Release

# Version 0.1.1

* Clean up main README Optimizers table
* Create detailed README docs for each Optimizer in a separate folder (add more details like continuations, etc.)
* Add README for synthetic tabular benchmarks (mfh_tabular and bbob_tabular)

# Version 0.1.2

* Add the following benchmarks:
    * HPO-B
    * Taskset Tabular
    * PD1 Tabular
* Add documentation for the above benchmarks

# Version 0.1.3

* Utility function for checking compatible Optimizer-Benchmark pairs without running
* Have the option for `all` in `hposuite.create_study()` to run an Optimizer on all available benchmarks and vice-versa

# Version 0.1.4

* Convert to Categorical Spaces for Optimizers that don't support Tabular Benchmarks