# Current version: `0.1.3`

# Upcoming versions

## Version 0.1.4

* Convert to Categorical Spaces for Optimizers that don't support Tabular Benchmarks

## Version 0.1.5

* Add the following benchmarks:
    * HPO-B
    * Taskset Tabular
    * PD1 Tabular
* Add documentation for the above benchmarks


## Version 0.1.6

* Add some basic tests

---
---


## Version 0.1.x

* Asynchronous Runs using MF-HPO simulatior


# Old releases

## Version 0.1.3
* Use continuations as Run budget
* Consider priors when plotting incumbent trace
* Add CITATION
* README corrections
* Minor bugfixes

## Version 0.1.2

* Utility function for checking compatible Optimizer-Benchmark pairs without running
* Have the option for `all` in `hposuite.create_study()` to run an Optimizer on all available benchmarks and vice-versa
* Add optimizers from NePS
* Bugfixes and style changes in `incumbent_trace.py`
* Adapt to changes in `hpoglue._run` pertaining to continuations

## Version 0.1.1

* Add option to add custom filenames for plots
* Clean up main README Optimizers table
* Add more details to larger Optimizers README table
* Add README for synthetic tabular benchmarks (mfh_tabular and bbob_tabular)
* Add steps to download LCBench-Tabular and PD1 benchmarks
* Create version tag

## Version 0.1.0

Initial Release


