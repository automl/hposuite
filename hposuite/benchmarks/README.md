## Detailed Benchmarks Table

| Benchmark       | Task                                         | Type       | Fidelities | Main Metrics             | Cost Metrics    |
|-----------------|----------------------------------------------|------------|------------|--------------------------|-----------------|
| Ackley          | -                                            | Functional | none       | value                    | -               |
| Branin          | -                                            | Functional | none       | value                    | -               |
| MF-Hartmann     | mfh3_good                                    | Synthetic  | z          | value                    | fid_cost        |
| -               | mfh3_bad                                     | -          | z          | value                    | fid_cost        |
| -               | mfh3_moderate                                | -          | z          | value                    | fid_cost        |
| -               | mfh3_terrible                                | -          | z          | value                    | fid_cost        |
| -               | mfh6_good                                    | -          | z          | value                    | fid_cost        |
| -               | mfh6_bad                                     | -          | z          | value                    | fid_cost        |
| -               | mfh6_moderate                                | -          | z          | value                    | fid_cost        |
| -               | mfh6_terrible                                | -          | z          | value                    | fid_cost        |
| PD1             | pd1-cifar100-wide_resnet-2048                | Surrogate  | epoch      | valid_error_rate         | train_cost      |
| -               | pd1-imagenet-resnet-512                      | -          | epoch      | valid_error_rate         | train_cost      |
| -               | pd1-lm1b-transformer-2048                    | -          | epoch      | valid_error_rate         | train_cost      |
| -               | pd1-translate_wmt-xformer_translate-64       | -          | epoch      | valid_error_rate         | train_cost      |
| JAHSbench       | jahs-CIFAR10                                 | Surrogate  | epoch      | valid_acc                | runtime         |
| -               | jahs-ColorectalHistology                     | -          | epoch      | valid_acc                | runtime         |
| -               | jahs-FashionMNIST                            | -          | epoch      | valid_acc                | runtime         |
| LCBench-Tabular | Task IDs ([listed below](#lcbench-task-ids)) | Tabular    | epoch      | val_accuracy, val_cross_entropy, val_balanced_accuracy        | time            |
| Pymoo           | Single Objective ([listed below](#pymoo-single-objective))          | Synthetic  | -          | value                    | -               |
| -               | Multi-Objective ([listed below](#pymoo-multi-objective))           | Synthetic  | -          | value1, value2           | -               |
| -               | Many-Objective ([listed below](#pymoo-many-objective))            | Synthetic  | -          | value1, value2, value3   | -               |
| BBOB            | 24 single objective, noiseless functions in 6 dimensions and 3 instances ([lsited below](#bbob-functions)) | Synthetic | -        | value | -           |



### LCBench task IDs:

> [!TIP]
> * get the corresponding benchmark in `hposuite` as `lcbench_tabular-{task_id}`

- adult  
- airlines  
- albert  
- Amazon_employee_access  
- APSFailure  
- Australian  
- bank-marketing  
- blood-transfusion-service-center  
- car  
- christine  
- cnae-9  
- connect-4  
- covertype  
- credit-g  
- dionis  
- fabert  
- Fashion-MNIST  
- helena  
- higgs  
- jannis  
- jasmine  
- jungle_chess_2pcs_raw_endgame_complete  
- kc1  
- KDDCup09_appetency  
- kr-vs-kp  
- mfeat-factors  
- MiniBooNE  
- nomao  
- numerai28.6  
- phoneme  
- segment  
- shuttle  
- sylvine  
- vehicle  
- volkert  

----------------------------------------------------------------

### Pymoo Problems

#### Pymoo Single-Objective

- ackley  
- griewank  
- himmelblau  
- rastrigin  
- rosenbrock  
- schwefel  
- sphere  
- zakharov


#### Pymoo Multi-Objective

- kursawe  
- zdt1  
- zdt2  
- zdt3  
- zdt4  
- zdt5  
- zdt6  
- omnitest  
- sympart  
- sympart_rotated  


#### Pymoo Many-Objective

- dtlz1  
- dtlz2  
- dtlz3  
- dtlz4  
- dtlz5  
- dtlz6  
- dtlz7  

-----------------------------------------------------------------------


### BBOB Functions

> [!TIP]
> * In `hposuite` BBOB functions are available in dimensions: (2, 3, 5, 10, 20, 40) and instances: (0, 1, 2)
> * get the corresponding benchmark in `hposuite` as `bbob-{function_id}-{dimension}-{instance}`


| Function ID | Function Name                  | Type                                         |
|-------------|--------------------------------|----------------------------------------------|
| f1          | Sphere                         | Separable                                    |
| f2          | Ellipsoidal                    | Separable                                    |
| f3          | Rastrigin                      | Separable                                    |
| f4          | Bueche-Rastrigin               | Separable                                    |
| f5          | Linear Slope                   | Separable                                    |
| f6          | Attractive Sector              | Low or moderate conditioning                 |
| f7          | Step Ellipsoidal               | Low or moderate conditioning                 |
| f8          | Rosenbrock Rotated             | Low or moderate conditioning                 |
| f9          | Rosenbrock                     | Low or moderate conditioning                 |
| f10         | Ellipsoidal Rotated            | High conditioning and unimodal               |
| f11         | Discus                         | High conditioning and unimodal               |
| f12         | Bent Cigar                     | High conditioning and unimodal               |
| f13         | Sharp Ridge                    | High conditioning and unimodal               |
| f14         | Different Powers               | High conditioning and unimodal               |
| f15         | Rastrigin Rotated              | Multi-modal with adequate global structure   |
| f16         | Weierstrass                    | Multi-modal with adequate global structure   |
| f17         | Schaffers F7                   | Multi-modal with adequate global structure   |
| f18         | Schaffers F7 Condition 10      | Multi-modal with adequate global structure   |
| f19         | Griewank-Rosenbrock            | Multi-modal with adequate global structure   |
| f20         | Schwefel                       | Multi-modal with weak global structure       |
| f21         | Gallagher 101 Peaks            | Multi-modal with weak global structure       |
| f22         | Gallagher 21 Peaks             | Multi-modal with weak global structure       |
| f23         | Katsuura                       | Multi-modal with weak global structure       |
| f24         | Lunacek Bi-Rastrigin           | Multi-modal with weak global structure       |



