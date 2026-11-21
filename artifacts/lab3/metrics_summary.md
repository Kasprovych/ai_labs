# Laboratory Work #3: Genetic Algorithm for Hyperparameter Optimization

Model: HistGradientBoostingRegressor (scikit-learn 1.9.1); dataset: California Housing; fitness: validation RMSE; GA seeds: [1, 2, 3, 4, 5]

## Search Space

|   Gene | Hyperparameter    | Type   | Range                   |   Default |
|-------:|:------------------|:-------|:------------------------|----------:|
|      0 | learning_rate     | float  | [0.01, 0.5], log scale  |       0.1 |
|      1 | max_iter          | int    | [50, 500]               |     100   |
|      2 | max_leaf_nodes    | int    | [4, 64]                 |      31   |
|      3 | min_samples_leaf  | int    | [1, 100]                |      20   |
|      4 | l2_regularization | float  | [0.0001, 10], log scale |       0   |
|      5 | max_features      | float  | [0.3, 1]                |       1   |
|      6 | max_bins          | int    | [16, 255]               |     255   |

## Summary over Seeds

| Method                  |   Runs | Val RMSE (mean ± std)   | Test RMSE (mean ± std)   |   Best test RMSE |   Test R2 (mean) |
|:------------------------|-------:|:------------------------|:-------------------------|-----------------:|-----------------:|
| Default hyperparameters |      1 | 0.4775 ± 0.0000         | 0.4620 ± 0.0000          |           0.462  |           0.8371 |
| Random search           |      5 | 0.4542 ± 0.0019         | 0.4401 ± 0.0020          |           0.437  |           0.8522 |
| Genetic algorithm       |      5 | 0.4502 ± 0.0004         | 0.4392 ± 0.0023          |           0.4363 |           0.8528 |

## Validation and Test Metrics per Run

| Configuration         |   Val RMSE |   Test RMSE |   Test MAE |   Test R2 |
|:----------------------|-----------:|------------:|-----------:|----------:|
| Default               |     0.4775 |      0.462  |     0.3074 |    0.8371 |
| Random search, seed 1 |     0.4512 |      0.4383 |     0.2857 |    0.8534 |
| Random search, seed 2 |     0.4544 |      0.437  |     0.2858 |    0.8543 |
| Random search, seed 3 |     0.4548 |      0.4415 |     0.2882 |    0.8512 |
| Random search, seed 4 |     0.4537 |      0.442  |     0.2876 |    0.8509 |
| Random search, seed 5 |     0.457  |      0.4415 |     0.2894 |    0.8513 |
| GA, seed 1            |     0.4503 |      0.4392 |     0.2856 |    0.8528 |
| GA, seed 2            |     0.4504 |      0.437  |     0.2855 |    0.8543 |
| GA, seed 3            |     0.4507 |      0.4414 |     0.2902 |    0.8513 |
| GA, seed 4            |     0.4502 |      0.4421 |     0.2875 |    0.8509 |
| GA, seed 5            |     0.4495 |      0.4363 |     0.2831 |    0.8547 |

## Best Hyperparameters per Run

| Configuration         |   learning_rate |   max_iter |   max_leaf_nodes |   min_samples_leaf |   l2_regularization |   max_features |   max_bins |
|:----------------------|----------------:|-----------:|-----------------:|-------------------:|--------------------:|---------------:|-----------:|
| Default               |         0.1     |        100 |               31 |                 20 |           0         |         1      |        255 |
| Random search, seed 1 |         0.06321 |        383 |               54 |                 26 |           0.7148    |         0.5969 |        244 |
| Random search, seed 2 |         0.07183 |        472 |               49 |                 58 |           0.122     |         0.6546 |        247 |
| Random search, seed 3 |         0.0817  |        436 |               42 |                 65 |           0.00901   |         0.5496 |        188 |
| Random search, seed 4 |         0.0471  |        447 |               64 |                 23 |           0.7986    |         0.3733 |        174 |
| Random search, seed 5 |         0.05305 |        378 |               54 |                 96 |           0.7922    |         0.8738 |        240 |
| GA, seed 1            |         0.05927 |        479 |               59 |                  1 |          10         |         0.5696 |        240 |
| GA, seed 2            |         0.06728 |        477 |               45 |                 28 |           0.01429   |         0.6587 |        247 |
| GA, seed 3            |         0.09005 |        500 |               41 |                 73 |           0.0008173 |         0.5158 |        185 |
| GA, seed 4            |         0.07873 |        340 |               62 |                 20 |           0.002418  |         0.486  |        188 |
| GA, seed 5            |         0.07109 |        477 |               46 |                 36 |           0.4654    |         0.7367 |        189 |

## Generations of the Best GA Run (seed 5)

|   Generation |   Best RMSE |   Mean RMSE |   Worst RMSE |   Diversity |   Evaluations |
|-------------:|------------:|------------:|-------------:|------------:|--------------:|
|            0 |      0.4594 |      0.514  |       0.7422 |       0.289 |            20 |
|            1 |      0.4572 |      0.4722 |       0.5104 |       0.255 |            38 |
|            2 |      0.4572 |      0.4625 |       0.4839 |       0.195 |            56 |
|            3 |      0.454  |      0.4599 |       0.4709 |       0.17  |            74 |
|            4 |      0.453  |      0.4603 |       0.4777 |       0.184 |            92 |
|            5 |      0.453  |      0.4594 |       0.4792 |       0.181 |           110 |
|            6 |      0.4529 |      0.4608 |       0.4751 |       0.201 |           128 |
|            7 |      0.4529 |      0.4575 |       0.4671 |       0.193 |           146 |
|            8 |      0.4526 |      0.4576 |       0.4632 |       0.184 |           164 |
|            9 |      0.4504 |      0.4554 |       0.4597 |       0.107 |           182 |
|           10 |      0.4504 |      0.4542 |       0.4572 |       0.074 |           200 |
|           11 |      0.4504 |      0.4551 |       0.4626 |       0.061 |           218 |
|           12 |      0.4504 |      0.4547 |       0.4586 |       0.067 |           236 |
|           13 |      0.4495 |      0.4538 |       0.4573 |       0.064 |           254 |
|           14 |      0.4495 |      0.4548 |       0.4591 |       0.074 |           272 |

## Search Cost

| Method            |   Seed |   Fitness evaluations |   Models trained |   Seconds |
|:------------------|-------:|----------------------:|-----------------:|----------:|
| Genetic algorithm |      1 |                   272 |              251 |     218.1 |
| Genetic algorithm |      2 |                   272 |              247 |     209.7 |
| Genetic algorithm |      3 |                   272 |              250 |     170.4 |
| Genetic algorithm |      4 |                   272 |              251 |     160.3 |
| Genetic algorithm |      5 |                   272 |              261 |     163   |
| Random search     |      1 |                   272 |              272 |     110.1 |
| Random search     |      2 |                   272 |              272 |     105.1 |
| Random search     |      3 |                   272 |              272 |     109.9 |
| Random search     |      4 |                   272 |              272 |     105.4 |
| Random search     |      5 |                   272 |              272 |     107   |

## Visual Artifacts

- [GA convergence and diversity](plots/ga_convergence.png)
- [GA vs random search](plots/ga_vs_random_search.png)
- [Population evolution](plots/population_evolution.png)
- [Final comparison](plots/final_comparison.png)
