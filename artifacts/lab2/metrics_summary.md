# Laboratory Work #2: Model Evaluation Summary (PyTorch)

PyTorch 2.15.0, device: cpu, seed: 42, epochs: 15, optimizer: adam, lr: 0.001, batch size: 512

## Test Metrics @ threshold 0.5

| Model                                |   Threshold |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   PR-AUC (AP) |   TP |   FP |    TN |   FN |
|:-------------------------------------|------------:|-----------:|------------:|---------:|-----------:|----------:|--------------:|-----:|-----:|------:|-----:|
| Baseline_FFN_relu                    |         0.5 |     0.9992 |      0.7432 |   0.7746 |     0.7586 |    0.9525 |        0.7757 |   55 |   19 | 42469 |   16 |
| Baseline_FFN_leaky_relu              |         0.5 |     0.9988 |      0.625  |   0.7746 |     0.6918 |    0.9463 |        0.7392 |   55 |   33 | 42455 |   16 |
| Baseline_FFN_gelu                    |         0.5 |     0.999  |      0.6667 |   0.7606 |     0.7105 |    0.9438 |        0.7009 |   54 |   27 | 42461 |   17 |
| Regularized_BN_Dropout_L2_relu       |         0.5 |     0.9989 |      0.6444 |   0.8169 |     0.7205 |    0.9508 |        0.7326 |   58 |   32 | 42456 |   13 |
| Regularized_BN_Dropout_L2_leaky_relu |         0.5 |     0.9985 |      0.5364 |   0.831  |     0.6519 |    0.9561 |        0.6939 |   59 |   51 | 42437 |   12 |
| Regularized_BN_Dropout_L2_gelu       |         0.5 |     0.9984 |      0.5182 |   0.8028 |     0.6298 |    0.9596 |        0.7409 |   57 |   53 | 42435 |   14 |

## Test Metrics @ F1-optimal threshold (tuned on validation set)

| Model                                |   Threshold |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   PR-AUC (AP) |   TP |   FP |    TN |   FN |
|:-------------------------------------|------------:|-----------:|------------:|---------:|-----------:|----------:|--------------:|-----:|-----:|------:|-----:|
| Baseline_FFN_relu                    |       0.886 |     0.9993 |      0.8475 |   0.7042 |     0.7692 |    0.9525 |        0.7757 |   50 |    9 | 42479 |   21 |
| Baseline_FFN_leaky_relu              |       0.805 |     0.9991 |      0.75   |   0.7183 |     0.7338 |    0.9463 |        0.7392 |   51 |   17 | 42471 |   20 |
| Baseline_FFN_gelu                    |       0.978 |     0.9993 |      0.8525 |   0.7324 |     0.7879 |    0.9438 |        0.7009 |   52 |    9 | 42479 |   19 |
| Regularized_BN_Dropout_L2_relu       |       0.98  |     0.9994 |      0.8615 |   0.7887 |     0.8235 |    0.9508 |        0.7326 |   56 |    9 | 42479 |   15 |
| Regularized_BN_Dropout_L2_leaky_relu |       0.966 |     0.9993 |      0.8088 |   0.7746 |     0.7914 |    0.9561 |        0.6939 |   55 |   13 | 42475 |   16 |
| Regularized_BN_Dropout_L2_gelu       |       0.98  |     0.9993 |      0.8281 |   0.7465 |     0.7852 |    0.9596 |        0.7409 |   53 |   11 | 42477 |   18 |

## Over/Underfitting Diagnostics (train vs validation loss)

| Model                                |   Final train loss |   Final val loss |   Best val loss |   Best epoch | Val loss rise after best   |   Final val PR-AUC |
|:-------------------------------------|-------------------:|-----------------:|----------------:|-------------:|:---------------------------|-------------------:|
| Baseline_FFN_relu                    |             0.0013 |           0.0098 |          0.0068 |            5 | +44.0%                     |             0.7597 |
| Baseline_FFN_leaky_relu              |             0.0028 |           0.0078 |          0.0069 |           10 | +13.1%                     |             0.7865 |
| Baseline_FFN_gelu                    |             0.0013 |           0.0119 |          0.0082 |            5 | +44.0%                     |             0.7663 |
| Regularized_BN_Dropout_L2_relu       |             0.0117 |           0.0117 |          0.0115 |           12 | +2.0%                      |             0.761  |
| Regularized_BN_Dropout_L2_leaky_relu |             0.0232 |           0.0132 |          0.0132 |           14 | +0.5%                      |             0.7056 |
| Regularized_BN_Dropout_L2_gelu       |             0.0127 |           0.0133 |          0.0124 |           11 | +7.4%                      |             0.7622 |

## TensorFlow (Lab 1) vs PyTorch (Lab 2) @ tuned threshold

| Model                                |   TF F1 |   PT F1 |   ΔF1 (PT-TF) |   TF PR-AUC |   PT PR-AUC |   TF ROC-AUC |   PT ROC-AUC | TF TP/FP/FN   | PT TP/FP/FN   |
|:-------------------------------------|--------:|--------:|--------------:|------------:|------------:|-------------:|-------------:|:--------------|:--------------|
| Baseline_FFN_relu                    |  0.75   |  0.7692 |        0.0192 |      0.7503 |      0.7757 |       0.957  |       0.9525 | 51/14/20      | 50/9/21       |
| Baseline_FFN_leaky_relu              |  0.7737 |  0.7338 |       -0.0399 |      0.7675 |      0.7392 |       0.9527 |       0.9463 | 53/13/18      | 51/17/20      |
| Baseline_FFN_gelu                    |  0.7647 |  0.7879 |        0.0232 |      0.761  |      0.7009 |       0.9456 |       0.9438 | 52/13/19      | 52/9/19       |
| Regularized_BN_Dropout_L2_relu       |  0.7794 |  0.8235 |        0.0441 |      0.7585 |      0.7326 |       0.959  |       0.9508 | 53/12/18      | 56/9/15       |
| Regularized_BN_Dropout_L2_leaky_relu |  0.7971 |  0.7914 |       -0.0057 |      0.7047 |      0.6939 |       0.9565 |       0.9561 | 55/12/16      | 55/13/16      |
| Regularized_BN_Dropout_L2_gelu       |  0.7761 |  0.7852 |        0.0091 |      0.7674 |      0.7409 |       0.9584 |       0.9596 | 52/11/19      | 53/11/18      |

## Wall-clock Time per Experiment

| Model                                |   TF experiment, s |   PT experiment, s |   PT training loop, s |
|:-------------------------------------|-------------------:|-------------------:|----------------------:|
| Baseline_FFN_relu                    |                  8 |                 13 |                  12.7 |
| Baseline_FFN_leaky_relu              |                  8 |                 13 |                  12.7 |
| Baseline_FFN_gelu                    |                 11 |                 14 |                  14.1 |
| Regularized_BN_Dropout_L2_relu       |                 12 |                 21 |                  21.6 |
| Regularized_BN_Dropout_L2_leaky_relu |                 12 |                 22 |                  22   |
| Regularized_BN_Dropout_L2_gelu       |                 15 |                 23 |                  22.6 |

## Visual Artifacts

- [Learning Curves (loss)](plots/learning_curves_comparison.png)
- [Learning Curves (PR-AUC)](plots/learning_curves_pr_auc.png)
- [ROC & PR Curves](plots/roc_and_pr_curves.png)
- [Confusion Matrices @ 0.5](plots/confusion_matrices.png)
- [Confusion Matrices @ tuned threshold](plots/confusion_matrices_tuned.png)
- [Metrics Comparison Bar Chart](plots/metrics_comparison_barchart.png)
- [TensorFlow vs PyTorch](plots/framework_comparison.png)
