# Laboratory Work #1: Model Evaluation Summary (TensorFlow)

Seed: 42, epochs: 15, optimizer: adam, lr: 0.001, batch size: 512

## Test Metrics @ threshold 0.5

| Model                                |   Threshold |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   PR-AUC (AP) |   TP |   FP |    TN |   FN |
|:-------------------------------------|------------:|-----------:|------------:|---------:|-----------:|----------:|--------------:|-----:|-----:|------:|-----:|
| Baseline_FFN_relu                    |         0.5 |     0.9991 |      0.7067 |   0.7465 |     0.726  |    0.957  |        0.7503 |   53 |   22 | 42466 |   18 |
| Baseline_FFN_leaky_relu              |         0.5 |     0.9988 |      0.5957 |   0.7887 |     0.6788 |    0.9527 |        0.7675 |   56 |   38 | 42450 |   15 |
| Baseline_FFN_gelu                    |         0.5 |     0.9987 |      0.5761 |   0.7465 |     0.6503 |    0.9456 |        0.761  |   53 |   39 | 42449 |   18 |
| Regularized_BN_Dropout_L2_relu       |         0.5 |     0.9987 |      0.5816 |   0.8028 |     0.6746 |    0.959  |        0.7585 |   57 |   41 | 42447 |   14 |
| Regularized_BN_Dropout_L2_leaky_relu |         0.5 |     0.9985 |      0.5263 |   0.8451 |     0.6486 |    0.9565 |        0.7047 |   60 |   54 | 42434 |   11 |
| Regularized_BN_Dropout_L2_gelu       |         0.5 |     0.9984 |      0.5043 |   0.831  |     0.6277 |    0.9584 |        0.7674 |   59 |   58 | 42430 |   12 |

## Test Metrics @ F1-optimal threshold (tuned on validation set)

| Model                                |   Threshold |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   PR-AUC (AP) |   TP |   FP |    TN |   FN |
|:-------------------------------------|------------:|-----------:|------------:|---------:|-----------:|----------:|--------------:|-----:|-----:|------:|-----:|
| Baseline_FFN_relu                    |       0.871 |     0.9992 |      0.7846 |   0.7183 |     0.75   |    0.957  |        0.7503 |   51 |   14 | 42474 |   20 |
| Baseline_FFN_leaky_relu              |       0.992 |     0.9993 |      0.803  |   0.7465 |     0.7737 |    0.9527 |        0.7675 |   53 |   13 | 42475 |   18 |
| Baseline_FFN_gelu                    |       0.954 |     0.9992 |      0.8    |   0.7324 |     0.7647 |    0.9456 |        0.761  |   52 |   13 | 42475 |   19 |
| Regularized_BN_Dropout_L2_relu       |       0.964 |     0.9993 |      0.8154 |   0.7465 |     0.7794 |    0.959  |        0.7585 |   53 |   12 | 42476 |   18 |
| Regularized_BN_Dropout_L2_leaky_relu |       0.983 |     0.9993 |      0.8209 |   0.7746 |     0.7971 |    0.9565 |        0.7047 |   55 |   12 | 42476 |   16 |
| Regularized_BN_Dropout_L2_gelu       |       0.965 |     0.9993 |      0.8254 |   0.7324 |     0.7761 |    0.9584 |        0.7674 |   52 |   11 | 42477 |   19 |

## Over/Underfitting Diagnostics (train vs validation loss)

| Model                                |   Final train loss |   Final val loss |   Best val loss |   Best epoch | Val loss rise after best   |   Final val PR-AUC |
|:-------------------------------------|-------------------:|-----------------:|----------------:|-------------:|:---------------------------|-------------------:|
| Baseline_FFN_relu                    |             0.0009 |           0.0102 |          0.0077 |            6 | +32.3%                     |             0.7214 |
| Baseline_FFN_leaky_relu              |             0.0034 |           0.0082 |          0.0082 |           15 | +0.0%                      |             0.7242 |
| Baseline_FFN_gelu                    |             0.0013 |           0.0116 |          0.0079 |            5 | +46.0%                     |             0.7187 |
| Regularized_BN_Dropout_L2_relu       |             0.0128 |           0.0117 |          0.0106 |           12 | +11.0%                     |             0.753  |
| Regularized_BN_Dropout_L2_leaky_relu |             0.0258 |           0.0126 |          0.0126 |           15 | +0.0%                      |             0.7433 |
| Regularized_BN_Dropout_L2_gelu       |             0.0137 |           0.0134 |          0.012  |            8 | +11.3%                     |             0.7731 |

## Visual Artifacts

- [Learning Curves (loss)](plots/learning_curves_comparison.png)
- [Learning Curves (PR-AUC)](plots/learning_curves_pr_auc.png)
- [ROC & PR Curves](plots/roc_and_pr_curves.png)
- [Confusion Matrices @ 0.5](plots/confusion_matrices.png)
- [Confusion Matrices @ tuned threshold](plots/confusion_matrices_tuned.png)
- [Metrics Comparison Bar Chart](plots/metrics_comparison_barchart.png)
