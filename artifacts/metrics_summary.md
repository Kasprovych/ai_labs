# Laboratory Work #1: Model Evaluation Summary (TensorFlow)

## Performance Metrics Table

| Model                          |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   PR-AUC (AP) |   TP |   FP |    TN |   FN |
|:-------------------------------|-----------:|------------:|---------:|-----------:|----------:|--------------:|-----:|-----:|------:|-----:|
| Baseline_FFN_relu              |     0.9989 |      0.6667 |   0.7838 |     0.7205 |    0.9622 |        0.7746 |   58 |   29 | 42619 |   16 |
| Baseline_FFN_gelu              |     0.999  |      0.6905 |   0.7838 |     0.7342 |    0.9708 |        0.7766 |   58 |   26 | 42622 |   16 |
| Regularized_BN_Dropout_L2_relu |     0.9982 |      0.4917 |   0.7973 |     0.6082 |    0.9724 |        0.7845 |   59 |   61 | 42587 |   15 |
| Regularized_BN_Dropout_L2_gelu |     0.9978 |      0.4308 |   0.7568 |     0.549  |    0.957  |        0.7361 |   56 |   74 | 42574 |   18 |

## Visual Artifacts

- [Learning Curves](learning_curves_comparison.png)
- [ROC & PR Curves](roc_and_pr_curves.png)
- [Confusion Matrices](confusion_matrices.png)
- [Metrics Comparison Bar Chart](metrics_comparison_barchart.png)
