# AI Deep Learning Models: Laboratory Work #1 (TensorFlow)

Building, optimizing, and evaluating deep learning models for credit card fraud detection with severe class imbalance using **TensorFlow / Keras**.

## 📌 Project Overview
- **Dataset**: Kaggle Credit Card Fraud Detection (284,807 transactions, 0.173% fraud class).
- **Core Architecture**: Clean Architecture & SOLID principles (OOP, Strategy Pattern, Builder Pattern, DTOs, Dependency Injection).
- **Techniques**:
  - Exploratory Data Analysis (EDA) with KDE, log-scale distributions, and Pearson correlation matrices.
  - Stratified 70/15/15 train/val/test splitting.
  - Removal of 1,081 exact duplicate rows (prevents train/test leakage).
  - Robust scaling fitted strictly on train set to prevent data leakage.
  - Synthetic Minority Over-sampling Technique (SMOTE) applied to training fold.
  - Feedforward baseline vs. Regularized architecture (Batch Normalization, Dropout 0.3, L2 weight decay).
  - Comparative analysis of activation functions (**ReLU**, **LeakyReLU**, **GELU**) for both architectures (6 experiments).
  - F1-optimal decision threshold tuned on the validation set (SMOTE miscalibrates probabilities).
  - Over/underfitting diagnostics (best val-loss epoch, val-loss rise) per model.
  - Fully reproducible runs (global seed 42).
  - Imbalanced classification metrics: Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix.

## 📁 Repository Structure
```text
├── main.py                     # Application entry point / orchestrator
├── requirements.txt            # Python dependencies
├── REPORT_LAB1.md              # Detailed laboratory report (source)
├── report/                     # Printable report: REPORT_LAB1.docx, REPORT_LAB1.pdf
├── scripts/build_report.py     # Markdown -> DOCX / PDF build (pandoc + headless Chrome)
├── src/
│   ├── config/
│   │   └── app_config.py       # Immutable dataclass configuration
│   ├── data/
│   │   ├── loader.py           # Dataset loading & caching (IDataLoader)
│   │   └── preprocessor.py     # Stratified split, scalers, SMOTE (Strategy)
│   ├── eda/
│   │   └── analyzer.py         # Exploratory Data Analysis & visual plots
│   ├── models/
│   │   └── builder.py          # Baseline & Regularized Builders (IModelBuilder)
│   ├── training/
│   │   ├── trainer.py          # Model compilation & training execution
│   │   └── evaluator.py        # Comprehensive imbalanced metrics evaluation
│   └── visualization/
│       └── plots.py            # Diagnostic charts generation
└── artifacts/
    ├── plots/                  # Visual plots (EDA, learning curves, ROC/PR, confusion matrices)
    ├── screenshots/            # Terminal screenshots of a real run
    ├── saved_models/           # Serialized .keras trained models
    └── metrics_summary.md      # Performance metrics table
```

## 📊 Experimental Results (Test Set: 42,559 Samples, 71 Frauds)

Test metrics at the F1-optimal threshold tuned on the validation set (full tables incl. threshold 0.5 in `artifacts/metrics_summary.md`):

| Model | Threshold | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (AP) | TP | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline_FFN_relu | 0.871 | 0.7846 | 0.7183 | 0.7500 | 0.9570 | 0.7503 | 51 | 14 | 20 |
| Baseline_FFN_leaky_relu | 0.992 | 0.8030 | 0.7465 | 0.7737 | 0.9527 | 0.7675 | 53 | 13 | 18 |
| Baseline_FFN_gelu | 0.954 | 0.8000 | 0.7324 | 0.7647 | 0.9456 | 0.7610 | 52 | 13 | 19 |
| Regularized_BN_Dropout_L2_relu | 0.964 | 0.8154 | 0.7465 | 0.7794 | 0.9590 | 0.7585 | 53 | 12 | 18 |
| **Regularized_BN_Dropout_L2_leaky_relu** | 0.983 | 0.8209 | **0.7746** | **0.7971** | 0.9565 | 0.7047 | **55** | 12 | **16** |
| Regularized_BN_Dropout_L2_gelu | 0.965 | **0.8254** | 0.7324 | 0.7761 | 0.9584 | 0.7674 | 52 | **11** | 19 |

> **Best model**: `Regularized_BN_Dropout_L2_leaky_relu` (F1 0.797). Baseline networks overfit after epoch 5–6 (val loss +32–46%); regularization removes this.

## 🚀 How to Run
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline
python main.py

# 4. (Optional) Build printable report: report/REPORT_LAB1.docx and report/REPORT_LAB1.pdf
#    Requires pandoc (brew install pandoc) and Google Chrome for the PDF.
python scripts/build_report.py
```
