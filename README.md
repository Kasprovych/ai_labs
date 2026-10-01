# AI Deep Learning Models: Laboratory Work #1 (TensorFlow)

Building, optimizing, and evaluating deep learning models for credit card fraud detection with severe class imbalance using **TensorFlow / Keras**.

## 📌 Project Overview
- **Dataset**: Kaggle Credit Card Fraud Detection (284,807 transactions, 0.173% fraud class).
- **Core Architecture**: Clean Architecture & SOLID principles (OOP, Strategy Pattern, Builder Pattern, DTOs, Dependency Injection).
- **Techniques**:
  - Exploratory Data Analysis (EDA) with KDE, log-scale distributions, and Pearson correlation matrices.
  - Stratified 70/15/15 train/val/test splitting.
  - Robust scaling fitted strictly on train set to prevent data leakage.
  - Synthetic Minority Over-sampling Technique (SMOTE) applied to training fold.
  - Feedforward baseline vs. Regularized architecture (Batch Normalization, Dropout 0.3, L2 weight decay).
  - Comparative analysis of activation functions (**ReLU** vs. **GELU**).
  - Imbalanced classification metrics: Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix.

## 📁 Repository Structure
```text
├── main.py                     # Application entry point / orchestrator
├── requirements.txt            # Python dependencies
├── REPORT_LAB1.md              # Detailed laboratory report
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
    ├── saved_models/           # Serialized .keras trained models
    └── metrics_summary.md      # Performance metrics table
```

## 📊 Experimental Results (Test Set: 42,722 Samples)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (AP) | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline_FFN_relu** | 0.9989 | 0.6667 | 0.7838 | 0.7205 | 0.9622 | 0.7746 | 58 | 29 | 42619 | 16 |
| **Baseline_FFN_gelu** | **0.9990** | **0.6905** | 0.7838 | **0.7342** | 0.9708 | 0.7766 | 58 | **26** | **42622** | 16 |
| **Regularized_BN_Dropout_L2_relu** | 0.9982 | 0.4917 | **0.7973** | 0.6082 | **0.9724** | **0.7845** | **59** | 61 | 42587 | **15** |
| **Regularized_BN_Dropout_L2_gelu** | 0.9978 | 0.4308 | 0.7568 | 0.5490 | 0.9570 | 0.7361 | 56 | 74 | 42574 | 18 |

> **Best Overall Model**: `Baseline_FFN_gelu` achieved the highest F1-Score (0.7342) and highest Precision (69.05%) with only 26 false positives on the independent test set.

## 🚀 How to Run
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline
python main.py
```
