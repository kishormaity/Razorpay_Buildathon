# Phase 2: LightGBM Transaction-Only Baseline Evaluation Report

Generated on: 2026-08-24 02:49:28

This report documents the training parameters and evaluation performance of the **Transaction-Only Baseline Model** (Phase 2). This baseline establishes the detection capacity using only features derived from the transaction transaction-details, serving as a benchmark for subsequent behavioral and graph feature iterations.

---

## 1. Split & Configuration Parameters

* **Temporal Split Strategy**: Time-aware chronological partition on sorted `TransactionDT`.
* **Dataset Boundary**:
  * **Train Set (Earliest 80%)**: 472,432 rows (TransactionDT < 12,192,900)
  * **Validation Set (Latest 20%)**: 118,108 rows (TransactionDT >= 12,192,900)
* **Hyperparameters**:
  * `scale_pos_weight = 1.0` (Unweighted baseline setup)
  * `learning_rate = 0.05`
  * `num_leaves = 31`
  * `boosting_type = gdbt`
  * Categorical variables: Handled natively by LightGBM using fisher-splits.

---

## 2. Model Performance Summary

| Metric | Score | Description |
| :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | **0.57181** | Primary performance score (imbalance-robust) |
| **ROC-AUC** | **0.91223** | Overall classification performance |

### Classification Decision Trade-offs:

| Threshold | F1-Score | Precision | Recall | False Positive Rate (FPR) |
| :--- | :--- | :--- | :--- | :--- |
| **Reference (0.50)** | 0.53733 | 0.78588 | 0.40822 | 0.00396 |
| **Optimal F1 (0.2740)** | 0.56645 | 0.65286 | 0.50025 | 0.00948 |

---

## 3. Confusion Matrices

### Reference Threshold (0.50)
* **True Negatives (TN)**: 113,592
* **False Positives (FP)**: 452
* **False Negatives (FN)**: 2,405
* **True Positives (TP)**: 1,659

### Optimal F1 Threshold (0.2740)
* **True Negatives (TN)**: 112,963
* **False Positives (FP)**: 1,081
* **False Negatives (FN)**: 2,031
* **True Positives (TP)**: 2,033

---

## 4. Top 20 Feature Importance (By Gain)

| Rank | Feature Name | Gain Importance | Description |
| :--- | :--- | :--- | :--- |
| 1 | `card1` | 304369.28 | Transaction-level metric |
| 2 | `V258` | 56106.36 | Transaction-level metric |
| 3 | `addr1` | 48654.51 | Transaction-level metric |
| 4 | `DeviceInfo` | 45910.21 | Transaction-level metric |
| 5 | `C1` | 44583.93 | Transaction-level metric |
| 6 | `C14` | 39730.04 | Transaction-level metric |
| 7 | `V257` | 38083.01 | Transaction-level metric |
| 8 | `card2` | 34156.38 | Transaction-level metric |
| 9 | `V294` | 22901.21 | Transaction-level metric |
| 10 | `R_emaildomain` | 21979.40 | Transaction-level metric |
| 11 | `C13` | 21869.28 | Transaction-level metric |
| 12 | `D2` | 19951.59 | Transaction-level metric |
| 13 | `TransactionDT` | 19826.53 | Transaction-level metric |
| 14 | `TransactionAmt` | 15897.13 | Transaction-level metric |
| 15 | `C4` | 14262.05 | Transaction-level metric |
| 16 | `P_emaildomain` | 11576.86 | Transaction-level metric |
| 17 | `C8` | 10726.24 | Transaction-level metric |
| 18 | `V189` | 10286.76 | Transaction-level metric |
| 19 | `V308` | 9892.47 | Transaction-level metric |
| 20 | `C11` | 9282.85 | Transaction-level metric |

---

## 5. Artifact Directory Inventory

The following outputs have been serialized and saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Model Booster**: [`baseline_lgb_model.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/baseline_lgb_model.txt)
2. **Evaluation Metrics**: [`baseline_evaluation_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/baseline_evaluation_report.md)
3. **PR Curve Plot**: [`baseline_pr_curve.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/baseline_pr_curve.png)
4. **Validation Predictions**: [`baseline_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/baseline_predictions.parquet)
