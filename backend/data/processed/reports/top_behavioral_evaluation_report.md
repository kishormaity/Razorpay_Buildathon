# Phase 4B: LightGBM Transaction + Top Behavioral Model Evaluation Report

Generated on: 2026-08-24 03:02:15

This report compares **Model C (Transaction + Top-6 Importance-Ranked Behavioral Features)** directly against **Model A (Transaction-Only)** and **Model B (Transaction + All 12 Behavioral Features)** to determine if filtering out noisy behavioral columns reclaims model predictive strength.

---

## 1. Metrics Comparison Table

| Metric | Model A (Transaction-Only) | Model B (Tx + All Behavior) | Model C (Tx + Top-6 Behavior) | Delta (C - A) |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Average Precision)** | `0.57181` | `0.54628` | `0.56722` | `-0.00459` |
| **ROC-AUC** | `0.91223` | `0.90603` | `0.91233` | `+0.00010` |
| **F1 @ 0.50 (Reference)** | `0.53733` | `0.51134` | `0.52585` | `-0.01148` |
| **Precision @ 0.50** | `0.78588` | `0.78070` | `0.79453` | `+0.00864` |
| **Recall @ 0.50** | `0.40822` | `0.38017` | `0.39296` | `-0.01526` |
| **Optimal F1-Score** | `0.56645` | `0.55230` | `0.56120` | `-0.00526` |
| **Optimal Threshold** | `0.2740` | `0.2813` | `0.3202` | `+0.0462` |
| **FPR @ Optimal Threshold** | `0.00948` | `0.00958` | `0.00735` | `-0.00213` |

---

## 2. Confusion Matrices Comparison

### Reference Threshold (0.50)
* **Model A (TN / FP / FN / TP)**: 113,592 / 452 / 2,405 / 1,659
* **Model B (TN / FP / FN / TP)**: 113,610 / 434 / 2,519 / 1,545 (Values from Model B Predictions)
* **Model C (TN / FP / FN / TP)**: 113,631 / 413 / 2,467 / 1,597

### Optimal Threshold
* **Model A (TN / FP / FN / TP @ 0.2740)**: 112,963 / 1,081 / 2,031 / 2,033
* **Model B (TN / FP / FN / TP @ 0.2813)**: 112,952 / 1,092 / 2,097 / 1,967
* **Model C (TN / FP / FN / TP @ 0.3202)**: 113,206 / 838 / 2,152 / 1,912

---

## 3. Top 20 Feature Importance (By Gain)

This inventory shows which attributes contributed the most gain to Model C:

| Rank | Feature Name | Gain Importance | Description |
| :--- | :--- | :--- | :--- |
| 1 | `card1` | 296332.73 | Model feature |
| 2 | `V258` | 76847.70 | Model feature |
| 3 | `addr1` | 48344.76 | Model feature |
| 4 | `DeviceInfo` | 44478.06 | Model feature |
| 5 | `C1` | 41319.22 | Model feature |
| 6 | `card2` | 40484.75 | Model feature |
| 7 | `C14` | 34412.17 | Model feature |
| 8 | `V257` | 24607.16 | Model feature |
| 9 | `C13` | 23009.66 | Model feature |
| 10 | `R_emaildomain` | 18813.02 | Model feature |
| 11 | `D2` | 18006.54 | Model feature |
| 12 | `V294` | 17994.51 | Model feature |
| 13 | `TransactionDT` | 17695.08 | Model feature |
| 14 | `C4` | 16409.62 | Model feature |
| 15 | `TransactionAmt` | 14816.24 | Model feature |
| 16 | `P_emaildomain` | 11338.81 | Model feature |
| 17 | `C11` | 9735.12 | Model feature |
| 18 | `card_time_since_prev` | 8966.67 | Model feature |
| 19 | `D15` | 8407.74 | Model feature |
| 20 | `C7` | 8094.89 | Model feature |

---

## 4. Artifact Directory Inventory

The following outputs have been serialized and saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Model Booster**: [`top_behavioral_lgb_model.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/top_behavioral_lgb_model.txt)
2. **Evaluation Metrics**: [`top_behavioral_evaluation_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/top_behavioral_evaluation_report.md)
3. **Comparative PR Curves**: [`comparative_pr_curve.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/comparative_pr_curve.png)
4. **Validation Predictions**: [`top_behavioral_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/top_behavioral_predictions.parquet)
