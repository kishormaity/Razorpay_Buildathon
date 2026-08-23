# Phase 4: LightGBM Transaction + Behavioral Model Evaluation Report

Generated on: 2026-08-24 02:51:15

This report documents the performance comparison between the **Transaction-Only Baseline Model** (Phase 2) and the **Transaction + Behavioral Model** (Phase 4), allowing us to isolate the predictive value of historical card velocities, deviations, and novelty checks.

---

## 1. Metrics Comparison Table

| Metric | Baseline (Transaction-Only) | Behavioral Model (Tx + Behavior) | Absolute Delta (Δ) |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Average Precision)** | `0.57181` | `0.54628` | `-0.02553` |
| **ROC-AUC** | `0.91223` | `0.90603` | `-0.00619` |
| **F1 @ 0.50 (Reference)** | `0.53733` | `0.51134` | `-0.02599` |
| **Precision @ 0.50** | `0.78588` | `0.78070` | `-0.00519` |
| **Recall @ 0.50** | `0.40822` | `0.38017` | `-0.02805` |
| **Optimal F1-Score** | `0.56645` | `0.55230` | `-0.01416` |
| **Optimal Threshold** | `0.2740` | `0.2813` | `+0.0073` |
| **FPR @ Optimal Threshold** | `0.00948` | `0.00958` | `+0.00010` |

---

## 2. Confusion Matrices Comparison

### Reference Threshold (0.50)
* **Baseline (TN / FP / FN / TP)**: 113,592 / 452 / 2,405 / 1,659
* **Behavioral (TN / FP / FN / TP)**: 113,610 / 434 / 2,519 / 1,545

### Optimal Threshold
* **Baseline (TN / FP / FN / TP @ 0.2740)**: 112,963 / 1,081 / 2,031 / 2,033
* **Behavioral (TN / FP / FN / TP @ 0.2813)**: 112,952 / 1,092 / 2,097 / 1,967

---

## 3. Top 20 Feature Importance (By Gain)

This inventory shows which attributes contributed the most gain to the model. Note if any of our newly engineered 12 behavioral/temporal features rank in the top 20:

| Rank | Feature Name | Gain Importance | Description |
| :--- | :--- | :--- | :--- |
| 1 | `card1` | 252311.98 | Model feature |
| 2 | `V258` | 91298.72 | Model feature |
| 3 | `C1` | 41060.67 | Model feature |
| 4 | `DeviceInfo` | 39513.74 | Model feature |
| 5 | `addr1` | 37712.50 | Model feature |
| 6 | `card2` | 36314.74 | Model feature |
| 7 | `C14` | 35729.33 | Model feature |
| 8 | `D2` | 18935.98 | Model feature |
| 9 | `R_emaildomain` | 18054.53 | Model feature |
| 10 | `C13` | 17613.64 | Model feature |
| 11 | `V294` | 16408.95 | Model feature |
| 12 | `C4` | 15004.07 | Model feature |
| 13 | `TransactionDT` | 13476.39 | Model feature |
| 14 | `TransactionAmt` | 13436.44 | Model feature |
| 15 | `V308` | 10430.84 | Model feature |
| 16 | `V317` | 10365.33 | Model feature |
| 17 | `C8` | 9321.93 | Model feature |
| 18 | `P_emaildomain` | 9200.85 | Model feature |
| 19 | `V201` | 9176.95 | Model feature |
| 20 | `C11` | 7894.22 | Model feature |

---

## 4. Artifact Directory Inventory

The following outputs have been serialized and saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Model Booster**: [`behavioral_lgb_model.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/behavioral_lgb_model.txt)
2. **Evaluation Metrics**: [`behavioral_evaluation_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/behavioral_evaluation_report.md)
3. **Comparative PR Curve**: [`comparative_pr_curve.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/comparative_pr_curve.png)
4. **Validation Predictions**: [`behavioral_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/behavioral_predictions.parquet)
