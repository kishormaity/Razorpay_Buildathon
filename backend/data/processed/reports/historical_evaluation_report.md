# Phase 5C: LightGBM Transaction + Historical Features Model Evaluation Report

Generated on: 2026-08-24 04:19:44

This report compares **Model D (Transaction + Chronological expanding Historical Features)** directly against **Model A (Transaction-Only Baseline)** to measure the predictive impact of leakage-free frequencies and target encodings for key entities.

---

## 1. Metrics Comparison Table

| Metric | Model A (Transaction-Only) | Model D (Tx + Historical Features) | Absolute Delta (Δ) |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Average Precision)** | `0.57181` | `0.58144` | `+0.00963` |
| **ROC-AUC** | `0.91223` | `0.90507` | `-0.00715` |
| **F1 @ 0.50 (Reference)** | `0.53733` | `0.56125` | `+0.02392` |
| **Precision @ 0.50** | `0.78588` | `0.78104` | `-0.00484` |
| **Recall @ 0.50** | `0.40822` | `0.43799` | `+0.02977` |
| **Optimal F1-Score** | `0.56645` | `0.58178` | `+0.01533` |
| **Optimal Threshold** | `0.2740` | `0.3040` | `+0.0300` |
| **FPR @ Optimal Threshold** | `0.00948` | `0.00880` | `-0.00068` |

---

## 2. Confusion Matrices Comparison

### Reference Threshold (0.50)
* **Model A (TN / FP / FN / TP)**: 113,592 / 452 / 2,405 / 1,659
* **Model D (TN / FP / FN / TP)**: 113,545 / 499 / 2,284 / 1,780

### Optimal Threshold
* **Model A (TN / FP / FN / TP @ 0.2740)**: 112,963 / 1,081 / 2,031 / 2,033
* **Model D (TN / FP / FN / TP @ 0.3040)**: 113,040 / 1,004 / 1,985 / 2,079

---

## 3. Top 20 Feature Importance (By Gain)

This inventory shows which attributes contributed the most gain to Model D. Note if any of our newly engineered 10 historical features rank in the top 20:

| Rank | Feature Name | Gain Importance | Description |
| :--- | :--- | :--- | :--- |
| 1 | `card1` | 180695.73 | Model feature |
| 2 | `card_addr_combo_historical_fraud_rate` | 170799.33 | Model feature |
| 3 | `card_email_combo_historical_fraud_rate` | 97136.33 | Model feature |
| 4 | `C1` | 32937.54 | Model feature |
| 5 | `C14` | 32339.43 | Model feature |
| 6 | `DeviceInfo` | 29554.36 | Model feature |
| 7 | `card_device_combo_historical_fraud_rate` | 27189.85 | Model feature |
| 8 | `addr1` | 26963.84 | Model feature |
| 9 | `card2` | 21744.78 | Model feature |
| 10 | `V133` | 18646.55 | Model feature |
| 11 | `C13` | 18207.27 | Model feature |
| 12 | `card_addr_past_count` | 17645.43 | Model feature |
| 13 | `V258` | 15405.66 | Model feature |
| 14 | `V257` | 15326.11 | Model feature |
| 15 | `V243` | 14949.43 | Model feature |
| 16 | `TransactionAmt` | 14343.15 | Model feature |
| 17 | `D2` | 13783.64 | Model feature |
| 18 | `TransactionDT` | 13671.33 | Model feature |
| 19 | `V307` | 11455.09 | Model feature |
| 20 | `card_device_past_count` | 10511.57 | Model feature |

---

## 4. Artifact Directory Inventory

The following outputs have been serialized and saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Model Booster**: [`historical_lgb_model.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/historical_lgb_model.txt)
2. **Evaluation Metrics**: [`historical_evaluation_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/historical_evaluation_report.md)
3. **Comparative PR Curve**: [`historical_pr_curves.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/historical_pr_curves.png)
4. **Validation Predictions**: [`historical_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/historical_predictions.parquet)
