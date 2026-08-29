# Phase 5E: LightGBM Combined Model Evaluation Report

Generated on: 2026-08-24 04:06:33

This report compares **Model F (LightGBM Transaction + Chronological expanding Historical Features + Top-6 Behavioral)** directly against **Model A (Transaction-Only Baseline)** and **Model D (Transaction + Historical)** to determine if adding temporal behavioral features offers complementary signal.

---

## 1. Metrics Leaderboard

| Model Config | Algorithm | Features | PR-AUC | ROC-AUC | Optimal F1 | Optimal Threshold | FPR @ Optimal | Training Time | Best Iteration |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | LightGBM | Transaction | `0.57181` | `0.91223` | `0.56645` | `0.2740` | `0.00948` | `94.91s` | `831` |
| **Model D** | LightGBM | Transaction + Historical | `0.58144` | `0.90507` | `0.58178` | `0.3040` | `0.00880` | `119.27s` | `998` |
| **Model F** | **LightGBM** | Transaction + Hist + Behav-6 | `0.57394` | `0.90275` | `0.57162` | `0.2863` | `0.01005` | `77.35s` | `562` |

---

## 2. Confusion Matrices Comparison (Optimal Threshold)

* **Model A (TN / FP / FN / TP @ 0.2740)**: 112,963 / 1,081 / 2,031 / 2,033
* **Model D (TN / FP / FN / TP @ 0.3040)**: 113,040 / 1,004 / 1,985 / 2,079
* **Model F (TN / FP / FN / TP @ 0.2863)**: 112,898 / 1,146 / 1,979 / 2,085

---

## 3. Top 20 Feature Importance (By Gain)

This list shows the top 20 attributes contributing the most predictive weight to Model F:

| Rank | Feature Name | Gain Importance | Description |
| :--- | :--- | :--- | :--- |
| 1 | `card_addr_combo_historical_fraud_rate` | 163669.50 | Model feature |
| 2 | `card1` | 158783.19 | Model feature |
| 3 | `card_email_combo_historical_fraud_rate` | 85934.99 | Model feature |
| 4 | `V258` | 48701.57 | Model feature |
| 5 | `C1` | 35289.67 | Model feature |
| 6 | `C14` | 26444.96 | Model feature |
| 7 | `V307` | 25092.98 | Model feature |
| 8 | `DeviceInfo` | 24485.79 | Model feature |
| 9 | `C13` | 24028.82 | Model feature |
| 10 | `card_device_combo_historical_fraud_rate` | 23962.37 | Model feature |
| 11 | `addr1` | 20533.55 | Model feature |
| 12 | `card2` | 19502.64 | Model feature |
| 13 | `card_addr_past_count` | 13376.13 | Model feature |
| 14 | `D2` | 12709.67 | Model feature |
| 15 | `TransactionAmt` | 10620.73 | Model feature |
| 16 | `V189` | 10520.95 | Model feature |
| 17 | `TransactionDT` | 9881.80 | Model feature |
| 18 | `DeviceInfo_historical_fraud_rate` | 9471.80 | Model feature |
| 19 | `card_device_past_count` | 8716.30 | Model feature |
| 20 | `C11` | 8312.04 | Model feature |

---

## 4. Artifact Directory Inventory

The following outputs have been serialized and saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Model Booster**: [`combined_lgb_model.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/combined_lgb_model.txt)
2. **Evaluation Metrics**: [`combined_evaluation_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/combined_evaluation_report.md)
3. **Comparative PR Curves**: [`combined_pr_curves.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/combined_pr_curves.png)
4. **Validation Predictions**: [`combined_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/combined_predictions.parquet)
