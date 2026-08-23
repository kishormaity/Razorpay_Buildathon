# Phase 5D: XGBoost Transaction + Historical Features Model Evaluation Report

Generated on: 2026-08-24 03:59:53

This report compares **Model E (XGBoost Transaction + Chronological expanding Historical Features)** directly against **Model A (LightGBM Transaction-Only Baseline)** and **Model D (LightGBM Transaction + Historical)** to determine which boosting algorithm leverages our historical features best.

---

## 1. Metrics Leaderboard

| Model Config | Algorithm | Features | PR-AUC | ROC-AUC | Optimal F1 | Optimal Threshold | FPR @ Optimal | Training Time | Best Iteration |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | LightGBM | Transaction | `0.57181` | `0.91223` | `0.56645` | `0.2740` | `0.00948` | `94.91s` | `831` |
| **Model D** | LightGBM | Transaction + Historical | `0.58144` | `0.90507` | `0.58178` | `0.3040` | `0.00880` | `119.27s` | `998` |
| **Model E** | **XGBoost** | Transaction + Historical | `0.55965` | `0.89799` | `0.56932` | `0.2423` | `0.00943` | `413.15s` | `444` |

---

## 2. Confusion Matrices Comparison (Optimal Threshold)

* **Model A (TN / FP / FN / TP @ 0.2740)**: 112,963 / 1,081 / 2,031 / 2,033
* **Model D (TN / FP / FN / TP @ 0.3040)**: 113,040 / 1,004 / 1,985 / 2,079
* **Model E (TN / FP / FN / TP @ 0.2423)**: 112,969 / 1,075 / 2,019 / 2,045

---

## 3. Top 20 Feature Importance (XGBoost Split Gains)

This list shows the top 20 attributes contributing the most predictive weight to Model E:

| Rank | Feature Name | Gain Weight | Description |
| :--- | :--- | :--- | :--- |
| 1 | `V258` | 0.134528 | Model feature |
| 2 | `V97` | 0.124280 | Model feature |
| 3 | `V175` | 0.064411 | Model feature |
| 4 | `card_addr_combo_historical_fraud_rate` | 0.040969 | Model feature |
| 5 | `V189` | 0.038810 | Model feature |
| 6 | `V128` | 0.033469 | Model feature |
| 7 | `V149` | 0.032523 | Model feature |
| 8 | `V102` | 0.024471 | Model feature |
| 9 | `V187` | 0.023552 | Model feature |
| 10 | `card_email_combo_historical_fraud_rate` | 0.023302 | Model feature |
| 11 | `V307` | 0.020154 | Model feature |
| 12 | `V133` | 0.016868 | Model feature |
| 13 | `V317` | 0.016046 | Model feature |
| 14 | `V91` | 0.015497 | Model feature |
| 15 | `C14` | 0.013695 | Model feature |
| 16 | `V308` | 0.012127 | Model feature |
| 17 | `V294` | 0.011035 | Model feature |
| 18 | `C1` | 0.010081 | Model feature |
| 19 | `C4` | 0.009717 | Model feature |
| 20 | `V323` | 0.007967 | Model feature |

---

## 4. Artifact Directory Inventory

The following outputs have been serialized and saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Model Booster**: [`historical_xgb_model.json`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/historical_xgb_model.json)
2. **Evaluation Metrics**: [`historical_xgb_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/historical_xgb_report.md)
3. **Comparative PR Curves**: [`historical_comparative_curves.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/historical_comparative_curves.png)
4. **Validation Predictions**: [`historical_xgb_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/historical_xgb_predictions.parquet)
