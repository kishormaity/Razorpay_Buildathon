# Phase 6: Model G (Behavioral Deviation Features) Evaluation Report

Generated on: 2026-08-24 18:43:58

This report evaluates **Model G (LightGBM Transaction + Historical + Core 11 Deviation features)** against the baseline Model A and the current champion Model D.

---

## 1. Metrics Comparison

| Metric | Model A (Tx-Only) | Model D (Tx + Hist) | Model G (Tx + Hist + Dev) | Delta (G vs D) |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Primary)** | `0.57181` | `0.58144` | `0.57247` | `-0.00897` |
| **ROC-AUC** | `0.91223` | `0.90507` | `0.90465` | `-0.00042` |
| **Optimal F1-Score** | `0.56645` | `0.58178` | `0.57252` | `-0.00926` |
| **Optimal Threshold** | `0.2740` | `0.3040` | `0.2763` | `-0.0277` |
| **FPR @ Optimal** | `0.00948` | `0.00880` | `0.01015` | `+0.00134` |

---

## 2. Champion Verdict

* **Success Status**: ⚠️ Model D remains the pipeline champion.
* **Scientific Insights**: Behavioral deviations relative to each entity's own history did not exceed the performance of the Bayes-smoothed chronological fraud-rate baseline in this configuration.

---

## 3. Top 20 Feature Importance (By Gain)

The top predictors in Model G by gain importance:

| Rank | Feature Name | Gain Importance | Description |
| :--- | :--- | :--- | :--- |
| 1 | `card1` | 165377.73 | Model Feature |
| 2 | `card_addr_combo_historical_fraud_rate` | 164261.29 | Model Feature |
| 3 | `card_email_combo_historical_fraud_rate` | 84777.76 | Model Feature |
| 4 | `V258` | 48690.10 | Model Feature |
| 5 | `C14` | 30657.58 | Model Feature |
| 6 | `card_device_combo_historical_fraud_rate` | 27398.88 | Model Feature |
| 7 | `V307` | 26962.07 | Model Feature |
| 8 | `DeviceInfo` | 26274.34 | Model Feature |
| 9 | `C1` | 22506.88 | Model Feature |
| 10 | `addr1` | 20639.87 | Model Feature |
| 11 | `C13` | 18533.50 | Model Feature |
| 12 | `card2` | 18079.85 | Model Feature |
| 13 | `D2` | 13253.91 | Model Feature |
| 14 | `V189` | 12098.79 | Model Feature |
| 15 | `card_addr_past_count` | 11371.39 | Model Feature |
| 16 | `DeviceInfo_historical_fraud_rate` | 10368.61 | Model Feature |
| 17 | `TransactionDT` | 10329.01 | Model Feature |
| 18 | `card_device_past_count` | 8820.98 | Model Feature |
| 19 | `C4` | 8648.80 | Model Feature |
| 20 | `TransactionAmt` | 8598.69 | Model Feature |

---

## 4. File Assets

The following artifacts have been created:
1. **Model G Booster**: [`deviation_lgb_model.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/deviation_lgb_model.txt)
2. **Comparative PR Curve**: [`deviation_pr_curves.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/deviation_pr_curves.png)
3. **Validation Predictions**: [`deviation_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/deviation_predictions.parquet)
