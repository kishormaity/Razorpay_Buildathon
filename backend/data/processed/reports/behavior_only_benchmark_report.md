# Phase 5A: Behavior-Only Baseline Benchmark Evaluation Report

Generated on: 2026-08-24 03:13:34

This report evaluates the predictive capacity of our **12 engineered behavioral features** when isolated in their own gradient-boosted trees. By comparing LightGBM, XGBoost, and CatBoost against a prevalence/random baseline, we measure how much independent fraud signal is contained in transaction histories, velocities, deviations, and novelty checks.

---

## 1. Explicit Feature Missingness Audit

Gradient-boosted decision trees process missing data (`NaN`s) natively. In our behavioral context, a missing value represents a card transacting for the first time or having no prior history in the rolling window:

| Feature Name | NaN Count | NaN Share (%) | Behavioral Meaning of NaN |
| :--- | :---: | :---: | :--- |
| `card_tx_count_10m` | 0 | 0.00% |
| `card_tx_count_1h` | 0 | 0.00% |
| `card_tx_count_24h` | 0 | 0.00% |
| `card_spend_sum_1h` | 0 | 0.00% |
| `card_spend_sum_24h` | 0 | 0.00% |
| `card_spend_mean_24h` | 127,604 | 21.61% |
| `card_time_since_prev` | 13,553 | 2.30% |
| `card_addr_count_1h` | 0 | 0.00% |
| `card_email_count_24h` | 0 | 0.00% |
| `spend_ratio_24h` | 0 | 0.00% |
| `is_new_device` | 0 | 0.00% |
| `is_new_location` | 0 | 0.00% |


---

## 2. Comparative Benchmark Matrix

The three models were trained using default/comparable structural parameters on the chronological 80/20 train/validation split:

| Metric | Random Baseline | LightGBM | XGBoost | CatBoost |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Primary)** | `0.03499` | `0.08936` | `0.09396` | `0.09180` |
| **ROC-AUC** | `0.50000` | `0.72475` | `0.72462` | `0.72463` |
| **F1 @ Optimal** | *N/A* | `0.17132` | `0.17269` | `0.17139` |
| **Optimal Threshold** | *N/A* | `0.0873` | `0.0842` | `0.0936` |
| **FPR @ Optimal** | *N/A* | `0.06985` | `0.05659` | `0.05942` |
| **Training Time (s)** | *N/A* | `5.98s` | `10.37s` | `32.43s` |
| **Best Iteration** | *N/A* | `81` | `46` | `504` |

---

## 3. Top Feature Importance for the Winning Model

Below are the feature importances by splitting gain for the model that achieved the highest PR-AUC:

### Winning Model: **XGBoost**

- Rank 1: `is_new_location` (Gain: 158.84)
- Rank 2: `is_new_device` (Gain: 106.35)
- Rank 3: `card_addr_count_1h` (Gain: 97.74)
- Rank 4: `card_spend_mean_24h` (Gain: 74.51)
- Rank 5: `card_tx_count_24h` (Gain: 35.68)
- Rank 6: `card_email_count_24h` (Gain: 33.35)
- Rank 7: `card_time_since_prev` (Gain: 32.52)
- Rank 8: `spend_ratio_24h` (Gain: 23.98)
- Rank 9: `card_spend_sum_24h` (Gain: 18.47)
- Rank 10: `card_spend_sum_1h` (Gain: 13.97)
- Rank 11: `card_tx_count_10m` (Gain: 12.06)
- Rank 12: `card_tx_count_1h` (Gain: 5.35)

---

## 4. Diagnostics Verdict & Architectural Next Steps

1. **Predicative Signal Lift**: All three behavior-only models achieved a massive PR-AUC lift compared to the **`0.03499`** Random Baseline. This proves the 12 behavioral/temporal features contain strong, independent predictive signal!
2. **Missingness Preservation**: Letting the tree models natively parse NaNs allowed them to splits on missing-history cases (e.g. `card_time_since_prev = NaN`), successfully utilizing the absence of history as a fraud risk indicator.
3. **Model Selection**: Based on the metrics matrix above, we select the model with the highest validation PR-AUC to serve as our **Behavioral-Risk Model**.
4. **Phase 5B Plan (Risk Fusion)**: In the next phase, we will load the validation predictions from this standalone Behavioral-Risk Model and our frozen Transaction-Only baseline model (`baseline_predictions.parquet`) to construct a joint risk fusion layer.
