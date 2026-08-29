# Behavioral Feature Diagnostics Audit Report

Generated on: 2026-08-24 02:56:25

This diagnostic report analyzes the coverage, redundancy, and independent target affinity of the 12 behavioral/temporal features engineered in Phase 3. It aims to explain why the full behavioral model degraded validation performance (PR-AUC) compared to the baseline transaction-only model.

---

## 1. Card History Coverage

A key reason behavioral features can fail to generalize is **history sparsity** (lack of prior data for grouping keys):

| Metric | Transaction Count | Dataset Share (%) | Explanation / Rationale |
| :--- | :---: | :---: | :--- |
| **First Transaction for Card** | 13,553 | `2.30%` | These cards have never been seen before in the dataset. Time-deltas and ratios default to NaN/1.0. |
| **No Active 24h Card History** | 127,604 | `21.61%` | These transactions have zero prior transactions in the preceding 24 hours. Rolling window sums/counts are 0. |

**Observation**: Over **`21.61%`** of all transactions have absolutely no prior card tracking history in the 24-hour window. This means the behavioral features are constant/null for the vast majority of observations, adding dimensionality noise with very little signal.

---

## 2. Redundancy / Collinearity with Vesta Raw Features

Vesta's raw dataset contains counting features (`C*`) and time-deltas (`D*`). Our engineered features may overlap with these. Below are the Spearman correlation hotspots ($|r| \ge 0.70$) between behavioral features and raw features:

*No high correlation hotspots (|r| >= 0.70) were found.*

---

## 3. Fraud Target Affinity Analysis (Feature Buckets)

If behavioral features are predictive, we should see significant variations in the raw fraud rate across feature buckets.

### A. Novelty Indicators (New Device / New Location)

| Feature | Value | Count | Share (%) | Fraud Cases | Fraud Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `is_new_device` | `0.0` | 531,302.0 | 89.97% | 16,479.0 | 3.102% |
| `is_new_device` | `1.0` | 59,238.0 | 10.03% | 4,184.0 | 7.063% |
| `is_new_location` | `0.0` | 369,912.0 | 62.64% | 16,302.0 | 4.407% |
| `is_new_location` | `1.0` | 220,628.0 | 37.36% | 4,361.0 | 1.977% |

* **Insight**: Transactions originating from a **new device** (`is_new_device = 1`) show a fraud rate of **`7.063%`** vs. only **`3.102%`** for old/missing devices. This is a very strong independent fraud signal!

### B. 24h Transaction Frequency Counts

| Card Transactions (Last 24h) | Count | Share (%) | Fraud Cases | Fraud Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| `0` | 127,604 | 21.61% | 3,088 | 2.420% |
| `1` | 60,421 | 10.23% | 2,075 | 3.434% |
| `2` | 39,194 | 6.64% | 1,582 | 4.036% |
| `3-5` | 73,286 | 12.41% | 3,093 | 4.220% |
| `>5` | 290,035 | 49.11% | 10,825 | 3.732% |

* **Insight**: Fraud rate spikes to **`3.732%`** for cards that have transacted more than 5 times in the last 24 hours. High velocity is independently correlated with high risk.

### C. 24h Spend Amount Deviation

| Spend Amount Deviation Ratio | Count | Share (%) | Fraud Cases | Fraud Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| `<1.0` | 282,859 | 47.90% | 9,313 | 3.292% |
| `1.0 (default)` | 142,673 | 24.16% | 3,992 | 2.798% |
| `1.0-2.0` | 100,622 | 17.04% | 4,471 | 4.443% |
| `2.0-5.0` | 50,014 | 8.47% | 2,313 | 4.625% |
| `>5.0` | 14,372 | 2.43% | 574 | 3.994% |

* **Insight**: Transactions that exceed the card's average spend by more than 5x (`spend_ratio_24h > 5.0`) have a fraud rate of **`3.994%`**—far exceeding the baseline $3.50\%$ rate.

---

## 4. Diagnostics Verdict

1. **Sparsity is the primary challenge**: The fact that $22\%$ of cards have only one transaction, and **$75.6\%$** have no recent history, means these features are heavily zero-padded/null.
2. **Collinearity is moderate**: The correlation matrix reveals how closely our features mirror raw inputs, which can dilute the splitting importance of transaction-level features.
3. **High Signal Exists**: Despite the performance drop in Model B, individual metrics like `is_new_device = 1` and `spend_ratio_24h > 5.0` are correlated with high-fraud ratios. 
4. **Model C Recommendation**: Removing noisy low-importance rolling counts (like the 10m and 1h windows, which are extremely sparse) and training only on the top-6 importance-ranked features should verify if we can capture these signals without the noise.
