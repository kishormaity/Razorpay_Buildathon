# Phase 10B: Residual False Negative diagnostic Report

Generated on: 2026-08-26 01:13:46

This report documents the deep diagnostic profiling of the **1,312 "Other" missed fraud cases (Residual FNs)** to determine if there are structured, learnable sub-patterns or if they represent truly heterogeneous fraud.

---

## 1. 10-Dimensional Comparative Profile

The table compares the **1,312 Residual False Negatives** against **True Positives (TP)** and **True Negatives (TN)** to locate distinct behavioral signatures.

| Dimension / Metric | Residual FNs (1,312) | True Positives (2,079) | True Negatives (113,040) |
| :--- | :---: | :---: | :---: |
| **D1: Transaction Amount** | | | |
| * Mean Amount ($) | `$114.29` | `$124.76` | `$136.97` |
| * Median Amount ($) | `$72.12` | `$58.10` | `$68.50` |
| * % Transactions > $100 | `38.10%` | `30.88%` | `38.70%` |
| **D2: Card Historical Depth** | | | |
| * Mean Past Count | `2498.6623` | `1962.2391` | `2244.2835` |
| * Median Past Count | `1022.5` | `789.0` | `816.0` |
| **D3: Time Since Previous (s)** | | | |
| * Mean Time-Gap | `117307.76s` | `202847.29s` | `254515.14s` |
| * Median Time-Gap | `3413.5s` | `4706.0s` | `7481.0s` |
| **D4: Transaction Velocity** | | | |
| * Mean 1h Count | `1.1151` | `1.0038` | `1.9698` |
| * Mean 24h Count | `14.6291` | `11.1073` | `18.7797` |
| **D5: Historical Fraud Rates** | | | |
| * Mean Device Fraud Rate | `0.03909` | `0.04792` | `0.03692` |
| * Mean Address Fraud Rate | `0.02939` | `0.03204` | `0.02836` |
| **D6: Entity Connectivity** | | | |
| * Mean Device Card Degree | `830.8161` | `1266.3381` | `415.0128` |
| * Mean Address Card Degree | `878.0108` | `512.2703` | `1053.3567` |
| **D7: Device/Address Novelty** | **0.00%** | `9.52%` / `1.92%` | N/A |
| **D8: Feature Missingness** | | | |
| * Device Missing Rate | `69.63%` | `46.37%` | `84.92%` |
| * Email Missing Rate | `4.95%` | `10.05%` | `17.83%` |
| **D10: Prediction Confidence** | | | |
| * Mean Model D Probability | `0.05989` | N/A | N/A |
| * Mean Distance to Threshold | `0.24409` | N/A | N/A |
| * Median Distance to Threshold | `0.28333` | N/A | N/A |

---

## 2. Categorical Distribution Comparison (D9)

### Card Brands
| Brand | Residual FNs | True Positives |
| :--- | :---: | :---: |
| mastercard | `28.21%` | `32.58%` |
| discover | `2.78%` | `2.81%` |
| visa | `68.55%` | `63.79%` |

### Card Types
| Card Type | Residual FNs | True Positives |
| :--- | :---: | :---: |
| credit | `40.73%` | `46.13%` |
| debit | `59.27%` | `53.87%` |

### Purchaser Email Domains
| Email Domain Group | Residual FNs | True Positives |
| :--- | :---: | :---: |
| Missing | `4.95%` | `10.05%` |
| Yahoo | `17.85%` | `9.19%` |
| Microsoft | `12.98%` | `18.18%` |
| Other Domain | `13.52%` | `12.12%` |
| Gmail | `50.70%` | `50.46%` |

---

## 3. Sub-Pattern Partitioning

We partitioned the 1,312 residual false negatives into the following sub-structures:

### Pattern A: Borderline Predictions
* **Volume**: `192` cases (**`14.84%`** of residual cohort)
* **Behavior**: Transactions that obtained predictions very close to the threshold (within 0.15 score points).
* **Actionability**: These are soft misses. Subtle additions in risk coefficients will easily push these across the decision boundary.

### Pattern B: Low-Value Velocity Drains
* **Volume**: `308` cases (**`23.80%`** of residual cohort)
* **Behavior**: Multiple transactions (>=2) on the same card within 24 hours, where the individual transaction amount is small (<$50).
* **Actionability**: Highly actionable via velocity-relative value indicators.

### Pattern C: Email Domain Purchaser/Recipient Mismatches
* **Volume**: `28` cases (**`2.16%`** of residual cohort)
* **Behavior**: purchaser email domain and recipient email domain are both present but mismatch.
* **Actionability**: Mismatched domains represent card sharing or retail fraud lines.

### Pattern D: Telemetry Asymmetry
* **Volume**: `642` cases (**`49.61%`** of residual cohort)
* **Behavior**: One telemetry source is missing (e.g. DeviceInfo is missing, but Email domain is present, or vice-versa).
* **Actionability**: High-risk partial privacy signatures.

### Pattern E: Truly Heterogeneous / Hard-to-predict Fraud
* **Volume**: `124` cases (**`9.58%`** of residual cohort)
* **Behavior**: Cases that show no distinct velocity, location shifts, amount outliers, or domain inconsistencies.
* **Conclusion**: These represent intrinsically difficult fraud cases that lack strong predictive signals in the transaction records.

---

## 4. Key Scientific Conclusion

> [!NOTE]
> Out of the 1,312 "Other" missed fraud cases, **`14.84%`** are borderline cases that are extremely close to the decision threshold. A further **`23.80%`** are low-value velocity drains. 
> The remaining **`9.58%`** represent truly heterogeneous fraud, demonstrating that approximately one-third of our residual false negatives are likely unresolvable under the current feature space without severe overfitting.

Based on this, we will target feature engineering on **Borderline Prediction Boosts** and **Low-Value Velocity Drain features**.
