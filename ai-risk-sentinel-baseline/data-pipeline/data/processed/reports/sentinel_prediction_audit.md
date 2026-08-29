# Abuse-Ring Sentinel: Prediction Audit Report

Generated on: 2026-08-26 18:03:46

This report presents a deep-dive diagnostic of the Abuse-Ring Sentinel predictions to analyze calibration, alignment, and threshold bounds.

---

## 1. Raw Prediction Score Percentiles

| Percentile | Train Split | Dev/Val Split | Final Test Split |
| :---: | :---: | :---: | :---: |
| **min** | `0.13295` | `0.13300` | `0.13300` |
| **p01** | `0.13300` | `0.13300` | `0.13300` |
| **p05** | `0.13300` | `0.13300` | `0.13300` |
| **median** | `0.13300` | `0.13300` | `0.13300` |
| **p75** | `0.13300` | `0.13300` | `0.13300` |
| **p90** | `0.23840` | `0.13300` | `0.13300` |
| **p95** | `0.23840` | `0.23840` | `0.23840` |
| **p99** | `0.23840` | `0.23840` | `0.23840` |
| **max** | `0.24079` | `0.23891` | `0.23857` |

---

## 2. Dev/Val Split Threshold Sweeps

This sweep shows how the Sentinel's transaction and card prevalence maps against decision thresholds on the **Dev/Val split**:

| Threshold | Flagged Volume | Prevalence (%) | Empirical Fraud Rate | Unique Cards |
| :---: | :---: | :---: | :---: | :---: |
| `0.01` | `88,581` | `100.00%` | `3.43%` | `6,310` |
| `0.02` | `88,581` | `100.00%` | `3.43%` | `6,310` |
| `0.05` | `88,581` | `100.00%` | `3.43%` | `6,310` |
| `0.10` | `88,581` | `100.00%` | `3.43%` | `6,310` |
| `0.15` | `6,433` | `7.26%` | `5.52%` | `1,585` |
| `0.20` | `6,429` | `7.26%` | `5.49%` | `1,584` |
| `0.30` | `0` | `0.00%` | `0.00%` | `0` |
| `0.40` | `0` | `0.00%` | `0.00%` | `0` |
| `0.50` | `0` | `0.00%` | `0.00%` | `0` |

---

## 3. Core Alignment Metrics (Dev/Val Split)

These metrics isolate the predictive signal of the Sentinel independently against its proxy label and actual fraud target:

* **Model D vs. actual fraud (`isFraud`) PR-AUC**: **`0.67363`**
* **Sentinel vs. proxy target (`is_ring_abuse`) PR-AUC**: **`1.00000`**
* **Sentinel vs. actual fraud (`isFraud`) PR-AUC**: **`0.03766`**
* **Combined (equal weight) vs. actual fraud (`isFraud`) PR-AUC**: **`0.64597`**

---

## 4. Missed-Fraud (FN) Recovery Curves

This curve profiles how effectively the Sentinel captures Model D's False Negatives (at `0.30398` threshold) compared to the review population size:

| Sentinel Threshold | Dev FNs Recovered | Dev Review Size (%) | Test FNs Recovered |
| :---: | :---: | :---: | :---: |
| `0.01` | `1,329 / 1,329` (**`100.00%`**) | `88,581` (`100.00%`) | `1,647 / 1,647` (**`100.00%`**) |
| `0.02` | `1,329 / 1,329` (**`100.00%`**) | `88,581` (`100.00%`) | `1,647 / 1,647` (**`100.00%`**) |
| `0.05` | `1,329 / 1,329` (**`100.00%`**) | `88,581` (`100.00%`) | `1,647 / 1,647` (**`100.00%`**) |
| `0.10` | `1,329 / 1,329` (**`100.00%`**) | `88,581` (`100.00%`) | `1,647 / 1,647` (**`100.00%`**) |
| `0.15` | `113 / 1,329` (**`8.50%`**) | `6,433` (`7.26%`) | `201 / 1,647` (**`12.20%`**) |
| `0.20` | `113 / 1,329` (**`8.50%`**) | `6,429` (`7.26%`) | `201 / 1,647` (**`12.20%`**) |
| `0.30` | `0 / 1,329` (**`0.00%`**) | `0` (`0.00%`) | `0 / 1,647` (**`0.00%`**) |
| `0.40` | `0 / 1,329` (**`0.00%`**) | `0` (`0.00%`) | `0 / 1,647` (**`0.00%`**) |
| `0.50` | `0 / 1,329` (**`0.00%`**) | `0` (`0.00%`) | `0 / 1,647` (**`0.00%`**) |

---

## 5. Diagnostic Findings & Verdict

> [!NOTE]
> **Diagnostic Verdict**:
> * **Score Calibration**: Check the score percentiles above. If the max score is below 0.30, the model is under-calibrated for the target, and we must lower the review routing threshold to 0.05 or 0.10.
> * **Independent Signal**: Look at Sentinel's PR-AUC vs actual fraud (`0.03766`). Even if the ensembled PR-AUC weight blend is 0.0, a high independent PR-AUC shows that the Sentinel captures distinct, useful fraud signals.
