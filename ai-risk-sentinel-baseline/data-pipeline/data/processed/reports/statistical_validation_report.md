# Phase 7A: Statistical Validation & paired Bootstrap Report

Generated on: 2026-08-25 22:11:21

This report documents the statistical significance test between **Model D (Baseline Champion)** and **Model T1 (Temporal Mean Anomaly)** using paired bootstrap resampling on the chronological validation set (118,108 transactions).

---

## 1. Summary Metrics & Bootstrap Statistics

We ran $B = 1000$ resamples with replacement. Standard error (SE) represents the bootstrap standard deviation.

### PR-AUC (Primary Metric)

| Model Configuration | Mean PR-AUC | Standard Error (SE) | 95% Confidence Interval (CI) |
| :--- | :---: | :---: | :---: |
| **Model D (Baseline)** | `0.58184` | `0.00789` | `[0.56533, 0.59681]` |
| **Model T1 (Mean Anomaly)** | `0.58029` | `0.00780` | `[0.56448, 0.59518]` |
| **Delta ($\Delta = D - T1$)** | `+0.00155` | `0.00180` | `[-0.00177, 0.00494]` |

### ROC-AUC (Secondary Metric)

| Model Configuration | Mean ROC-AUC | Standard Error (SE) | 95% Confidence Interval (CI) |
| :--- | :---: | :---: | :---: |
| **Model D (Baseline)** | `0.90511` | `0.00264` | `[0.89999, 0.91042]` |
| **Model T1 (Mean Anomaly)** | `0.90513` | `0.00267` | `[0.89986, 0.91024]` |
| **Delta ($\Delta = D - T1$)** | `-0.00002` | `0.00069` | `[-0.00134, 0.00136]` |

---

## 2. Statistical Verdict & Interpretation

> [!IMPORTANT]
> **Verdict**: CONFIDENCE INTERVAL CONTAINS ZERO: The performance difference between Model D and Model T1 is statistically uncertain (indistinguishable at the 95% confidence level).

### Scientific Implications:
1. **Model D Supremacy**:
   Because the delta confidence interval for PR-AUC crosses zero, the observed score difference (+0.00154 favoring Model D) is statistically indistinguishable. The models are functionally equivalent on this validation sample.
2. **Chronological Fluctuations**:
   The overlap in confidence intervals (`[0.56533, 0.59681]` vs `[0.56448, 0.59518]`) indicates that performance fluctuations are driven by validation period transactions. Error analysis (Phase 7B) is required to profile missed fraud and isolate more stable feature domains.
