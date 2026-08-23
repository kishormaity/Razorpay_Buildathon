# Phase 5B: Risk Score Fusion Evaluation Report

Generated on: 2026-08-24 03:19:41

This report evaluates stacked combinations of our transaction-risk and behavioral-risk models. By separating transaction-level logic (Model A LightGBM) and card history context (Model B XGBoost) and fusing their output probabilities, we measure whether stacked fusion improves predictive power compared to a transaction-only classifier.

---

## 1. Experimental Protocol & Splits

To ensure strict validation and prevent data leakage:
* **Validation Subset Size**: 118,108 rows.
* **Meta-Development Set (Earliest 50%)**: 59,054 rows. Used for linear weight grid search and logistic stacking training.
* **Meta-Test Set (Latest 50%)**: 59,054 rows. Used for final comparative evaluations.

All model metrics below are calculated on the **Meta-Test Set** for a clean, apples-to-apples baseline comparison.

---

## 2. Comparative Performance Matrix (Evaluated on Meta-Test)

| Metric | Model A (Transaction-Only) | Behavioral Model (XGBoost) | Weighted Fusion ($w_{opt}$ = 1.00) | Logistic Stack Meta-Classifier |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Primary)** | `0.54831` | `0.10583` | `0.54831` | `0.54529` |
| **ROC-AUC** | `0.90667` | `0.73221` | `0.90667` | `0.90729` |
| **F1 @ Optimal** | `0.54939` | `0.19172` | `0.54939` | `0.54991` |
| **Optimal Threshold** | `0.2740` | `0.0840` | `0.2740` | `0.1133` |
| **FPR @ Optimal** | `0.01054` | `0.06129` | `0.01054` | `0.01052` |

---

## 3. Key Observations & Findings

1. **Optimal Linear Combination Weight**: The grid-search selected **`w_opt = 1.00`** as the weight for the transaction baseline, meaning the fused score is computed as:
   `fused_prob = 1.00 * p_tx + 0.00 * p_behav`
   This shows that the transaction model remains the dominant feature, but is supplemented by a `0.00` weight of the behavioral classifier.
2. **Logistic Stack Calibration**: The meta-model coefficients are `p_tx = 8.1760` and `p_behav = 0.3418` with an intercept of `-4.3224`.
3. **PR-AUC Improvement**: Check the Delta (Fusion vs. Model A) in the matrix above. If Fused Model PR-AUC exceeds Model A's `0.54831`, it demonstrates that historical behavioral risk is complementary to transaction-level logic.

---

## 4. Serialization Outputs

The following files have been saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Fused Probabilities Parquet**: [`fused_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/fused_predictions.parquet) (contains columns for transaction probabilities, behavioral probabilities, weighted fusion, and logistic stack fusion)
2. **Evaluation Metrics**: [`risk_fusion_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/risk_fusion_report.md)
3. **Comparative PR Curves Graph**: [`fused_pr_curves.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/fused_pr_curves.png)
