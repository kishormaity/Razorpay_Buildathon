# Phase 10C: Targeted Feature Validation Report

Generated on: 2026-08-26 01:29:10

This report documents the results of the Phase 10C targeted feature engineering ablation study designed from the residual missed-fraud profile.

---

## 1. 80/20 Chronological Split Leaderboard (Stage 1)

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model D (Baseline)** | `403` | `0.58144` | `+0.00000` | `0.90507` | `0.58178` | `0.00880` | N/A |
| 2 | **E1 (Device Missing & Email Present)** | `404` | `0.57772` | `-0.00373` | `0.90597` | `0.57924` | `0.00835` | N/A |
| 3 | **E2 (Device Missing Value Weight)** | `404` | `0.57654` | `-0.00490` | `0.90587` | `0.57463` | `0.00904` | N/A |
| 4 | **E4 (Velocity Value Drain)** | `404` | `0.57529` | `-0.00615` | `0.90499` | `0.57442` | `0.00560` | N/A |
| 5 | **E5 (Best 2 Combined)** | `405` | `0.57494` | `-0.00650` | `0.90479` | `0.57548` | `0.01153` | N/A |
| 6 | **E3 (Short Gap Velocity Ratio)** | `404` | `0.56523` | `-0.01621` | `0.90446` | `0.56079` | `0.00894` | N/A |

---

## 2. 70/30 Chronological Split Leaderboard (Stage 2)

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model D (Baseline)** | `403` | `0.59882` | `+0.00000` | `0.91462` | `0.58992` | `0.00934` | N/A |
| 2 | **E1 (Device Missing & Email Present)** | `404` | `0.59733` | `-0.00149` | `0.91501` | `0.58549` | `0.00688` | N/A |
| 3 | **E5 (Best 2 Combined)** | `405` | `0.59095` | `-0.00787` | `0.91421` | `0.57641` | `0.00907` | N/A |

---

## 3. Statistical Verification

---

## 4. Final Verdict & Conclusion

> [!IMPORTANT]
> **Final Verdict**:
> MODEL D REMAINS CHAMPION: Model D remains the overall pipeline champion at PR-AUC `0.58144` (80/20) and `0.59882` (70/30). No error-driven configuration consistently outpaced it.

### Project Champion Status:
Model D remains the overall platform champion at **`0.58144` PR-AUC** (80/20 split) and **`0.59882` PR-AUC** (70/30 split). 
