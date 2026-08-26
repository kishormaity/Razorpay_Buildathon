# Phase 9: Network Feature Refinement Report

Generated on: 2026-08-26 01:03:01

This report documents the results of the Phase 9 refined network-risk experiment, including focused aggregates (mean, max, gap, product) and novelty detection.

---

## 1. 80/20 Chronological Split Leaderboard (Stage 1)

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model D (Baseline)** | `403` | `0.58144` | `+0.00000` | `0.90507` | `0.58178` | `0.00880` | N/A |
| 2 | **N6 (Address Novelty)** | `404` | `0.57812` | `-0.00333` | `0.90534` | `0.57644` | `0.00776` | N/A |
| 3 | **N7 (Best 2 Combined)** | `405` | `0.57667` | `-0.00477` | `0.90512` | `0.57620` | `0.00926` | N/A |
| 4 | **N4 (Network Risk Product)** | `404` | `0.57517` | `-0.00627` | `0.90395` | `0.57769` | `0.00847` | N/A |
| 5 | **N1 (Network Risk Mean)** | `404` | `0.57474` | `-0.00670` | `0.90445` | `0.57736` | `0.00976` | N/A |
| 6 | **N5 (Device Novelty)** | `404` | `0.57453` | `-0.00692` | `0.90498` | `0.57528` | `0.00788` | N/A |
| 7 | **N3 (Network Risk Gap)** | `404` | `0.56505` | `-0.01639` | `0.90401` | `0.56229` | `0.00869` | N/A |
| 8 | **N2 (Network Risk Max)** | `404` | `0.56219` | `-0.01925` | `0.90457` | `0.56060` | `0.00899` | N/A |

---

## 2. 70/30 Chronological Split Leaderboard (Stage 2)

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model D (Baseline)** | `403` | `0.59882` | `+0.00000` | `0.91462` | `0.58992` | `0.00934` | N/A |
| 2 | **N7 (Best 2 Combined)** | `405` | `0.59702` | `-0.00179` | `0.91496` | `0.58307` | `0.00934` | N/A |
| 3 | **N6 (Address Novelty)** | `404` | `0.59681` | `-0.00201` | `0.91439` | `0.58191` | `0.00779` | N/A |

---

## 3. Leakage & Significance Verification

* **Pre-Training Leakage Check**: **PASSED**.

---

## 4. Scientific Insights & Conclusion

> [!IMPORTANT]
> **Final Verdict**:
> MODEL D REMAINS CHAMPION: Model D remains the overall pipeline champion at PR-AUC `0.58144` (80/20) and `0.59882` (70/30). No graph configuration consistently outpaced it.

### Project Champion Status:
Model D remains the overall platform champion at **`0.58144` PR-AUC** (80/20 split) and **`0.59882` PR-AUC** (70/30 split). 

* **H3 (Connected Risk)** was our best observed graph candidate on the 80/20 split (`0.58457`), but its improvement was not statistically significant and did not reproduce on the 70/30 split (`0.59654`).
* **Phase 9 Refinements** (including mean, max, and novelty check aggregates) did not consistently outperform Model D across both splits. This confirms that the Bayes-smoothed chronological fraud rates in Model D are highly optimal and that network-derived features, while carrying similar signal, do not yield a statistically significant stable boost.
