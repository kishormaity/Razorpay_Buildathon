# Phase 8D: Model H3 Ablation & Chronological Stability Report

Generated on: 2026-08-25 23:02:22

This report documents the ablation study (decoupling Device vs Address risk), chronological stability checking (70/30 split), and paired bootstrap significance tests.

---

## 1. 80/20 Chronological Split Leaderboard

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model D (Baseline)** | `nan` | `0.58144` | `+0.00000` | `0.90507` | `0.58178` | `0.00880` | N/A |
| 2 | **Model H3a** | `404.0` | `0.57354` | `-0.00790` | `0.90439` | `0.56963` | `0.00922` | N/A |
| 3 | **Model H3b** | `404.0` | `0.57930` | `-0.00214` | `0.90363` | `0.57455` | `0.00905` | N/A |
| 4 | **Model H3** | `405.0` | `0.58457` | `+0.00313` | `0.90563` | `0.58465` | `0.00673` | `[-0.00041, +0.00691]` |
| 5 | **Model H3i** | `406.0` | `0.57386` | `-0.00758` | `0.90427` | `0.57543` | `0.00763` | N/A |

---

## 2. 70/30 Chronological Split Leaderboard

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model D (Baseline)** | `403` | `0.59882` | `+0.00000` | `0.91462` | `0.58992` | `0.00934` | N/A |
| 2 | **Model H3a** | `404` | `0.59631` | `-0.00250` | `0.91421` | `0.58579` | `0.00999` | N/A |
| 3 | **Model H3b** | `404` | `0.59463` | `-0.00419` | `0.91397` | `0.58150` | `0.00690` | N/A |
| 4 | **Model H3** | `405` | `0.59654` | `-0.00228` | `0.91416` | `0.58614` | `0.00891` | `[-0.00512, +0.00063]` |
| 5 | **Model H3i** | `406` | `0.60011` | `+0.00130` | `0.91463` | `0.58979` | `0.00820` | N/A |

---

## 3. Key Scientific Findings & Answers

### Question 1: Which network entity matters?
* **Result**: Model H3a (Device) PR-AUC is `0.57354` compared to Model H3b (Address) PR-AUC of `0.57930`. 
* **Conclusion**: **`addr_connected_fraud_rate (Model H3b)`** is the stronger network signal of the two.

### Question 2: Do they complement each other?
* **Result**: Model H3 (both features) PR-AUC is `0.58457`.
* **Conclusion**: Yes, combining both features produces a higher score than either feature individually, showing complementary utility.
* **Interaction Check**: Model H3i (which includes the product product-interaction) scored `0.57386` on the 80/20 split, indicating that adding the interaction term does not improve performance.

### Question 3: Is H3 actually better than Model D? (Stability Check)
* **Chronological Stability**: Is H3 consistently better than Model D across splits? **`NO`**.
  * On the 80/20 split: H3 = `0.58457` vs Model D = `0.58144` (Delta: `+0.00313`).
  * On the 70/30 split: H3 = `0.59654` vs Model D = `0.59882` (Delta: `-0.00228`).
* **Bootstrap Significance**:
  * 80/20 split delta 95% CI: `[-0.00041, +0.00691]` (**CONFIDENCE INTERVAL CONTAINS ZERO**).
  * 70/30 split delta 95% CI: `[-0.00512, +0.00063]` (**CONFIDENCE INTERVAL CONTAINS ZERO**).

---

## 4. Final Promotion Recommendation

Based on these results:
* **Frozen Benchmark**: **Model D** remains the frozen champion benchmark (`0.58144` on 80/20, `0.59882` on 70/30).
* **H3 Status**: **Model H3** is our **Best Observed Graph Candidate**. 
* **Recommendation**: Since the performance is unstable across splits, we freeze Model D as the champion and do not promote H3.
