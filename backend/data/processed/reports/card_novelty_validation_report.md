# Phase 12B: Reliability-Weighted Novelty Validation Report

Generated on: 2026-08-26 01:56:25

This report documents the results of the Phase 12B reliability-weighted novelty feature ablation study.

---

## 1. 80/20 Chronological Split Leaderboard (Stage 1)

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **C5 (Card-Email Novelty Strength)** | `404` | `0.58246` | `+0.00102` | `0.90351` | `0.58135` | `0.00791` | N/A |
| 2 | **Model D (Baseline)** | `403` | `0.58144` | `+0.00000` | `0.90507` | `0.58178` | `0.00880` | N/A |
| 3 | **C1 (Card-Address Unseen)** | `404` | `0.57773` | `-0.00371` | `0.90479` | `0.57458` | `0.00680` | N/A |
| 4 | **C6 (Best 2 Combined)** | `405` | `0.57619` | `-0.00525` | `0.90565` | `0.57490` | `0.00862` | N/A |
| 5 | **C2 (Card-Email Unseen)** | `404` | `0.57569` | `-0.00575` | `0.90424` | `0.57669` | `0.00810` | N/A |
| 6 | **C3 (Card-Device Unseen)** | `404` | `0.57147` | `-0.00998` | `0.90187` | `0.57013` | `0.00848` | N/A |
| 7 | **C4 (Card-Address Novelty Strength)** | `404` | `0.56730` | `-0.01414` | `0.90648` | `0.56552` | `0.00847` | N/A |

---

## 2. 70/30 Chronological Split Leaderboard (Stage 2)

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model D (Baseline)** | `403` | `0.59882` | `+0.00000` | `0.91462` | `0.58992` | `0.00934` | N/A |
| 2 | **C6 (Best 2 Combined)** | `405` | `0.59845` | `-0.00037` | `0.91485` | `0.57924` | `0.00948` | N/A |
| 3 | **C5 (Card-Email Novelty Strength)** | `404` | `0.59491` | `-0.00390` | `0.91455` | `0.57966` | `0.00813` | N/A |

---

## 3. Statistical Verification

---

## 4. Final Verdict & Conclusion

> [!IMPORTANT]
> **Final Verdict**:
> MODEL D REMAINS CHAMPION: Model D remains the overall pipeline champion at PR-AUC `0.58144` (80/20) and `0.59882` (70/30). No novelty configuration consistently outpaced it.

### Project Champion Status:
Model D remains the overall platform champion at **`0.58144` PR-AUC** (80/20 split) and **`0.59882` PR-AUC** (70/30 split). 
