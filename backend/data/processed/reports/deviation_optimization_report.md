# Phase 6B & 6C: Focused Behavioral Deviation Optimization Report

Generated on: 2026-08-25 21:55:12

This report documents the results of the second-stage screening, combination search, and stability checks on the temporal acceleration and spend-temporal interaction features.

---

## 1. Comparative Leaderboard (80/20 Chronological Split)

| Rank | Model Configuration | Features | PR-AUC (Primary) | Delta vs G3 | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | Best Iter |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 |  **Model D (Baseline)** | `403` | `0.58144` | `+0.00495` | `+0.00000` | `0.90507` | `0.58178` | `0.00880` | `998` |
| 2 |  **T1 (Isolate Mean)** | `404` | `0.57991` | `+0.00342` | `-0.00154` | `0.90510` | `0.57885` | `0.00925` | `972` |
| 3 |  **Temporal + Entity + Frequency** | `409` | `0.57945` | `+0.00297` | `-0.00199` | `0.90583` | `0.58008` | `0.00794` | `704` |
| 4 |  **Temporal + Velocity + Frequency** | `409` | `0.57901` | `+0.00252` | `-0.00243` | `0.90409` | `0.57384` | `0.00810` | `535` |
| 5 |  **C5 (Temporal + Entity + Amount + Velocity)** | `412` | `0.57792` | `+0.00143` | `-0.00352` | `0.90460` | `0.57850` | `0.00894` | `864` |
| 6 |  **TV3 (Temporal + Spending)** | `407` | `0.57705` | `+0.00056` | `-0.00439` | `0.90439` | `0.57686` | `0.00739` | `863` |
| 7 |  **TE3 (Temporal + Entity)** | `407` | `0.57693` | `+0.00044` | `-0.00451` | `0.90387` | `0.57784` | `0.00910` | `864` |
| 8 |  **Model G3 (Temporal Baseline)** | `405` | `0.57649` | `+0.00000` | `-0.00495` | `0.90597` | `0.58166` | `0.00811` | `691` |
| 9 |  **TF3 (Temporal + Frequency)** | `407` | `0.57459` | `-0.00190` | `-0.00685` | `0.90327` | `0.57364` | `0.00886` | `803` |
| 10 |  **T2 (Isolate Median)** | `404` | `0.57316` | `-0.00333` | `-0.00828` | `0.90496` | `0.56836` | `0.00876` | `546` |
| 11 |  **Temporal + Velocity + Entity + Frequency** | `411` | `0.57193` | `-0.00456` | `-0.00951` | `0.90519` | `0.57216` | `0.00813` | `667` |
| 12 |  **TR2 (Temporal + Interaction)** | `407` | `0.57156` | `-0.00493` | `-0.00988` | `0.90556` | `0.57090` | `0.00811` | `440` |
| 13 |  **TR3 (Temporal + New Features)** | `409` | `0.57001` | `-0.00648` | `-0.01143` | `0.90412` | `0.56869` | `0.00954` | `442` |
| 14 |  **TR1 (Temporal + Acceleration)** | `407` | `0.56980` | `-0.00669` | `-0.01164` | `0.90429` | `0.56985` | `0.00853` | `379` |
| 15 |  **TA4 (Temporal + Amount)** | `408` | `0.56705` | `-0.00944` | `-0.01440` | `0.90298` | `0.56383` | `0.00792` | `451` |
| 16 |  **Temporal + Velocity + Entity** | `409` | `0.56702` | `-0.00947` | `-0.01443` | `0.90521` | `0.56135` | `0.01176` | `359` |

---

## 2. Split Stability Check (70/30 Chronological Split)

To ensure that the performance improvements are not split-specific or overfit to the 80/20 chronological validation boundary, we evaluated the baseline models and the top candidate configuration on an alternative **70/30 split** (using the earliest 70% for training and the latest 30% for validation):

| Model Configuration (70/30 Split) | PR-AUC | Optimal F1 | FPR @ Optimal | Best Iteration |
| :--- | :---: | :---: | :---: | :---: |
| **Model D (Baseline)** | `0.59882` | `0.58992` | `0.00934` | `404` |
| **Model G3 (Temporal Baseline)** | `0.58999` | `0.57724` | `0.00958` | `370` |
| **T1 (Isolate Mean)** | `0.59837` | `0.58356` | `0.00963` | `395` |

---

## 3. Scientific Insights & Conclusion

1. **Spend-Temporal Multiplication Interactions**:
   * Inspect the rank of **TR2 (Temporal + Interaction)**. This configuration tests the product of the log-scaled amount deviation and the log-scaled temporal acceleration. 
   * If it outpaces the baseline G3, it confirms that combining timing and size multiplicatively provides a cleaner, spike-free fraud signature.
   
2. **Temporal Acceleration Transformation**:
   * Inspect **TR1 (Temporal + Acceleration)**. Reversing and log-transforming the time gaps to measure acceleration can highlight high-frequency card drains.
   
3. **Optimized Selection Verdict**:
### ⚠️ FINAL VERDICT: MODEL D REMAINS CHAMPION
Model D remains the overall pipeline champion with a PR-AUC of 0.58144. T1 (Temporal Mean Deviation) is the strongest deviation-based candidate, achieving 0.57991 on the 80/20 split and 0.59837 on the 70/30 split, but it does not surpass Model D.
