# Phase 6: Behavioral Deviation Feature Ablation Report

Generated on: 2026-08-24 19:01:22

This study systematically isolates and measures the performance impact of each anomaly category when added individually and collectively to the **Model D (Transaction + Historical)** baseline features.

---

## 1. Comparative Leaderboard

| Rank | Model Configuration | Features | PR-AUC (Primary) | ROC-AUC | Optimal F1 | Optimal Threshold | FPR @ Optimal | Best Iter | Train Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 🏆 **Model G3 (Temporal Anomaly)** | `405` | `0.58232` | `0.90630` | `0.58176` | `0.2291` | `0.01133` | `972` | `133.2s` |
| 2 |  **Model D (Baseline)** | `403` | `0.58144` | `0.90507` | `0.58178` | `0.3040` | `0.00880` | `998` | `164.4s` |
| 3 |  **Model G5 (Entity Association Anomaly)** | `405` | `0.57985` | `0.90483` | `0.57809` | `0.3093` | `0.00847` | `921` | `127.0s` |
| 4 |  **Model G1 (Amount Anomaly)** | `406` | `0.57599` | `0.90585` | `0.57463` | `0.3424` | `0.00782` | `619` | `119.3s` |
| 5 |  **Model G2 (Frequency Anomaly)** | `405` | `0.57531` | `0.90496` | `0.57285` | `0.2888` | `0.00954` | `534` | `104.0s` |
| 6 |  **Model G4 (Spending Velocity Anomaly)** | `405` | `0.57266` | `0.90502` | `0.56958` | `0.3424` | `0.00799` | `456` | `74.5s` |
| 7 |  **Model G6 (All 11 Deviations)** | `414` | `0.57247` | `0.90465` | `0.57252` | `0.2763` | `0.01015` | `599` | `116.1s` |
| 8 |  **Model G7 (All 11 + Diversity)** | `416` | `0.56534` | `0.90266` | `0.56162` | `0.3223` | `0.00935` | `352` | `72.0s` |

---

## 2. Key Scientific Findings & Analysis

1. **Ablation Performance Contribution**:
   * Inspect the ranks above. The category that provides the largest PR-AUC improvement is the primary driver of behavioral contextualization.
   * If **Model G6 (All 11 Deviations)** outperforms the individual models, the anomaly categories act in a complementary manner.
   
2. **Contextualization vs. Standalone Frequencies**:
   * Earlier experiments combining raw rolling behavioral features failed due to noise. 
   * If Model G6 beats Model D (`0.58144`), it proves that **entity-relative deviation metrics (contextualized anomaly scores)** bypass high-frequency rolling noise and successfully enrich chronological Bayes-smoothed historical models.

3. **Diversity Metric Evaluation (Model G7)**:
   * Model G7 measures the marginal benefit of adding non-deviation contextual context (`card_device_diversity`, `card_location_diversity`). Check if G7 outperforms G6 to decide if diversity should be retained in the champion pipeline.

---

## 3. Serialization Log
All models trained inside this ablation study run with identical seed parameters (`random_state=42`) and an 80/20 chronological train/validation split.
