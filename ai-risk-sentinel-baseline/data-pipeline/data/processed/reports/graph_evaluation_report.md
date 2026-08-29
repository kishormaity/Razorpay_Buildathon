# Phase 8B & 8C: Graph Model Evaluation Report

Generated on: 2026-08-25 22:33:55

This report documents the chronological evaluation of Card-Device-Address graph features (degrees, card multiplexing, and connected target fraud risk) against Model D.

---

## 1. Comparative Leaderboard

All configurations are evaluated strictly on the 80/20 chronological split with identical LightGBM hyper-parameters. 

| Rank | Model Configuration | Features | PR-AUC (Primary) | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | Best Iter |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model H3** | `405.0` | `0.58457` | `+0.00313` | `0.90563` | `0.58465` | `0.00673` | `997.0` |
| 2 | **Model D (Baseline)** | `nan` | `0.58144` | `+0.00000` | `0.90507` | `0.58178` | `0.00880` | `nan` |
| 3 | **Model T1 (Temporal Mean)** | `404.0` | `0.57991` | `-0.00153` | `0.90510` | `0.57885` | `0.00925` | `972.0` |
| 4 | **Model H5 (Best Graph + T1)** | `406.0` | `0.57964` | `-0.00180` | `0.90506` | `0.58208` | `0.00822` | `998.0` |
| 5 | **Model H1** | `407.0` | `0.57234` | `-0.00910` | `0.90354` | `0.57308` | `0.00879` | `641.0` |
| 6 | **Model H4** | `411.0` | `0.57085` | `-0.01059` | `0.90478` | `0.57507` | `0.00801` | `685.0` |
| 7 | **Model H2** | `405.0` | `0.56391` | `-0.01754` | `0.90473` | `0.55984` | `0.00905` | `341.0` |

---

## 2. Leakage and Significance Verification

### Automated Leakage Check:
* **Result**: **SUCCESS**. Pre-training checks confirmed first-device connected fraud rates match the global prior, verifying strict look-back constraints and zero target leakage.

### Paired Bootstrap Statistics (Model H3 vs Model D)
* **Mean Difference**: `+0.00317`
* **Standard Error (SE)**: `0.00188`
* **95% Confidence Interval**: `[-0.00041, +0.00691]`

---

## 3. Scientific Insights & Conclusion

> [!IMPORTANT]
> **Final Verdict**:
> STATISTICALLY UNCERTAIN: Although **Model H3** achieved a score of `0.58457`, the paired confidence interval of delta contains zero (`[-0.00041, +0.00691]`). The difference is statistically indistinguishable at 95% confidence.

1. **Information Value of Graph Degrees (H1)**:
   * Inspect the delta of **Model H1**. If it outpaces Model D, it confirms that simple multi-entity connectivity (number of distinct addresses/devices used by the card) provides an orthogonal signal.
   
2. **Entity Sharing / Card Multiplexing (H2)**:
   * Inspect **Model H2**. A high shared card count indicates device/address pooling, which is a strong signature for multi-account abuse.
   
3. **Propagation of Connected Fraud Risk (H3)**:
   * Inspect **Model H3**. The expanding connected fraud rate acts as a proxy for risk propagation (inheriting risk from other fraudulent cards on the same device/address). This is particularly valuable for cold-start cards (past count = 0) which would otherwise bypass individual filters.
