# Production Risk Policy Scorecard
## System: Abuse-Ring Sentinel (`abuse-ring-sentinel`)
### Evaluation Basis: Locked Chronological Test Partition (15% split · 3,003 txns)

---

## 1. Frozen Production Policy Specification

The production policy routes each incoming transaction through a two-tiered decision hierarchy combining transaction-level tabular risk ($r_{\text{gbm}}$) and network-level community ring risk ($s_t$):

```text
Incoming Transaction
       ↓
Model D Individual Score (r_gbm)
       ├── r_gbm >= 0.50 (tau_d_block) ────────► BLOCK (High-Risk Individual)
       └── r_gbm < 0.50
               ↓
          Sentinel Abuse-Ring Escalation (s_t)
               ├── s_t >= 0.45 (sentinel_t) ───► MANUAL_REVIEW (Coordinated Network Risk)
               ├── r_gbm >= 0.05 (tau_review) ─► MANUAL_REVIEW (Moderate Individual Risk)
               └── otherwise ──────────────────► ALLOW (Low Risk)
```

### Threshold Selection Split:
- **Tuned strictly on Validation Split ($n=3,003$)**:
  - $\tau_{d\_block} = 0.50$ (standard operational block)
  - $\tau_{d\_review} = 0.05$ (optimizing validation $F_1 = 0.2418$)
  - $s_t = 0.45$ (maximizing validation incremental fraud capture while penalizing review volume)
- **Status on Test Evaluation**: **PERMANENTLY FROZEN**. Zero parameter adjustments were made on the test partition.

---

## 2. Policy Operating-Point Analysis (Validation Split)

The table below illustrates the sensitivity trade-offs across three explicit operational profiles evaluated on the validation partition:

| Operating Point Profile | Sentinel Threshold ($s_t$) | Validation Recall | Validation Precision | Validation FPR | Fraud Caught / Total | Incremental Fraud Caught | False Positives | Operational Guidance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **High Precision (Conservative)** | 0.50 | 43.02% | **16.82%** | **6.27%** | 37 / 86 | 0 | **183** | Minimal queue overhead. For merchants with tight review staffing; misses coordinated device rings. |
| **Balanced (Production Champion)**| **0.45** | **61.63%** | 9.31% | 17.69% | **53 / 86** | **16** | 516 | **Recommended Champion Policy**. Intercepts Community 10 device farm coordination with manageable queue. |
| **High Recall (Defensive Lockdown)**| 0.38 | 88.37% | 3.32% | 75.90% | 76 / 86 | 39 | 2,214 | Emergency defensive posture activated during active automated bot attacks to stop inventory drain. |

---

## 3. Authoritative Performance Ladder (Locked Test Set)

Evaluated once on the locked chronological test split ($n = 3,003$; 96 fraud cases; 2,907 legitimate transactions) under the **Balanced Production Policy**:

| Dimension / Metric | Model D Alone (GBM) | Model D + Abuse-Ring Sentinel | Lift / Delta |
| :--- | :--- | :--- | :--- |
| **Detection Recall** | **44.79%** | **58.33%** | **+13.54 percentage points** |
| **Precision** | 15.87% | 9.76% | -6.11 percentage points |
| **F1-Score** | 0.2343 | 0.1672 | -0.0672 |
| **PR-AUC** | 0.1636 | 0.0957 | -0.0678 |
| **False Positive Rate (FPR)** | 7.84% | 17.82% | +9.98 percentage points |
| **Fraud Cases Captured / Total** | **43 / 96** | **56 / 96** | **+13 fraud cases caught** |
| **Fraud Cases Missed** | 53 | 40 | -13 fraud cases missed |
| **Estimated Fraud Value Prevented** | **INR 6,917.32** | **INR 7,614.08** | **+INR 696.76 (+10.1%) Lift** |
| **Fraud Value Missed Exposure** | INR 7,037.47 | INR 6,340.71 | -INR 696.76 (-9.9%) |
| **Total False Positives Escalated** | 228 txns | 518 txns | +290 reviews |
| **- False Blocks (Friction Loss)** | 3 txns | 3 txns | +0 blocks |
| **- False Reviews (SLA Triage)** | 225 txns | 515 txns | +290 reviews |
| **Est. FP Cost (Strict INR 1,500)** | INR 342,000.00 | INR 777,000.00 | +INR 435,000.00 |
| **Est. FP Cost (Tiered Operational)**| INR 117,000.00 | INR 262,000.00 | +INR 145,000.00 |

---

## 4. Sentinel Incremental Value & Operational Trade-off

- **Incremental Fraud Captured**: **13 transactions** (INR 696.76).
- **Missed by Model D**: 53 transactions (INR 7,037.48).
- **Incremental Capture Rate**: **24.53%** of fraud that slipped through Model D was intercepted by Sentinel.
- **Additional Legitimate Escalations**: 290 transactions.
- **Operational Trade-off Ratio**: **22.3 legitimate manual reviews per incremental fraud caught**.
- **Crucial Operational Insight**: Sentinel **does not blindly block legitimate buyers**. All 290 additional escalations are routed to `MANUAL_REVIEW`, where analysts can release legitimate buyers without causing checkout abandonment.

---

## 5. Cost Framework Classification

To ensure commercial and scientific honesty, all metrics are classified into three distinct categories:

1. **Observed Ledger Data**:
   - `TransactionAmt` (actual monetary value of transaction)
   - `isFraud` (verified ground-truth fraud outcome)
   - `TransactionDT` (temporal order of transactions)
2. **Derived Policy Metrics**:
   - `Estimated Fraud Value Prevented`: Gross volume of fraud transactions routed to defensive intervention (`BLOCK` or `MANUAL_REVIEW`).
   - `Incremental Capture Rate`: Percentage of Model D-missed fraud caught by Sentinel.
   - `False Positives Inconvenienced`: Exact count of legitimate transactions escalated.
3. **Business Model Assumptions (`is_assumption: True`)**:
   - `fp_block_friction_cost`: INR 1,500.00 (assumed cart abandonment / customer friction cost).
   - `investigation_sla_cost`: INR 500.00 (assumed analyst triage overhead per manual review).
   - `chargeback_fee`: INR 1,200.00 (assumed gateway chargeback execution fee).
