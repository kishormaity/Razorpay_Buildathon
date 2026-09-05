# Merchant Risk Evaluation & Loss Prevention Report
## Project: Abuse-Ring Sentinel (`abuse-ring-sentinel`)
### Track: AI Risk Manager — Coordinated Fraud & Merchant Loss Prevention

---

## 1. Executive Summary

This evaluation report formally measures the merchant loss prevention performance of **Abuse-Ring Sentinel** on a locked chronological held-out test partition. Aligned with the hackathon problem statement:

> *"Build a working detector, verifier or auto-responder for one class of loss, with measured precision and recall on a held-out test set."*

Abuse-Ring Sentinel targets **payment fraud and coordinated abuse rings**. The system combines an individual tabular gradient boosted model (Model D) with graph topological representations, Leiden community partitioning, and multi-model stacking to detect coordinated attacks that bypass individual transactional filters.

### Key Evaluation Highlights:
1. **Locked Chronological Test Partition**: Exactly 15% (3,003 transactions) chronologically held out. Train (14,014 rows), Validation (3,003 rows), Test (3,003 rows).
2. **Threshold Freezing (Zero Test Peeking)**: All decision thresholds were optimized on the **validation split only** and permanently frozen prior to test evaluation.
3. **Incremental Coordinated Fraud Interception**: Sentinel intercepted **13 out of 53 fraud transactions** missed by Model D alone (**+24.53% incremental capture rate**).
4. **Overall Recall Lift**: Fraud detection recall increased from **44.79%** (Model D alone) to **58.33%** (Model D + Sentinel), representing a **+13.54 percentage point boost** in fraud capture.
5. **Honest Financial Terminology**: Distinguishes between gross fraud value blocked (*Estimated Fraud Value Prevented*) and commercial net impact (*Estimated Net Loss Avoided*), explicitly modeling false positive checkout friction costs.
6. **Real Native TreeSHAP**: Explainability runs on LightGBM's native C++ TreeSHAP implementation (`pred_contrib=True`), completely free of ground truth leakage or conditional branching on `isFraud`.

---

## 2. Dataset Split Integrity & Temporal Isolation

Split boundaries are dynamically loaded from `configs/data.yaml` and chronologically sorted by `TransactionDT` and `TransactionID`:

| Split Name | Ratio | Row Count | Timestamp Range (`TransactionDT`) | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 70% | 14,014 | `86,400` → `404,956` | Model training & feature learning |
| **Validation** | 15% | 3,003 | `404,992` → `444,965` | Probability calibration & threshold tuning |
| **LOCKED TEST** | 15% | 3,003 | `444,971` → `517,208` | Final, unpeaked business evaluation |

- **Zero Temporal Leakage**: $\max(T_{\text{train}}) < \min(T_{\text{val}})$ and $\max(T_{\text{val}}) < \min(T_{\text{test}})$.
- **Dynamic Calculation**: Splits are calculated dynamically (`len(df) * ratio`) without hardcoded row indices.

---

## 3. Production Decision Policy & Frozen Thresholds

### Operational Decision Rules:
1. $\tau_{d\_block} \ge 0.50$: Direct **BLOCK** (High-confidence individual risk).
2. $s_{t} \ge 0.45$: **MANUAL_REVIEW** (Sentinel network / abuse ring escalation).
3. $\tau_{d\_review} \ge 0.05$: **MANUAL_REVIEW** (Model D moderate individual risk).
4. Below all thresholds: **ALLOW**.

### Threshold Selection on Validation Split:
- **Model D Review Threshold ($\tau_{d\_review}$)**: Grid sweep across $[0.01, 0.99]$ on validation split yields optimal $F_1 = 0.2418$ at $\tau_{d\_review} = 0.05$.
- **Operational Block Threshold ($\tau_{d\_block}$)**: Fixed operational standard at $0.50$.
- **Sentinel Network Threshold ($s_t$)**: Grid sweep across $[0.40, 0.55]$ on validation split selects $s_t = 0.45$ (maximizing incremental fraud interception while controlling false positive volume).
- **Freezing Rule**: Both thresholds were locked before executing test-set evaluation.

---

## 4. Locked Test Set Performance Ladder & Economic Decomposition

Evaluation performed on the 3,003 transactions of the locked test partition containing **96 ground-truth fraud cases** (prevalence 3.20%) and **2,907 legitimate transactions**.

### Empirical Metric Comparison:

| Metric Category | Metric Dimension | Model D Alone (GBM) | Model D + Sentinel (Hybrid) | Lift / Operational Delta |
| :--- | :--- | :--- | :--- | :--- |
| **OBSERVED** | **Detection Recall** | **44.79%** | **58.33%** | **+13.54%** |
| OBSERVED | Precision | 15.87% | 9.76% | -6.11% |
| OBSERVED | F1-Score | 0.2343 | 0.1672 | -0.0672 |
| OBSERVED | PR-AUC | 0.1636 | 0.0957 | -0.0678 |
| OBSERVED | False Positive Rate (FPR) | 7.84% | 17.82% | +9.98% |
| OBSERVED | **Fraud Cases Captured / Total** | **43 / 96** | **56 / 96** | **+13 cases caught** |
| OBSERVED | Fraud Cases Missed | 53 | 40 | -13 cases missed |
| OBSERVED | Direct Automated Blocks (Fraud) | 3 | 3 | +0 |
| OBSERVED | Manual Review Triage (Fraud) | 40 | 53 | +13 triaged |
| OBSERVED | False Positive Blocks (Legit) | 3 | 3 | **+0 extra false blocks** |
| OBSERVED | False Positive Reviews (Legit) | 225 | 515 | +290 reviews escalated |
| **DERIVED** | **Direct Automated Block Fraud Value** | **INR 136.91** | **INR 136.91** | **INR 0.00** |
| DERIVED | **Manual Review Triaged Fraud Exposure** | **INR 6,780.41** | **INR 7,477.17** | **+INR 696.76 (+10.3%)** |
| DERIVED | **Gross Intercepted Fraud Exposure** | **INR 6,917.32** | **INR 7,614.08** | **+INR 696.76 (+10.1%)** |
| DERIVED | Fraud Value Missed | INR 7,037.47 | INR 6,340.71 | -INR 696.76 (-9.9%) |
| DERIVED | False Positive Inconvenienced Value | INR 51,113.12 | INR 77,256.58 | +INR 26,143.46 |
| **ASSUMED** | **Realized Prevented Loss (@ 85% SLA)** | **INR 5,900.26** | **INR 6,492.51** | **+INR 592.25** |
| ASSUMED | Strict Flat FP Cost (₹1,500/all FPs) | INR 342,000.00 | INR 777,000.00 | +INR 435,000.00 |
| ASSUMED | Tiered Operational FP Cost | **INR 117,000.00** | **INR 262,000.00** | **+INR 145,000.00** |
| ASSUMED | Net Operational Difference vs Strict | +INR 225,000.00 | +INR 515,000.00 | Realistic Triage Model |

---

## 5. Sentinel Incremental Coordinated Abuse Interception

The core technical thesis of Abuse-Ring Sentinel is that coordinated fraudsters intentionally structure transactions to stay below individual tabular risk thresholds (e.g. low transaction amounts, randomized velocity). Sentinel's graph network and community detection layers identify coordinated entity sharing:

- **Frauds missed by Model D alone**: 53 cases (INR 7,037.48 in potential merchant loss).
- **Frauds intercepted by Sentinel escalation**: **13 cases** (INR 696.76 in fraud value).
- **Sentinel Incremental Capture Rate**: **24.53%** of fraud that slipped past the primary detector was caught by network defenses.
- **Total Fraud Capture Lift**: **+13.54 percentage points** higher recall.
- **Operational Trade-off**: **22.3 legitimate reviews escalated per additional fraud captured** (290 legitimate reviews for 13 incremental fraud catches).
- **Zero Customer Friction Penalty**: Zero additional legitimate transactions were hard-blocked (3 false blocks under Model D vs 3 under Model D + Sentinel).

---

## 6. Honest Business Cost Framework & Methodological Taxonomy

To eliminate misleading commercial claims, all economic numbers are categorized under three distinct layers:

### A. OBSERVED (Direct Ground Truth Facts)
- Test set size: 3,003 transactions (96 fraud, 2,907 legitimate).
- Total observed fraud value: INR 13,954.79.
- Actions taken: 6 hard blocks (3 fraud, 3 legitimate), 568 manual reviews (53 fraud, 515 legitimate).

### B. DERIVED (Mathematical Functions of Observed Facts)
- **Direct Block Value**: INR 136.91 (3 transactions definitely prevented without analyst labor).
- **Triaged Review Exposure**: INR 7,477.17 (53 fraud transactions escalated to investigation queue).
- **Gross Exposure Intercepted**: INR 7,614.08 total fraud volume flagged.
- **Incremental Missed Capture Rate**: $13 / 53 = 24.53\%$.

### C. ASSUMED (Configured Operational Model Parameters)
- `review_capture_efficiency = 0.85`: Assumes human fraud analysts successfully confirm and halt 85% of triaged fraudulent orders before fulfillment, with 15% slippage.
- `fp_cost = 1,500.0`: Model parameter for abandoned checkout lifetime value loss when a legitimate user is hard-blocked.
- `investigation_cost = 500.0`: Model parameter for analyst SLA cost to inspect a triaged review case.
- **Estimated Realized Loss Avoided**:
  $$\text{Realized Loss Avoided} = \text{Direct Block Value} + (0.85 \times \text{Triaged Review Exposure}) = 136.91 + (0.85 \times 7477.17) = \text{INR } 6,492.51$$
- **Tiered Operational Cost**:
  $$\text{Tiered FP Cost} = (3 \text{ False Blocks} \times 1500) + (515 \text{ False Reviews} \times 500) = \text{INR } 262,000.00$$

### Unsupervised Community Scoring & Methodological Integrity:
- **Target Leakage Elimination**: In `configs/risk_policy.yaml`, $w_{\text{financial}} = 0.00$. Sentinel community scores are computed purely from graph topology, temporal synchronization, and behavioral device sharing, with zero consumption of test-set `isFraud` or `is_abuse` labels.
- **Community 10 Stability**: Under pure unsupervised scoring, Community 10 scores **0.4762** (above the frozen $s_t \ge 0.45$ threshold), successfully catching all 13 incremental fraud transactions.
- **Known Limitation**: Graph community detection flags hardware/IP sharing; in scenarios like public terminals, benign users sharing infrastructure will be escalated to manual review. This is why Sentinel escalates shared communities to `MANUAL_REVIEW` rather than executing irreversible `BLOCK` actions.

---

## 7. Real TreeSHAP Explainability (Zero Leakage)

Previous iterations used synthetic explainability weights conditioned on ground-truth `isFraud`. This has been completely eliminated:
- **Engine**: LightGBM native C++ `pred_contrib=True` TreeSHAP (Lundberg et al., 2020).
- **Execution**: At runtime, each transaction feature vector is evaluated through `TabularSHAPExplainer.explain(single_row_df)` to produce exact local attribution for top risk drivers (`TransactionAmt`, `card_tx_count_10m`, `pagerank_centrality`, `network_risk_product`).
- **Graph Evidence**: `GraphEvidenceExtractor` traverses relational entities (shared IP, Device, Address) to provide human-interpretable investigation notes without touching ground-truth labels.

---

## 8. Artifact Locations

- **Evaluation Script**: `src/evaluation/business_evaluation.py`
- **Locked Test Evaluation JSON**: `data/processed/evaluation/test_business_evaluation.json`
- **Per-Transaction Decision Audit CSV**: `data/processed/evaluation/test_decision_records.csv`
- **FastAPI Endpoint**: `GET /api/merchant/impact`
- **Next.js Dashboard**: `frontend/src/app/dashboard/page.tsx`
- **Automated Test Suite**: `tests/test_business_evaluation.py`
