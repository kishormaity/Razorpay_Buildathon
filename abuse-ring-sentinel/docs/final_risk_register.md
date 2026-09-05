# Final Competition Risk Register & Hostile Review Vulnerabilities
## Project: `abuse-ring-sentinel`
### Classification: 🔴 CRITICAL | 🟠 HIGH | 🟡 MEDIUM | 🟢 LOW
### Evaluation Scope: Full Repository Forensic Review Prior to Submission

---

## 1. Executive Summary

This risk register catalogs every architectural, methodological, economic, and demonstration vulnerability discovered during the hostile-reviewer forensic audit. Each item is scored by severity, its exact code location is cited, and its concrete competition risk is evaluated alongside the recommended fix.

---

## 2. Risk Register Matrix

```
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| #   | Severity | Vulnerability Description           | Exact File & Line Location  | Competition Impact & Auditor Risk Assessment |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R1  | CRITICAL | Full-DB Target Query in Ring Score  | src/graph/ring_detector.py  | Queries `is_abuse` across all 20,000 rows.   |
|     |          | (Potential Label Leakage)           | lines 55-57, 120-126        | Mitigation: Unsupervised score of Community  |
|     |          |                                     |                             | 10 is 0.4762 >= 0.45; catches remain intact. |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R2  | CRITICAL | Manual Review Counted as Guaranteed | src/evaluation/business_    | ₹7,477.17 of ₹7,614.08 is triaged to review. |
|     |          | Prevented Merchant Loss             | evaluation.py:L115-116      | Claiming 100% loss prevented is economically |
|     |          |                                     |                             | unrealistic unless human triage is modeled.  |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R3  | HIGH     | Demo Endpoint Fallback Key Mismatch | api/main.py:L531-554        | `get_demo_cases` queries keys that differ    |
|     |          | in /api/demo/cases                  |                             | from `get_alert_detail`, causing Case A to   |
|     |          |                                     |                             | fall back to ₹482.10 instead of ₹47.73.      |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R4  | HIGH     | Policy Mismatch in Single Alert API | api/main.py:L236-245        | `get_alert_detail` uses legacy `r_final`     |
|     |          | Endpoint                            |                             | (0.50/0.20), returning ALLOW for TXN-3004262 |
|     |          |                                     |                             | instead of production MANUAL_REVIEW.         |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R5  | HIGH     | Static Mock Data in Top Dashboard   | frontend/src/app/dashboard/ | Top charts show 1,284 events / ₹24.8L, while |
|     |          | Charts                              | page.tsx lines 50-100       | test benchmark shows 3,003 txns / ₹4.3L.     |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R6  | MEDIUM   | False-Confidence Unit Test 03       | tests/test_business_        | Inverts `isFraud` on pre-computed DataFrame  |
|     |          |                                     | evaluation.py:L79-88        | without re-running `booster.predict()`.      |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R7  | MEDIUM   | False-Confidence Unit Test 10       | tests/test_business_        | Checks HTTP 200 on metadata endpoints        |
|     |          |                                     | evaluation.py:L195-212      | without sending live transaction payloads.   |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R8  | MEDIUM   | Coarse Device Hash Collisions       | src/data/ingestion.py       | `DEV-{hash(device + os) % 100000}` groups    |
|     |          |                                     | lines 114-115               | generic "Windows 10" users into same token.  |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R9  | MEDIUM   | Instantaneous Test Target Update in | src/features/graph.py       | `card_stats['fraud'] += targets[i]` updates  |
|     |          | Feature Engineering                 | lines 118-120               | dynamically without chargeback lag.          |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R10 | LOW      | FastAPI Startup Deprecation Warning | api/main.py:L52             | Uses `@app.on_event("startup")` deprecated   |
|     |          |                                     |                             | in newer FastAPI versions (favor lifespan).  |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
| R11 | LOW      | Missing `auto_decision` in Alerts   | api/main.py:L169-178        | `/api/alerts` does not return `auto_decision`|
|     |          | Endpoint                            |                             | causing UI table to default to ALLOW badge.  |
+-----+----------+-------------------------------------+-----------------------------+----------------------------------------------+
```

---

## 3. Deep Analysis of Critical & High Risks

### R1 [CRITICAL]: Sentinel Financial Score Target Leakage
- **Location**: `src/graph/ring_detector.py:L55-57, L120-126`
- **Code**:
  ```python
  tx_df = pd.read_sql_query("SELECT user_id, amount, timestamp, is_abuse FROM transactions;", conn)
  ...
  s_fin = float(np.mean(fraud_labels)) if fraud_labels else 0.0
  r_final = w_struct * s_struct + w_temp * s_temp + w_behav * s_behav + w_fin * s_fin
  ```
- **Auditor Critique**: Ingesting `is_abuse` across the entire SQLite database into `s_fin` means test-set labels were accessible during ring scoring if $w_{\text{financial}} > 0$.
- **Empirical Defense**:
  In Community 10:
  - $s_{\text{struct}} = 0.90734 \times 0.35 = 0.31757$
  - $s_{\text{temp}} = 0.19647 \times 0.25 = 0.04912$
  - $s_{\text{behav}} = 0.54754 \times 0.20 = 0.10951$
  - $\mathbf{s_{\text{unsupervised}} = 0.4762 \ge 0.45}$
  Because the unsupervised score exceeds 0.45, **setting $w_{\text{financial}} = 0.0$ changes zero test decisions**. All 13 incremental fraud catches remain 100% intact.
- **Recommended Action**: Set `w_financial: 0.0` in `configs/risk_policy.yaml` and re-score rings to permanently extinguish this attack vector.

### R2 [CRITICAL]: Overstatement of Prevented Merchant Loss
- **Location**: `src/evaluation/business_evaluation.py:L115-116`
- **Audit Findings**:
  - Total reported "Estimated Fraud Value Prevented": **₹7,614.08**
  - Direct Automated Block Value (`BLOCK`): **₹136.91** (3 transactions)
  - Manual Review Triage Value (`MANUAL_REVIEW`): **₹7,477.17** (53 transactions)
  - Direct blocks constitute only **1.8%** of the prevented value; **98.2%** depends on human triage.
- **Auditor Critique**: Stating that ₹7,614.08 was "prevented" implies that sending an alert to a human queue guarantees money is saved. In reality, human analysts have finite SLA windows and triage capture rates (~80–90%).
- **Recommended Action**: Re-label the KPI as **"Estimated Fraud Exposure Intercepted (Triaged)"**. State explicitly: *"₹136.91 directly blocked + ₹7,477.17 triaged for review (@ 85% analyst capture efficiency = ₹6,492.50 net savings)"*.

### R3 [HIGH]: Demo Endpoint Key Mismatch in `/api/demo/cases`
- **Location**: `api/main.py:L531-554`
- **Code**:
  ```python
  case_a_detail.get("amount_inr", 482.10)
  case_a_detail.get("model_d_score", 0.8931)
  case_a_detail.get("sentinel_score", 0.0)
  ```
- **Auditor Critique**: `get_alert_detail()` returns `amount` (not `amount_inr`) and `risk_factors['r_gbm']` (not `model_d_score`). Because the keys did not match, `/api/demo/cases` returned the fallback values (`482.10` and `0.8931`) instead of the true parquet values (`47.73` and `0.7537`).
- **Recommended Action**: Update dictionary access in `api/main.py:get_demo_cases` to read `case_a_detail['amount']` and `case_a_detail['risk_factors']['r_gbm']`.

### R4 [HIGH]: Policy Discrepancy in Single Alert Detail API
- **Location**: `api/main.py:L236-245`
- **Auditor Critique**: When an analyst clicks on `TXN-3004262` (Case B) in `/api/transaction/{id}`, the endpoint uses legacy thresholds against `r_final = 0.1004` and displays `action: "ALLOW"`, contradicting the production hybrid policy where Sentinel escalates it to `MANUAL_REVIEW`.
- **Recommended Action**: Refactor `get_alert_detail()` to evaluate the authoritative hybrid policy rules ($r_{\text{gbm}} \ge 0.50 \to \text{BLOCK}, r_{\text{ring}} \ge 0.45 \to \text{MANUAL\_REVIEW}$).

---

## 4. Prioritized Fix Action Plan

| Priority | Action Item | Target File | Impact on Metrics |
| :---: | :--- | :--- | :---: |
| **1** | Set `w_financial: 0.0` in `configs/risk_policy.yaml` and re-score rings | `configs/risk_policy.yaml`, `src/graph/ring_detector.py` | Zero change to decisions; 100% airtight scientific defensibility. |
| **2** | Separate Direct Blocks vs Review Exposure in economic reporting | `src/evaluation/business_evaluation.py` | Completely eliminates the "prevented vs flagged" economic critique. |
| **3** | Fix dynamic key extraction in `/api/demo/cases` | `api/main.py` | Guarantees Case A amounts and scores match raw test records exactly. |
| **4** | Align `get_alert_detail` decision logic with hybrid policy | `api/main.py` | Eliminates policy contradictions across API endpoints. |
| **5** | Strengthen unit test `test_03` to call `booster.predict()` | `tests/test_business_evaluation.py` | Converts false-confidence test into genuine behavioral inference test. |
