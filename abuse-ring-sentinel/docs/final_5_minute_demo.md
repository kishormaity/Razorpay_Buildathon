# The 5-Minute Skeptical-Judge Demonstration Script
## Project: `abuse-ring-sentinel`
### Live Demo Strategy: Step-by-Step Defense, Verification, and Walkthrough

---

## 1. Demonstration Overview & Judge Profile

- **Target Audience**: Skeptical Senior Machine Learning / Fraud Risk Competition Judge.
- **Goal**: In under 5 minutes, prove that `abuse-ring-sentinel` solves an authentic single-transaction blindspot through graph community intelligence on a locked chronological test set, backed by real TreeSHAP and tiered economics.
- **Prerequisites**:
  - Backend running: `uvicorn api.main:app --port 8001`
  - Frontend running: `npm run dev` (on port 3000)

---

## 2. Step-by-Step 5-Minute Script

```
+---------------------------------------------------------------------------------------------------------------+
|                                            5-MINUTE JUDGE DEMO PATH                                            |
+-----+----------+----------------------+--------------------+--------------------------------------------------+
| Min | Page     | Action               | Expected Result    | Defensible Talking Point                         |
+-----+----------+----------------------+--------------------+--------------------------------------------------+
| 0:00| Overview | Open Dashboard       | UI loads with Live | "Individual GBDT models evaluate transactions in |
|  -  | /        | (localhost:3000/     | Banner and Locked  | isolation. Fraud syndicates bypass this using     |
| 0:45|          | dashboard)           | Test Set Badge     | micro-amounts and synthetic identities."         |
+-----+----------+----------------------+--------------------+--------------------------------------------------+
| 0:45| Why      | Scroll to "Why       | 3-column card      | "Model D sees 37 tabular features and misses 53  |
|  -  | Sentinel?| Sentinel?" Card      | comparing tabular  | test frauds. Sentinel projects hardware sharing  |
| 1:30|          |                      | vs graph signals   | to intercept 13 of those missed frauds."         |
+-----+----------+----------------------+--------------------+--------------------------------------------------+
| 1:30| Demo     | Click "Case B:       | Card highlights;   | "Here is TXN-3004262 (₹85.49). Model D scores it |
|  -  | Panel    | Coordinated Abuse    | Case B details and | 0.037 (ALLOW). But Sentinel flags Community 10   |
| 2:45|          | (Sentinel Caught)"   | evidence expand    | at 0.491, escalating it to MANUAL_REVIEW."       |
+-----+----------+----------------------+--------------------+--------------------------------------------------+
| 2:45| Explana- | Click "Inspect Tree- | Native TreeSHAP &  | "SHAP shows tabular features look benign. But    |
|  -  | bility   | SHAP & Evidence"     | multi-hop shared   | graph evidence reveals it shares hardware        |
| 3:30|          |                      | DEV-29295 nodes    | DEV-29295 with confirmed fraud node CUS-15885."  |
+-----+----------+----------------------+--------------------+--------------------------------------------------+
| 3:30| Merchant | Scroll to "Merchant  | Side-by-side Model | "On our locked 15% chronological test set, recall|
|  -  | Impact   | Loss Prevention"     | D vs Hybrid table; | rises from 44.79% to 58.33% with 0 extra blocks  |
| 4:15|          |                      | Operating selector | at 22.3 reviews per incremental fraud."          |
+-----+----------+----------------------+--------------------+--------------------------------------------------+
| 4:15| Conclu-  | Click "Balanced" vs  | Trade-off metrics  | "Thresholds were frozen on validation data only. |
|  -  | sion     | "Conservative"       | update; summary    | Backed by 17 automated tests, Sentinel delivers  |
| 5:00|          | operating points     | confirms lift      | measured, commercially honest loss prevention."  |
+-----+----------+----------------------+--------------------+--------------------------------------------------+
```

---

## 3. Detailed Minute-by-Minute Execution

### Minute 0:00 – 0:45: The Problem & The Baseline
- **Screen**: `http://localhost:3000/dashboard`
- **Action**: Direct the judge's attention to the top banner: *"Locked Chronological Test Set Benchmark (15% Split · 3,003 Transactions · 96 Confirmed Frauds)"*.
- **Judge Narrative**:
  > *"When an enterprise payment system relies solely on single-transaction anomaly detection (Model D), coordinated abuse syndicates easily slip through. Fraudsters run distributed card-testing attacks using micro-ticket amounts and distinct cards across multiple synthetic user accounts. Each transaction looks isolated and benign. On our locked chronological test set, Model D missed 53 confirmed fraud cases."*

### Minute 0:45 – 1:30: The Architecture ("Why Sentinel?")
- **Screen**: "Why Sentinel?" Architectural Card.
- **Action**: Highlight the contrast between the Tabular GBDT column and the Graph Modularity column.
- **Judge Narrative**:
  > *"Model D evaluates 37 stationary tabular features—velocity, spend ratios, and time gaps. Abuse-Ring Sentinel adds an asynchronous verification layer: it projects an entity-sharing bipartite graph linking users, devices, and IP subnets, applying Leiden community modularity clustering. It detects coordinated rings before money leaves the merchant."*

### Minute 1:30 – 2:45: The Incremental Proof (Case B)
- **Screen**: Interactive 3-Case Demo Panel.
- **Action**: Click the middle tab: **`Case B: Coordinated Abuse Intercepted by Sentinel`**.
- **Observation**:
  - Transaction ID: `TXN-3004262`
  - Amount: `₹85.49`
  - Model D Score: `0.0366` $\implies$ `ALLOW` (Missed Fraud)
  - Sentinel Risk: `0.4911` $\implies$ `MANUAL_REVIEW` (Intercepted)
- **Judge Narrative**:
  > *"This is a confirmed fraud transaction in our locked test set. Because the amount was only ₹85.49 and card velocity was low, Model D predicted a 0.037 risk score and routed it to ALLOW. However, Sentinel mapped the user (CUS-16746) to Community 10 with a risk score of 0.491. Under our frozen production policy, Sentinel intercepted this fraud by routing it to manual review."*

### Minute 2:45 – 3:30: Authentic Explainability (TreeSHAP & Graph Evidence)
- **Screen**: Expand the explanation panel for Case B.
- **Observation**:
  - TreeSHAP attributions: `network_risk_product (+0.693)`, `device_connected_fraud_rate (+0.501)`.
  - Network Evidence: Shares hardware device `DEV-29295` with known flagged accounts `CUS-15885` and `CUS-5583`.
- **Judge Narrative**:
  > *"Our explainability is completely authentic: TreeSHAP is generated directly by LightGBM's C++ engine without any ground-truth branching. The network evidence extracts true multi-hop neighbors directly from the relational graph. The fraudster could disguise their transaction amount, but they could not hide their shared hardware footprint."*

### Minute 3:30 – 4:15: Honest Merchant Economic Impact
- **Screen**: Merchant Loss Prevention & Performance Ladder section.
- **Action**: Point out the performance comparison table.
- **Observation**:
  - Model D Recall: `44.79%` (43/96)
  - Hybrid Recall: `58.33%` (56/96) $\implies \mathbf{+13.54\%\ lift}$
  - Incremental Fraud Caught: `13 cases` ($24.53\%$ of missed fraud)
  - Intercepted Fraud Exposure: `₹7,614.08`
  - False Blocks: `3` (Model D) vs `3` (Hybrid) $\implies \mathbf{0\ additional\ false\ blocks}$
  - Reviews per Incremental Fraud: `22.3`
- **Judge Narrative**:
  > *"Sentinel does not blindly block customers. It routes coordinated risk to manual review or step-up authentication. Across the 3,003 test transactions, Sentinel caused zero additional customer false blocks. It required 290 additional reviews to catch 13 additional frauds—an exchange rate of 22.3 reviews per fraud caught. All cost parameters are explicitly marked as operational assumptions."*

### Minute 4:15 – 5:00: Validation-Only Tuning & Conclusion
- **Screen**: Operating Point Selector (Balanced vs Conservative vs High Recall).
- **Action**: Toggle between "Balanced (Production Champion)" and "High Precision".
- **Judge Narrative**:
  > *"Critically, all thresholds—Model D block at 0.50, review at 0.05, and Sentinel at 0.45—were optimized strictly on the 15% validation partition and frozen before evaluating test data. Backed by 17 passing automated unit tests and a zero-error Next.js production build, Abuse-Ring Sentinel delivers measured, reproducible, and commercially honest loss prevention."*

---

## 4. Rapid Fallback & Recovery Matrix

| Potential Issue | Immediate Fallback Action during Demo |
| :--- | :--- |
| **API Server not running on Port 8001** | Run in terminal: `python -m api.main` (or run automated tests: `python -m unittest discover -s tests -p "test_*.py"` to prove all 17 pass in 4s). |
| **Frontend dev server port conflict** | Run: `npm run start` in `frontend/` (pre-compiled Next.js production server on port 3000). |
| **Judge asks to verify raw data offline** | Open `data/processed/evaluation/sentinel_incremental_cases.csv` and show all 13 rows with exact transaction IDs and scores. |
| **Judge questions TreeSHAP authenticity** | Open `src/explainability/shap.py` and point directly to line 26: `self.model.predict(X, pred_contrib=True)`. |
| **Judge asks if test labels were tuned** | Run `python -m unittest tests.test_business_evaluation.TestPhaseP2CompetitionHardening.test_02_no_test_label_threshold_optimization`. |
