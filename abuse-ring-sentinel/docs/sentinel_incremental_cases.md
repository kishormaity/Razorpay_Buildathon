# Forensic Report: The 13 Incremental Fraud Cases
## Investigating Fraud Missed by Model D Alone and Intercepted by Abuse-Ring Sentinel
### Project: `abuse-ring-sentinel` | Split: Locked Chronological Test Partition

---

## 1. Executive Summary

A central proof of the **Abuse-Ring Sentinel** architecture is its empirical ability to intercept coordinated payment fraud that bypasses traditional transaction-level machine learning models. 

On the locked chronological test set ($n = 3,003$ transactions), **Model D alone missed 53 fraud transactions** because each individual transaction appeared legitimate in isolation. **Abuse-Ring Sentinel intercepted 13 of those 53 missed frauds**, boosting detection recall from **44.79% to 58.33%** (+13.54 percentage points, a **24.53% incremental capture rate**).

This document provides a transaction-by-transaction forensic breakdown of all 13 intercepted cases, answering three critical questions:
1. **Why did Model D miss this transaction?**
2. **Why did Sentinel catch it?**
3. **What empirical network evidence existed?**

---

## 2. Master Table of 13 Intercepted Transactions

| # | Transaction ID | Amount | User ID | Model D Risk ($r_{\text{gbm}}$) | Sentinel Ring Risk ($r_{\text{ring}}$) | Shared Device | Linked Flagged Accounts | Top Risk Features (TreeSHAP) |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `TXN-3004262` | ₹85.49 | `CUS-16746` | 0.0366 | 0.4911 | `DEV-29295` | `CUS-15885`, `CUS-13832`, `CUS-5583` | `network_risk_product` (0.69), `device_fraud_rate` (0.50) |
| 2 | `TXN-3004645` | ₹38.67 | `CUS-10876` | 0.0345 | 0.4911 | `DEV-29295` | `CUS-15885`, `CUS-13832`, `CUS-5583` | `network_risk_product` (1.12), `device_fraud_rate` (0.29) |
| 3 | `TXN-3004648` | ₹38.67 | `CUS-2256` | 0.0261 | 0.4911 | `DEV-29295` | `CUS-15885`, `CUS-13832`, `CUS-5583` | `network_risk_product` (0.86), `device_fraud_rate` (0.30) |
| 4 | `TXN-3004720` | ₹47.73 | `CUS-4504` | 0.0167 | 0.4911 | `DEV-274` | `CUS-15885`, `CUS-4461`, `CUS-16578` | `device_fraud_rate` (0.45), `card1` (0.33) |
| 5 | `TXN-3005147` | ₹123.44 | `CUS-4329` | 0.0223 | 0.4911 | `DEV-24744` | `CUS-15885`, `CUS-16062`, `CUS-9917` | `network_risk_product` (0.51), `TransactionAmt` (0.33) |
| 6 | `TXN-3005411` | ₹12.57 | `CUS-17942` | 0.0104 | 0.4911 | `DEV-274` | `CUS-15885`, `CUS-4461`, `CUS-16578` | `network_risk_product` (0.66), `device_fraud_rate` (0.40) |
| 7 | `TXN-3005416` | ₹12.57 | `CUS-17942` | 0.0479 | 0.4911 | `DEV-274` | `CUS-15885`, `CUS-4461`, `CUS-16578` | `network_risk_product` (1.24), `card_time_since_prev` (0.87) |
| 8 | `TXN-3005600` | ₹39.00 | `CUS-12695` | 0.0067 | 0.4911 | `DEV-274` | `CUS-15885`, `CUS-4461`, `CUS-16578` | `addr_card_degree` (0.31), `P_emaildomain` (0.21) |
| 9 | `TXN-3005623` | ₹78.79 | `CUS-7949` | 0.0314 | 0.4911 | `DEV-274` | `CUS-15885`, `CUS-4461`, `CUS-16578` | `network_risk_product` (0.46), `addr1` (0.26) |
| 10 | `TXN-3005661` | ₹39.00 | `CUS-12695` | 0.0047 | 0.4911 | `DEV-274` | `CUS-15885`, `CUS-4461`, `CUS-16578` | `addr_card_degree` (0.16), `card_addr_degree` (0.10) |
| 11 | `TXN-3005685` | ₹22.05 | `CUS-15885` | 0.0092 | 0.4911 | `DEV-29295` | `CUS-5583`, `CUS-8755`, `CUS-13832` | `network_risk_product` (0.58), `device_fraud_rate` (0.25) |
| 12 | `TXN-3005885` | ₹80.00 | `CUS-8695` | 0.0092 | 0.4911 | `DEV-518` | `CUS-3821`, `CUS-10486`, `CUS-6019` | `card_device_degree` (0.25), `card_addr_degree` (0.20) |
| 13 | `TXN-3006017` | ₹78.79 | `CUS-2256` | 0.0276 | 0.4911 | `DEV-29295` | `CUS-15885`, `CUS-13832`, `CUS-5583` | `device_fraud_rate` (0.31), `network_risk_product` (0.27) |

---

## 3. The Core Mechanism: Why Model D Missed vs. Why Sentinel Caught

### Why Did Model D Miss These Transactions?
1. **Micro-Amounts Below Alert Thresholds**: The ticket sizes ranged from ₹12.57 to ₹123.44 (mean ₹53.60). Standard tabular gradient boosting models learn that low ticket values correlate with everyday low-risk retail transactions.
2. **Normal Individual Card Velocity**: In all 13 transactions, `card_tx_count_10m = 0` and `card_tx_count_1h = 0`. Fraudsters deliberately spread transactions over time across different synthetic payment cards to avoid velocity triggers.
3. **Valid Credentials**: Clean email domains and billing addresses prevented rule-based blocks. Model D produced low predicted risk scores ($r_{\text{gbm}} \in [0.0047, 0.0479]$), far below the review threshold of $\tau_D = 0.05$. Under Model D alone, all 13 were routed to **ALLOW**.

### Why Did Sentinel Catch Them?
1. **Device Farm Clustering**: Although the cards and user accounts were ostensibly different, Sentinel's heterogeneous bipartite graph detected that the accounts were physically executing from shared hardware fingerprints (`DEV-29295`, `DEV-274`, `DEV-24744`, `DEV-518`).
2. **Graph Modularity Partitioning (Leiden Algorithm)**: Sentinel grouped these accounts into **Community 10**. The structural density, cross-entity reuse, and historical fraud rate of Community 10 resulted in a composite ring score of $s_t = 0.491111$.
3. **Operational Policy Escalation**: Under the frozen production policy, any transaction with $s_t \ge 0.45$ is escalated to **MANUAL_REVIEW**, preventing immediate settlement and catching the attack.

---

## 4. In-Depth Case Studies

### Case Study A: The Rapid Multi-Account Device Replay (`DEV-29295`)
- **Transactions**: `TXN-3004262` (₹85.49), `TXN-3004645` (₹38.67), `TXN-3004648` (₹38.67), `TXN-3005685` (₹22.05), `TXN-3006017` (₹78.79).
- **Accounts Involved**: `CUS-16746`, `CUS-10876`, `CUS-2256`, `CUS-15885`.
- **Forensics**:
  - `TXN-3004645` and `TXN-3004648` were executed almost simultaneously with the exact same amount (₹38.669) by two different user IDs (`CUS-10876` and `CUS-2256`) using different cards.
  - Model D scored them as $0.034$ and $0.026$ (safe).
  - Graph inspection revealed both users were active on device `DEV-29295`, which was previously associated with confirmed abuse account `CUS-15885`.
  - Sentinel's `device_connected_fraud_rate` spiked to `0.086`, and the community risk score triggered an immediate manual review hold.

### Case Study B: The Micro-Velocity Test (`DEV-274`)
- **Transactions**: `TXN-3005411` (₹12.57) and `TXN-3005416` (₹12.57).
- **Account Involved**: `CUS-17942`.
- **Forensics**:
  - User executed two back-to-back ₹12.57 card testing charges within seconds.
  - Model D gave scores of $0.010$ and $0.047$, failing to trigger the $0.05$ threshold.
  - The device `DEV-274` was connected to known fraud accounts `CUS-4461` and `CUS-16578`.
  - Sentinel caught both, preventing the card testing attack from scaling into high-value unauthorized purchases.

---

## 5. Artifact Reference
The complete dataset with all 21 feature columns is available at:
`data/processed/evaluation/sentinel_incremental_cases.csv`
