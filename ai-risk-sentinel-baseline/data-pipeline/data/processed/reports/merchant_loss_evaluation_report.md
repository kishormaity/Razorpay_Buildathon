# AI Risk Manager — Merchant Loss Prevention Evaluation Report

**Evaluation Split**: Locked Chronological Test Set (Last 15% of dataset; rows 501,959 to 590,540)  
**Operating Thresholds**: Block $\ge 0.15000$ | Review $\ge 0.08000$  
**Cost Model**: $\text{Friction Penalty} = 15\% \times \text{False Positive Transaction Value}$

---

## 1. Executive Summary: Merchant Financial Impact

| Metric | Measured Value | Business Interpretation |
| :--- | :---: | :--- |
| **Total Test Transactions** | `88,581` | Full chronological traffic volume evaluated |
| **Total Transaction Value** | `$12,148,754.19` | Total merchant checkout gross volume |
| **Total Fraud Volume** | `$469,608.52` | Baseline fraud loss if no risk engine intervened |
| **Fraud Loss Prevented** | **`$218,964.73`** | **Direct fraud checkout value blocked by Model D** |
| **Fraud Loss Missed** | `$0.00` | Fraud value auto-approved at checkout |
| **Fraud Value in Secondary Review** | `$250,643.79` | Fraud value intercepted by Abuse-Ring Sentinel |
| **False-Positive Transaction Value** | `$493,809.36` | Legitimate purchase volume blocked |
| **Estimated Friction Cost (15%)** | `$74,071.40` | Quantified merchant friction & lost lifetime value |
| **Estimated Net Loss Avoided** | **`$144,893.32`** | **Prevented Loss minus False-Positive Friction Cost** |

---

## 2. Model & Policy Operational Metrics

| Metric | Value |
| :--- | :---: |
| **Fraud Recall** | `56.57%` |
| **Fraud Value Capture Rate** | `46.63%` |
| **Precision** | `40.95%` |
| **False Positive Rate (FPR)** | `2.94%` |
| **F1-Score** | `0.4751` |
| **PR-AUC (Model D)** | `0.5122` |
| **ROC-AUC (Model D)** | `0.8892` |
| **Sentinel Intercepted Fraud Count** | `1,339 transactions` |
