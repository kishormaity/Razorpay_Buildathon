# Phase 10: Model D Second-Generation Error Analysis Report

Generated on: 2026-08-26 01:09:00

This report documents the deep profiling of **Model D**'s validation errors (1,985 False Negatives and 1,004 False Positives) on the 80/20 split to identify systematic failure modes.

---

## 1. Validation Predictions Summary

* **Optimal Model D Decision Threshold**: `0.30398` (Optimal F1: `0.58178`)
* **Total Validation Set Size**: `118,108`
* **Detected Fraud (True Positives)**: `2,079`
* **Missed Fraud (False Negatives)**: `1,985`
* **False Alarms (False Positives)**: `1,004`
* **Correct Allows (True Negatives)**: `113,040`

---

## 2. Deep Profiling by Fraud Detection Status

The table compares behavioral statistics of detected fraud (TP) vs missed fraud (FN).

| Feature Attribute / Metric | Detected Fraud (TP) | Missed Fraud (FN) | True Negatives (TN) | False Positives (FP) |
| :--- | :---: | :---: | :---: | :---: |
| **Transaction Count (`card1_past_count`)** | `1962.2391` | **`2568.0292`** | `2244.2835` | `2656.5637` |
| **Cold-Start Card Rate (Past Count <= 1)** | `1.83%` | **`1.26%`** | N/A | N/A |
| **Average Transaction Amount ($)** | `$124.76` | **`$176.60`** | `$136.97` | `$150.06` |
| **Ratio to Card Avg (`amount_vs_card_mean`)** | `1.2032` | **`1.2996`** | N/A | N/A |
| **Device Missingness Rate** | `46.37%` | **`70.58%`** | N/A | `62.15%` |
| **Double Missing (Device & Email)** | `7.07%` | **`15.92%`** | N/A | N/A |
| **Device Card Novelty** | `9.52%` | **`4.89%`** | N/A | N/A |
| **Address Card Novelty** | `1.92%` | **`1.66%`** | N/A | N/A |
| **Device Connected Fraud Rate** | `0.04792` | **`0.04224`** | N/A | N/A |
| **Address Connected Fraud Rate** | `0.03204` | **`0.02947`** | N/A | N/A |
| **Email Domain Mismatch Rate** | `2.74%` | **`2.92%`** | N/A | N/A |

---

## 3. Missed Fraud Failure Archetypes

We classified the `1,985` missed fraud cases into the following failure categories:

### Archetype 1: Cold-Start Fraud
* **Volume**: `25` cases (**`1.26%`** of missed fraud)
* **Definition**: Card has no prior transaction history (past count <= 1).
* **Why Model D Missed**: Model D relies heavily on historical card risk profiles. Without history, it defaults to a low-risk prediction.

### Archetype 2: Telemetry Blindspots
* **Volume**: `316` cases (**`15.92%`** of missed fraud)
* **Definition**: Both hardware details (`DeviceInfo`) and email domains are missing.
* **Why Model D Missed**: Without device fingerprints or email domains, the model has no telemetry hooks to correlate.

### Archetype 3: Device/Address Novelty
* **Volume**: `97` cases (**`4.89%`** of missed fraud)
* **Definition**: Card has prior transactions, but is transacting from a brand new device or address.
* **Why Model D Missed**: Model D cannot contextualize location or device shifts for a card in isolation.

### Archetype 4: Network Connected Risk (Risk Propagation)
* **Volume**: `60` cases (**`3.02%`** of missed fraud)
* **Definition**: Clean or new card transacting from a device or address location that has been used by other fraudulent cards.
* **Why Model D Missed**: Model D is card-focused and blind to device/address network links.

### Archetype 5: High-Value Outliers
* **Volume**: `175` cases (**`8.82%`** of missed fraud)
* **Definition**: Transaction amount is more than 3x the card's typical historic average.
* **Why Model D Missed**: Tree models do not extrapolate continuous numerical outliers well unless explicitly represented.

---

## 4. False Alarms (False Positives) Analysis

Model D made `1,004` false alarms. The profiles show:
* **High Amounts**: Average FP amount is `$150.06`, showing a strong bias towards flagging large transactions.
* **Device sharing**: FP cards share devices with an average of `887.02` other cards, showing that device pooling triggers false alarms on normal cards.
* **Address sharing**: FP cards share addresses with an average of `674.04` other cards.

---

## 5. Highly Motivated Feature Candidates for Phase 10

Based on this empirical profiling, here are the **4 most motivated feature candidates** to target Model D's failure modes:

1. **`is_cold_start_high_risk`** (Targets Archetype 1 & 5):
   * *Formula*: `(card1_past_count <= 1) * log1p(TransactionAmt)`
   * *Motivation*: Highlights new cards transacting high values, preventing the default low-risk classification.
2. **`telemetry_blindspot_severity`** (Targets Archetype 2):
   * *Formula*: `is_device_missing * is_email_missing * log1p(TransactionAmt)`
   * *Motivation*: penalizes transactions that lack telemetry footprint when transaction sizes increase.
3. **`network_risk_weight`** (Targets Archetype 4):
   * *Formula*: `log1p(card1_past_count) * device_connected_fraud_rate`
   * *Motivation*: Down-weights card history if the card connects to a device that has a high fraud rate.
4. **`novelty_risk_acceleration`** (Targets Archetype 3):
   * *Formula*: `(device_card_novelty + addr_card_novelty) * log1p(amount_vs_card_mean)`
   * *Motivation*: Flags transactions that represent a location/device shift combined with a transaction amount outlier.
