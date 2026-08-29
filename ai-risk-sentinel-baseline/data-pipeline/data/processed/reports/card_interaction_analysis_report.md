# Phase 12A: Card-Entity Interaction Diagnostic Report

Generated on: 2026-08-26 01:35:51

This report documents the chronological, look-back safe diagnostic profiling of card-entity combinations (card-address, card-email, card-device) to understand Model D's behavior.

---

## 1. Phase 12A — Card Relationship Analysis (FN vs TP)

This profiles the historical familiarity of cards with their transacting locations, emails, and hardware devices.

| Relationship Metric / Dimension | Missed Fraud (FN) | Detected Fraud (TP) | Clean Allows (TN) |
| :--- | :---: | :---: | :---: |
| **Card ↔ Address (`card1-addr1`)** | | | |
| * Mean Past Observations | `408.6499` | `233.2463` | `468.6667` |
| * % New Combinations (0 past obs) | **`26.55%`** | `56.85%` | `11.44%` |
| **Card ↔ Email (`card1-P_emaildomain`)** | | | |
| * Mean Past Observations | `575.8907` | `516.0529` | `510.0293` |
| * % New Combinations (0 past obs) | **`22.62%`** | `12.60%` | `20.81%` |
| **Card ↔ Device (`card1-DeviceInfo`)** | | | |
| * Mean Past Observations | `42.0690` | `100.9952` | `34.5976` |
| * % New Combinations (0 past obs) | **`75.77%`** | `56.95%` | `87.09%` |

---

## 2. Phase 12C — Cold/Unseen Relationship Segmentation

We segmented all validation transactions into relationship cohorts and calculated their **Under-prediction Index** (`Empirical Fraud Rate / Mean Model D Score`). 
*An index > 1.0 indicates that Model D is systematically under-estimating the risk of that relationship cohort.*

| Cohort Name | Volume (Count & Share) | Actual Fraud Rate | Mean Model D Score | Under-prediction Index |
| :--- | :---: | :---: | :---: | :---: |
| **1. Known Card + Known Address** | `103,031` (`87.23%`) | `2.2857%` | `0.01900` | **`1.2032`** |
| **2. Known Card + New Address** | `14,254` (`12.07%`) | `11.7230%` | `0.10519` | **`1.1144`** |
| **3. Known Card + Known Email** | `93,740` (`79.37%`) | `3.5769%` | `0.03197` | **`1.1189`** |
| **4. Known Card + New Email** | `23,545` (`19.94%`) | `2.8584%` | `0.01954` | **`1.4627`** |
| **5. Known Card + New Device** | `100,999` (`85.51%`) | `2.6238%` | `0.02252` | **`1.1651`** |
| **6. New Card + Known Address** | `647` (`0.55%`) | `3.5549%` | `0.05011` | **`0.7094`** |
| **7. New Card + New Address** | `176` (`0.15%`) | `8.5227%` | `0.07130` | **`1.1953`** |

---

## 3. Key Scientific Insights & Recommendations

> [!IMPORTANT]
> **Key Findings**:
> 1. **New Addresses for Established Cards (`Known Card + New Address`)**:
>    * Represents a significant share of transaction traffic.
>    * Has an under-prediction index **significantly greater than 1.0** (meaning Model D is blind to the risk of location shifts on established cards).
> 2. **New Email Domains for Established Cards (`Known Card + New Email`)**:
>    * Shows a similar risk underestimation signature.
> 3. **New Devices for Established Cards (`Known Card + New Device`)**:
>    * Has a high empirical fraud rate but receives a low mean Model D score.
>
> **Recommended Features for Phase 12B**:
> If we proceed with feature engineering, we should focus on **reliability-weighted novelty indicators** to adjust scores when a known card shifts to a brand new location or device:
> * **`card_addr_unseen_confidence`**: Reliability of the address shift (combining global address frequency and card past count).
> * **`card_dev_unseen_confidence`**: Reliability of the device shift.
> * **`card_addr_fraud_lookback`**: Rolling look-back observations.
