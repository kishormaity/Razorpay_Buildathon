# Abuse-Ring Sentinel: Standalone Routing Validation Report

Generated on: 2026-08-26 18:09:48

This report documents the standalone operational evaluation of the **Abuse-Ring Sentinel** as a secondary review routing layer (fixed decision threshold = **`0.15`**).

---

## 1. Split Configurations (70/15/15)
* **Train split** (0% to 70%): `413,378` transactions
* **Dev/Val split** (70% to 85%): `88,581` transactions
* **Final Test split** (85% to 100%): `88,581` transactions (Strictly locked during tuning)

---

## 2. Operational Evaluation Table (Threshold = 0.15)

| Operational Metric / Dimension | Dev/Val Split | Final Test Split (Locked) |
| :--- | :---: | :---: |
| **Review Volume (Flagged Transactions)** | `6,433` | `8,773` |
| **Review Population Share (%)** | `7.26%` | `9.90%` |
| **Fraud Count in Reviewed** | `355` | `426` |
| **Fraud Rate in Reviewed (%)** | `5.52%` | `4.86%` |
| **Model D FNs (Missed Fraud)** | `1,329` | `1,647` |
| **Sentinel-Captured FNs** | `113` | `201` |
| **FN Capture Rate (%)** | `8.50%` | `12.20%` |
| **FPs (Friction) among Reviewed** | `6,078` | `8,347` |
| **Precision of Review (%)** | `5.52%` | `4.86%` |
| **Fraud Concentration (%)** | `11.67%` | `13.82%` |
| **Unique Cards Covered** | `1,585` | `1,739` |
| **Unique Devices Covered** | `129` | `146` |
| **Unique Addresses Covered** | `63` | `63` |
| **FN Capture Efficiency** | **`1.1708`** | **`1.2322`** |

---

## 3. Key Findings & Rationale

> [!IMPORTANT]
> **FN Capture Efficiency Interpretation**:
> * **Dev Split Efficiency**: **`1.17`**
> * **Test Split Efficiency**: **`1.23`**
> * *Interpretation*: An efficiency of **`1.23`** on the locked test set indicates that the Sentinel is preferentially routing Model D's missed fraud for review compared to a random baseline selection (efficiency = 1.0).

---

## 4. Production Security Workflow

Rather than blending raw probabilities, the finalized security topology routes transaction flows as follows:

```text
                  Incoming Transaction
                           │
                           ▼
                    [ Model D GBDT ]
                     Threshold 0.30
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
          [ Block ]                   [ Allow ]
       (score >= 0.30)            (score < 0.30)
                                         │
                                         ▼
                               [ Sentinel Check ]
                                 Threshold 0.15
                                         │
                           ┌─────────────┴─────────────┐
                           ▼                           ▼
                       [ Review ]                  [ Approve ]
                    (score >= 0.15)             (score < 0.15)
```
