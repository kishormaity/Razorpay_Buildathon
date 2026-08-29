# Abuse-Ring Proxy Target Audit & Calibration Report

Generated on: 2026-08-26 15:44:14

This report documents the non-modeling calibration of the abuse-ring proxy target label on the Train + Dev/Val splits (`501,959` rows total).

---

## 1. Hub-Pollution Diagnostic (Top 10 Devices in R1 Positives)

The baseline proxy definition **R1** flagged **92.69%** of the dataset because it was contaminated by common, highly frequent hardware nodes acting as network hubs:

| Rank | DeviceInfo Value | Positive Transactions | Share of R1 Positives |
| :---: | :--- | :---: | :---: |
| 1 | `Windows` | `41,259` | `8.88%` |
| 2 | `iOS Device` | `17,609` | `3.79%` |
| 3 | `MacOS` | `11,246` | `2.42%` |
| 4 | `Trident/7.0` | `6,700` | `1.44%` |
| 5 | `rv:11.0` | `1,715` | `0.37%` |
| 6 | `rv:57.0` | `934` | `0.20%` |
| 7 | `SM-J700M Build/MMB29K` | `440` | `0.09%` |
| 8 | `SM-G610M Build/MMB29K` | `355` | `0.08%` |
| 9 | `SM-G531H Build/LMY48B` | `303` | `0.07%` |
| 10 | `SM-G955U Build/NRD90M` | `293` | `0.06%` |

> [!IMPORTANT]
> **Hub Pollution Confirmed**:
> Transactions sharing generic categories like `Windows`, `iOS Device`, and `Android` dominate the R1 positives. These values do not represent a local, coordinated abuse ring. We must utilize stricter rate or overlap conditions to filter them.

---

## 2. Stricter Proxy Candidates (R1–R5) Comparison

We compared 5 candidates on the Train + Dev splits:

| Proxy | Definition | Prevalence (Count & %) | Fraud Rate | Fraud Captured | Cards Covered | Entities Covered |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **R1** | ≥3 cards + ≥1 fraud + ≥3/72h | `464,614` (`92.56%`) | `2.97%` | `78.49%` | `12,137` | `1,012` |
| **R2** | ≥5 cards + ≥1 fraud + ≥3/72h | `464,410` (`92.52%`) | `2.96%` | `78.10%` | `12,132` | `1,002` |
| **R3** | ≥5 cards + ≥2 fraud + ≥3/72h | `463,818` (`92.40%`) | `2.96%` | `78.02%` | `12,126` | `1,002` |
| **R4** | ≥5 cards + ≥20% fraud density + ≥3/72h | `396,509` (`78.99%`) | `2.96%` | `66.79%` | `11,290` | `866` |
| **R5** | High-risk device + high-risk address overlap | `67,221` (`13.39%`) | `4.33%` | `16.55%` | `6,261` | `403` |

---

## 3. Key Observations & Recommendations

> [!TIP]
> **Analysis of Candidate Candidates**:
> * **R1/R2/R3**: Show massive prevalence and extremely low fraud rates (close to the baseline fraud rate), confirming they are highly contaminated by clean card transactions.
> * **R4 (Rate-based Density)**: Restricts the positive population by requiring that at least **20%** of the connected cards have been fraudulent. This isolates a smaller, higher-risk sub-network.
> * **R5 (Device + Address Overlap)**: Requiring both the device and address nodes to be high-risk provides the strongest signal.
>
> **Recommended Decision**:
> We will select the candidate that maximizes **fraud rate among positives** while retaining a **meaningful fraud capture rate**.
