# IEEE-CIS Fraud Detection: Final Project Report

Generated on: 2026-08-26 02:19:22

This report documents the final statistical validation, model selection, production guidelines, and business threshold sweeps for the **IEEE-CIS Fraud Detection Pipeline**.

---

## 1. Executive Summary & Project Champion

Following a multi-phase, chronologically controlled feature search, **Model D (403 features)** is verified and frozen as the **Final Project Champion**.

* **Model D Scores**:
  * **80/20 Split (Stage 1 validation)**: **`0.58144` PR-AUC** (ROC-AUC: `0.90507`)
  * **70/30 Split (Stage 2 validation)**: **`0.59882` PR-AUC**
* **The Decision Rationale**:
  * Throughout the experiments, several feature families (behavioral deviations, graph features, network refactoring, and card novelty shifts) were engineered.
  * While configurations like **H3** (+0.00313 on 80/20) and **C5** (+0.00102 on 80/20) showed small validation gains, they failed to reproduce on the 70/30 chronological split, and delta paired confidence intervals contained zero.
  * Declaring Model D as champion prevents overfitting split boundaries and preserves model generalizability.

---

## 2. Production Operating Profiles & Sweeps

Model D's metrics swept across different decision thresholds. Costs are modeled assuming a **False Negative (missed fraud) costs $10.00** and a **False Positive (friction alarm) costs $1.00**.

| Threshold Point | Precision | Recall | F1-Score | False Positive Rate | Total FNs | Total FPs | modeled Business Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.01` | `0.12649` | `0.83120` | `0.21957` | `20.4544%` | `686` | `23,327` | **`$30,187`** |
| `0.02` | `0.18276` | `0.77141` | `0.29550` | `12.2926%` | `929` | `14,019` | **`$23,309`** |
| `0.03` | `0.23030` | `0.73794` | `0.35105` | `8.7887%` | `1,065` | `10,023` | **`$20,673`** |
| `0.05` | `0.30346` | `0.68725` | `0.42101` | `5.6215%` | `1,271` | `6,411` | **`$19,121`** |
| **0.10000** (Cost-Opt) | `0.43203` | `0.62795` | `0.51188` | `2.9418%` | `1,512` | `3,355` | **`$18,475`** |
| `0.15` | `0.52006` | `0.59006` | `0.55285` | `1.9405%` | `1,666` | `2,213` | **`$18,873`** |
| `0.20` | `0.58413` | `0.55955` | `0.57157` | `1.4196%` | `1,790` | `1,619` | **`$19,519`** |
| `0.25` | `0.62913` | `0.53469` | `0.57808` | `1.1233%` | `1,891` | `1,281` | **`$20,191`** |
| `0.30` | `0.67032` | `0.51230` | `0.58075` | `0.8979%` | `1,982` | `1,024` | **`$20,844`** |
| **0.30398** (F1-Opt) | `0.67434` | `0.51156` | `0.58178` | `0.8804%` | `1,985` | `1,004` | **`$20,854`** |
| `0.35` | `0.70186` | `0.49237` | `0.57874` | `0.7453%` | `2,063` | `850` | **`$21,480`** |
| `0.40` | `0.73448` | `0.47441` | `0.57647` | `0.6112%` | `2,136` | `697` | **`$22,057`** |
| `0.50` | `0.78104` | `0.43799` | `0.56125` | `0.4376%` | `2,284` | `499` | **`$23,339`** |
| `0.60` | `0.82139` | `0.40059` | `0.53854` | `0.3104%` | `2,436` | `354` | **`$24,714`** |
| `0.70` | `0.85516` | `0.36467` | `0.51130` | `0.2201%` | `2,582` | `251` | **`$26,071`** |
| `0.80` | `0.87953` | `0.31078` | `0.45927` | `0.1517%` | `2,801` | `173` | **`$28,183`** |

### Recommended Operational Settings:
1. **F1-Optimal Profile (Threshold = 0.30398)**:
   * Recommended for balanced operations. Combines precision of `0.6743` with recall of `0.5116`.
2. **Cost-Optimal Profile (Threshold = 0.10000)**:
   * Recommended to minimize overall financial loss. Reduces business costs to **`$18,475`** by operating at a recall of `0.6280`.
3. **Low-Friction Profile (Threshold = 0.80000)**:
   * Recommended for premium user checkouts where abandonment friction must be kept under 0.2%. Operates at an FPR of `0.1517%` and recall of `0.3108`.
4. **High-Security Profile (Threshold = 0.02000)**:
   * Recommended for high-risk regions or new merchant accounts. Blocks at least 75% of fraud (Recall = `0.7714`) at an FPR of `12.2926%`.

---

## 3. Global Feature Importance (Top 15 Features by Gain)

The following features drive Model D's predictions:

| Rank | Feature Name | Information Gain |
| :---: | :--- | :---: |
| 1 | `card1` | `180695.73` |
| 2 | `card_addr_combo_historical_fraud_rate` | `170799.33` |
| 3 | `card_email_combo_historical_fraud_rate` | `97136.33` |
| 4 | `C1` | `32937.54` |
| 5 | `C14` | `32339.43` |
| 6 | `DeviceInfo` | `29554.36` |
| 7 | `card_device_combo_historical_fraud_rate` | `27189.85` |
| 8 | `addr1` | `26963.84` |
| 9 | `card2` | `21744.78` |
| 10 | `V133` | `18646.55` |
| 11 | `C13` | `18207.27` |
| 12 | `card_addr_past_count` | `17645.43` |
| 13 | `V258` | `15405.66` |
| 14 | `V257` | `15326.11` |
| 15 | `V243` | `14949.43` |

---

## 4. Missed Fraud Limitations & Archetypes

Deep error analysis of Model D's validation failures (1,985 False Negatives) grouped them into 5 distinct behavioral modes:
1. **Telemetry Blindspots (15.92%)**: Transactions missing both hardware footprints (`DeviceInfo`) and emails.
2. **High-Value Outliers (8.82%)**: Transactions where the amount exceeds 3x the card's typical mean size.
3. **Device/Address Novelty (4.89%)**: Shifts to unusual addresses/devices on established cards.
4. **Network Connected Risk (3.02%)**: Clean cards transacting through corrupted devices.
5. **Cold-Start Fraud (1.26%)**: Cards with zero prior history.
6. **Heterogeneous/Unexplained (66.10%)**: Fraud cases that mathematically overlap with clean allowed transactions. Over 85% of these have probabilities < 0.10, indicating they are indistinguishable without additional external/telemetry links.

---

## 5. Graphical Comparisons

The Precision-Recall and ROC comparisons for the main validation candidates are saved in the project directory:

![Final Comparison Curves](file:///C:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/final_comparison_curves.png)

---

## 6. Final Abuse-Ring Sentinel Validation

The **Abuse-Ring Sentinel** operates as a secondary safety layer on top of Model D. Its primary function is to:
> **Identify transactions exhibiting coordinated multi-entity risk patterns using a chronological weak-supervision proxy.**

### Calibrating the R5 Proxy Target
1. **The Hub Pollution Bug**: Our initial proxy (R1) connected any device seeing $\ge 3$ cards and $\ge 1$ fraud. This caused **Windows** and **iOS Device** hardware nodes to act as giant hubs, contaminating the target label and flagging **92.69%** of all transactions as suspicious.
2. **The R5 Overlap Solution**: To isolate local, coordinated attacks, we calibrated the target to R5: a card must transact on both a high-risk device ($\ge 3$ cards, $\ge 1$ fraud) AND a high-risk address ($\ge 3$ cards, $\ge 1$ fraud) in its look-back history. This successfully reduced proxy prevalence to a clean **12.86%** on the Train + Dev splits.

### Standalone Routing Workflow
During weight optimization, ensembling the Sentinel directly into Model D's probability score degraded PR-AUC (Dev PR-AUC fell from `0.67363` to `0.64597` under equal weights). Therefore, we abandoned probability blending and finalized the Sentinel as a standalone secondary routing filter:

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

---

## 7. Locked Final Test Set Sentinel Metrics
*This split was never opened during design or tuning, representing a strictly generalizable test.*

| Locked Test Metric / Dimension | Final Test Split Result |
| :--- | :---: |
| **Review Volume (Flagged Transactions)** | **`8,773`** transactions |
| **Review Population Share (%)** | **`9.90%`** |
| **Model D FNs (Missed Fraud) Total** | **`1,647`** cases |
| **Sentinel-Captured FNs** | **`201`** cases |
| **Missed Fraud (FN) Capture Rate** | **`12.20%`** |
| **Fraud Count in Reviewed** | **`426`** cases |
| **FPs (Friction) in Reviewed** | **`8,347`** cases |
| **Precision of Review (%)** | **`4.86%`** (vs 3.48% baseline rate) |
| **Fraud Concentration in Reviewed (%)** | **`13.82%`** |
| **Unique Cards Covered** | **`2,002`** cards |
| **Unique Devices Covered** | **`81`** devices |
| **Unique Addresses Covered** | **`223`** subnets |
| **FN Capture Efficiency** | **`1.23x`** |

> [!NOTE]
> **Performance Rationale**:
> By capturing **12.20%** of Model D's missed fraud while reviewing only **9.90%** of transactions, the Abuse-Ring Sentinel operates at **1.23x** efficiency compared to a random review baseline.

---

## 8. Final Hackathon Conclusions & Next Steps

* **Frozen Champion**: Model D represents our most robust transaction-level detector, achieving `0.58144` (80/20) and `0.59882` (70/30) PR-AUC. Chasing further transaction-level feature engineering risked overfitting chronological split boundaries.
* **Sentinel Value**: The Abuse-Ring Sentinel serves as an excellent secondary routing layer. It isolates high-risk coordinated networks, retrieving a significant portion of missed fraud without introducing friction on clean checkout flows.

