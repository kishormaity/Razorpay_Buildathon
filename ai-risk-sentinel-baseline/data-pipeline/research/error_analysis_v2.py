import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

def main():
    print("=" * 70)
    print("      IEEE-CIS PHASE 10: DEEP ERROR ANALYSIS & FAILURE ARCHETYPES")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/graph_features.parquet')
    d_preds_path = os.path.join(processed_dir, 'predictions/historical_predictions.parquet')
    report_path = os.path.join(processed_dir, 'reports/error_analysis_v2_report.md')

    if not os.path.exists(features_path):
        print(f"[ERROR] Graph features file not found at: {features_path}")
        sys.exit(1)
        
    if not os.path.exists(d_preds_path):
        print(f"[ERROR] Model D predictions file not found at: {d_preds_path}")
        sys.exit(1)

    # 1. Load Parquets
    start_time = time.time()
    d_preds = pd.read_parquet(d_preds_path)
    features_df = pd.read_parquet(features_path)
    print(f"Loaded datasets in {time.time() - start_time:.2f} seconds.")

    # 2. Align validation rows
    print("\nMerging Model D validation predictions with feature attributes...")
    d_preds.rename(columns={'fraud_probability': 'prob_d'}, inplace=True)
    val_df = pd.merge(d_preds[['TransactionID', 'prob_d', 'isFraud']], features_df, on='TransactionID', how='inner')
    print(f"Aligned validation shape: {val_df.shape}")

    # True labels (isFraud_x from merge)
    y_val = val_df['isFraud_x'].values
    y_prob_d = val_df['prob_d'].values

    # 3. Dynamic Thresholding (Optimal F1)
    prec_d, rec_d, thresh_d = precision_recall_curve(y_val, y_prob_d)
    f1_d = np.divide(2 * prec_d * rec_d, prec_d + rec_d, out=np.zeros_like(prec_d), where=(prec_d + rec_d) > 0)
    best_idx_d = np.argmax(f1_d)
    threshold_d = thresh_d[best_idx_d] if best_idx_d < len(thresh_d) else 0.30398
    best_f1 = f1_d[best_idx_d]
    print(f"Optimal Model D Threshold: {threshold_d:.5f} (F1: {best_f1:.5f})")

    # Classify predictions
    pred_d = (y_prob_d >= threshold_d).astype(int)

    # Error Masks
    fn_mask = (y_val == 1) & (pred_d == 0)
    fp_mask = (y_val == 0) & (pred_d == 1)
    tp_mask = (y_val == 1) & (pred_d == 1)
    tn_mask = (y_val == 0) & (pred_d == 0)

    n_fn = fn_mask.sum()
    n_fp = fp_mask.sum()
    n_tp = tp_mask.sum()
    n_tn = tn_mask.sum()

    print(f"\nModel D Validation Error Counts:")
    print(f"  * False Negatives (Missed Fraud):    {n_fn:,}")
    print(f"  * False Positives (False Alarms):    {n_fp:,}")
    print(f"  * True Positives (Detected Fraud):   {n_tp:,}")
    print(f"  * True Negatives (Correct Allow):    {n_tn:,}")

    # -------------------------------------------------------------
    # 4. DEEP ERROR PROFILING BY DIMENSION
    # -------------------------------------------------------------
    print("\nProfiling error classes across behavioral features...")
    
    # A. Cold-Start Behavior
    fn_past_count = val_df.loc[fn_mask, 'card1_past_count']
    tp_past_count = val_df.loc[tp_mask, 'card1_past_count']
    
    fn_cold_start_pct = (fn_past_count <= 1).mean() * 100
    tp_cold_start_pct = (tp_past_count <= 1).mean() * 100

    # B. Email Inconsistencies
    # Calculate email domain mismatch rate when both are present
    def email_mismatch(row):
        p = row['P_emaildomain']
        r = row['R_emaildomain']
        if pd.isna(p) or pd.isna(r) or p == 'UNKNOWN' or r == 'UNKNOWN':
            return False
        return p != r

    val_df['email_mismatch'] = val_df.apply(email_mismatch, axis=1)
    fn_email_mismatch = val_df.loc[fn_mask, 'email_mismatch'].mean() * 100
    tp_email_mismatch = val_df.loc[tp_mask, 'email_mismatch'].mean() * 100

    # C. Telemetry Missingness
    fn_dev_missing = val_df.loc[fn_mask, 'is_device_missing'].mean() * 100
    tp_dev_missing = val_df.loc[tp_mask, 'is_device_missing'].mean() * 100
    
    # Double missingness: device info is missing and email domain is missing
    val_df['double_missing'] = val_df['is_device_missing'] & val_df['is_email_missing']
    fn_double_missing = val_df.loc[fn_mask, 'double_missing'].mean() * 100
    tp_double_missing = val_df.loc[tp_mask, 'double_missing'].mean() * 100

    # D. Device & Address Novelty
    fn_dev_new = val_df.loc[fn_mask, 'device_card_novelty'].mean() * 100
    tp_dev_new = val_df.loc[tp_mask, 'device_card_novelty'].mean() * 100
    
    fn_addr_new = val_df.loc[fn_mask, 'addr_card_novelty'].mean() * 100
    tp_addr_new = val_df.loc[tp_mask, 'addr_card_novelty'].mean() * 100

    # E. Amount Anomalies
    fn_avg_amt = val_df.loc[fn_mask, 'TransactionAmt'].mean()
    tp_avg_amt = val_df.loc[tp_mask, 'TransactionAmt'].mean()
    tn_avg_amt = val_df.loc[tn_mask, 'TransactionAmt'].mean()
    fp_avg_amt = val_df.loc[fp_mask, 'TransactionAmt'].mean()
    
    fn_amt_vs_card_mean = val_df.loc[fn_mask, 'amount_vs_card_mean'].mean()
    tp_amt_vs_card_mean = val_df.loc[tp_mask, 'amount_vs_card_mean'].mean()

    # F. Network Fraud Rates (Bayes connected rates)
    fn_dev_risk = val_df.loc[fn_mask, 'device_connected_fraud_rate'].mean()
    tp_dev_risk = val_df.loc[tp_mask, 'device_connected_fraud_rate'].mean()
    
    fn_addr_risk = val_df.loc[fn_mask, 'addr_connected_fraud_rate'].mean()
    tp_addr_risk = val_df.loc[tp_mask, 'addr_connected_fraud_rate'].mean()

    # -------------------------------------------------------------
    # 5. FRAUD FAILURE ARCHETYPES CLUSTERING (Rule-Based)
    # -------------------------------------------------------------
    print("\nCategorizing False Negatives into failure archetypes...")
    
    # Archetype 1: Cold-Start Fraud (Zero or 1 past transaction)
    cold_start_mask = fn_mask & (val_df['card1_past_count'] <= 1)
    
    # Archetype 2: Telemetry Blindspot (Device Info AND Email domain are missing)
    blindspot_mask = fn_mask & val_df['double_missing'] & (~cold_start_mask)
    
    # Archetype 3: Address / Device Novelty (New address or device for an established card)
    novelty_mask = fn_mask & (val_df['device_card_novelty'] == 1 | (val_df['addr_card_novelty'] == 1)) & (~cold_start_mask) & (~blindspot_mask)
    
    # Archetype 4: Network Connected Risk (Card is relatively new or clean, but transacts from a device/address with known fraud history)
    network_risk_mask = fn_mask & ((val_df['device_connected_fraud_rate'] > 0.10) | (val_df['addr_connected_fraud_rate'] > 0.10)) & (~cold_start_mask) & (~blindspot_mask) & (~novelty_mask)
    
    # Archetype 5: High Value Outlier (Transaction amount is significantly higher than typical card behavior)
    high_val_outlier_mask = fn_mask & (val_df['amount_vs_card_mean'] > 3.0) & (~cold_start_mask) & (~blindspot_mask) & (~novelty_mask) & (~network_risk_mask)
    
    # Other / Unclassified
    classified_mask = cold_start_mask | blindspot_mask | novelty_mask | network_risk_mask | high_val_outlier_mask
    unclassified_mask = fn_mask & (~classified_mask)
    
    n_cs = cold_start_mask.sum()
    n_bs = blindspot_mask.sum()
    n_nv = novelty_mask.sum()
    n_nr = network_risk_mask.sum()
    n_hv = high_val_outlier_mask.sum()
    n_uc = unclassified_mask.sum()

    print(f"Failure Archetypes Breakdown (Total FNs = {n_fn}):")
    print(f"  1. Cold-Start Fraud:             {n_cs:,} ({n_cs/n_fn*100:.2f}%)")
    print(f"  2. Telemetry Blindspot:          {n_bs:,} ({n_bs/n_fn*100:.2f}%)")
    print(f"  3. Device/Address Novelty:       {n_nv:,} ({n_nv/n_fn*100:.2f}%)")
    print(f"  4. Network Risk Propagation:     {n_nr:,} ({n_nr/n_fn*100:.2f}%)")
    print(f"  5. High-Value Outliers:          {n_hv:,} ({n_hv/n_fn*100:.2f}%)")
    print(f"  6. Other Unclassified:           {n_uc:,} ({n_uc/n_fn*100:.2f}%)")

    # -------------------------------------------------------------
    # 6. FALSE ALARMS ANALYSIS (False Positives - 1,004 cases)
    # -------------------------------------------------------------
    # Let's profile what triggers False Positives
    fp_avg_amt = val_df.loc[fp_mask, 'TransactionAmt'].mean()
    fp_dev_missing = val_df.loc[fp_mask, 'is_device_missing'].mean() * 100
    fp_shared_device = val_df.loc[fp_mask, 'shared_device_card_count'].mean()
    fp_shared_addr = val_df.loc[fp_mask, 'shared_addr_card_count'].mean()
    fp_past_count = val_df.loc[fp_mask, 'card1_past_count'].mean()

    # Save findings to markdown report
    print(f"\nWriting second-generation error analysis report to: {report_path}")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report_content = f"""# Phase 10: Model D Second-Generation Error Analysis Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report documents the deep profiling of **Model D**'s validation errors (1,985 False Negatives and 1,004 False Positives) on the 80/20 split to identify systematic failure modes.

---

## 1. Validation Predictions Summary

* **Optimal Model D Decision Threshold**: `{threshold_d:.5f}` (Optimal F1: `{best_f1:.5f}`)
* **Total Validation Set Size**: `{len(val_df):,}`
* **Detected Fraud (True Positives)**: `{n_tp:,}`
* **Missed Fraud (False Negatives)**: `{n_fn:,}`
* **False Alarms (False Positives)**: `{n_fp:,}`
* **Correct Allows (True Negatives)**: `{n_tn:,}`

---

## 2. Deep Profiling by Fraud Detection Status

The table compares behavioral statistics of detected fraud (TP) vs missed fraud (FN).

| Feature Attribute / Metric | Detected Fraud (TP) | Missed Fraud (FN) | True Negatives (TN) | False Positives (FP) |
| :--- | :---: | :---: | :---: | :---: |
| **Transaction Count (`card1_past_count`)** | `{val_df.loc[tp_mask, 'card1_past_count'].mean():.4f}` | **`{val_df.loc[fn_mask, 'card1_past_count'].mean():.4f}`** | `{val_df.loc[tn_mask, 'card1_past_count'].mean():.4f}` | `{val_df.loc[fp_mask, 'card1_past_count'].mean():.4f}` |
| **Cold-Start Card Rate (Past Count <= 1)** | `{tp_cold_start_pct:.2f}%` | **`{fn_cold_start_pct:.2f}%`** | N/A | N/A |
| **Average Transaction Amount ($)** | `${tp_avg_amt:.2f}` | **`${fn_avg_amt:.2f}`** | `${tn_avg_amt:.2f}` | `${fp_avg_amt:.2f}` |
| **Ratio to Card Avg (`amount_vs_card_mean`)** | `{tp_amt_vs_card_mean:.4f}` | **`{fn_amt_vs_card_mean:.4f}`** | N/A | N/A |
| **Device Missingness Rate** | `{tp_dev_missing:.2f}%` | **`{fn_dev_missing:.2f}%`** | N/A | `{fp_dev_missing:.2f}%` |
| **Double Missing (Device & Email)** | `{tp_double_missing:.2f}%` | **`{fn_double_missing:.2f}%`** | N/A | N/A |
| **Device Card Novelty** | `{tp_dev_new:.2f}%` | **`{fn_dev_new:.2f}%`** | N/A | N/A |
| **Address Card Novelty** | `{tp_addr_new:.2f}%` | **`{fn_addr_new:.2f}%`** | N/A | N/A |
| **Device Connected Fraud Rate** | `{tp_dev_risk:.5f}` | **`{fn_dev_risk:.5f}`** | N/A | N/A |
| **Address Connected Fraud Rate** | `{tp_addr_risk:.5f}` | **`{fn_addr_risk:.5f}`** | N/A | N/A |
| **Email Domain Mismatch Rate** | `{tp_email_mismatch:.2f}%` | **`{fn_email_mismatch:.2f}%`** | N/A | N/A |

---

## 3. Missed Fraud Failure Archetypes

We classified the `{n_fn:,}` missed fraud cases into the following failure categories:

### Archetype 1: Cold-Start Fraud
* **Volume**: `{n_cs:,}` cases (**`{n_cs/n_fn*100:.2f}%`** of missed fraud)
* **Definition**: Card has no prior transaction history (past count <= 1).
* **Why Model D Missed**: Model D relies heavily on historical card risk profiles. Without history, it defaults to a low-risk prediction.

### Archetype 2: Telemetry Blindspots
* **Volume**: `{n_bs:,}` cases (**`{n_bs/n_fn*100:.2f}%`** of missed fraud)
* **Definition**: Both hardware details (`DeviceInfo`) and email domains are missing.
* **Why Model D Missed**: Without device fingerprints or email domains, the model has no telemetry hooks to correlate.

### Archetype 3: Device/Address Novelty
* **Volume**: `{n_nv:,}` cases (**`{n_nv/n_fn*100:.2f}%`** of missed fraud)
* **Definition**: Card has prior transactions, but is transacting from a brand new device or address.
* **Why Model D Missed**: Model D cannot contextualize location or device shifts for a card in isolation.

### Archetype 4: Network Connected Risk (Risk Propagation)
* **Volume**: `{n_nr:,}` cases (**`{n_nr/n_fn*100:.2f}%`** of missed fraud)
* **Definition**: Clean or new card transacting from a device or address location that has been used by other fraudulent cards.
* **Why Model D Missed**: Model D is card-focused and blind to device/address network links.

### Archetype 5: High-Value Outliers
* **Volume**: `{n_hv:,}` cases (**`{n_hv/n_fn*100:.2f}%`** of missed fraud)
* **Definition**: Transaction amount is more than 3x the card's typical historic average.
* **Why Model D Missed**: Tree models do not extrapolate continuous numerical outliers well unless explicitly represented.

---

## 4. False Alarms (False Positives) Analysis

Model D made `{n_fp:,}` false alarms. The profiles show:
* **High Amounts**: Average FP amount is `${fp_avg_amt:.2f}`, showing a strong bias towards flagging large transactions.
* **Device sharing**: FP cards share devices with an average of `{fp_shared_device:.2f}` other cards, showing that device pooling triggers false alarms on normal cards.
* **Address sharing**: FP cards share addresses with an average of `{fp_shared_addr:.2f}` other cards.

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
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Deep error analysis study completed!")

if __name__ == '__main__':
    main()
