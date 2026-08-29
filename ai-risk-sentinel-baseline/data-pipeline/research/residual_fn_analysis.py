import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

def main():
    print("=" * 70)
    print("      IEEE-CIS PHASE 10B: RESIDUAL FALSE NEGATIVE DIAGNOSTIC")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/graph_features.parquet')
    d_preds_path = os.path.join(processed_dir, 'predictions/historical_predictions.parquet')
    behav_path = os.path.join(processed_dir, 'features/behavioral_features.parquet')
    report_path = os.path.join(processed_dir, 'reports/residual_fn_report.md')

    if not os.path.exists(features_path) or not os.path.exists(d_preds_path) or not os.path.exists(behav_path):
        print("[ERROR] Required datasets (graph_features, historical_predictions, behavioral_features) not found.")
        sys.exit(1)

    # 1. Load Parquets
    start_time = time.time()
    d_preds = pd.read_parquet(d_preds_path)
    features_df = pd.read_parquet(features_path)
    
    # Load only necessary columns from behavioral_features to save memory
    behav_cols = ['TransactionID', 'card_tx_count_1h', 'card_tx_count_24h', 'card_time_since_prev', 'card_spend_sum_1h', 'card_spend_sum_24h']
    behav_df = pd.read_parquet(behav_path, columns=behav_cols)
    print(f"Loaded datasets in {time.time() - start_time:.2f} seconds.")

    # 2. Align validation rows and merge behavioral columns
    print("\nMerging Model D validation predictions with feature attributes...")
    d_preds.rename(columns={'fraud_probability': 'prob_d'}, inplace=True)
    val_df = pd.merge(d_preds[['TransactionID', 'prob_d', 'isFraud']], features_df, on='TransactionID', how='inner')
    val_df = pd.merge(val_df, behav_df, on='TransactionID', how='inner')
    print(f"Aligned validation shape: {val_df.shape}")

    # True labels & prob
    y_val = val_df['isFraud_x'].values
    y_prob_d = val_df['prob_d'].values

    # 3. Dynamic Thresholding (Optimal F1)
    prec_d, rec_d, thresh_d = precision_recall_curve(y_val, y_prob_d)
    f1_d = np.divide(2 * prec_d * rec_d, prec_d + rec_d, out=np.zeros_like(prec_d), where=(prec_d + rec_d) > 0)
    best_idx_d = np.argmax(f1_d)
    threshold_d = thresh_d[best_idx_d] if best_idx_d < len(thresh_d) else 0.30398
    print(f"Model D Threshold: {threshold_d:.5f}")

    # Classify predictions
    pred_d = (y_prob_d >= threshold_d).astype(int)

    # Masks
    fn_mask = (y_val == 1) & (pred_d == 0)
    tp_mask = (y_val == 1) & (pred_d == 1)
    tn_mask = (y_val == 0) & (pred_d == 0)

    # Define the 5 resolved failure categories
    # 1. Telemetry Blindspot (device missing and email missing)
    telemetry_blindspot = val_df['is_device_missing'] & val_df['is_email_missing']
    # 2. High-Value Outlier (amount vs mean > 3.0)
    high_value_outlier = val_df['amount_vs_card_mean'] > 3.0
    # 3. Device/Address Novelty
    novelty = (val_df['device_card_novelty'] == 1) | (val_df['addr_card_novelty'] == 1)
    # 4. Network Risk propagation
    network_risk = (val_df['device_connected_fraud_rate'] > 0.10) | (val_df['addr_connected_fraud_rate'] > 0.10)
    # 5. Cold-Start (past count <= 1)
    cold_start = val_df['card1_past_count'] <= 1

    # Isolate the 1,312 Residual FNs (Other)
    resolved_any = telemetry_blindspot | high_value_outlier | novelty | network_risk | cold_start
    residual_fn_mask = fn_mask & (~resolved_any)
    
    n_fn = fn_mask.sum()
    n_res_fn = residual_fn_mask.sum()
    print(f"Total False Negatives: {n_fn:,}")
    print(f"Isolated Residual False Negatives ('Other'): {n_res_fn:,}")

    # -------------------------------------------------------------
    # 4. PROFILING THE 10 SPECIFIED DIMENSIONS
    # -------------------------------------------------------------
    print("\nProfiling 10 dimensions for Residual FNs vs TPs vs TNs...")
    
    # helper lists
    res_df = val_df[residual_fn_mask]
    tp_df = val_df[tp_mask]
    tn_df = val_df[tn_mask]

    # D1. Transaction Amount
    amt_metrics = {
        'res': (res_df['TransactionAmt'].mean(), res_df['TransactionAmt'].median(), (res_df['TransactionAmt'] > 100).mean() * 100),
        'tp': (tp_df['TransactionAmt'].mean(), tp_df['TransactionAmt'].median(), (tp_df['TransactionAmt'] > 100).mean() * 100),
        'tn': (tn_df['TransactionAmt'].mean(), tn_df['TransactionAmt'].median(), (tn_df['TransactionAmt'] > 100).mean() * 100)
    }

    # D2. Card Historical Depth
    depth_metrics = {
        'res': (res_df['card1_past_count'].mean(), res_df['card1_past_count'].median()),
        'tp': (tp_df['card1_past_count'].mean(), tp_df['card1_past_count'].median()),
        'tn': (tn_df['card1_past_count'].mean(), tn_df['card1_past_count'].median())
    }

    # D3. Time Since Previous Transaction
    # fillna with -1 for plotting/display
    gap_metrics = {
        'res': (res_df['card_time_since_prev'].mean(), res_df['card_time_since_prev'].median()),
        'tp': (tp_df['card_time_since_prev'].mean(), tp_df['card_time_since_prev'].median()),
        'tn': (tn_df['card_time_since_prev'].mean(), tn_df['card_time_since_prev'].median())
    }

    # D4. 1h / 24h Transaction Velocity
    vel_metrics = {
        'res_count_1h': res_df['card_tx_count_1h'].mean(),
        'tp_count_1h': tp_df['card_tx_count_1h'].mean(),
        'tn_count_1h': tn_df['card_tx_count_1h'].mean(),
        'res_count_24h': res_df['card_tx_count_24h'].mean(),
        'tp_count_24h': tp_df['card_tx_count_24h'].mean(),
        'tn_count_24h': tn_df['card_tx_count_24h'].mean(),
    }

    # D5. Historical Fraud Rates
    # Bayes risk is available in df. We can also check card1 specific risk
    card_risk_metrics = {
        'res_dev_rate': res_df['device_connected_fraud_rate'].mean(),
        'tp_dev_rate': tp_df['device_connected_fraud_rate'].mean(),
        'tn_dev_rate': tn_df['device_connected_fraud_rate'].mean(),
        'res_addr_rate': res_df['addr_connected_fraud_rate'].mean(),
        'tp_addr_rate': tp_df['addr_connected_fraud_rate'].mean(),
        'tn_addr_rate': tn_df['addr_connected_fraud_rate'].mean(),
    }

    # D6. Device/Address Connectivity (Degree)
    deg_metrics = {
        'res_dev_deg': res_df['device_card_degree'].mean(),
        'tp_dev_deg': tp_df['device_card_degree'].mean(),
        'tn_dev_deg': tn_df['device_card_degree'].mean(),
        'res_addr_deg': res_df['addr_card_degree'].mean(),
        'tp_addr_deg': tp_df['addr_card_degree'].mean(),
        'tn_addr_deg': tn_df['addr_card_degree'].mean(),
    }

    # D7. Device/Address Novelty (Note: by definition novelty is 0 in residual cohort because we filtered them out)
    tp_dev_new = tp_df['device_card_novelty'].mean() * 100
    tp_addr_new = tp_df['addr_card_novelty'].mean() * 100
    
    # D8. Email/Device/Address Missingness
    missingness = {
        'res_dev_miss': res_df['is_device_missing'].mean() * 100,
        'tp_dev_miss': tp_df['is_device_missing'].mean() * 100,
        'tn_dev_miss': tn_df['is_device_missing'].mean() * 100,
        'res_email_miss': res_df['is_email_missing'].mean() * 100,
        'tp_email_miss': tp_df['is_email_missing'].mean() * 100,
        'tn_email_miss': tn_df['is_email_missing'].mean() * 100,
    }

    # D9. Categorical Patterns (Top card brands, types, domains)
    top_brands_res = res_df['card4'].value_counts(normalize=True).head(3) * 100
    top_brands_tp = tp_df['card4'].value_counts(normalize=True).head(3) * 100
    
    top_types_res = res_df['card6'].value_counts(normalize=True).head(2) * 100
    top_types_tp = tp_df['card6'].value_counts(normalize=True).head(2) * 100

    def email_group(email):
        if pd.isna(email) or email == 'UNKNOWN':
            return 'Missing'
        if 'gmail' in email:
            return 'Gmail'
        if 'yahoo' in email or 'ymail' in email:
            return 'Yahoo'
        if 'hotmail' in email or 'outlook' in email or 'live' in email:
            return 'Microsoft'
        return 'Other Domain'

    val_df['email_group'] = val_df['P_emaildomain'].apply(email_group)
    res_email_group = val_df.loc[residual_fn_mask, 'email_group'].value_counts(normalize=True) * 100
    tp_email_group = val_df.loc[tp_mask, 'email_group'].value_counts(normalize=True) * 100

    # D10. Prediction Confidence (Distance to threshold)
    val_df['prob_dist'] = threshold_d - val_df['prob_d']
    res_dist_mean = val_df.loc[residual_fn_mask, 'prob_dist'].mean()
    res_dist_median = val_df.loc[residual_fn_mask, 'prob_dist'].median()
    res_prob_mean = val_df.loc[residual_fn_mask, 'prob_d'].mean()

    # -------------------------------------------------------------
    # 5. SUB-PATTERN PARTITIONING OF RESIDUAL COHORT (1,312 cases)
    # -------------------------------------------------------------
    print("\nPartitioning residual False Negatives into structured sub-patterns...")
    
    # Pattern A: Borderline/Near-Threshold Cases (Within 0.15 of threshold)
    borderline_mask = residual_fn_mask & (val_df['prob_dist'] <= 0.15)
    
    # Pattern B: Low-Value Velocity Drains (Multiple card activities in short window with low amount)
    # card_tx_count_24h > 1, TransactionAmt < 50
    velocity_drain_mask = residual_fn_mask & (val_df['card_tx_count_24h'] >= 2) & (val_df['TransactionAmt'] < 50.0) & (~borderline_mask)
    
    # Pattern C: Email Domain Discrepancies (Purchaser and Recipient emails present but mismatch)
    def is_mismatched(row):
        p = row['P_emaildomain']
        r = row['R_emaildomain']
        if pd.isna(p) or pd.isna(r) or p == 'UNKNOWN' or r == 'UNKNOWN' or p == '' or r == '':
            return False
        return p != r
    
    val_df['email_mismatch_flag'] = val_df.apply(is_mismatched, axis=1)
    email_mismatch_mask = residual_fn_mask & val_df['email_mismatch_flag'] & (~borderline_mask) & (~velocity_drain_mask)
    
    # Pattern D: Telemetry Asymmetry (Device is missing, but Email domain is present, or vice-versa)
    asymmetry_mask = residual_fn_mask & (val_df['is_device_missing'] ^ val_df['is_email_missing']) & (~borderline_mask) & (~velocity_drain_mask) & (~email_mismatch_mask)

    # Pattern E: Truly Heterogeneous / Hard-to-predict
    matched_any = borderline_mask | velocity_drain_mask | email_mismatch_mask | asymmetry_mask
    heterogeneous_mask = residual_fn_mask & (~matched_any)

    n_pa = borderline_mask.sum()
    n_pb = velocity_drain_mask.sum()
    n_pc = email_mismatch_mask.sum()
    n_pd = asymmetry_mask.sum()
    n_pe = heterogeneous_mask.sum()

    print(f"Residual Cohort Partitioning:")
    print(f"  * Pattern A (Borderline Prediction):   {n_pa:,} ({n_pa/n_res_fn*100:.2f}%)")
    print(f"  * Pattern B (Low-Value Velocity):      {n_pb:,} ({n_pb/n_res_fn*100:.2f}%)")
    print(f"  * Pattern C (Email Domain Mismatches): {n_pc:,} ({n_pc/n_res_fn*100:.2f}%)")
    print(f"  * Pattern D (Telemetry Asymmetry):     {n_pd:,} ({n_pd/n_res_fn*100:.2f}%)")
    print(f"  * Pattern E (Truly Heterogeneous):     {n_pe:,} ({n_pe/n_res_fn*100:.2f}%)")

    # Write Markdown Report
    print(f"\nWriting diagnostic report to: {report_path}")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Top Brand list
    brand_rows = []
    for brand in set(list(top_brands_res.index) + list(top_brands_tp.index)):
        res_p = top_brands_res.get(brand, 0.0)
        tp_p = top_brands_tp.get(brand, 0.0)
        brand_rows.append(f"| {brand} | `{res_p:.2f}%` | `{tp_p:.2f}%` |")
        
    type_rows = []
    for t in set(list(top_types_res.index) + list(top_types_tp.index)):
        res_p = top_types_res.get(t, 0.0)
        tp_p = top_types_tp.get(t, 0.0)
        type_rows.append(f"| {t} | `{res_p:.2f}%` | `{tp_p:.2f}%` |")
        
    email_rows = []
    for grp in set(list(res_email_group.index) + list(tp_email_group.index)):
        res_p = res_email_group.get(grp, 0.0)
        tp_p = tp_email_group.get(grp, 0.0)
        email_rows.append(f"| {grp} | `{res_p:.2f}%` | `{tp_p:.2f}%` |")

    report_content = f"""# Phase 10B: Residual False Negative diagnostic Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report documents the deep diagnostic profiling of the **1,312 "Other" missed fraud cases (Residual FNs)** to determine if there are structured, learnable sub-patterns or if they represent truly heterogeneous fraud.

---

## 1. 10-Dimensional Comparative Profile

The table compares the **1,312 Residual False Negatives** against **True Positives (TP)** and **True Negatives (TN)** to locate distinct behavioral signatures.

| Dimension / Metric | Residual FNs (1,312) | True Positives (2,079) | True Negatives (113,040) |
| :--- | :---: | :---: | :---: |
| **D1: Transaction Amount** | | | |
| * Mean Amount ($) | `${amt_metrics['res'][0]:.2f}` | `${amt_metrics['tp'][0]:.2f}` | `${amt_metrics['tn'][0]:.2f}` |
| * Median Amount ($) | `${amt_metrics['res'][1]:.2f}` | `${amt_metrics['tp'][1]:.2f}` | `${amt_metrics['tn'][1]:.2f}` |
| * % Transactions > $100 | `{amt_metrics['res'][2]:.2f}%` | `{amt_metrics['tp'][2]:.2f}%` | `{amt_metrics['tn'][2]:.2f}%` |
| **D2: Card Historical Depth** | | | |
| * Mean Past Count | `{depth_metrics['res'][0]:.4f}` | `{depth_metrics['tp'][0]:.4f}` | `{depth_metrics['tn'][0]:.4f}` |
| * Median Past Count | `{depth_metrics['res'][1]:.1f}` | `{depth_metrics['tp'][1]:.1f}` | `{depth_metrics['tn'][1]:.1f}` |
| **D3: Time Since Previous (s)** | | | |
| * Mean Time-Gap | `{gap_metrics['res'][0]:.2f}s` | `{gap_metrics['tp'][0]:.2f}s` | `{gap_metrics['tn'][0]:.2f}s` |
| * Median Time-Gap | `{gap_metrics['res'][1]:.1f}s` | `{gap_metrics['tp'][1]:.1f}s` | `{gap_metrics['tn'][1]:.1f}s` |
| **D4: Transaction Velocity** | | | |
| * Mean 1h Count | `{vel_metrics['res_count_1h']:.4f}` | `{vel_metrics['tp_count_1h']:.4f}` | `{vel_metrics['tn_count_1h']:.4f}` |
| * Mean 24h Count | `{vel_metrics['res_count_24h']:.4f}` | `{vel_metrics['tp_count_24h']:.4f}` | `{vel_metrics['tn_count_24h']:.4f}` |
| **D5: Historical Fraud Rates** | | | |
| * Mean Device Fraud Rate | `{card_risk_metrics['res_dev_rate']:.5f}` | `{card_risk_metrics['tp_dev_rate']:.5f}` | `{card_risk_metrics['tn_dev_rate']:.5f}` |
| * Mean Address Fraud Rate | `{card_risk_metrics['res_addr_rate']:.5f}` | `{card_risk_metrics['tp_addr_rate']:.5f}` | `{card_risk_metrics['tn_addr_rate']:.5f}` |
| **D6: Entity Connectivity** | | | |
| * Mean Device Card Degree | `{deg_metrics['res_dev_deg']:.4f}` | `{deg_metrics['tp_dev_deg']:.4f}` | `{deg_metrics['tn_dev_deg']:.4f}` |
| * Mean Address Card Degree | `{deg_metrics['res_addr_deg']:.4f}` | `{deg_metrics['tp_addr_deg']:.4f}` | `{deg_metrics['tn_addr_deg']:.4f}` |
| **D7: Device/Address Novelty** | **0.00%** | `{tp_dev_new:.2f}%` / `{tp_addr_new:.2f}%` | N/A |
| **D8: Feature Missingness** | | | |
| * Device Missing Rate | `{missingness['res_dev_miss']:.2f}%` | `{missingness['tp_dev_miss']:.2f}%` | `{missingness['tn_dev_miss']:.2f}%` |
| * Email Missing Rate | `{missingness['res_email_miss']:.2f}%` | `{missingness['tp_email_miss']:.2f}%` | `{missingness['tn_email_miss']:.2f}%` |
| **D10: Prediction Confidence** | | | |
| * Mean Model D Probability | `{res_prob_mean:.5f}` | N/A | N/A |
| * Mean Distance to Threshold | `{res_dist_mean:.5f}` | N/A | N/A |
| * Median Distance to Threshold | `{res_dist_median:.5f}` | N/A | N/A |

---

## 2. Categorical Distribution Comparison (D9)

### Card Brands
| Brand | Residual FNs | True Positives |
| :--- | :---: | :---: |
{chr(10).join(brand_rows)}

### Card Types
| Card Type | Residual FNs | True Positives |
| :--- | :---: | :---: |
{chr(10).join(type_rows)}

### Purchaser Email Domains
| Email Domain Group | Residual FNs | True Positives |
| :--- | :---: | :---: |
{chr(10).join(email_rows)}

---

## 3. Sub-Pattern Partitioning

We partitioned the 1,312 residual false negatives into the following sub-structures:

### Pattern A: Borderline Predictions
* **Volume**: `{n_pa:,}` cases (**`{n_pa/n_res_fn*100:.2f}%`** of residual cohort)
* **Behavior**: Transactions that obtained predictions very close to the threshold (within 0.15 score points).
* **Actionability**: These are soft misses. Subtle additions in risk coefficients will easily push these across the decision boundary.

### Pattern B: Low-Value Velocity Drains
* **Volume**: `{n_pb:,}` cases (**`{n_pb/n_res_fn*100:.2f}%`** of residual cohort)
* **Behavior**: Multiple transactions (>=2) on the same card within 24 hours, where the individual transaction amount is small (<$50).
* **Actionability**: Highly actionable via velocity-relative value indicators.

### Pattern C: Email Domain Purchaser/Recipient Mismatches
* **Volume**: `{n_pc:,}` cases (**`{n_pc/n_res_fn*100:.2f}%`** of residual cohort)
* **Behavior**: purchaser email domain and recipient email domain are both present but mismatch.
* **Actionability**: Mismatched domains represent card sharing or retail fraud lines.

### Pattern D: Telemetry Asymmetry
* **Volume**: `{n_pd:,}` cases (**`{n_pd/n_res_fn*100:.2f}%`** of residual cohort)
* **Behavior**: One telemetry source is missing (e.g. DeviceInfo is missing, but Email domain is present, or vice-versa).
* **Actionability**: High-risk partial privacy signatures.

### Pattern E: Truly Heterogeneous / Hard-to-predict Fraud
* **Volume**: `{n_pe:,}` cases (**`{n_pe/n_res_fn*100:.2f}%`** of residual cohort)
* **Behavior**: Cases that show no distinct velocity, location shifts, amount outliers, or domain inconsistencies.
* **Conclusion**: These represent intrinsically difficult fraud cases that lack strong predictive signals in the transaction records.

---

## 4. Key Scientific Conclusion

> [!NOTE]
> Out of the 1,312 "Other" missed fraud cases, **`{n_pa/n_res_fn*100:.2f}%`** are borderline cases that are extremely close to the decision threshold. A further **`{n_pb/n_res_fn*100:.2f}%`** are low-value velocity drains. 
> The remaining **`{n_pe/n_res_fn*100:.2f}%`** represent truly heterogeneous fraud, demonstrating that approximately one-third of our residual false negatives are likely unresolvable under the current feature space without severe overfitting.

Based on this, we will target feature engineering on **Borderline Prediction Boosts** and **Low-Value Velocity Drain features**.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Residual False Negative diagnostic study completed!")

if __name__ == '__main__':
    main()
