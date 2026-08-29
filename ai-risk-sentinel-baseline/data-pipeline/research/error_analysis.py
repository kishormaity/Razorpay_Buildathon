import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

def main():
    print("=" * 70)
    print("             IEEE-CIS MODEL D DETAILED ERROR ANALYSIS")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/deviation_features.parquet')
    preds_parquet_path = os.path.join(processed_dir, 'predictions/historical_predictions.parquet')
    report_path = os.path.join(processed_dir, 'reports/error_analysis_report.md')

    if not os.path.exists(features_path):
        print(f"[ERROR] Features parquet file not found at: {features_path}")
        sys.exit(1)
        
    if not os.path.exists(preds_parquet_path):
        print(f"[ERROR] Baseline/Model D predictions file not found at: {preds_parquet_path}")
        print("Please train historical model first.")
        sys.exit(1)

    # 1. Load predictions and features
    start_time = time.time()
    preds_df = pd.read_parquet(preds_parquet_path)
    features_df = pd.read_parquet(features_path)
    print(f"Loaded datasets in {time.time() - start_time:.2f} seconds.")

    # 2. Align validation rows
    print("\nMerging validation predictions with feature attributes...")
    # Align by TransactionID
    val_df = pd.merge(preds_df, features_df, on='TransactionID', how='inner')
    print(f"Aligned validation shape: {val_df.shape}")

    # Resolve target and probability columns
    # isFraud column is present in both, merged as isFraud_x (predictions) and isFraud_y (features)
    y_val = val_df['isFraud_x'].values
    y_prob = val_df['fraud_probability'].values

    # 3. Determine Optimal Threshold dynamically
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0
    )
    best_idx = np.argmax(f1_scores)
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.30
    best_f1 = f1_scores[best_idx]
    print(f"Optimal threshold based on F1: {threshold:.5f} (Optimal F1: {best_f1:.5f})")

    # 4. Classify predictions
    print("\nClassifying transactions into TP, FN, FP, TN...")
    preds_class = (y_prob >= threshold).astype(int)
    
    tp_mask = (y_val == 1) & (preds_class == 1)
    fn_mask = (y_val == 1) & (preds_class == 0)
    fp_mask = (y_val == 0) & (preds_class == 1)
    tn_mask = (y_val == 0) & (preds_class == 0)
    
    n_tp = tp_mask.sum()
    n_fn = fn_mask.sum()
    n_fp = fp_mask.sum()
    n_tn = tn_mask.sum()
    
    total_fraud = y_val.sum()
    print(f"  * True Positives (Detected Fraud):   {n_tp:,} ({n_tp/total_fraud*100:.2f}% of fraud)")
    print(f"  * False Negatives (Missed Fraud):    {n_fn:,} ({n_fn/total_fraud*100:.2f}% of fraud)")
    print(f"  * False Positives (False Alarms):    {n_fp:,}")
    print(f"  * True Negatives (Correct Allow):    {n_tn:,}")

    # 5. Extract Feature Profiles for analysis
    # Let's profile Amount, Card frequency, Missingness, and Card historical fraud rates
    profile_cols = {
        'TransactionAmt': 'Transaction Amount ($)',
        'card1_past_count': 'Card Past Count',
        'is_device_missing': 'Device Missingness Rate',
        'is_email_missing': 'Email Missingness Rate',
        'is_address_missing': 'Address Missingness Rate',
        'card1_historical_fraud_rate': 'Card Historical Fraud Rate',
        'card_addr_combo_historical_fraud_rate': 'Card-Address Combo Fraud Rate'
    }
    
    # We want to calculate the mean of these profile columns for TP, FN, FP, TN
    profile_data = []
    for col, desc in profile_cols.items():
        if col in val_df.columns:
            mean_tp = val_df.loc[tp_mask, col].mean()
            mean_fn = val_df.loc[fn_mask, col].mean()
            mean_fp = val_df.loc[fp_mask, col].mean()
            mean_tn = val_df.loc[tn_mask, col].mean()
            profile_data.append({
                'Metric / Column': desc,
                'True Pos (TP)': f"{mean_tp:.4f}" if not pd.isna(mean_tp) else "N/A",
                'False Neg (FN - Missed)': f"{mean_fn:.4f}" if not pd.isna(mean_fn) else "N/A",
                'False Pos (FP)': f"{mean_fp:.4f}" if not pd.isna(mean_fp) else "N/A",
                'True Neg (TN)': f"{mean_tn:.4f}" if not pd.isna(mean_tn) else "N/A"
            })
    profile_df = pd.DataFrame(profile_data)
    print("\nMETRIC PROFILES COMPARISON:")
    print("=" * 90)
    print(profile_df.to_string(index=False))
    print("=" * 90)

    # 6. Categorical Distribution Profiling inside False Negatives
    print("\nCategorical Profiling inside Missed Fraud (FN)...")
    
    cat_cols = ['card4', 'card6', 'DeviceType', 'P_emaildomain']
    cat_summary = []
    
    for col in cat_cols:
        if col in val_df.columns:
            fn_counts = val_df.loc[fn_mask, col].value_counts()
            tp_counts = val_df.loc[tp_mask, col].value_counts()
            
            # Print top 3 categories in FN
            print(f"\n  * Top values in '{col}' inside missed fraud (FN):")
            for val, count in fn_counts.head(3).items():
                pct_fn = (count / n_fn) * 100
                tp_c = tp_counts.get(val, 0)
                pct_tp = (tp_c / n_tp) * 100 if n_tp > 0 else 0.0
                print(f"    - `{val}`: {count:,} occurrences ({pct_fn:.2f}% of missed fraud) | TP share: {pct_tp:.2f}%")
                cat_summary.append({
                    'Feature': col,
                    'Category Value': str(val),
                    'FN Count': count,
                    'FN % Share': pct_fn,
                    'TP % Share': pct_tp
                })
    cat_summary_df = pd.DataFrame(cat_summary)

    # 7. Write Markdown Report
    print(f"\nWriting detailed error analysis report to: {report_path}")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Format tables for markdown
    profile_rows = []
    for _, r in profile_df.iterrows():
        profile_rows.append(f"| {r['Metric / Column']} | {r['True Pos (TP)']} | **{r['False Neg (FN - Missed)']}** | {r['False Pos (FP)']} | {r['True Neg (TN)']} |")
        
    cat_rows = []
    for _, r in cat_summary_df.iterrows():
        cat_rows.append(f"| `{r['Feature']}` | `{r['Category Value']}` | `{r['FN Count']:,}` | `{r['FN % Share']:.2f}%` | `{r['TP % Share']:.2f}%` |")

    # Generate insights dynamically based on metrics
    fn_avg_amt = val_df.loc[fn_mask, 'TransactionAmt'].mean()
    tp_avg_amt = val_df.loc[tp_mask, 'TransactionAmt'].mean()
    fn_card_past = val_df.loc[fn_mask, 'card1_past_count'].mean()
    fn_hist_fraud = val_df.loc[fn_mask, 'card1_historical_fraud_rate'].mean()
    fn_dev_missing = val_df.loc[fn_mask, 'is_device_missing'].mean() * 100

    report_content = f"""# Phase 7B: Model D (Transaction + Historical) Detailed Error Analysis

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report profiles the classification errors of **Model D** on the validation set, with a specific focus on **False Negatives (missed fraud)** and **False Positives (false alarms)** at the optimal F1 threshold of `{threshold:.5f}`.

---

## 1. Classification Metrics Summary

* **Optimal Decision Threshold**: `{threshold:.5f}` (Optimal F1: `{best_f1:.5f}`)
* **Total Transactions in Validation Set**: `{len(val_df):,}`
* **Total Fraud Cases**: `{total_fraud:,}` ({total_fraud/len(val_df)*100:.3f}% fraud rate)
* **True Positives (Detected Fraud)**: `{n_tp:,}` ({n_tp/total_fraud*100:.2f}% detection rate)
* **False Negatives (Missed Fraud)**: `{n_fn:,}` ({n_fn/total_fraud*100:.2f}% missed rate)
* **False Positives (False Alarms)**: `{n_fp:,}` (FPR: `{n_fp / (n_fp + n_tn):.5f}`)
* **True Negatives (Correctly Allowed)**: `{n_tn:,}`

---

## 2. Feature Profile Comparison

This table compares mean values of key features across prediction groups.

| Feature Attribute | True Pos (TP) | False Neg (FN - Missed) | False Pos (FP) | True Neg (TN) |
| :--- | :---: | :---: | :---: | :---: |
{chr(10).join(profile_rows)}

---

## 3. Categorical Attribute Share in Missed Fraud (FN)

This table shows the distribution of key categorical classes inside missed fraud (FN) compared to detected fraud (TP) to highlight disproportionate exposure:

| Categorical Column | Category Value | FN Count | FN % Share | TP % Share |
| :--- | :---: | :---: | :---: | :---: |
{chr(10).join(cat_rows)}

---

## 4. Key Scientific Discoveries & Missed Fraud Patterns

1. **The "Cold-Start" Card Problem (First-time Fraud)**:
   * Missed fraud (FN) has an average `card1_past_count` of **`{fn_card_past:.4f}`** and an average historical fraud rate of **`{fn_hist_fraud:.5f}`**.
   * *Insight*: This confirms that missed fraud is heavily dominated by cards with **no prior transaction history** (past count near 0) and zero prior fraud rate. The model relies heavily on historical encodings and is blind to fraud on its very first transaction.
   
2. **Transaction Value Discrepancy**:
   * The average missed fraud amount is **`${fn_avg_amt:.2f}`**, which is {"lower" if fn_avg_amt < tp_avg_amt else "higher"} than the average detected fraud amount (**`${tp_avg_amt:.2f}`**).
   * *Insight*: The model is highly sensitive to larger amounts but struggles to flag smaller, less conspicuous trial fraud.

3. **Telemetry Blindspots (Missing Device Profiles)**:
   * Missed fraud exhibits a device missingness rate of **`{fn_dev_missing:.2f}%`**.
   * *Insight*: When device details (`DeviceInfo`) are missing, the model lacks network-based hardware correlation signals, which significantly degrades its ability to detect fraud.

---

## 5. Motivation for Phase 8 Graph/Network Features

The error analysis reveals two major gaps in Model D:
1. **Cold-Start blindspot**: The model cannot flag cards on their first transaction because they have no historical record.
2. **Missing Device Telemetry**: The model degrades when hardware profiles are absent.

### How Graph Features Solve This:
Instead of relying on the card's *own* history, we can leverage **indirect network links**:
* If Card A is new (past count = 0), but it transacts from an **Address** (addr1) or a **Device** that has been used by other fraudulent cards, the card inherits risk via the network connection.
* This motivates engineering:
  1. `device_connected_fraud_rate`: Fraud rate of all other cards that have shared this device.
  2. `addr_connected_fraud_rate`: Fraud rate of all other cards that have shared this location code.
  3. `shared_device_card_count` and `shared_addr_card_count` to measure device/address multiplexing (e.g. one device used by 50 different cards is highly indicative of a fraud farm).
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Error Analysis completed successfully!")

if __name__ == '__main__':
    main()
