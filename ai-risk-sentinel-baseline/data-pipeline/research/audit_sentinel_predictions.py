import os
import sys
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from sklearn.metrics import average_precision_score, roc_auc_score

def train_lgb(X_train, y_train, X_val, y_val, categorical_cols, objective='binary', metric='average_precision'):
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset, categorical_feature=categorical_cols)
    
    params = {
        'objective': objective,
        'metric': metric,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'scale_pos_weight': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[train_dataset, val_dataset],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    probs_train = model.predict(X_train, num_iteration=model.best_iteration)
    probs_val = model.predict(X_val, num_iteration=model.best_iteration)
    return model, probs_train, probs_val

def main():
    print("=" * 70)
    print("      IEEE-CIS PHASE 14C.1: SENTINEL PREDICTION AUDIT & DIAGNOSTIC")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/abuse_ring_features.parquet')
    report_path = os.path.join(processed_dir, 'reports/sentinel_prediction_audit.md')

    if not os.path.exists(features_path):
        print(f"[ERROR] Features file not found.")
        sys.exit(1)

    # 1. Load Data
    df = pd.read_parquet(features_path)
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)
    
    split_70 = int(total_rows * 0.70)
    split_85 = int(total_rows * 0.85)

    # Define Feature Sets
    core_deviations = [
        'amount_vs_card_mean', 'amount_vs_card_median', 'amount_zscore',
        'tx_frequency_deviation_24h', 'tx_frequency_deviation_1h',
        'time_gap_deviation_median', 'time_gap_deviation_mean',
        'spend_velocity_deviation_1h', 'spend_velocity_deviation_24h',
        'card_device_frequency', 'card_location_frequency',
        'time_gap_acceleration_median', 'time_gap_acceleration_mean',
        'amount_temporal_interaction', 'amount_temporal_interaction_mean'
    ]
    diversity_cols = ['card_device_diversity', 'card_location_diversity']
    metadata_cols = ['TransactionID', 'isFraud']
    
    graph_base_cols = [
        'card_device_degree', 'card_addr_degree', 'device_card_degree', 'addr_card_degree',
        'shared_device_card_count', 'shared_addr_card_count',
        'device_connected_fraud_rate', 'addr_connected_fraud_rate'
    ]
    graph_refined_cols = [
        'network_risk_mean', 'network_risk_max', 'network_risk_gap', 'network_risk_product',
        'device_card_novelty', 'addr_card_novelty'
    ]
    novelty_features = [
        'card_addr_unseen', 'card_email_unseen', 'card_device_unseen',
        'card_addr_novelty_confidence', 'card_email_novelty_confidence'
    ]
    ring_features = [
        'device_unique_card_count', 'addr_unique_card_count', 'email_unique_card_count',
        'device_connected_fraud_rate', 'addr_connected_fraud_rate',
        'rapid_card_convergence', 'cross_entity_convergence', 'ring_fraud_density'
    ]

    all_excluded = (core_deviations + diversity_cols + metadata_cols + 
                    graph_base_cols + graph_refined_cols + novelty_features + ring_features + ['is_ring_abuse'])
    base_features = [col for col in df.columns if col not in all_excluded]

    # Train Model D
    print("Training Model D...")
    X_train_d = df.iloc[:split_70][base_features]
    y_train_d = df.iloc[:split_70]['isFraud'].values
    X_dev_d = df.iloc[split_70:split_85][base_features]
    y_dev_d = df.iloc[split_70:split_85]['isFraud'].values
    
    categorical_cols = list(X_train_d.select_dtypes(include=['category']).columns)
    model_d, probs_d_train, probs_d_dev = train_lgb(X_train_d, y_train_d, X_dev_d, y_dev_d, categorical_cols)
    
    # Train Sentinel
    print("Training Sentinel...")
    X_train_sentinel = df.iloc[:split_70][ring_features]
    y_train_sentinel = df.iloc[:split_70]['is_ring_abuse'].values
    X_dev_sentinel = df.iloc[split_70:split_85][ring_features]
    y_dev_sentinel = df.iloc[split_70:split_85]['is_ring_abuse'].values
    
    model_sentinel, probs_sentinel_train, probs_sentinel_dev = train_lgb(
        X_train_sentinel, y_train_sentinel, 
        X_dev_sentinel, y_dev_sentinel, 
        categorical_cols=[]
    )

    # Generate test predictions
    X_test_d = df.iloc[split_85:][base_features]
    y_test_d = df.iloc[split_85:]['isFraud'].values
    X_test_sentinel = df.iloc[split_85:][ring_features]
    y_test_sentinel = df.iloc[split_85:]['is_ring_abuse'].values

    probs_d_test = model_d.predict(X_test_d, num_iteration=model_d.best_iteration)
    probs_sentinel_test = model_sentinel.predict(X_test_sentinel, num_iteration=model_sentinel.best_iteration)

    # -------------------------------------------------------------
    # 1. Prediction Percentile Analysis
    # -------------------------------------------------------------
    print("\n[Phase 14C.1] Running percentile profiling...")
    percentiles = [0, 1, 5, 50, 75, 90, 95, 99, 100]
    
    pct_train = np.percentile(probs_sentinel_train, percentiles)
    pct_dev = np.percentile(probs_sentinel_dev, percentiles)
    pct_test = np.percentile(probs_sentinel_test, percentiles)

    pct_rows = []
    names = ['min', 'p01', 'p05', 'median', 'p75', 'p90', 'p95', 'p99', 'max']
    for idx, name in enumerate(names):
        pct_rows.append(f"| **{name}** | `{pct_train[idx]:.5f}` | `{pct_dev[idx]:.5f}` | `{pct_test[idx]:.5f}` |")

    # -------------------------------------------------------------
    # 2. Threshold Sweep & Prevalence Mapping
    # -------------------------------------------------------------
    print("\n[Phase 14C.1] Sweeping Sentinel thresholds...")
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    dev_sweep_rows = []
    for th in thresholds:
        dev_flagged = (probs_sentinel_dev >= th).sum()
        dev_prev = (dev_flagged / len(y_dev_d)) * 100
        
        dev_pos_df = df.iloc[split_70:split_85][probs_sentinel_dev >= th]
        dev_fraud_rate = dev_pos_df['isFraud'].mean() * 100 if dev_flagged > 0 else 0.0
        dev_cards = len(set(dev_pos_df['card1'].values)) if dev_flagged > 0 else 0
        
        dev_sweep_rows.append(f"| `{th:.2f}` | `{dev_flagged:,}` | `{dev_prev:.2f}%` | `{dev_fraud_rate:.2f}%` | `{dev_cards:,}` |")

    test_sweep_rows = []
    for th in thresholds:
        test_flagged = (probs_sentinel_test >= th).sum()
        test_prev = (test_flagged / len(y_test_d)) * 100
        
        test_pos_df = df.iloc[split_85:][probs_sentinel_test >= th]
        test_fraud_rate = test_pos_df['isFraud'].mean() * 100 if test_flagged > 0 else 0.0
        test_cards = len(set(test_pos_df['card1'].values)) if test_flagged > 0 else 0
        
        test_sweep_rows.append(f"| `{th:.2f}` | `{test_flagged:,}` | `{test_fraud_rate:.2f}%` | `{test_prev:.2f}%` | `{test_cards:,}` |")

    # -------------------------------------------------------------
    # 3. Core Alignment PR-AUC Scores
    # -------------------------------------------------------------
    print("\n[Phase 14C.1] Calculating alignment PR-AUC scores...")
    
    # 4 Core measurements
    pr_d_vs_fraud = average_precision_score(y_dev_d, probs_d_dev)
    pr_sentinel_vs_proxy = average_precision_score(y_dev_sentinel, probs_sentinel_dev)
    pr_sentinel_vs_fraud = average_precision_score(y_dev_d, probs_sentinel_dev)
    
    # Combined with equal weights
    probs_combined_dev = 0.5 * probs_d_dev + 0.5 * probs_sentinel_dev
    pr_comb_vs_fraud = average_precision_score(y_dev_d, probs_combined_dev)

    # -------------------------------------------------------------
    # 4. Model D FNs Recovery Curve Mapping
    # -------------------------------------------------------------
    print("\n[Phase 14C.1] Mapping FN recovery curves...")
    threshold_d = 0.30398
    fn_d_dev = (y_dev_d == 1) & (probs_d_dev < threshold_d)
    fn_d_test = (y_test_d == 1) & (probs_d_test < threshold_d)
    
    recovery_rows = []
    for th in thresholds:
        dev_flagged_mask = probs_sentinel_dev >= th
        test_flagged_mask = probs_sentinel_test >= th
        
        recovered_dev = fn_d_dev & dev_flagged_mask
        recovered_test = fn_d_test & test_flagged_mask
        
        pct_dev = (recovered_dev.sum() / fn_d_dev.sum()) * 100 if fn_d_dev.sum() > 0 else 0.0
        pct_test = (recovered_test.sum() / fn_d_test.sum()) * 100 if fn_d_test.sum() > 0 else 0.0
        
        dev_total_flagged = dev_flagged_mask.sum()
        dev_total_prev = (dev_total_flagged / len(y_dev_d)) * 100
        
        recovery_rows.append(f"| `{th:.2f}` | `{recovered_dev.sum():,} / {fn_d_dev.sum():,}` (**`{pct_dev:.2f}%`**) | `{dev_total_flagged:,}` (`{dev_total_prev:.2f}%`) | `{recovered_test.sum():,} / {fn_d_test.sum():,}` (**`{pct_test:.2f}%`**) |")

    # Clean up memory
    del X_train_d, X_dev_d, X_test_d
    del X_train_sentinel, X_dev_sentinel, X_test_sentinel
    gc.collect()

    # 5. Write Report
    print(f"\nWriting Sentinel Prediction Audit report to: {report_path}")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report_content = f"""# Abuse-Ring Sentinel: Prediction Audit Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report presents a deep-dive diagnostic of the Abuse-Ring Sentinel predictions to analyze calibration, alignment, and threshold bounds.

---

## 1. Raw Prediction Score Percentiles

| Percentile | Train Split | Dev/Val Split | Final Test Split |
| :---: | :---: | :---: | :---: |
{chr(10).join(pct_rows)}

---

## 2. Dev/Val Split Threshold Sweeps

This sweep shows how the Sentinel's transaction and card prevalence maps against decision thresholds on the **Dev/Val split**:

| Threshold | Flagged Volume | Prevalence (%) | Empirical Fraud Rate | Unique Cards |
| :---: | :---: | :---: | :---: | :---: |
{chr(10).join(dev_sweep_rows)}

---

## 3. Core Alignment Metrics (Dev/Val Split)

These metrics isolate the predictive signal of the Sentinel independently against its proxy label and actual fraud target:

* **Model D vs. actual fraud (`isFraud`) PR-AUC**: **`{pr_d_vs_fraud:.5f}`**
* **Sentinel vs. proxy target (`is_ring_abuse`) PR-AUC**: **`{pr_sentinel_vs_proxy:.5f}`**
* **Sentinel vs. actual fraud (`isFraud`) PR-AUC**: **`{pr_sentinel_vs_fraud:.5f}`**
* **Combined (equal weight) vs. actual fraud (`isFraud`) PR-AUC**: **`{pr_comb_vs_fraud:.5f}`**

---

## 4. Missed-Fraud (FN) Recovery Curves

This curve profiles how effectively the Sentinel captures Model D's False Negatives (at `0.30398` threshold) compared to the review population size:

| Sentinel Threshold | Dev FNs Recovered | Dev Review Size (%) | Test FNs Recovered |
| :---: | :---: | :---: | :---: |
{chr(10).join(recovery_rows)}

---

## 5. Diagnostic Findings & Verdict

> [!NOTE]
> **Diagnostic Verdict**:
> * **Score Calibration**: Check the score percentiles above. If the max score is below 0.30, the model is under-calibrated for the target, and we must lower the review routing threshold to 0.05 or 0.10.
> * **Independent Signal**: Look at Sentinel's PR-AUC vs actual fraud (`{pr_sentinel_vs_fraud:.5f}`). Even if the ensembled PR-AUC weight blend is 0.0, a high independent PR-AUC shows that the Sentinel captures distinct, useful fraud signals.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Sentinel prediction audit report written successfully!")

if __name__ == '__main__':
    main()
