import os
import sys
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

def is_valid_device(dev):
    return not pd.isna(dev) and dev != 'UNKNOWN' and dev != ''

def is_valid_addr(a):
    return not pd.isna(a) and a != -999 and a != -999.0

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
    
    probs_val = model.predict(X_val, num_iteration=model.best_iteration)
    return model, probs_val

def compute_metrics(y_true, y_prob_d, y_prob_sentinel, df_subset, threshold_d, threshold_sentinel):
    n_total = len(y_true)
    total_fraud = y_true.sum()
    
    # Sentinel routing flags
    flagged_mask = y_prob_sentinel >= threshold_sentinel
    review_volume = flagged_mask.sum()
    review_pct = (review_volume / n_total) * 100
    
    # Fraud in reviewed
    flagged_y = y_true[flagged_mask]
    fraud_count = flagged_y.sum()
    fraud_rate = (fraud_count / review_volume) * 100 if review_volume > 0 else 0.0
    
    # Model D FNs (at threshold_d)
    fn_d_mask = (y_true == 1) & (y_prob_d < threshold_d)
    fn_d_total = fn_d_mask.sum()
    
    # FNs captured by Sentinel
    captured_fn = (fn_d_mask & flagged_mask).sum()
    fn_capture_rate = (captured_fn / fn_d_total) * 100 if fn_d_total > 0 else 0.0
    
    # FPs in reviewed
    fp_count = review_volume - fraud_count
    precision_review = (fraud_count / review_volume) * 100 if review_volume > 0 else 0.0
    fraud_concentration = (fraud_count / total_fraud) * 100 if total_fraud > 0 else 0.0
    
    # Unique entities
    flagged_df = df_subset[flagged_mask]
    unique_cards = len(set(flagged_df['card1'].values)) if review_volume > 0 else 0
    
    devs = flagged_df['DeviceInfo'].dropna().values
    unique_devices = len(set(d for d in devs if is_valid_device(d))) if review_volume > 0 else 0
    
    addrs = flagged_df['addr1'].dropna().values
    unique_addrs = len(set(a for a in addrs if is_valid_addr(a))) if review_volume > 0 else 0
    
    # FN Capture Efficiency
    fn_capture_efficiency = fn_capture_rate / review_pct if review_pct > 0 else 0.0
    
    return {
        'volume': review_volume,
        'pct': review_pct,
        'fraud_count': fraud_count,
        'fraud_rate': fraud_rate,
        'model_d_fn': fn_d_total,
        'captured_fn': captured_fn,
        'fn_capture_rate': fn_capture_rate,
        'fp_reviewed': fp_count,
        'precision': precision_review,
        'concentration': fraud_concentration,
        'unique_cards': unique_cards,
        'unique_devices': unique_devices,
        'unique_addrs': unique_addrs,
        'efficiency': fn_capture_efficiency
    }

def main():
    print("=" * 70)
    print("      IEEE-CIS PHASE 14D: SENTINEL STANDALONE ROUTING VALIDATION")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/abuse_ring_features.parquet')
    report_path = os.path.join(processed_dir, 'reports/sentinel_routing_validation_report.md')

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
    print("Training Model D on Train split...")
    X_train_d = df.iloc[:split_70][base_features]
    y_train_d = df.iloc[:split_70]['isFraud'].values
    X_dev_d = df.iloc[split_70:split_85][base_features]
    y_dev_d = df.iloc[split_70:split_85]['isFraud'].values
    
    categorical_cols = list(X_train_d.select_dtypes(include=['category']).columns)
    model_d, probs_d_dev = train_lgb(X_train_d, y_train_d, X_dev_d, y_dev_d, categorical_cols)
    
    # Train Sentinel
    print("Training Sentinel on Train split...")
    X_train_sentinel = df.iloc[:split_70][ring_features]
    y_train_sentinel = df.iloc[:split_70]['is_ring_abuse'].values
    X_dev_sentinel = df.iloc[split_70:split_85][ring_features]
    y_dev_sentinel = df.iloc[split_70:split_85]['is_ring_abuse'].values
    
    model_sentinel, probs_sentinel_dev = train_lgb(
        X_train_sentinel, y_train_sentinel, 
        X_dev_sentinel, y_dev_sentinel, 
        categorical_cols=[]
    )

    # Inferences on locked Test split
    X_test_d = df.iloc[split_85:][base_features]
    y_test_d = df.iloc[split_85:]['isFraud'].values
    X_test_sentinel = df.iloc[split_85:][ring_features]
    y_test_sentinel = df.iloc[split_85:]['is_ring_abuse'].values

    print("\nRunning inferences on Final Test split...")
    probs_d_test = model_d.predict(X_test_d, num_iteration=model_d.best_iteration)
    probs_sentinel_test = model_sentinel.predict(X_test_sentinel, num_iteration=model_sentinel.best_iteration)

    # 2. Compute metrics
    print("\nComputing operational routing metrics (threshold = 0.15)...")
    threshold_d = 0.30398
    threshold_sentinel = 0.15
    
    df_dev = df.iloc[split_70:split_85]
    df_test = df.iloc[split_85:]
    
    m_dev = compute_metrics(y_dev_d, probs_d_dev, probs_sentinel_dev, df_dev, threshold_d, threshold_sentinel)
    m_test = compute_metrics(y_test_d, probs_d_test, probs_sentinel_test, df_test, threshold_d, threshold_sentinel)

    print("\n" + "=" * 60)
    print("OPERATIONAL COMPARISON TABLE (Sentinel Threshold = 0.15):")
    print("=" * 60)
    print(f"Dimension                     | Dev Split      | Test Split")
    print("-" * 60)
    print(f"Review Volume (Count)         | {m_dev['volume']:<14,} | {m_test['volume']:<14,}")
    print(f"Review Population Share (%)   | {m_dev['pct']:<14.2f}% | {m_test['pct']:<14.2f}%")
    print(f"Fraud Count in Reviewed       | {m_dev['fraud_count']:<14,} | {m_test['fraud_count']:<14,}")
    print(f"Fraud Rate in Reviewed (%)    | {m_dev['fraud_rate']:<14.2f}% | {m_test['fraud_rate']:<14.2f}%")
    print(f"Model D FNs Total             | {m_dev['model_d_fn']:<14,} | {m_test['model_d_fn']:<14,}")
    print(f"Sentinel-Captured FNs         | {m_dev['captured_fn']:<14,} | {m_test['captured_fn']:<14,}")
    print(f"FN Capture Rate (%)           | {m_dev['fn_capture_rate']:<14.2f}% | {m_test['fn_capture_rate']:<14.2f}%")
    print(f"FPs among Reviewed            | {m_dev['fp_reviewed']:<14,} | {m_test['fp_reviewed']:<14,}")
    print(f"Precision of Review (%)       | {m_dev['precision']:<14.2f}% | {m_test['precision']:<14.2f}%")
    print(f"Fraud Concentration (%)       | {m_dev['concentration']:<14.2f}% | {m_test['concentration']:<14.2f}%")
    print(f"Unique Cards Covered          | {m_dev['unique_cards']:<14,} | {m_test['unique_cards']:<14,}")
    print(f"Unique Devices Covered        | {m_dev['unique_devices']:<14,} | {m_test['unique_devices']:<14,}")
    print(f"Unique Addresses Covered      | {m_dev['unique_addrs']:<14,} | {m_test['unique_addrs']:<14,}")
    print("-" * 60)
    print(f"FN Capture Efficiency         | {m_dev['efficiency']:<14.2f}  | {m_test['efficiency']:<14.2f}")
    print("=" * 60)

    # Clean up memory
    del X_train_d, X_dev_d, X_test_d
    del X_train_sentinel, X_dev_sentinel, X_test_sentinel
    gc.collect()

    # 3. Write report
    print(f"\nWriting routing validation report to: {report_path}")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report_content = f"""# Abuse-Ring Sentinel: Standalone Routing Validation Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report documents the standalone operational evaluation of the **Abuse-Ring Sentinel** as a secondary review routing layer (fixed decision threshold = **`0.15`**).

---

## 1. Split Configurations (70/15/15)
* **Train split** (0% to 70%): `{split_70:,}` transactions
* **Dev/Val split** (70% to 85%): `{split_85 - split_70:,}` transactions
* **Final Test split** (85% to 100%): `{total_rows - split_85:,}` transactions (Strictly locked during tuning)

---

## 2. Operational Evaluation Table (Threshold = 0.15)

| Operational Metric / Dimension | Dev/Val Split | Final Test Split (Locked) |
| :--- | :---: | :---: |
| **Review Volume (Flagged Transactions)** | `{m_dev['volume']:,}` | `{m_test['volume']:,}` |
| **Review Population Share (%)** | `{m_dev['pct']:.2f}%` | `{m_test['pct']:.2f}%` |
| **Fraud Count in Reviewed** | `{m_dev['fraud_count']:,}` | `{m_test['fraud_count']:,}` |
| **Fraud Rate in Reviewed (%)** | `{m_dev['fraud_rate']:.2f}%` | `{m_test['fraud_rate']:.2f}%` |
| **Model D FNs (Missed Fraud)** | `{m_dev['model_d_fn']:,}` | `{m_test['model_d_fn']:,}` |
| **Sentinel-Captured FNs** | `{m_dev['captured_fn']:,}` | `{m_test['captured_fn']:,}` |
| **FN Capture Rate (%)** | `{m_dev['fn_capture_rate']:.2f}%` | `{m_test['fn_capture_rate']:.2f}%` |
| **FPs (Friction) among Reviewed** | `{m_dev['fp_reviewed']:,}` | `{m_test['fp_reviewed']:,}` |
| **Precision of Review (%)** | `{m_dev['precision']:.2f}%` | `{m_test['precision']:.2f}%` |
| **Fraud Concentration (%)** | `{m_dev['concentration']:.2f}%` | `{m_test['concentration']:.2f}%` |
| **Unique Cards Covered** | `{m_dev['unique_cards']:,}` | `{m_test['unique_cards']:,}` |
| **Unique Devices Covered** | `{m_dev['unique_devices']:,}` | `{m_test['unique_devices']:,}` |
| **Unique Addresses Covered** | `{m_dev['unique_addrs']:,}` | `{m_test['unique_addrs']:,}` |
| **FN Capture Efficiency** | **`{m_dev['efficiency']:.4f}`** | **`{m_test['efficiency']:.4f}`** |

---

## 3. Key Findings & Rationale

> [!IMPORTANT]
> **FN Capture Efficiency Interpretation**:
> * **Dev Split Efficiency**: **`{m_dev['efficiency']:.2f}`**
> * **Test Split Efficiency**: **`{m_test['efficiency']:.2f}`**
> * *Interpretation*: An efficiency of **`{m_test['efficiency']:.2f}`** on the locked test set indicates that the Sentinel is preferentially routing Model D's missed fraud for review compared to a random baseline selection (efficiency = 1.0).

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
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Sentinel routing validation report written successfully!")

if __name__ == '__main__':
    main()
