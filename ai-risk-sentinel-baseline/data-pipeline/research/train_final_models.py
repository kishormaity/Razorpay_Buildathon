import os
import sys
import time
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    return model

def main():
    print("=" * 70)
    print("      IEEE-CIS FINAL MODELS OFFLINE TRAINING & FREEZING")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    features_path = os.path.join(processed_dir, 'features/abuse_ring_features.parquet')
    models_dir = os.path.join(current_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(features_path):
        print(f"[ERROR] Features file not found at: {features_path}")
        sys.exit(1)

    # 1. Load Data
    print("Loading aligned features dataset...")
    df = pd.read_parquet(features_path)
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)
    
    # Ensure stationary cyclical time features are present
    if 'hour_of_day' not in df.columns:
        print("Deriving stationary cyclical time features from TransactionDT...")
        df['hour_of_day'] = ((df['TransactionDT'] // 3600) % 24).astype('float32')
        df['day_of_week'] = ((df['TransactionDT'] // 86400) % 7).astype('float32')
        df['hour_sin'] = np.sin(2 * np.pi * (df['TransactionDT'] % 86400) / 86400).astype('float32')
        df['hour_cos'] = np.cos(2 * np.pi * (df['TransactionDT'] % 86400) / 86400).astype('float32')
        df['day_of_week_sin'] = np.sin(2 * np.pi * ((df['TransactionDT'] // 86400) % 7) / 7).astype('float32')
        df['day_of_week_cos'] = np.cos(2 * np.pi * ((df['TransactionDT'] // 86400) % 7) / 7).astype('float32')
    
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
    # NOTE: TransactionDT is in metadata_cols and all_excluded so it NEVER enters Model D feature matrix
    metadata_cols = ['TransactionID', 'TransactionDT', 'isFraud']
    
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
    # NOTE: ring_fraud_density removed to eliminate circular target reconstruction of is_ring_abuse (R5)
    ring_features = [
        'device_unique_card_count', 'addr_unique_card_count', 'email_unique_card_count',
        'device_connected_fraud_rate', 'addr_connected_fraud_rate',
        'rapid_card_convergence', 'cross_entity_convergence'
    ]

    all_excluded = (core_deviations + diversity_cols + metadata_cols + 
                    graph_base_cols + graph_refined_cols + novelty_features + ring_features + ['is_ring_abuse', 'ring_fraud_density'])
    base_features = [col for col in df.columns if col not in all_excluded]

    # Save feature lists to ensure order compliance in inference
    with open(os.path.join(models_dir, 'model_d_features.json'), 'w') as f:
        json.dump(base_features, f)
    with open(os.path.join(models_dir, 'sentinel_features.json'), 'w') as f:
        json.dump(ring_features, f)
        
    print(f"-> Model D Features: {len(base_features)} (TransactionDT verified excluded)")
    print(f"-> Sentinel Features: {len(ring_features)} (ring_fraud_density verified excluded)")

    # 2. Train Model D
    print("\nTraining Model D (Baseline Transaction Model)...")
    X_train_d = df.iloc[:split_70][base_features]
    y_train_d = df.iloc[:split_70]['isFraud'].values
    X_dev_d = df.iloc[split_70:split_85][base_features]
    y_dev_d = df.iloc[split_70:split_85]['isFraud'].values
    categorical_cols = list(X_train_d.select_dtypes(include=['category']).columns)
    
    model_d = train_lgb(X_train_d, y_train_d, X_dev_d, y_dev_d, categorical_cols)
    model_d.save_model(os.path.join(models_dir, 'model_d_final.txt'))
    print("[OK] Model D booster saved successfully to models/model_d_final.txt")

    # 3. Train Abuse-Ring Sentinel
    print("\nTraining Abuse-Ring Sentinel Model...")
    X_train_sentinel = df.iloc[:split_70][ring_features]
    y_train_sentinel = df.iloc[:split_70]['is_ring_abuse'].values
    X_dev_sentinel = df.iloc[split_70:split_85][ring_features]
    y_dev_sentinel = df.iloc[split_70:split_85]['is_ring_abuse'].values
    
    model_sentinel = train_lgb(X_train_sentinel, y_train_sentinel, X_dev_sentinel, y_dev_sentinel, categorical_cols=[])
    model_sentinel.save_model(os.path.join(models_dir, 'abuse_ring_sentinel_final.txt'))
    print("[OK] Abuse-Ring Sentinel booster saved successfully to models/abuse_ring_sentinel_final.txt")

    # 4. Final Evaluation on Held-Out Locked Test Split (85%-100%)
    print("\n" + "=" * 70)
    print("      EVALUATING ON LOCKED HELD-OUT CHRONOLOGICAL TEST SET (15%)")
    print("=" * 70)
    from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, precision_score, recall_score, f1_score

    X_test_d = df.iloc[split_85:][base_features]
    y_test_d = df.iloc[split_85:]['isFraud'].values
    X_test_sentinel = df.iloc[split_85:][ring_features]
    y_test_sentinel = df.iloc[split_85:]['is_ring_abuse'].values

    raw_probs_d_test = model_d.predict(X_test_d)
    raw_probs_sentinel_test = model_sentinel.predict(X_test_sentinel)

    # Model D Evaluation
    p_curve_d, r_curve_d, _ = precision_recall_curve(y_test_d, raw_probs_d_test)
    pr_auc_d = auc(r_curve_d, p_curve_d)
    roc_auc_d = roc_auc_score(y_test_d, raw_probs_d_test)
    pred_bin_d = (raw_probs_d_test >= 0.30398).astype(int)
    prec_d = precision_score(y_test_d, pred_bin_d, zero_division=0)
    rec_d = recall_score(y_test_d, pred_bin_d, zero_division=0)
    f1_d = f1_score(y_test_d, pred_bin_d, zero_division=0)

    print(f"\nModel D (Baseline Threshold = 0.30398):")
    print(f"  * PR-AUC:    {pr_auc_d:.4f}")
    print(f"  * ROC-AUC:   {roc_auc_d:.4f}")
    print(f"  * Precision: {prec_d*100:.2f}%")
    print(f"  * Recall:    {rec_d*100:.2f}%")
    print(f"  * F1-Score:  {f1_d:.4f}")

    # Sentinel Proxy-Label Evaluation (Weak Supervision)
    p_curve_s, r_curve_s, _ = precision_recall_curve(y_test_sentinel, raw_probs_sentinel_test)
    pr_auc_s = auc(r_curve_s, p_curve_s)
    roc_auc_s = roc_auc_score(y_test_sentinel, raw_probs_sentinel_test)
    pred_bin_s = (raw_probs_sentinel_test >= 0.15000).astype(int)
    prec_s = precision_score(y_test_sentinel, pred_bin_s, zero_division=0)
    rec_s = recall_score(y_test_sentinel, pred_bin_s, zero_division=0)
    f1_s = f1_score(y_test_sentinel, pred_bin_s, zero_division=0)

    # Missed fraud interception (Defense-in-depth)
    fn_mask = (y_test_d == 1) & (raw_probs_d_test < 0.30398)
    total_fn = int(np.sum(fn_mask))
    captured_fn = int(np.sum(fn_mask & (raw_probs_sentinel_test >= 0.15000)))
    fn_rate = captured_fn / total_fn if total_fn > 0 else 0.0

    print(f"\nAbuse-Ring Sentinel (WEAK SUPERVISION / PROXY-LABEL EVALUATION):")
    print(f"  * Proxy PR-AUC:    {pr_auc_s:.4f}")
    print(f"  * Proxy ROC-AUC:   {roc_auc_s:.4f}")
    print(f"  * Proxy Precision: {prec_s*100:.2f}%")
    print(f"  * Proxy Recall:    {rec_s*100:.2f}%")
    print(f"  * Proxy F1:        {f1_s:.4f}")
    print(f"  * Ground-Truth Missed Fraud Interception: {captured_fn}/{total_fn} ({fn_rate*100:.2f}%)")

    # Save metrics JSON
    metrics_summary = {
        "model_d": {
            "pr_auc": float(pr_auc_d),
            "roc_auc": float(roc_auc_d),
            "precision": float(prec_d),
            "recall": float(rec_d),
            "f1": float(f1_d),
            "threshold": 0.30398
        },
        "sentinel_weak_supervision_proxy": {
            "pr_auc": float(pr_auc_s),
            "roc_auc": float(roc_auc_s),
            "precision": float(prec_s),
            "recall": float(rec_s),
            "f1": float(f1_s),
            "threshold": 0.15000,
            "evaluation_type": "WEAK-SUPERVISION / PROXY-LABEL EVALUATION"
        },
        "sentinel_ground_truth_interception": {
            "model_d_false_negatives_captured": captured_fn,
            "total_model_d_false_negatives": total_fn,
            "capture_rate": float(fn_rate)
        }
    }
    with open(os.path.join(models_dir, 'p0_test_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[OK] P0 test metrics exported to: {os.path.join(models_dir, 'p0_test_metrics.json')}")

if __name__ == '__main__':
    main()

