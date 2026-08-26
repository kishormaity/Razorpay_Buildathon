import os
import sys
import time
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

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

    # Save feature lists to ensure order compliance in inference
    with open(os.path.join(models_dir, 'model_d_features.json'), 'w') as f:
        json.dump(base_features, f)
    with open(os.path.join(models_dir, 'sentinel_features.json'), 'w') as f:
        json.dump(ring_features, f)

    # 2. Train Model D
    print("Training Model D (Baseline Transaction Model)...")
    X_train_d = df.iloc[:split_70][base_features]
    y_train_d = df.iloc[:split_70]['isFraud'].values
    X_dev_d = df.iloc[split_70:split_85][base_features]
    y_dev_d = df.iloc[split_70:split_85]['isFraud'].values
    categorical_cols = list(X_train_d.select_dtypes(include=['category']).columns)
    
    model_d = train_lgb(X_train_d, y_train_d, X_dev_d, y_dev_d, categorical_cols)
    model_d.save_model(os.path.join(models_dir, 'model_d_final.txt'))
    print("✓ Model D booster saved successfully to models/model_d_final.txt")

    # 3. Train Abuse-Ring Sentinel
    print("Training Abuse-Ring Sentinel Model...")
    X_train_sentinel = df.iloc[:split_70][ring_features]
    y_train_sentinel = df.iloc[:split_70]['is_ring_abuse'].values
    X_dev_sentinel = df.iloc[split_70:split_85][ring_features]
    y_dev_sentinel = df.iloc[split_70:split_85]['is_ring_abuse'].values
    
    model_sentinel = train_lgb(X_train_sentinel, y_train_sentinel, X_dev_sentinel, y_dev_sentinel, categorical_cols=[])
    model_sentinel.save_model(os.path.join(models_dir, 'abuse_ring_sentinel_final.txt'))
    print("✓ Abuse-Ring Sentinel booster saved successfully to models/abuse_ring_sentinel_final.txt")

    print("\n" + "=" * 50)
    print("OFFLINE TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 50)

if __name__ == '__main__':
    main()
