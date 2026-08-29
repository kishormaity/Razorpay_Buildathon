import os
import sys
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from sklearn.metrics import average_precision_score

def evaluate_predictions(y_true, y_prob):
    return average_precision_score(y_true, y_prob)

def train_eval_config(X_train, y_train, X_val, y_val, categorical_cols):
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset, categorical_feature=categorical_cols)
    
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
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
    
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    del train_dataset
    del val_dataset
    del model
    gc.collect()
    
    return val_preds

def run_paired_bootstrap(y_true, y_prob_d, y_prob_cand, num_resamples=1000, random_seed=42):
    np.random.seed(random_seed)
    n_samples = len(y_true)
    boot_delta = np.zeros(num_resamples)
    
    for i in range(num_resamples):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_b = y_true[boot_idx]
        y_prob_d_b = y_prob_d[boot_idx]
        y_prob_cand_b = y_prob_cand[boot_idx]
        
        pr_d = average_precision_score(y_true_b, y_prob_d_b)
        pr_cand = average_precision_score(y_true_b, y_prob_cand_b)
        
        boot_delta[i] = pr_cand - pr_d
        
    mean_delta = np.mean(boot_delta)
    std_delta = np.std(boot_delta)
    ci_lower = np.percentile(boot_delta, 2.5)
    ci_upper = np.percentile(boot_delta, 97.5)
    
    return mean_delta, std_delta, ci_lower, ci_upper

def main():
    print("=" * 70)
    print("      IEEE-CIS PHASE 12C: C5 VS MODEL D PAIRED BOOTSTRAP STUDY")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/card_novelty_features.parquet')
    d_preds_8020_path = os.path.join(processed_dir, 'predictions/historical_predictions.parquet')

    if not os.path.exists(features_path):
        print(f"[ERROR] Card novelty features file not found at: {features_path}")
        sys.exit(1)

    # 1. Load Data
    df = pd.read_parquet(features_path)
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)
    split_80 = int(total_rows * 0.8)
    
    y = df['isFraud']
    y_val_80 = y.iloc[split_80:].values

    # Columns definitions
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
    
    all_excluded = (core_deviations + diversity_cols + metadata_cols + 
                    graph_base_cols + graph_refined_cols + novelty_features)
    base_features = [col for col in df.columns if col not in all_excluded]

    # Load Model D predictions
    if os.path.exists(d_preds_8020_path):
        print("Loading Model D predictions...")
        d_preds_df = pd.read_parquet(d_preds_8020_path)
        probs_d = d_preds_df['fraud_probability'].values
        del d_preds_df
        gc.collect()
    else:
        print("Training Model D...")
        X_train_d = df.iloc[:split_80][base_features]
        X_val_d = df.iloc[split_80:][base_features]
        categorical_cols = list(X_train_d.select_dtypes(include=['category']).columns)
        probs_d = train_eval_config(X_train_d, y.iloc[:split_80].values, X_val_d, y_val_80, categorical_cols)
        del X_train_d
        del X_val_d
        gc.collect()

    # Train C5 (Model D + card_email_novelty_confidence)
    print("Training Candidate C5 (Model D + card_email_novelty_confidence)...")
    X_train_c5 = df.iloc[:split_80][base_features + ['card_email_novelty_confidence']]
    X_val_c5 = df.iloc[split_80:][base_features + ['card_email_novelty_confidence']]
    categorical_cols = list(X_train_c5.select_dtypes(include=['category']).columns)
    
    probs_c5 = train_eval_config(X_train_c5, y.iloc[:split_80].values, X_val_c5, y_val_80, categorical_cols)
    
    del X_train_c5
    del X_val_c5
    gc.collect()

    pr_d = evaluate_predictions(y_val_80, probs_d)
    pr_c5 = evaluate_predictions(y_val_80, probs_c5)
    
    print(f"\nModel D PR-AUC: {pr_d:.5f}")
    print(f"C5 PR-AUC:      {pr_c5:.5f}")
    print(f"Delta:           {pr_c5 - pr_d:+.5f}")

    # Run paired bootstrap
    print("\nRunning 1,000 paired bootstrap resamples on the 80/20 validation set...")
    mean_d, std_d, ci_l, ci_u = run_paired_bootstrap(y_val_80, probs_d, probs_c5, num_resamples=1000)
    
    print("\n" + "=" * 60)
    print("BOOTSTRAP SIGNIFICANCE STUDY METRICS:")
    print("=" * 60)
    print(f"Mean Delta (C5 - D):   {mean_d:+.5f}")
    print(f"Std Deviation:         {std_d:.5f}")
    print(f"95% Confidence Bounds: [{ci_l:+.5f}, {ci_u:+.5f}]")
    print("=" * 60)
    
    if ci_l > 0:
        print("VERDICT: C5 outpaces Model D by a statistically significant margin on the 80/20 split.")
    else:
        print("VERDICT: The difference between C5 and Model D is statistically uncertain at 95% confidence.")
    print("=" * 60)

if __name__ == '__main__':
    main()
