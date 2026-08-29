import os
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))

from src.evaluation.metrics import evaluate_predictions

def load_config():
    config_path = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "configs", "data.yaml"))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_lightgbm_training():
    print("=" * 70)
    print("          ABUSE-RING SENTINEL LIGHTGBM V2 TRAINING")
    print("=" * 70)
    
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    features_parquet = os.path.join(proj_root, "data", "processed", "features", "features.parquet")
    
    if not os.path.exists(features_parquet):
        print(f"[ERROR] Features file not found at: {features_parquet}")
        sys.exit(1)
        
    df = pd.read_parquet(features_parquet)
    print(f"Loaded features matrix: {df.shape}")
    
    # 1. Set chronological train / val / test indices
    config = load_config()
    train_ratio = config["splits"]["train_ratio"]
    val_ratio = config["splits"]["val_ratio"]
    
    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    target = 'isFraud'
    
    # Define feature lists
    tabular_features = [
        'TransactionAmt', 'card1', 'card2', 'addr1',
        'card_tx_count_10m', 'card_tx_count_1h', 'card_tx_count_24h',
        'card_spend_sum_1h', 'card_spend_sum_24h', 'card_spend_mean_24h',
        'spend_ratio_24h', 'card_time_since_prev', 'is_new_device', 'is_new_location',
        'time_gap', 'time_gap_deviation_median', 'time_gap_deviation_mean',
        'time_gap_acceleration_median'
    ]
    
    graph_features = [
        'card_device_degree', 'card_addr_degree', 'device_card_degree', 'addr_card_degree',
        'shared_device_card_count', 'shared_addr_card_count', 
        'device_connected_fraud_rate', 'addr_connected_fraud_rate',
        'network_risk_mean', 'network_risk_max', 'network_risk_gap', 'network_risk_product',
        'device_card_novelty', 'addr_card_novelty', 'pagerank_centrality', 'clustering_coefficient'
    ]
    
    # Handle category encoding
    for col in ['ProductCD', 'addr2', 'P_emaildomain']:
        for split in [train_df, val_df, test_df]:
            split[col] = split[col].astype('category')
        tabular_features.append(col)
        
    print(f"Tabular features: {len(tabular_features)} columns")
    print(f"Graph topological features: {len(graph_features)} columns")
    
    # Model parameters
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    
    y_train = train_df[target].values
    y_val = val_df[target].values
    y_test = test_df[target].values
    
    # ----------------------------------------------------
    # Experiment B: Tabular Only
    # ----------------------------------------------------
    print("\n[Experiment B] Training Tabular-Only GBDT baseline...")
    X_train_tab = train_df[tabular_features]
    X_val_tab = val_df[tabular_features]
    X_test_tab = test_df[tabular_features]
    
    train_data_tab = lgb.Dataset(X_train_tab, label=y_train)
    val_data_tab = lgb.Dataset(X_val_tab, label=y_val, reference=train_data_tab)
    
    model_tab = lgb.train(
        params,
        train_data_tab,
        num_boost_round=500,
        valid_sets=[train_data_tab, val_data_tab],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    val_preds_tab = model_tab.predict(X_val_tab)
    test_preds_tab = model_tab.predict(X_test_tab)
    
    metrics_val_tab = evaluate_predictions(y_val, val_preds_tab)
    metrics_test_tab = evaluate_predictions(y_test, test_preds_tab)
    
    # ----------------------------------------------------
    # Experiment C: Tabular + Graph
    # ----------------------------------------------------
    print("\n[Experiment C] Training Tabular + Graph GBDT champion...")
    combined_features = tabular_features + graph_features
    X_train_comb = train_df[combined_features]
    X_val_comb = val_df[combined_features]
    X_test_comb = test_df[combined_features]
    
    train_data_comb = lgb.Dataset(X_train_comb, label=y_train)
    val_data_comb = lgb.Dataset(X_val_comb, label=y_val, reference=train_data_comb)
    
    model_comb = lgb.train(
        params,
        train_data_comb,
        num_boost_round=500,
        valid_sets=[train_data_comb, val_data_comb],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    val_preds_comb = model_comb.predict(X_val_comb)
    test_preds_comb = model_comb.predict(X_test_comb)
    
    metrics_val_comb = evaluate_predictions(y_val, val_preds_comb)
    metrics_test_comb = evaluate_predictions(y_test, test_preds_comb)
    
    # Save the trained combined champion model
    models_dir = os.path.join(proj_root, "models", "lightgbm")
    os.makedirs(models_dir, exist_ok=True)
    
    booster_path = os.path.join(models_dir, "sentinel_gbm_booster.txt")
    model_comb.save_model(booster_path)
    
    # Save columns json for API inference feature alignment
    features_json = os.path.join(models_dir, "model_d_features.json")
    with open(features_json, "w") as f:
        json.dump(combined_features, f)
        
    print("\n" + "=" * 70)
    print("                 GBDT COMPARATIVE PERFORMANCE")
    print("=" * 70)
    print(f"Metric        | Tabular Val  | Hybrid Val   | Tabular Test | Hybrid Test")
    print("-" * 70)
    print(f"PR-AUC        | {metrics_val_tab['pr_auc']:.5f}      | {metrics_val_comb['pr_auc']:.5f}      | {metrics_test_tab['pr_auc']:.5f}      | {metrics_test_comb['pr_auc']:.5f}")
    print(f"ROC-AUC       | {metrics_val_tab['roc_auc']:.5f}      | {metrics_val_comb['roc_auc']:.5f}      | {metrics_test_tab['roc_auc']:.5f}      | {metrics_test_comb['roc_auc']:.5f}")
    print(f"Best F1 Score | {metrics_val_tab['best_f1']:.5f}      | {metrics_val_comb['best_f1']:.5f}      | {metrics_test_tab['best_f1']:.5f}      | {metrics_test_comb['best_f1']:.5f}")
    print(f"FPR @ F1      | {metrics_val_tab['fpr']:.5f}      | {metrics_val_comb['fpr']:.5f}      | {metrics_test_tab['fpr']:.5f}      | {metrics_test_comb['fpr']:.5f}")
    print("=" * 70)
    print(f"Champion booster saved to: {booster_path}")
    print(f"Features mapping file saved to: {features_json}")
    
    # Return predictions for downstream fusion tests
    return {
        'val_preds_tab': val_preds_tab,
        'test_preds_tab': test_preds_tab,
        'val_preds_comb': val_preds_comb,
        'test_preds_comb': test_preds_comb,
        'test_y': y_test
    }

if __name__ == "__main__":
    run_lightgbm_training()
