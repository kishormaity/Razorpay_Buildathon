import os
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))

from src.data.connection import get_connection
from src.evaluation.metrics import evaluate_predictions

def load_config():
    config_path = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "configs", "models.yaml"))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_fusion():
    print("=" * 70)
    print("           ABUSE-RING SENTINEL MULTI-MODEL RISK FUSION")
    print("=" * 70)
    
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    
    # 1. Load features matrix
    features_parquet = os.path.join(proj_root, "data", "processed", "features", "features.parquet")
    if not os.path.exists(features_parquet):
        print(f"[ERROR] Features file not found at: {features_parquet}")
        sys.exit(1)
        
    df = pd.read_parquet(features_parquet)
    print(f"Loaded feature store matrix: {df.shape}")
    
    # Sort chronologically
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    # 2. Get predictions from each layer
    print("Compiling component risk predictions...")
    
    # A. GBDT predictions
    booster_path = os.path.join(proj_root, "models", "lightgbm", "sentinel_gbm_booster.txt")
    features_json = os.path.join(proj_root, "models", "lightgbm", "model_d_features.json")
    
    if not os.path.exists(booster_path) or not os.path.exists(features_json):
        print("[ERROR] LightGBM model booster not found. Run lightgbm_model.py first.")
        sys.exit(1)
        
    model = lgb.Booster(model_file=booster_path)
    with open(features_json, "r") as f:
        cols = json.load(f)
        
    # Prepare categories
    X_all = df[cols].copy()
    for col in ['ProductCD', 'addr2', 'P_emaildomain']:
        if col in X_all.columns:
            X_all[col] = X_all[col].astype('category')
            
    df['r_gbm'] = model.predict(X_all)
    
    # B. GraphSAGE predictions
    sage_path = os.path.join(proj_root, "models", "graphsage", "graphsage_user_risks.json")
    if not os.path.exists(sage_path):
        print("[ERROR] GraphSAGE user risks not found. Run graphsage.py first.")
        sys.exit(1)
        
    with open(sage_path, "r") as f:
        sage_risks = json.load(f)
    df['r_gnn'] = df['user_id'].map(sage_risks).fillna(0.10).astype(float)
    
    # C. Isolation Forest predictions
    anomaly_path = os.path.join(proj_root, "models", "anomaly", "calibrated_anomaly_risks.json")
    if not os.path.exists(anomaly_path):
        print("[ERROR] Anomaly scores not found. Run anomaly.py first.")
        sys.exit(1)
        
    with open(anomaly_path, "r") as f:
        anomaly_risks = json.load(f)
    df['r_anomaly'] = np.array(anomaly_risks, dtype=float)
    
    # D. Ring Scorer predictions
    ring_path = os.path.join(proj_root, "data", "processed", "ring_risk_scores.json")
    comm_json_path = os.path.join(proj_root, "data", "processed", "champion_communities.json")
    if not os.path.exists(ring_path) or not os.path.exists(comm_json_path):
        print("[ERROR] Ring scores not found. Run ring_detector.py first.")
        sys.exit(1)
        
    with open(comm_json_path, "r") as f:
        comm_data = json.load(f)
    node_to_comm = comm_data["node_to_comm"]
    
    with open(ring_path, "r") as f:
        ring_scores = json.load(f)
        
    user_to_ring_score = {}
    for user_id, comm_idx in node_to_comm.items():
        comm_str = str(comm_idx)
        user_to_ring_score[user_id] = ring_scores.get(comm_str, {}).get('score', 0.05)
        
    df['r_ring'] = df['user_id'].map(user_to_ring_score).fillna(0.05).astype(float)
    
    # 3. Set chronological splits
    data_config_path = os.path.join(proj_root, "configs", "data.yaml")
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
        
    train_ratio = data_config["splits"]["train_ratio"]
    val_ratio = data_config["splits"]["val_ratio"]
    
    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    print(f"Chronological Splits compiled for Stacking.")
    
    # ----------------------------------------------------
    # Strategy A: Weighted Blended Average
    # ----------------------------------------------------
    print("\n[Strategy A] Evaluating Weighted Average Fusion...")
    m_config = load_config()["risk_fusion"]
    weights = m_config["weights"]
    w_gbm = weights["w_gbm"]
    w_gnn = weights["w_gnn"]
    w_anomaly = weights["w_anomaly"]
    w_ring = weights["w_ring"]
    
    def compute_weighted_blend(split):
        return (w_gbm * split['r_gbm'] + 
                w_gnn * split['r_gnn'] + 
                w_anomaly * split['r_anomaly'] + 
                w_ring * split['r_ring'])
                
    test_preds_blend = compute_weighted_blend(test_df).values
    metrics_blend = evaluate_predictions(test_df['isFraud'].values, test_preds_blend)
    
    # ----------------------------------------------------
    # Strategy B: Logistic Regression Stacking Meta-Model
    # ----------------------------------------------------
    print("[Strategy B] Training Logistic Stacking Meta-Model on Validation split...")
    features_list = ['r_gbm', 'r_gnn', 'r_anomaly', 'r_ring']
    
    X_val = val_df[features_list].values
    y_val = val_df['isFraud'].values
    X_test = test_df[features_list].values
    y_test = test_df['isFraud'].values
    
    stacker = LogisticRegression(penalty=None, solver='lbfgs')
    stacker.fit(X_val, y_val)
    
    test_preds_stack = stacker.predict_proba(X_test)[:, 1]
    metrics_stack = evaluate_predictions(y_test, test_preds_stack)
    
    print("\n" + "=" * 70)
    print("                 FUSION STRATEGY BENCHMARK")
    print("=" * 70)
    print(f"Metric        | Strategy A (Weighted) | Strategy B (Stacking) | Improvement")
    print("-" * 70)
    pr_auc_diff = metrics_stack['pr_auc'] - metrics_blend['pr_auc']
    roc_auc_diff = metrics_stack['roc_auc'] - metrics_blend['roc_auc']
    print(f"PR-AUC        | {metrics_blend['pr_auc']:.5f}                 | {metrics_stack['pr_auc']:.5f}                  | {pr_auc_diff:+.5f}")
    print(f"ROC-AUC       | {metrics_blend['roc_auc']:.5f}                 | {metrics_stack['roc_auc']:.5f}                  | {roc_auc_diff:+.5f}")
    print(f"Best F1 Score | {metrics_blend['best_f1']:.5f}                 | {metrics_stack['best_f1']:.5f}                  | {metrics_stack['best_f1'] - metrics_blend['best_f1']:+.5f}")
    print(f"FPR @ F1      | {metrics_blend['fpr']:.5f}                 | {metrics_stack['fpr']:.5f}                  | {metrics_stack['fpr'] - metrics_blend['fpr']:+.5f}")
    print("=" * 70)
    
    # Stacking outperforms or matches, and offers learned weight vectors. Let's use it as the champion!
    # Apply to all splits and save final parquet predictions
    print("Generating stacked predictions for the entire dataset...")
    X_all = df[features_list].values
    df['r_final'] = stacker.predict_proba(X_all)[:, 1]
    
    predictions_dir = os.path.join(proj_root, "data", "processed", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    out_path = os.path.join(predictions_dir, "sentinel_fused_preds.parquet")
    df[[
        'TransactionID', 'TransactionAmt', 'TransactionDT', 'user_id', 'isFraud', 
        'r_gbm', 'r_gnn', 'r_anomaly', 'r_ring', 'r_final'
    ]].to_parquet(out_path, engine="pyarrow", index=False)
    
    # Save stacker model pkl
    models_dir = os.path.join(proj_root, "models", "fusion")
    os.makedirs(models_dir, exist_ok=True)
    stacker_path = os.path.join(models_dir, "fusion_stacker.pkl")
    with open(stacker_path, "wb") as f:
        import pickle
        pickle.dump(stacker, f)
        
    print(f"Final predictions saved to: {out_path}")
    print(f"Fusion stacking model saved to: {stacker_path}")

if __name__ == "__main__":
    run_fusion()
