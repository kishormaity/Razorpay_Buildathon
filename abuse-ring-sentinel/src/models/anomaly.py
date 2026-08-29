import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))

from src.models.calibration import min_max_calibrate

def train_anomaly_detector(df):
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)
    
    # Chronological training split index (70%)
    split_idx = int(total_rows * 0.70)
    train_df = df.iloc[:split_idx]
    
    # Behavioral and velocity variables for isolation
    features = [
        'TransactionAmt',
        'card_tx_count_10m', 'card_tx_count_1h', 'card_tx_count_24h',
        'card_spend_sum_1h', 'card_spend_sum_24h', 'card_spend_mean_24h',
        'spend_ratio_24h', 'time_gap', 'time_gap_deviation_median'
    ]
    
    print(f"Fitting Isolation Forest on training split ({train_df.shape[0]} rows)...")
    X_train = train_df[features].copy()
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    
    # Load configuration
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    config_path = os.path.join(proj_root, "configs", "models.yaml")
    
    n_est = 100
    cont = "auto"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            m_config = yaml.safe_load(f)
            n_est = m_config["isolation_forest"].get("n_estimators", 100)
            cont = m_config["isolation_forest"].get("contamination", "auto")
            
    clf = IsolationForest(
        n_estimators=n_est,
        contamination=cont,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train)
    
    print("Scoring all transactions for velocity anomaly indices...")
    X_all = df[features].copy().fillna(medians)
    raw_anomaly_scores = clf.decision_function(X_all)
    
    # Min-max scale the decision values
    calibrated_risks = min_max_calibrate(raw_anomaly_scores, inverse=True)
    
    return clf, calibrated_risks

def main():
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    features_parquet = os.path.join(proj_root, "data", "processed", "features", "features.parquet")
    
    if not os.path.exists(features_parquet):
        print(f"[ERROR] Features file not found: {features_parquet}")
        sys.exit(1)
        
    df = pd.read_parquet(features_parquet)
    clf, calibrated_risks = train_anomaly_detector(df)
    print("Anomaly Isolation Forest training complete.")
    
    # Save the model pickle
    models_dir = os.path.join(proj_root, "models", "anomaly")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "isolation_forest.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
        
    # Save scores array for downstream stacking
    scores_path = os.path.join(models_dir, "calibrated_anomaly_risks.json")
    with open(scores_path, "w") as f:
        json.dump(calibrated_risks.tolist(), f)
        
    print(f"Isolation Forest model saved to: {model_path}")
    print(f"Calibrated anomaly scores saved to: {scores_path}")

if __name__ == "__main__":
    import json
    main()
