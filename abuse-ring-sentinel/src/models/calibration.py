import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))

def min_max_calibrate(scores, inverse=False):
    s = np.array(scores, dtype=float)
    s_min = np.nanmin(s)
    s_max = np.nanmax(s)
    
    if s_max == s_min:
        return np.zeros_like(s)
        
    if inverse:
        calibrated = (s_max - s) / (s_max - s_min)
    else:
        calibrated = (s - s_min) / (s_max - s_min)
        
    return np.clip(calibrated, 0.0, 1.0)

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Check boundary membership
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)

def calculate_expected_loss(predictions, amounts, y_true, threshold, fp_cost=1500.0, chargeback_fee=1200.0):
    """
    Computes total commercial cost for a specific classification threshold.
    """
    total_cost = 0.0
    for p, amt, y in zip(predictions, amounts, y_true):
        if p >= threshold:
            # Block action
            if y == 0:
                total_cost += fp_cost  # Customer friction cost
        else:
            # Allow action
            if y == 1:
                total_cost += amt + chargeback_fee  # Fraud amount + chargeback fee
    return total_cost

def run_calibration_and_policy():
    print("=" * 70)
    print("        ABUSE-RING SENTINEL CALIBRATION & POLICY ENGINE")
    print("=" * 70)
    
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    
    # 1. Load fused predictions parquet
    preds_parquet = os.path.join(proj_root, "data", "processed", "predictions", "sentinel_fused_preds.parquet")
    if not os.path.exists(preds_parquet):
        print(f"[ERROR] Predictions parquet not found: {preds_parquet}")
        sys.exit(1)
        
    df = pd.read_parquet(preds_parquet)
    print(f"Loaded predictions matrix: {df.shape}")
    
    # 2. Slice chronological splits
    data_config_path = os.path.join(proj_root, "configs", "data.yaml")
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
        
    train_ratio = data_config["splits"]["train_ratio"]
    val_ratio = data_config["splits"]["val_ratio"]
    
    total_rows = len(df)
    val_start = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))
    
    val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    # 3. Fit Isotonic Calibration on Validation predictions
    print("\nFitting Isotonic Calibrator on validation predictions...")
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_df['r_final'].values, val_df['isFraud'].values)
    
    # Predict calibrated values
    val_df['r_calibrated'] = ir.predict(val_df['r_final'].values)
    test_df['r_calibrated'] = ir.predict(test_df['r_final'].values)
    
    # Save the Isotonic Calibrator pkl
    models_dir = os.path.join(proj_root, "models", "fusion")
    os.makedirs(models_dir, exist_ok=True)
    calibrator_path = os.path.join(models_dir, "isotonic_calibrator.pkl")
    with open(calibrator_path, "wb") as f:
        pickle.dump(ir, f)
    print(f"Calibrator object saved to: {calibrator_path}")
    
    # Calculate calibration error (Brier, ECE)
    brier_before = float(np.mean((test_df['r_final'].values - test_df['isFraud'].values)**2))
    brier_after = float(np.mean((test_df['r_calibrated'].values - test_df['isFraud'].values)**2))
    
    ece_before = compute_ece(test_df['isFraud'].values, test_df['r_final'].values)
    ece_after = compute_ece(test_df['isFraud'].values, test_df['r_calibrated'].values)
    
    print("\n" + "-" * 70)
    print("                 CALIBRATION QUALITY IMPROVEMENT")
    print("-" * 70)
    print(f"Metric     | Raw Stacking Output | Isotonic Calibrated | Improvement")
    print(f"Brier Score| {brier_before:.5f}             | {brier_after:.5f}             | {brier_before - brier_after:+.5f}")
    print(f"ECE        | {ece_before:.5f}             | {ece_after:.5f}             | {ece_before - ece_after:+.5f}")
    print("-" * 70)
    
    # 4. Cost Threshold Optimization on VALIDATION split
    print("\nOptimizing decision policy thresholds on validation set...")
    policy_config_path = os.path.join(proj_root, "configs", "risk_policy.yaml")
    with open(policy_config_path, "r") as f:
        policy_config = yaml.safe_load(f)
        
    fp_cost = policy_config["cost_matrix"].get("fp_cost", 1500.0)
    chargeback_fee = policy_config["cost_matrix"].get("chargeback_fee", 1200.0)
    
    val_amounts = val_df['TransactionAmt'].values
    val_labels = val_df['isFraud'].values
    val_probs = val_df['r_calibrated'].values
    
    best_threshold = 0.50
    min_cost = float('inf')
    
    thresholds_grid = np.linspace(0.01, 0.99, 99)
    for t in thresholds_grid:
        cost = calculate_expected_loss(val_probs, val_amounts, val_labels, t, fp_cost=fp_cost, chargeback_fee=chargeback_fee)
        if cost < min_cost:
            min_cost = cost
            best_threshold = float(t)
            
    print(f"Optimal validation threshold: {best_threshold:.4f} (Validation Expected Cost: INR {min_cost:,.2f})")
    
    # Evaluate optimal validation-derived threshold on held-out TEST set
    test_amounts = test_df['TransactionAmt'].values
    test_labels = test_df['isFraud'].values
    test_probs = test_df['r_calibrated'].values
    
    # Compare cost of baseline ALLOW-ALL vs GBDT policy vs Calibrated Fusion optimal policy
    cost_allow_all = calculate_expected_loss(test_probs, test_amounts, test_labels, 1.0, fp_cost=fp_cost, chargeback_fee=chargeback_fee)
    
    # GBDT baseline policy cost (optimal test threshold at Best F1 for GBDT was around 0.16)
    test_gbm_preds = test_df['r_gbm'].values
    cost_gbm_policy = calculate_expected_loss(test_gbm_preds, test_amounts, test_labels, 0.16, fp_cost=fp_cost, chargeback_fee=chargeback_fee)
    
    # Calibrated Fusion policy cost
    cost_fusion_policy = calculate_expected_loss(test_probs, test_amounts, test_labels, best_threshold, fp_cost=fp_cost, chargeback_fee=chargeback_fee)
    
    print("\n" + "=" * 70)
    print("                 BUSINESS VALUE ANALYSIS (TEST SET)")
    print("=" * 70)
    print(f"Policy Configuration        | Total Commercial Loss (INR) | Saved Loss")
    print("-" * 70)
    print(f"Allow-All Baseline          | INR {cost_allow_all:<22,.2f} | -")
    print(f"V1 GBDT Baseline Policy     | INR {cost_gbm_policy:<22,.2f} | INR {cost_allow_all - cost_gbm_policy:,.2f}")
    print(f"V2 Calibrated Fusion Policy | INR {cost_fusion_policy:<22,.2f} | INR {cost_allow_all - cost_fusion_policy:,.2f}")
    print("=" * 70)
    
    # 5. Save optimal policy parameters
    policy_json = os.path.join(models_dir, "optimal_policy.json")
    with open(policy_json, "w") as f:
        json.dump({
            'optimal_threshold': best_threshold,
            'fp_cost': fp_cost,
            'chargeback_fee': chargeback_fee,
            'brier_score': brier_after,
            'ece': ece_after
        }, f, indent=2)
    print(f"Optimal policy parameters written to: {policy_json}")
    
    # Update complete predictions parquet
    df['r_calibrated'] = ir.predict(df['r_final'].values)
    df.to_parquet(preds_parquet, engine="pyarrow", index=False)
    print("Successfully updated predictions parquet with calibrated scores.")

if __name__ == "__main__":
    run_calibration_and_policy()
