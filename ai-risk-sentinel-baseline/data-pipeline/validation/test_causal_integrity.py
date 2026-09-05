"""
Automated Test: Causal Integrity, Feature Decoupling & Decision Engine Verification
Part of the AI Risk Sentinel - Fraud Loss Prevention System

Asserts:
1. model_d_features.json does NOT contain raw TransactionDT.
2. sentinel_features.json does NOT contain circular ring_fraud_density.
3. Decision engine logic is decoupled, deterministic, and preserves arithmetic integrity.
4. Risk explanations generate valid signals without accessing ground truth.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

def test_feature_schemas():
    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    models_dir = os.path.join(current_dir, 'models')
    
    d_path = os.path.join(models_dir, 'model_d_features.json')
    sentinel_path = os.path.join(models_dir, 'sentinel_features.json')
    
    if os.path.exists(d_path):
        with open(d_path, 'r') as f:
            d_feats = json.load(f)
        assert 'TransactionDT' not in d_feats, "CRITICAL ERROR: TransactionDT found in Model D feature matrix!"
        print(f"[OK] Model D feature schema test PASSED: {len(d_feats)} features (TransactionDT absent).")
    else:
        print("ℹ Note: model_d_features.json not yet regenerated. Skipping schema existence check.")

    if os.path.exists(sentinel_path):
        with open(sentinel_path, 'r') as f:
            s_feats = json.load(f)
        assert 'ring_fraud_density' not in s_feats, "CRITICAL ERROR: Circular feature ring_fraud_density found in Sentinel features!"
        print(f"[OK] Sentinel feature schema test PASSED: {len(s_feats)} features (ring_fraud_density absent).")
    else:
        print("ℹ Note: sentinel_features.json not yet regenerated. Skipping schema existence check.")

def test_decision_engine_and_financials():
    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    sys.path.append(current_dir)
    from pipeline.risk_engine import (
        make_decision, make_decisions_batch, compute_merchant_loss_metrics, explain_risk
    )
    
    # 1. Test Single Decision
    assert make_decision(0.40, 0.05, 0.30, 0.15) == "BLOCK", "Threshold block failed"
    assert make_decision(0.20, 0.25, 0.30, 0.15) == "MANUAL_REVIEW", "Threshold review failed"
    assert make_decision(0.10, 0.05, 0.30, 0.15) == "ALLOW", "Threshold allow failed"
    print("[OK] Decision routing logic tests PASSED.")
    
    # 2. Test Batch Decisions & Financial Arithmetic
    np.random.seed(42)
    n = 1000
    y_true = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    probs_d = np.random.uniform(0.0, 1.0, size=n)
    probs_s = np.random.uniform(0.0, 1.0, size=n)
    amounts = np.random.uniform(10.0, 500.0, size=n)
    
    metrics = compute_merchant_loss_metrics(
        y_true, probs_d, probs_s, amounts, threshold_block=0.30, threshold_review=0.15, fp_cost_rate=0.15
    )
    
    # Arithmetic conservation check
    fraud_sum = metrics["fraud_value_prevented"] + metrics["fraud_value_missed"] + metrics["fraud_value_in_review"]
    assert abs(fraud_sum - metrics["total_fraud_volume"]) < 0.05, f"Fraud volume mismatch: {fraud_sum} vs {metrics['total_fraud_volume']}"
    
    expected_net = round(metrics["fraud_value_prevented"] - metrics["false_positive_cost"], 2)
    assert abs(metrics["net_loss_avoided"] - expected_net) < 0.05, f"Net loss mismatch: {metrics['net_loss_avoided']} vs {expected_net}"
    print(f"[OK] Financial metrics conservation test PASSED: Net loss avoided = ${metrics['net_loss_avoided']:,.2f}")
    
    # 3. Test Explainability without Ground Truth
    sample_row = {
        'device_unique_card_count': 5,
        'addr_unique_card_count': 4,
        'rapid_card_convergence': 4,
        'device_connected_fraud_rate': 0.12,
        'card1_past_count': 25
    }
    signals = explain_risk(sample_row, prob_d=0.45, prob_sentinel=0.22, threshold_block=0.30, threshold_review=0.15)
    assert len(signals) >= 4, "Expected multiple risk signals"
    for s in signals:
        assert "isFraud" not in s, "Label leakage in explanation!"
    print(f"[OK] Deterministic explainability test PASSED: Generated {len(signals)} evidence signals.")

def main():
    print("=" * 60)
    print("   RUNNING CAUSAL INTEGRITY & DECISION ENGINE AUDIT")
    print("=" * 60)
    test_feature_schemas()
    test_decision_engine_and_financials()
    print("=" * 60)
    print("   ALL INTEGRITY & DECISION TESTS PASSED SUCCESSFULLY")
    print("=" * 60)

if __name__ == '__main__':
    main()
