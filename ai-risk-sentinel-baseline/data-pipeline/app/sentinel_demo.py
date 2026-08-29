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
    return model

def render_dashboard(row, prob_d, prob_sentinel, threshold_d, threshold_sentinel):
    tx_id = int(row['TransactionID'])
    amt = float(row['TransactionAmt'])
    card_id = int(row['card1'])
    email = str(row['P_emaildomain']) if not pd.isna(row['P_emaildomain']) else 'N/A'
    device = str(row['DeviceInfo']) if is_valid_device(row['DeviceInfo']) else 'N/A'
    
    is_fraud = int(row['isFraud'])
    is_proxy = int(row['is_ring_abuse'])
    
    # Decisions
    d_decision = "BLOCK ❌" if prob_d >= threshold_d else "ALLOW  "
    sentinel_decision = "ROUTE TO SECONDARY REVIEW 🔍" if prob_sentinel >= threshold_sentinel else "AUTO-APPROVE ✅"
    
    # Final Flow Status
    flow_status = ""
    if prob_d >= threshold_d:
        flow_status = "TRANSACTION BLOCKED (Directly by Model D)"
    elif prob_sentinel >= threshold_sentinel:
        flow_status = "SENT TO MANUAL REVIEW (Model D Allowed, Sentinel Flagged Network Risk)"
    else:
        flow_status = "TRANSACTION APPROVED & CAPTURED"

    print("\n" + "=" * 80)
    print("                 IEEE-CIS FRAUD PIPELINE: DECISION DASHBOARD")
    print("=" * 80)
    print(f" TRANSACTION ID: {tx_id:<15} | CARD ID: {card_id:<15} | AMOUNT: ${amt:.2f}")
    print(f" EMAIL DOMAIN:   {email:<15} | DEVICE INFO: {device:<20}")
    print("-" * 80)
    print(f" [STAGE 1] TRANSACTION RISK (Model D)")
    print(f"   * Raw Fraud Score:  {prob_d:.5f}  (Decision Bound: {threshold_d:.5f})")
    print(f"   * Stage 1 Decision: {d_decision}")
    
    if prob_d < threshold_d:
        print("-" * 80)
        print(f" [STAGE 2] COORDINATED NETWORK RISK (Abuse-Ring Sentinel)")
        print(f"   * Raw Network Risk Score: {prob_sentinel:.5f}  (Review Bound: {threshold_sentinel:.5f})")
        print(f"   * Stage 2 Decision:       {sentinel_decision}")
        
    print("-" * 80)
    print(" NETWORK METADATA & EVIDENCE PORTFOLIO (Strictly Look-Back):")
    print(f"   * Device Connected Cards Count:  {int(row['device_unique_card_count']):,}")
    print(f"   * Address Connected Cards Count: {int(row['addr_unique_card_count']):,}")
    print(f"   * Device Connected Fraud Rate:   {row['device_connected_fraud_rate'] * 100:.2f}%")
    print(f"   * Address Connected Fraud Rate:  {row['addr_connected_fraud_rate'] * 100:.2f}%")
    print(f"   * 72h Coordinated Convergence:   {int(row['rapid_card_convergence']):,} cards")
    print(f"   * Cross-Entity Hub Overlap:      {'YES (Card shares dev+addr with >=2 cards)' if row['cross_entity_convergence'] > 0 else 'NO'}")
    print(f"   * Ring Connected Fraud Density:  {row['ring_fraud_density'] * 100:.2f}%")
    print("-" * 80)
    print(f" ACTUAL GROUND TRUTH STATE:")
    print(f"   * Is Fraudulent Transaction?     {'YES 🚨' if is_fraud == 1 else 'NO '}")
    print(f"   * Weak Supervision Proxy Flag?  {'ACTIVE (R5)' if is_proxy == 1 else 'INACTIVE'}")
    print("-" * 80)
    print(f" PIPELINE WORKFLOW STATUS:")
    print(f" >>> ** {flow_status} ** <<<")
    print("=" * 80)

    # Narrative explanation
    narrative = ""
    if prob_d >= threshold_d:
        narrative = f"This transaction exhibits extreme single-transaction risk profile (Model D Score: {prob_d:.4f}), triggering immediate automated blocking at the checkout gateway."
    elif prob_sentinel >= threshold_sentinel:
        narrative = f"Although Model D score ({prob_d:.4f}) is below block threshold, the card is transacting on a subnet (address) and device cluster that exhibits coordinated abuse-ring patterns: {int(row['device_unique_card_count'])} unique cards are shared on the device, with a {row['device_connected_fraud_rate']*100:.1f}% historical fraud exposure. The Sentinel overrides the auto-approve and routes it for review."
    else:
        narrative = "The transaction presents clean baseline behavior and does not show signs of coordinated device or address sharing. It is auto-approved."
        
    print(f"RATIONALE: {narrative}")
    print("=" * 80 + "\n")

def main():
    print("=" * 70)
    print("      IEEE-CIS PHASE 15B: ABUSE-RING SENTINEL INTERACTIVE DEMO")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/abuse_ring_features.parquet')

    if not os.path.exists(features_path):
        print(f"[ERROR] Features file not found.")
        sys.exit(1)

    # 1. Load and Align
    print("Initializing demo framework and loading dataset...")
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
    print("Training Model D GBDT...")
    X_train_d = df.iloc[:split_70][base_features]
    y_train_d = df.iloc[:split_70]['isFraud'].values
    X_dev_d = df.iloc[split_70:split_85][base_features]
    y_dev_d = df.iloc[split_70:split_85]['isFraud'].values
    categorical_cols = list(X_train_d.select_dtypes(include=['category']).columns)
    
    model_d = train_lgb(X_train_d, y_train_d, X_dev_d, y_dev_d, categorical_cols)
    
    # Train Sentinel
    print("Training Sentinel GBDT...")
    X_train_sentinel = df.iloc[:split_70][ring_features]
    y_train_sentinel = df.iloc[:split_70]['is_ring_abuse'].values
    X_dev_sentinel = df.iloc[split_70:split_85][ring_features]
    y_dev_sentinel = df.iloc[split_70:split_85]['is_ring_abuse'].values
    
    model_sentinel = train_lgb(X_train_sentinel, y_train_sentinel, X_dev_sentinel, y_dev_sentinel, categorical_cols=[])

    # Extract test rows
    test_df = df.iloc[split_85:].copy().reset_index(drop=True)
    X_test_d = test_df[base_features]
    X_test_sentinel = test_df[ring_features]
    
    probs_d = model_d.predict(X_test_d, num_iteration=model_d.best_iteration)
    probs_sentinel = model_sentinel.predict(X_test_sentinel, num_iteration=model_sentinel.best_iteration)
    
    test_df['prob_d'] = probs_d
    test_df['prob_sentinel'] = probs_sentinel

    # Clean up memory
    del X_train_d, X_dev_d, X_train_sentinel, X_dev_sentinel
    gc.collect()

    # Pre-select samples for the interactive demo
    threshold_d = 0.30398
    threshold_sentinel = 0.15

    # 1. Clean Approved
    clean_mask = (test_df['isFraud'] == 0) & (test_df['prob_d'] < 0.10) & (test_df['prob_sentinel'] < 0.15)
    clean_sample = test_df[clean_mask].head(1).iloc[0]

    # 2. Directly Blocked
    blocked_mask = (test_df['isFraud'] == 1) & (test_df['prob_d'] >= threshold_d)
    blocked_sample = test_df[blocked_mask].head(1).iloc[0]

    # 3. Model D Miss -> Sentinel Review (Coordinated Risk captured)
    recovered_mask = (test_df['isFraud'] == 1) & (test_df['prob_d'] < threshold_d) & (test_df['prob_sentinel'] >= threshold_sentinel)
    recovered_sample = test_df[recovered_mask].head(1).iloc[0]

    # 4. Sentinel Flagged Clean (Review Friction)
    friction_mask = (test_df['isFraud'] == 0) & (test_df['prob_d'] < threshold_d) & (test_df['prob_sentinel'] >= threshold_sentinel)
    friction_sample = test_df[friction_mask].head(1).iloc[0]

    samples = {
        '1': ('Model D Miss -> Routed to Review (Coordinated Abuse Ring Pattern)', recovered_sample),
        '2': ('Directly Blocked by Model D (High Transaction Risk)', blocked_sample),
        '3': ('Clean Approved Checkout (Low Risk)', clean_sample),
        '4': ('Clean Transaction Flagged for Review (Friction/Investigation case)', friction_sample)
    }

    # Interactive Loop
    while True:
        print("\n" + "=" * 70)
        print(" CHOOSE A SCENARIO TO DEMONSTRATE:")
        print("=" * 70)
        for key, (desc, row) in samples.items():
            print(f"  [{key}] {desc} (ID: {int(row['TransactionID'])})")
        print("  [q] Quit Demo")
        print("-" * 70)
        
        choice = input("Enter choice: ").strip()
        if choice.lower() == 'q':
            print("\nExiting Sentinel demo. Thank you!")
            break
            
        if choice in samples:
            desc, row = samples[choice]
            render_dashboard(row, row['prob_d'], row['prob_sentinel'], threshold_d, threshold_sentinel)
        else:
            print("Invalid choice, please select again.")

if __name__ == '__main__':
    main()
