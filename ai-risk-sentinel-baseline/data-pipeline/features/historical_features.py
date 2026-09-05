import os
import sys
import time
import pandas as pd
import numpy as np

def compute_expanding_counts(df, group_cols):
    """
    Computes chronological cumulative count of group key occurrences
    strictly before the current row (look-back only).
    """
    counts = np.zeros(len(df))
    # df is already chronologically sorted, so grouped indices are sorted.
    for _, indices in df.groupby(group_cols).groups.items():
        idx_arr = np.array(indices)
        counts[idx_arr] = np.arange(len(idx_arr))
    return counts

def compute_expanding_target_rates(df_train, group_col, prior, m):
    """
    Computes expanding, Bayesian-smoothed target fraud rates on the training set
    strictly looking at past rows (0 to i-1) to avoid look-ahead leakage.
    """
    rates = np.zeros(len(df_train))
    targets = df_train['isFraud'].values
    
    for _, indices in df_train.groupby(group_col).groups.items():
        idx_arr = np.array(indices)
        group_size = len(idx_arr)
        
        # Cumulative fraud cases strictly before current index
        group_targets = targets[idx_arr]
        past_fraud_cases = np.cumsum(group_targets) - group_targets
        
        # Cumulative transaction count strictly before current index
        past_transactions = np.arange(group_size)
        
        # Bayesian smoothed rate
        rates[idx_arr] = (past_fraud_cases + prior * m) / (past_transactions + m)
        
    return rates

def build_train_lookup(df_train, group_col, prior, m):
    """
    Calculates final cumulative smoothed fraud rates at the end of the training set
    to use as a static lookup for the validation set.
    """
    agg = df_train.groupby(group_col)['isFraud'].agg(['count', 'sum'])
    agg['smoothed_rate'] = (agg['sum'] + prior * m) / (agg['count'] + m)
    return agg['smoothed_rate'].to_dict()

def main():
    print("=" * 70)
    print("       IEEE-CIS LEAKAGE-FREE HISTORICAL FEATURE ENGINEERING")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    features_path = os.path.join(processed_dir, 'features/transaction_features.parquet')
    output_path = os.path.join(processed_dir, 'features/historical_features.parquet')

    # 1. Load Dataset
    print("\n[1/5] Loading preprocessed transaction features...")
    if not os.path.exists(features_path):
        print(f"[ERROR] Transaction features parquet not found at: {features_path}")
        sys.exit(1)

    start_time = time.time()
    df = pd.read_parquet(features_path)
    print(f"Loaded dataset: {df.shape} in {time.time() - start_time:.2f} seconds.")

    # 2. Chronological Sorting
    print("\n[2/5] Preparing temporal index (sorting TransactionDT)...")
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    total_rows = len(df)
    # Align strictly to the 70% train split (413,378 rows) to prevent cross-split target leakage
    split_idx = int(total_rows * 0.7)
    
    # 3. Create Temporary Combo Columns for Complex Entities
    print("\n[3/5] Constructing combination keys...")
    df['temp_addr'] = df['addr1'].astype('string').fillna('-999')
    df['temp_device'] = df['DeviceInfo'].astype('string').fillna('UNKNOWN')
    df['temp_email'] = df['P_emaildomain'].astype('string').fillna('MISSING')
    
    df['card_addr_combo'] = df['card1'].astype(str) + "_" + df['temp_addr']
    df['card_device_combo'] = df['card1'].astype(str) + "_" + df['temp_device']
    df['card_email_combo'] = df['card1'].astype(str) + "_" + df['temp_email']

    # 4. Compute Frequency Features (Cumulative Past Counts)
    print("\n[4/5] Computing cumulative transaction counts (expanding look-back)...")
    t0 = time.time()
    
    # Past counts are calculated over the ENTIRE dataset chronologically
    df['card1_past_count'] = compute_expanding_counts(df, 'card1')
    df['card_addr_past_count'] = compute_expanding_counts(df, 'card_addr_combo')
    df['card_device_past_count'] = compute_expanding_counts(df, 'card_device_combo')
    df['card_email_past_count'] = compute_expanding_counts(df, 'card_email_combo')
    
    print(f"  * Engineered 4 past count features in {time.time() - t0:.2f} seconds.")

    # 5. Compute Target / Fraud Rate Encodings (Leakage-Free Bayesian Smoothed)
    print("\n[5/5] Computing expanding target fraud rates (Bayesian Smoothed)...")
    t0 = time.time()
    
    # Prior training fraud rate
    y_train = df.iloc[:split_idx]['isFraud'].values
    prior = y_train.mean() # ~ 0.03514
    m = 10 # smoothing weight
    print(f"  * Global training fraud rate prior: {prior:.5f} (using smoothing weight m={m})")

    # Split into Train / Val for target encoding calculations
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()

    target_keys = [
        'card1', 'addr1', 'DeviceInfo',
        'card_addr_combo', 'card_device_combo', 'card_email_combo'
    ]

    for key in target_keys:
        feature_name = f"{key}_historical_fraud_rate"
        print(f"  * Processing target rate for: {key} -> {feature_name}")
        
        # Train set gets chronological expanding rates
        df_train[feature_name] = compute_expanding_target_rates(df_train, key, prior, m)
        
        # Validation set gets training set final lookup mapping
        lookup_dict = build_train_lookup(df_train, key, prior, m)
        df_val[feature_name] = df_val[key].map(lookup_dict).fillna(prior)

    # Recombine Train and Validation
    df_enriched = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    
    # Drop temporary columns
    columns_to_drop = [
        'temp_addr', 'temp_device', 'temp_email',
        'card_addr_combo', 'card_device_combo', 'card_email_combo'
    ]
    df_enriched.drop(columns=columns_to_drop, inplace=True)
    
    print(f"  * Engineered 6 target fraud rate features in {time.time() - t0:.2f} seconds.")

    # Save Output
    df_enriched.to_parquet(output_path, engine='pyarrow', index=False)
    
    # Verification check
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    historical_cols = [
        'card1_past_count', 'card_addr_past_count', 'card_device_past_count', 'card_email_past_count',
        'card1_historical_fraud_rate', 'addr1_historical_fraud_rate', 'DeviceInfo_historical_fraud_rate',
        'card_addr_combo_historical_fraud_rate', 'card_device_combo_historical_fraud_rate', 'card_email_combo_historical_fraud_rate'
    ]
    
    print("\n" + "=" * 70)
    print("HISTORICAL FEATURE ENGINEERING COMPLETED")
    print("=" * 70)
    print(f"Output Path:      {output_path}")
    print(f"Output Shape:     {df_enriched.shape}")
    print(f"File Size:        {file_size_mb:.2f} MB")
    print(f"Execution Time:   {time.time() - start_time:.2f} seconds")
    print("-" * 70)
    print("Historical Columns Check (NaN Counts):")
    print(df_enriched[historical_cols].isna().sum())
    print("=" * 70)

if __name__ == "__main__":
    main()
