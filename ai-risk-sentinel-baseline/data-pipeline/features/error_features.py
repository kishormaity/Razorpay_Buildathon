import os
import sys
import time
import pandas as pd
import numpy as np

def main():
    start_time = time.time()
    print("=" * 70)
    print("      IEEE-CIS PHASE 10C: ERROR-DRIVEN FEATURE CALCULATIONS")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/graph_features.parquet')
    behav_path = os.path.join(processed_dir, 'features/behavioral_features.parquet')
    output_path = os.path.join(processed_dir, 'features/error_features.parquet')

    if not os.path.exists(features_path) or not os.path.exists(behav_path):
        print("[ERROR] Required datasets (graph_features, behavioral_features) not found.")
        sys.exit(1)

    # 1. Load Parquets
    print("Loading graph features and behavioral features...")
    df = pd.read_parquet(features_path)
    
    behav_cols = ['TransactionID', 'card_tx_count_24h', 'card_time_since_prev']
    behav_df = pd.read_parquet(behav_path, columns=behav_cols)
    print(f"Loaded datasets in {time.time() - start_time:.2f} seconds.")

    # 2. Merge on TransactionID
    print("Merging features...")
    df = pd.merge(df, behav_df, on='TransactionID', how='inner')
    print(f"Merged Shape: {df.shape}")

    # 3. Calculate 4 Targeted Error Features
    print("\nCalculating error-driven features...")
    t0 = time.time()

    # Get missingness flags
    is_device_missing = df['is_device_missing'].values
    is_email_missing = df['is_email_missing'].values
    tx_amt = df['TransactionAmt'].values
    tx_count_24h = df['card_tx_count_24h'].values
    time_since_prev = df['card_time_since_prev'].fillna(86400.0).values # Neutral value: 1 day gap

    # E1: device_missing_email_present
    df['device_missing_email_present'] = is_device_missing * (1.0 - is_email_missing)

    # E2: device_missing_value_weight
    df['device_missing_value_weight'] = is_device_missing * np.log1p(tx_amt)

    # E3: short_gap_velocity_ratio
    df['short_gap_velocity_ratio'] = tx_count_24h / np.log1p(time_since_prev)

    # E4: velocity_value_drain
    df['velocity_value_drain'] = tx_count_24h / np.log1p(tx_amt)

    print(f"Features created in {time.time() - t0:.2f} seconds.")

    # 4. Save Parquet
    print(f"\nWriting error features Parquet to: {output_path}")
    t0 = time.time()
    
    # Drop intermediate columns if they weren't in graph_features
    # card_tx_count_24h and card_time_since_prev are intermediate columns, we can drop them or keep them.
    # To be safe and keep baseline features matching Model D exactly, let's drop them.
    df.drop(columns=['card_tx_count_24h', 'card_time_since_prev'], inplace=True, errors='ignore')
    
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Parquet write completed in {time.time() - t0:.2f} seconds.")

    # Verify column count
    # 436 columns from graph_features + 4 targeted columns = 440 columns total
    print("\n" + "=" * 70)
    print("ERROR-DRIVEN FEATURE CALCULATIONS COMPLETED")
    print("=" * 70)
    print(f"Output Shape:     {df.shape}")
    print(f"Columns Count:    {df.shape[1]} (Expected 440)")
    print(f"Total Time:       {time.time() - start_time:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    main()
