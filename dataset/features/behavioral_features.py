import os
import sys
import time
import pandas as pd
import numpy as np

def compute_rolling_stats(df, group_cols, window_seconds):
    """
    Computes rolling transaction counts, sum, and mean over a look-back window
    that strictly excludes the current transaction (closed='left').
    Uses numpy searchsorted for high-performance and safety against duplicate label issues.
    """
    n_rows = len(df)
    counts = np.zeros(n_rows)
    sums = np.zeros(n_rows)
    means = np.full(n_rows, np.nan)
    
    times = df['TransactionDT'].values
    amounts = df['TransactionAmt'].values
    
    # Iterate over the grouped indices
    for _, indices in df.groupby(group_cols).groups.items():
        idx_arr = np.array(indices)
        
        # Get sorted times and amounts for this group
        group_times = times[idx_arr]
        group_amounts = amounts[idx_arr]
        
        # Compute cumulative sum of amounts for fast interval sum calculations
        cumsum_amounts = np.zeros(len(group_amounts) + 1)
        cumsum_amounts[1:] = np.cumsum(group_amounts)
        
        # Find index of first element >= current_t - window (excluding current_t)
        # side='left' returns the first index j where group_times[j] >= current_t - window
        j = np.searchsorted(group_times, group_times - window_seconds, side='left')
        
        # Current index in the group sequence (0, 1, ..., group_size-1)
        group_indices = np.arange(len(group_times))
        
        # Calculate stats for the look-back window [j, group_indices - 1]
        group_counts = group_indices - j
        group_sums = cumsum_amounts[group_indices] - cumsum_amounts[j]
        group_means = np.where(group_counts > 0, group_sums / group_counts, np.nan)
        
        # Map values back to original indices
        counts[idx_arr] = group_counts
        sums[idx_arr] = group_sums
        means[idx_arr] = group_means
        
    return counts, sums, means

def compute_rolling_counts(df, group_cols, window_seconds):
    """
    Computes rolling transaction counts only (excluding current transaction).
    """
    n_rows = len(df)
    counts = np.zeros(n_rows)
    times = df['TransactionDT'].values
    
    for _, indices in df.groupby(group_cols).groups.items():
        idx_arr = np.array(indices)
        group_times = times[idx_arr]
        j = np.searchsorted(group_times, group_times - window_seconds, side='left')
        group_indices = np.arange(len(group_times))
        counts[idx_arr] = group_indices - j
        
    return counts

def main():
    print("=" * 70)
    print("        IEEE-CIS BEHAVIORAL & TEMPORAL FEATURE ENGINEERING")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    features_path = os.path.join(processed_dir, 'features/transaction_features.parquet')
    output_path = os.path.join(processed_dir, 'features/behavioral_features.parquet')

    # 1. Load Preprocessed Transaction Features
    print("\n[1/5] Loading transaction features dataset...")
    if not os.path.exists(features_path):
        print(f"[ERROR] Transaction features parquet not found at: {features_path}")
        print("Please run transaction_features.py first.")
        sys.exit(1)

    start_time = time.time()
    df = pd.read_parquet(features_path)
    print(f"Loaded dataset: {df.shape} in {time.time() - start_time:.2f} seconds.")

    # 2. Setup Temporal Index
    print("\n[2/5] Preparing temporal index (sorting TransactionDT)...")
    # Chronological sort is critical for rolling windows
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    print("Sorted dataset chronologically by TransactionDT.")

    # 3. Compute Card Velocity and Spending Behavior (Excluding Current Transaction)
    print("\n[3/5] Computing rolling card velocities & spending (look-back only)...")
    
    # A. Card Transaction Frequencies (10m, 1h, 24h)
    print("  * Computing rolling transaction counts (10m, 1h, 24h)...")
    t0 = time.time()
    df['card_tx_count_10m'] = compute_rolling_counts(df, 'card1', 600)
    df['card_tx_count_1h'] = compute_rolling_counts(df, 'card1', 3600)
    df['card_tx_count_24h'] = compute_rolling_counts(df, 'card1', 86400)
    print(f"    Completed frequency velocity in {time.time() - t0:.2f} seconds.")

    # B. Card Spend Velocities
    print("  * Computing rolling spending aggregates (1h sum, 24h sum, 24h mean)...")
    t0 = time.time()
    _, spend_sum_1h, _ = compute_rolling_stats(df, 'card1', 3600)
    _, spend_sum_24h, spend_mean_24h = compute_rolling_stats(df, 'card1', 86400)
    
    df['card_spend_sum_1h'] = spend_sum_1h
    df['card_spend_sum_24h'] = spend_sum_24h
    df['card_spend_mean_24h'] = spend_mean_24h
    print(f"    Completed spend velocity in {time.time() - t0:.2f} seconds.")

    # C. Card Time-Delta Behavior
    print("  * Computing seconds elapsed since previous card transaction...")
    t0 = time.time()
    df['card_time_since_prev'] = df.groupby('card1')['TransactionDT'].diff() # NaN if first transaction
    print(f"    Completed time-deltas in {time.time() - t0:.2f} seconds.")

    # 4. Compute Cross-Entity Behavior and Novelty
    print("\n[4/5] Computing cross-entity velocities & novelty flags...")
    
    # A. Cross-Entity counts (Card + Location and Card + Email)
    print("  * Computing cross-entity rolling frequencies...")
    t0 = time.time()
    df['temp_addr'] = df['addr1'].astype('string').fillna('MISSING')
    df['temp_email'] = df['P_emaildomain'].astype('string').fillna('MISSING')
    
    df['card_addr_count_1h'] = compute_rolling_counts(df, ['card1', 'temp_addr'], 3600)
    df['card_email_count_24h'] = compute_rolling_counts(df, ['card1', 'temp_email'], 86400)
    
    # Drop temp columns used for null-safe grouping
    df.drop(columns=['temp_addr', 'temp_email'], inplace=True)
    print(f"    Completed cross-entity velocity in {time.time() - t0:.2f} seconds.")

    # B. Spend Deviation
    print("  * Computing spend amount deviation ratios...")
    df['spend_ratio_24h'] = df['TransactionAmt'] / df['card_spend_mean_24h']
    df['spend_ratio_24h'] = df['spend_ratio_24h'].fillna(1.0).replace([np.inf, -np.inf], 1.0)

    # C. Device and Location Novelty (Excluding cards with no history)
    print("  * Computing Device and Location novelty flags...")
    t0 = time.time()
    # is_new_device (1 if device not seen in last 24h and prior history exists)
    delta_device = df.groupby(['card1', 'DeviceInfo'])['TransactionDT'].diff()
    df['is_new_device'] = np.where(
        df['DeviceInfo'].isna() | df['card_time_since_prev'].isna(),
        0,
        np.where(delta_device.isna() | (delta_device > 86400), 1, 0)
    ).astype('int8')
    
    # is_new_location (1 if location not seen in last 24h and prior history exists)
    delta_loc = df.groupby(['card1', 'addr1'])['TransactionDT'].diff()
    df['is_new_location'] = np.where(
        df['addr1'].isna() | df['card_time_since_prev'].isna(),
        0,
        np.where(delta_loc.isna() | (delta_loc > 86400), 1, 0)
    ).astype('int8')
    print(f"    Completed novelty indicators in {time.time() - t0:.2f} seconds.")

    # 5. Save Output
    print("\n[5/5] Writing behavioral features Parquet...")
    
    # Define behavioral columns to verify
    behavioral_cols = [
        'card_tx_count_10m', 'card_tx_count_1h', 'card_tx_count_24h',
        'card_spend_sum_1h', 'card_spend_sum_24h', 'card_spend_mean_24h',
        'card_time_since_prev', 'card_addr_count_1h', 'card_email_count_24h',
        'spend_ratio_24h', 'is_new_device', 'is_new_location'
    ]
    
    t0 = time.time()
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Successfully saved behavioral features dataset to: {output_path}")
    print(f"Parquet write completed in {time.time() - t0:.2f} seconds.")

    # 6. Verification Metrics
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print("\n" + "=" * 70)
    print("BEHAVIORAL FEATURE ENGINEERING COMPLETED")
    print("=" * 70)
    print(f"Output Path:      {output_path}")
    print(f"Output Shape:     {df.shape}")
    print(f"File Size:        {file_size_mb:.2f} MB")
    print(f"Execution Time:   {elapsed:.2f} seconds")
    print("-" * 70)
    print("Behavioral Column Check (NaN Counts):")
    print(df[behavioral_cols].isna().sum())
    print("=" * 70)

if __name__ == "__main__":
    main()
