import os
import sys
import time
import pandas as pd
import numpy as np

def compute_expanding_mean_std(df, group_col, val_col):
    """
    Computes chronological cumulative mean and sample standard deviation
    strictly looking at past rows (0 to i-1) per group.
    """
    vals = df[val_col].values
    means = np.full(len(df), np.nan)
    stds = np.full(len(df), np.nan)
    
    for _, indices in df.groupby(group_col).groups.items():
        idx_arr = np.array(indices)
        if len(idx_arr) <= 1:
            continue
        group_vals = vals[idx_arr]
        
        # Cumulative sum strictly before current index
        cumsum = np.cumsum(group_vals) - group_vals
        counts = np.arange(len(idx_arr))
        means[idx_arr[1:]] = cumsum[1:] / counts[1:]
        
        # Sum of squares strictly before current index
        cumsum_sq = np.cumsum(group_vals**2) - group_vals**2
        
        # We need at least 2 previous samples to compute sample standard deviation (N-1 denominator)
        j_prev = np.arange(len(idx_arr))
        valid_mask = j_prev >= 2
        if np.any(valid_mask):
            var = (cumsum_sq[valid_mask] - (cumsum[valid_mask]**2 / j_prev[valid_mask])) / (j_prev[valid_mask] - 1)
            stds[idx_arr[valid_mask]] = np.sqrt(np.maximum(var, 0))
            
    return means, stds

def compute_expanding_mean(df, group_col, val_col):
    """
    Computes chronological cumulative mean strictly looking at past rows (0 to i-1) per group.
    Special handling for 'card_time_since_prev' where the first element is NaN.
    """
    vals = df[val_col].values
    means = np.full(len(df), np.nan)
    is_time_gap = (val_col == 'card_time_since_prev')
    
    for _, indices in df.groupby(group_col).groups.items():
        idx_arr = np.array(indices)
        if len(idx_arr) <= 1:
            continue
        group_vals = vals[idx_arr]
        
        if is_time_gap:
            # The first value group_vals[0] is NaN, so we start from index 1.
            # If the group only has <= 2 transactions, we cannot compute a past non-NaN mean.
            if len(group_vals) > 2:
                non_nan_vals = group_vals[1:]
                cumsum = np.cumsum(non_nan_vals) - non_nan_vals
                counts = np.arange(len(non_nan_vals))
                means[idx_arr[2:]] = cumsum[1:] / counts[1:]
        else:
            cumsum = np.cumsum(group_vals) - group_vals
            counts = np.arange(len(idx_arr))
            means[idx_arr[1:]] = cumsum[1:] / counts[1:]
            
    return means

def compute_expanding_median(df, group_col, val_col):
    """
    Computes chronological cumulative median strictly looking at past rows (0 to i-1) per group.
    Special handling for 'card_time_since_prev' where the first element is NaN.
    """
    vals = df[val_col].values
    medians = np.full(len(df), np.nan)
    is_time_gap = (val_col == 'card_time_since_prev')
    
    for _, indices in df.groupby(group_col).groups.items():
        idx_arr = np.array(indices)
        if len(idx_arr) <= 1:
            continue
        group_vals = vals[idx_arr]
        
        for j in range(1, len(idx_arr)):
            if is_time_gap:
                slice_vals = group_vals[1:j]
            else:
                slice_vals = group_vals[:j]
                
            if len(slice_vals) > 0:
                medians[idx_arr[j]] = np.median(slice_vals)
                
    return medians

def compute_expanding_unique_counts(df, group_col, val_col):
    """
    Computes chronological unique count of values strictly before the current index.
    Ignores null and placeholder values.
    """
    vals = df[val_col].values
    unique_counts = np.zeros(len(df))
    is_addr = (val_col == 'addr1')
    
    for _, indices in df.groupby(group_col).groups.items():
        idx_arr = np.array(indices)
        seen = set()
        for j in range(len(idx_arr)):
            unique_counts[idx_arr[j]] = len(seen)
            val = vals[idx_arr[j]]
            if is_addr:
                if not pd.isna(val) and val != -999 and val != -999.0:
                    seen.add(val)
            else:
                if not pd.isna(val) and val != 'UNKNOWN' and val != '':
                    seen.add(str(val))
                    
    return unique_counts

def main():
    print("=" * 70)
    print("       IEEE-CIS BEHAVIORAL DEVIATION FEATURE ENGINEERING")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    historical_path = os.path.join(processed_dir, 'features/historical_features.parquet')
    behavioral_path = os.path.join(processed_dir, 'features/behavioral_features.parquet')
    output_path = os.path.join(processed_dir, 'features/deviation_features.parquet')

    # 1. Load Datasets
    print("\n[1/5] Loading historical and behavioral features...")
    if not os.path.exists(historical_path):
        print(f"[ERROR] Historical features parquet not found at: {historical_path}")
        sys.exit(1)
    if not os.path.exists(behavioral_path):
        print(f"[ERROR] Behavioral features parquet not found at: {behavioral_path}")
        sys.exit(1)

    start_time = time.time()
    df_hist = pd.read_parquet(historical_path)
    df_behav = pd.read_parquet(behavioral_path)
    print(f"Loaded datasets in {time.time() - start_time:.2f} seconds.")
    print(f"  * Historical shape: {df_hist.shape}")
    print(f"  * Behavioral shape: {df_behav.shape}")

    # 2. Chronological Sorting & Clean Alignment
    print("\n[2/5] Preparing chronological sort and alignment...")
    df_hist = df_hist.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    df_behav = df_behav.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)

    if not np.array_equal(df_hist['TransactionID'].values, df_behav['TransactionID'].values):
        print("[WARNING] Parquet files not aligned by ID. Performing inner merge...")
        behav_cols_needed = ['TransactionID', 'card_tx_count_1h', 'card_tx_count_24h', 'card_time_since_prev', 'card_spend_sum_1h', 'card_spend_sum_24h']
        df = pd.merge(df_hist, df_behav[behav_cols_needed], on='TransactionID', how='inner')
    else:
        print("✅ Parquets are aligned. Appending behavioral columns directly.")
        df = df_hist.copy()
        df['card_tx_count_1h'] = df_behav['card_tx_count_1h']
        df['card_tx_count_24h'] = df_behav['card_tx_count_24h']
        df['card_time_since_prev'] = df_behav['card_time_since_prev']
        df['card_spend_sum_1h'] = df_behav['card_spend_sum_1h']
        df['card_spend_sum_24h'] = df_behav['card_spend_sum_24h']

    # 3. Compute Deviation Features
    print("\n[3/5] Computing deviation features (look-back only)...")
    t_feat_start = time.time()

    # --- 1. Amount Anomaly ---
    print("  * Engineering Amount Anomalies...")
    t0 = time.time()
    # Compute baseline intermediate stats
    card_amount_mean, card_amount_std = compute_expanding_mean_std(df, 'card1', 'TransactionAmt')
    card_amount_median = compute_expanding_median(df, 'card1', 'TransactionAmt')
    
    # Compute anomaly scores
    df['amount_vs_card_mean'] = df['TransactionAmt'] / card_amount_mean
    df['amount_vs_card_mean'] = df['amount_vs_card_mean'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    
    df['amount_vs_card_median'] = df['TransactionAmt'] / card_amount_median
    df['amount_vs_card_median'] = df['amount_vs_card_median'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    
    df['amount_zscore'] = (df['TransactionAmt'] - card_amount_mean) / card_amount_std
    df['amount_zscore'] = df['amount_zscore'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    print(f"    Completed Amount Anomalies in {time.time() - t0:.2f} seconds.")

    # --- 2. Frequency Anomaly ---
    print("  * Engineering Frequency Anomalies...")
    t0 = time.time()
    card_tx_count_24h_hist_mean = compute_expanding_mean(df, 'card1', 'card_tx_count_24h')
    df['tx_frequency_deviation_24h'] = df['card_tx_count_24h'] / card_tx_count_24h_hist_mean
    df['tx_frequency_deviation_24h'] = df['tx_frequency_deviation_24h'].fillna(1.0).replace([np.inf, -np.inf], 1.0)

    card_tx_count_1h_hist_mean = compute_expanding_mean(df, 'card1', 'card_tx_count_1h')
    df['tx_frequency_deviation_1h'] = df['card_tx_count_1h'] / card_tx_count_1h_hist_mean
    df['tx_frequency_deviation_1h'] = df['tx_frequency_deviation_1h'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    print(f"    Completed Frequency Anomalies in {time.time() - t0:.2f} seconds.")

    # --- 3. Temporal Anomaly ---
    print("  * Engineering Temporal Anomalies...")
    t0 = time.time()
    card_time_since_prev_hist_median = compute_expanding_median(df, 'card1', 'card_time_since_prev')
    df['time_gap_deviation_median'] = df['card_time_since_prev'] / card_time_since_prev_hist_median
    df['time_gap_deviation_median'] = df['time_gap_deviation_median'].fillna(1.0).replace([np.inf, -np.inf], 1.0)

    card_time_since_prev_hist_mean = compute_expanding_mean(df, 'card1', 'card_time_since_prev')
    df['time_gap_deviation_mean'] = df['card_time_since_prev'] / card_time_since_prev_hist_mean
    df['time_gap_deviation_mean'] = df['time_gap_deviation_mean'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    print(f"    Completed Temporal Anomalies in {time.time() - t0:.2f} seconds.")

    # --- 3B. Temporal Acceleration & Spend-Temporal Interactions ---
    print("  * Engineering Temporal Acceleration & Interactions...")
    t0_acc = time.time()
    hist_gap_median_safe = np.nan_to_num(card_time_since_prev_hist_median, nan=0.0, posinf=0.0, neginf=0.0)
    hist_gap_mean_safe = np.nan_to_num(card_time_since_prev_hist_mean, nan=0.0, posinf=0.0, neginf=0.0)
    current_gap_safe = df['card_time_since_prev'].fillna(0.0).values
    
    df['time_gap_acceleration_median'] = np.log1p(hist_gap_median_safe / (current_gap_safe + 1.0))
    df['time_gap_acceleration_median'] = df['time_gap_acceleration_median'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    df['time_gap_acceleration_mean'] = np.log1p(hist_gap_mean_safe / (current_gap_safe + 1.0))
    df['time_gap_acceleration_mean'] = df['time_gap_acceleration_mean'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    safe_amt_median = np.clip(df['amount_vs_card_median'].fillna(1.0).replace([np.inf, -np.inf], 1.0).values, 0, None)
    safe_amt_mean = np.clip(df['amount_vs_card_mean'].fillna(1.0).replace([np.inf, -np.inf], 1.0).values, 0, None)
    
    df['amount_temporal_interaction'] = np.log1p(safe_amt_median) * df['time_gap_acceleration_median']
    df['amount_temporal_interaction'] = df['amount_temporal_interaction'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    df['amount_temporal_interaction_mean'] = np.log1p(safe_amt_mean) * df['time_gap_acceleration_mean']
    df['amount_temporal_interaction_mean'] = df['amount_temporal_interaction_mean'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    print(f"    Completed Acceleration & Interactions in {time.time() - t0_acc:.2f} seconds.")

    # --- 4. Spending Velocity Anomaly ---
    print("  * Engineering Spending Velocity Anomalies...")
    t0 = time.time()
    card_spend_sum_1h_hist_mean = compute_expanding_mean(df, 'card1', 'card_spend_sum_1h')
    df['spend_velocity_deviation_1h'] = df['card_spend_sum_1h'] / card_spend_sum_1h_hist_mean
    df['spend_velocity_deviation_1h'] = df['spend_velocity_deviation_1h'].fillna(1.0).replace([np.inf, -np.inf], 1.0)

    card_spend_sum_24h_hist_mean = compute_expanding_mean(df, 'card1', 'card_spend_sum_24h')
    df['spend_velocity_deviation_24h'] = df['card_spend_sum_24h'] / card_spend_sum_24h_hist_mean
    df['spend_velocity_deviation_24h'] = df['spend_velocity_deviation_24h'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    print(f"    Completed Spending Velocity Anomalies in {time.time() - t0:.2f} seconds.")

    # --- 5. Entity Association Anomaly ---
    print("  * Engineering Entity Association Anomalies...")
    t0 = time.time()
    card1_past_count = df['card1_past_count'].values
    safe_denom = np.where(card1_past_count > 0, card1_past_count, 1.0)
    df['card_device_frequency'] = np.where(card1_past_count > 0, df['card_device_past_count'].values / safe_denom, 0.0)
    df['card_device_frequency'] = df['card_device_frequency'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    df['card_location_frequency'] = np.where(card1_past_count > 0, df['card_addr_past_count'].values / safe_denom, 0.0)
    df['card_location_frequency'] = df['card_location_frequency'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    print(f"    Completed Entity Association Anomalies in {time.time() - t0:.2f} seconds.")

    # --- 6. Diversity Metrics (Optional) ---
    print("  * Engineering Optional Diversity Metrics...")
    t0 = time.time()
    df['card_device_diversity'] = compute_expanding_unique_counts(df, 'card1', 'DeviceInfo')
    df['card_location_diversity'] = compute_expanding_unique_counts(df, 'card1', 'addr1')
    print(f"    Completed Diversity Metrics in {time.time() - t0:.2f} seconds.")

    print(f"Total feature calculation finished in {time.time() - t_feat_start:.2f} seconds.")

    # 4. Clean Feature Space (Drop intermediates and temporary behavioral features)
    print("\n[4/5] Dropping temporary columns and finalizing feature space...")
    cols_to_drop = [
        'card_tx_count_1h', 'card_tx_count_24h', 'card_time_since_prev', 'card_spend_sum_1h', 'card_spend_sum_24h'
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    # 5. Save Output
    print(f"\n[5/5] Writing deviation features Parquet to: {output_path}")
    t0 = time.time()
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Parquet write completed in {time.time() - t0:.2f} seconds.")

    # Verify column count
    # 393 transaction features + 10 historical features + 11 core deviation + 4 new features + 2 diversity = 420 feature columns
    # + 2 columns for metadata (TransactionID, isFraud) = 422 columns total
    print("\n" + "=" * 70)
    print("BEHAVIORAL DEVIATION FEATURE ENGINEERING COMPLETED")
    print("=" * 70)
    print(f"Output Shape:     {df.shape}")
    print(f"Columns Count:    {df.shape[1]} (Expected 422)")
    print(f"Total Time:       {time.time() - start_time:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    main()
