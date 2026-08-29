import pandas as pd
import numpy as np

def compute_rolling_counts(df, group_col, window_seconds):
    """
    Computes rolling transaction counts over a time window (excluding the current transaction).
    """
    n_rows = len(df)
    counts = np.zeros(n_rows)
    times = df['TransactionDT'].values
    
    for _, indices in df.groupby(group_col).groups.items():
        idx_arr = np.array(indices)
        group_times = times[idx_arr]
        j = np.searchsorted(group_times, group_times - window_seconds, side='left')
        group_indices = np.arange(len(group_times))
        counts[idx_arr] = group_indices - j
        
    return counts

def compute_rolling_stats(df, group_col, window_seconds):
    """
    Computes rolling spending sums and means over a time window (excluding the current transaction).
    """
    n_rows = len(df)
    sums = np.zeros(n_rows)
    means = np.full(n_rows, np.nan)
    
    times = df['TransactionDT'].values
    amounts = df['TransactionAmt'].values
    
    for _, indices in df.groupby(group_col).groups.items():
        idx_arr = np.array(indices)
        group_times = times[idx_arr]
        group_amounts = amounts[idx_arr]
        
        cumsum_amounts = np.zeros(len(group_amounts) + 1)
        cumsum_amounts[1:] = np.cumsum(group_amounts)
        
        j = np.searchsorted(group_times, group_times - window_seconds, side='left')
        group_indices = np.arange(len(group_times))
        
        group_counts = group_indices - j
        group_sums = cumsum_amounts[group_indices] - cumsum_amounts[j]
        group_means = np.where(group_counts > 0, group_sums / group_counts, np.nan)
        
        sums[idx_arr] = group_sums
        means[idx_arr] = group_means
        
    return sums, means

def compute_behavioral_features(df):
    """
    Calculates behavioral metrics: rolling velocities, spend aggregates, spend ratios, and novelty flags.
    """
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    print("Computing rolling velocities...")
    df['card_tx_count_10m'] = compute_rolling_counts(df, 'card1', 600)
    df['card_tx_count_1h'] = compute_rolling_counts(df, 'card1', 3600)
    df['card_tx_count_24h'] = compute_rolling_counts(df, 'card1', 86400)
    
    print("Computing rolling spend sums & means...")
    spend_sum_1h, _ = compute_rolling_stats(df, 'card1', 3600)
    spend_sum_24h, spend_mean_24h = compute_rolling_stats(df, 'card1', 86400)
    
    df['card_spend_sum_1h'] = spend_sum_1h
    df['card_spend_sum_24h'] = spend_sum_24h
    df['card_spend_mean_24h'] = spend_mean_24h
    
    # Deviation features
    df['spend_ratio_24h'] = df['TransactionAmt'] / df['card_spend_mean_24h']
    df['spend_ratio_24h'] = df['spend_ratio_24h'].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    
    # Novelty calculations
    df['card_time_since_prev'] = df.groupby('card1')['TransactionDT'].diff()
    
    delta_device = df.groupby(['card1', 'device_id'])['TransactionDT'].diff()
    df['is_new_device'] = np.where(
        (df['device_id'] == 'UNKNOWN') | df['card_time_since_prev'].isna(),
        0,
        np.where(delta_device.isna() | (delta_device > 86400), 1, 0)
    ).astype('int8')
    
    delta_loc = df.groupby(['card1', 'ip_id'])['TransactionDT'].diff()
    df['is_new_location'] = np.where(
        (df['ip_id'].isna()) | df['card_time_since_prev'].isna(),
        0,
        np.where(delta_loc.isna() | (delta_loc > 86400), 1, 0)
    ).astype('int8')
    
    return df
