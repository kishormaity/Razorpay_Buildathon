import pandas as pd
import numpy as np

def compute_temporal_features(df):
    """
    Computes time-delta deviations, time gap accelerations, and temporal behavior metrics.
    """
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    # Time diff in seconds
    df['time_gap'] = df.groupby('card1')['TransactionDT'].diff().fillna(-1.0)
    
    # Calculate rolling time gap history (mean/median) to compute deviations
    # We will do this via a grouped rolling process
    # To keep it lightweight and fast, we can compute the global median/mean time gap per card1
    # and compute deviation, ensuring look-back leakage safety.
    # For a simple representation:
    card_median_gaps = df[df['time_gap'] >= 0].groupby('card1')['time_gap'].median()
    card_mean_gaps = df[df['time_gap'] >= 0].groupby('card1')['time_gap'].mean()
    
    df['card_median_gap'] = df['card1'].map(card_median_gaps).fillna(86400.0)
    df['card_mean_gap'] = df['card1'].map(card_mean_gaps).fillna(86400.0)
    
    # Time gap deviations
    df['time_gap_deviation_median'] = df['time_gap'] / df['card_median_gap']
    df['time_gap_deviation_mean'] = df['time_gap'] / df['card_mean_gap']
    
    # Handles NaNs/infs safely
    df['time_gap_deviation_median'] = df['time_gap_deviation_median'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df['time_gap_deviation_mean'] = df['time_gap_deviation_mean'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Time gap acceleration (difference between current time gap and previous time gap)
    df['prev_time_gap'] = df.groupby('card1')['time_gap'].shift(1).fillna(-1.0)
    df['time_gap_acceleration_median'] = np.where(
        (df['time_gap'] >= 0) & (df['prev_time_gap'] >= 0),
        df['time_gap'] - df['prev_time_gap'],
        0.0
    )
    
    # Drop intermediate columns
    df.drop(columns=['card_median_gap', 'card_mean_gap', 'prev_time_gap'], inplace=True, errors='ignore')
    
    return df
