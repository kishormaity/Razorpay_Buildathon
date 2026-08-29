import os
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict

def main():
    start_time = time.time()
    print("=" * 70)
    print("      IEEE-CIS PHASE 12B: CARD NOVELTY FEATURE GENERATION")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/graph_features.parquet')
    output_path = os.path.join(processed_dir, 'features/card_novelty_features.parquet')

    if not os.path.exists(features_path):
        print(f"[ERROR] Baseline graph features file not found at: {features_path}")
        sys.exit(1)

    # 1. Load Data
    print("Loading graph features dataset...")
    df = pd.read_parquet(features_path)
    print(f"Loaded dataset in {time.time() - start_time:.2f} seconds.")

    # Sort chronologically to simulate chronological look-back states
    print("Sorting chronologically...")
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)

    # 2. Chronological state tracking
    print("\nRunning chronological look-back simulation...")
    t0 = time.time()

    card_addr_stats = defaultdict(lambda: {'total': 0})
    card_email_stats = defaultdict(lambda: {'total': 0})
    card_dev_stats = defaultdict(lambda: {'total': 0})
    
    card_stats = defaultdict(lambda: {'total': 0})
    addr_stats = defaultdict(lambda: {'total': 0})
    email_stats = defaultdict(lambda: {'total': 0})

    # Preallocate recording arrays
    card_past_count_global = np.zeros(total_rows)
    addr_past_count_global = np.zeros(total_rows)
    email_past_count_global = np.zeros(total_rows)
    
    card_addr_past_count = np.zeros(total_rows)
    card_email_past_count = np.zeros(total_rows)
    card_dev_past_count = np.zeros(total_rows)

    cards = df['card1'].values
    addrs = df['addr1'].values
    emails = df['P_emaildomain'].values
    devices = df['DeviceInfo'].values
    targets = df['isFraud'].values

    # Run chronological simulation loop (look-back state must be read BEFORE updating state)
    for i in range(total_rows):
        c = cards[i]
        a = addrs[i]
        e = emails[i]
        d = devices[i]
        
        # Read look-back values
        card_past_count_global[i] = card_stats[c]['total']
        
        if not pd.isna(a) and a != -999 and a != -999.0:
            addr_past_count_global[i] = addr_stats[a]['total']
            card_addr_past_count[i] = card_addr_stats[(c, a)]['total']
            
        if not pd.isna(e) and e != 'UNKNOWN' and e != '':
            email_past_count_global[i] = email_stats[e]['total']
            card_email_past_count[i] = card_email_stats[(c, e)]['total']
            
        if not pd.isna(d) and d != 'UNKNOWN' and d != '':
            card_dev_past_count[i] = card_dev_stats[(c, d)]['total']

        # Update states
        card_stats[c]['total'] += 1
        
        if not pd.isna(a) and a != -999 and a != -999.0:
            addr_stats[a]['total'] += 1
            card_addr_stats[(c, a)]['total'] += 1
            
        if not pd.isna(e) and e != 'UNKNOWN' and e != '':
            email_stats[e]['total'] += 1
            card_email_stats[(c, e)]['total'] += 1
            
        if not pd.isna(d) and d != 'UNKNOWN' and d != '':
            card_dev_stats[(c, d)]['total'] += 1

    print(f"Simulation completed in {time.time() - t0:.2f} seconds.")

    # 3. Calculate C1 - C5 features
    print("\nCalculating C1-C5 novelty features...")
    
    # C1: card_addr_unseen
    df['card_addr_unseen'] = ((card_past_count_global > 0) & (card_addr_past_count == 0)).astype(float)
    
    # C2: card_email_unseen
    df['card_email_unseen'] = ((card_past_count_global > 0) & (card_email_past_count == 0)).astype(float)
    
    # C3: card_device_unseen
    df['card_device_unseen'] = ((card_past_count_global > 0) & (card_dev_past_count == 0)).astype(float)
    
    # C4: card_addr_novelty_confidence (Address Novelty Strength)
    df['card_addr_novelty_confidence'] = df['card_addr_unseen'] * np.log1p(card_past_count_global) * np.log1p(addr_past_count_global)
    
    # C5: card_email_novelty_confidence (Email Novelty Strength)
    df['card_email_novelty_confidence'] = df['card_email_unseen'] * np.log1p(card_past_count_global) * np.log1p(email_past_count_global)

    # 4. Save Parquet
    print(f"\nWriting card novelty features Parquet to: {output_path}")
    t0 = time.time()
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Parquet write completed in {time.time() - t0:.2f} seconds.")

    # Verify column count
    # 436 columns from graph_features + 5 novelty columns = 441 columns total
    print("\n" + "=" * 70)
    print("CARD NOVELTY FEATURE GENERATION COMPLETED")
    print("=" * 70)
    print(f"Output Shape:     {df.shape}")
    print(f"Columns Count:    {df.shape[1]} (Expected 441)")
    print(f"Total Time:       {time.time() - start_time:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    main()
