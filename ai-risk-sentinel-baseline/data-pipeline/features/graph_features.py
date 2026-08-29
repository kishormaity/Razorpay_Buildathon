import os
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict

def main():
    print("=" * 70)
    print("          IEEE-CIS CARD-DEVICE-ADDRESS GRAPH FEATURES")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/deviation_features.parquet')
    output_path = os.path.join(processed_dir, 'features/graph_features.parquet')

    if not os.path.exists(features_path):
        print(f"[ERROR] Features parquet file not found at: {features_path}")
        sys.exit(1)

    # 1. Load Parquet
    start_time = time.time()
    df = pd.read_parquet(features_path)
    print(f"Loaded deviation dataset shape: {df.shape} in {time.time() - start_time:.2f}s")

    # 2. Sort chronologically by TransactionDT
    print("\nPreparing chronological sort (TransactionDT)...")
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)

    # 3. Graph/Network feature parameters
    total_rows = len(df)
    split_idx = int(total_rows * 0.8)
    
    # Calculate global prior training fraud rate for Bayes smoothing
    prior = df.iloc[:split_idx]['isFraud'].mean()
    m = 10 # smoothing factor
    print(f"Global training fraud rate prior: {prior:.5f} (using smoothing weight m={m})")

    # 4. Chronological State Map Loop
    print("\nEngineering graph degree, sharing, and connected risk features...")
    t0 = time.time()

    card_devices = defaultdict(set)
    card_addrs = defaultdict(set)
    device_cards = defaultdict(set)
    addr_cards = defaultdict(set)
    card_stats = defaultdict(lambda: {'total': 0, 'fraud': 0})

    # Pre-allocate numpy arrays for speed
    card_device_degree = np.zeros(total_rows)
    card_addr_degree = np.zeros(total_rows)
    device_card_degree = np.zeros(total_rows)
    addr_card_degree = np.zeros(total_rows)
    shared_device_card_count = np.zeros(total_rows)
    shared_addr_card_count = np.zeros(total_rows)
    device_connected_fraud_rate = np.full(total_rows, prior)
    addr_connected_fraud_rate = np.full(total_rows, prior)

    # 6 new features for Phase 9
    network_risk_mean = np.full(total_rows, prior)
    network_risk_max = np.full(total_rows, prior)
    network_risk_gap = np.zeros(total_rows)
    network_risk_product = np.full(total_rows, prior * prior)
    device_card_novelty = np.zeros(total_rows)
    addr_card_novelty = np.zeros(total_rows)

    cards = df['card1'].values
    devices = df['DeviceInfo'].values
    addrs = df['addr1'].values
    targets = df['isFraud'].values

    # Run chronological simulation
    for i in range(total_rows):
        c = cards[i]
        d = devices[i]
        a = addrs[i]

        # A. Card Degrees
        card_device_degree[i] = len(card_devices[c])
        card_addr_degree[i] = len(card_addrs[c])

        # B. Device Features
        if not pd.isna(d) and d != 'UNKNOWN' and d != '':
            device_card_degree[i] = len(device_cards[d])
            shared_device_card_count[i] = len(device_cards[d] - {c})
            
            if len(device_cards[d]) > 0:
                conn_total = sum(card_stats[card]['total'] for card in device_cards[d])
                conn_fraud = sum(card_stats[card]['fraud'] for card in device_cards[d])
                device_connected_fraud_rate[i] = (conn_fraud + prior * m) / (conn_total + m)

        # C. Address Features
        if not pd.isna(a) and a != -999 and a != -999.0:
            addr_card_degree[i] = len(addr_cards[a])
            shared_addr_card_count[i] = len(addr_cards[a] - {c})
            
            if len(addr_cards[a]) > 0:
                conn_total = sum(card_stats[card]['total'] for card in addr_cards[a])
                conn_fraud = sum(card_stats[card]['fraud'] for card in addr_cards[a])
                addr_connected_fraud_rate[i] = (conn_fraud + prior * m) / (conn_total + m)

        # E. Phase 9: Network Risk & Novelty Aggregates
        network_risk_mean[i] = (device_connected_fraud_rate[i] + addr_connected_fraud_rate[i]) / 2.0
        network_risk_max[i] = max(device_connected_fraud_rate[i], addr_connected_fraud_rate[i])
        network_risk_gap[i] = abs(device_connected_fraud_rate[i] - addr_connected_fraud_rate[i])
        network_risk_product[i] = device_connected_fraud_rate[i] * addr_connected_fraud_rate[i]
        
        has_transacted_before = card_stats[c]['total'] > 0
        if has_transacted_before and not pd.isna(d) and d != 'UNKNOWN' and d != '':
            if d not in card_devices[c]:
                device_card_novelty[i] = 1.0
                
        if has_transacted_before and not pd.isna(a) and a != -999 and a != -999.0:
            if a not in card_addrs[c]:
                addr_card_novelty[i] = 1.0

        # D. Update States (strictly look-back, i.e. current transaction updates state only AFTER features are read)
        if not pd.isna(d) and d != 'UNKNOWN' and d != '':
            device_cards[d].add(c)
            card_devices[c].add(d)

        if not pd.isna(a) and a != -999 and a != -999.0:
            addr_cards[a].add(c)
            card_addrs[c].add(a)

        card_stats[c]['total'] += 1
        card_stats[c]['fraud'] += targets[i]

    print(f"Graph feature calculations finished in {time.time() - t0:.2f} seconds.")

    # 5. Append features to dataframe
    print("\nAppending features to dataframe...")
    df['card_device_degree'] = card_device_degree
    df['card_addr_degree'] = card_addr_degree
    df['device_card_degree'] = device_card_degree
    df['addr_card_degree'] = addr_card_degree
    df['shared_device_card_count'] = shared_device_card_count
    df['shared_addr_card_count'] = shared_addr_card_count
    df['device_connected_fraud_rate'] = device_connected_fraud_rate
    df['addr_connected_fraud_rate'] = addr_connected_fraud_rate

    # Phase 9 refined features
    df['network_risk_mean'] = network_risk_mean
    df['network_risk_max'] = network_risk_max
    df['network_risk_gap'] = network_risk_gap
    df['network_risk_product'] = network_risk_product
    df['device_card_novelty'] = device_card_novelty
    df['addr_card_novelty'] = addr_card_novelty

    # 6. Save Parquet
    print(f"\nWriting graph features Parquet to: {output_path}")
    t0 = time.time()
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Parquet write completed in {time.time() - t0:.2f} seconds.")

    # Verify column count
    # 422 columns from deviation_features + 14 graph columns = 436 columns total
    print("\n" + "=" * 70)
    print("GRAPH FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Output Shape:     {df.shape}")
    print(f"Columns Count:    {df.shape[1]} (Expected 436)")
    print(f"Total Time:       {time.time() - start_time:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    main()
