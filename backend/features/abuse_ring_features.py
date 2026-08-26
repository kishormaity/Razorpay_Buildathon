import os
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict, deque

def is_valid_device(dev):
    return not pd.isna(dev) and dev != 'UNKNOWN' and dev != ''

def is_valid_addr(a):
    return not pd.isna(a) and a != -999 and a != -999.0

def is_valid_email(e):
    return not pd.isna(e) and e != 'UNKNOWN' and e != ''

def main():
    start_time = time.time()
    print("=" * 70)
    print("      IEEE-CIS PHASE 14B: ABUSE-RING SENTINEL FEATURE GENERATION")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/card_novelty_features.parquet')
    output_path = os.path.join(processed_dir, 'features/abuse_ring_features.parquet')

    if not os.path.exists(features_path):
        print(f"[ERROR] Card novelty features file not found at: {features_path}")
        sys.exit(1)

    # 1. Load Data
    print("Loading card novelty features dataset...")
    df = pd.read_parquet(features_path)
    print(f"Loaded dataset in {time.time() - start_time:.2f} seconds.")

    # Sort chronologically
    print("Sorting chronologically...")
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)

    # 2. Chronological simulation states
    print("\nRunning chronological look-back graph simulation...")
    t0 = time.time()

    card_stats = defaultdict(lambda: {'fraud': 0})
    
    device_cards = defaultdict(set)
    device_obs = defaultdict(int)
    device_fraud = defaultdict(int)
    device_history = defaultdict(deque)  # queue of (TransactionDT, card1)
    
    addr_cards = defaultdict(set)
    addr_obs = defaultdict(int)
    addr_fraud = defaultdict(int)
    addr_history = defaultdict(deque)  # queue of (TransactionDT, card1)
    
    email_cards = defaultdict(set)

    # Preallocate recording arrays
    device_unique_card_count = np.zeros(total_rows)
    addr_unique_card_count = np.zeros(total_rows)
    email_unique_card_count = np.zeros(total_rows)
    
    device_connected_fraud_rate = np.zeros(total_rows)
    addr_connected_fraud_rate = np.zeros(total_rows)
    
    rapid_card_convergence = np.zeros(total_rows)
    cross_entity_convergence = np.zeros(total_rows)
    ring_fraud_density = np.zeros(total_rows)
    is_ring_abuse = np.zeros(total_rows)

    cards = df['card1'].values
    addrs = df['addr1'].values
    emails = df['P_emaildomain'].values
    devices = df['DeviceInfo'].values
    times = df['TransactionDT'].values
    targets = df['isFraud'].values

    # Run chronological simulation
    for i in range(total_rows):
        c = cards[i]
        a = addrs[i]
        e = emails[i]
        d = devices[i]
        t = times[i]
        y = targets[i]

        valid_dev = is_valid_device(d)
        valid_addr = is_valid_addr(a)
        valid_email = is_valid_email(e)

        # ---------------------------------------------------------
        # Read look-back statistics (before updating states)
        # ---------------------------------------------------------
        
        # 1. Unique card counts
        if valid_dev:
            device_unique_card_count[i] = len(device_cards[d])
        if valid_addr:
            addr_unique_card_count[i] = len(addr_cards[a])
        if valid_email:
            email_unique_card_count[i] = len(email_cards[e])

        # 2. Connected fraud rates
        if valid_dev and device_obs[d] > 0:
            device_connected_fraud_rate[i] = (device_fraud[d] + 0.1) / (device_obs[d] + 1.0)
        if valid_addr and addr_obs[a] > 0:
            addr_connected_fraud_rate[i] = (addr_fraud[a] + 0.1) / (addr_obs[a] + 1.0)

        # 3. Rapid convergence in past 72 hours (259200 seconds)
        dev_rapid_cards = 0
        if valid_dev:
            dev_q = device_history[d]
            while dev_q and dev_q[0][0] < t - 259200:
                dev_q.popleft()
            dev_rapid_cards = len(set(card for _, card in dev_q))
            
        addr_rapid_cards = 0
        if valid_addr:
            addr_q = addr_history[a]
            while addr_q and addr_q[0][0] < t - 259200:
                addr_q.popleft()
            addr_rapid_cards = len(set(card for _, card in addr_q))

        rapid_card_convergence[i] = max(dev_rapid_cards, addr_rapid_cards)

        # 4. Cross entity convergence (shares dev + addr with >= 2 other cards)
        if valid_dev and valid_addr:
            shared_cards = device_cards[d] & addr_cards[a]
            other_shared_count = len(shared_cards - {c})
            cross_entity_convergence[i] = 1.0 if other_shared_count >= 2 else 0.0

        # 5. Ring fraud density
        dev_density = 0.0
        if valid_dev and len(device_cards[d]) > 0:
            dev_density = sum(1 for card in device_cards[d] if card_stats[card]['fraud'] > 0) / len(device_cards[d])
            
        addr_density = 0.0
        if valid_addr and len(addr_cards[a]) > 0:
            addr_density = sum(1 for card in addr_cards[a] if card_stats[card]['fraud'] > 0) / len(addr_cards[a])
            
        ring_fraud_density[i] = max(dev_density, addr_density)

        # 6. Formulate weak proxy label is_ring_abuse (R5: High-risk device AND high-risk address overlap)
        dev_fraud_cards_count = sum(1 for card in device_cards[d] if card_stats[card]['fraud'] > 0) if valid_dev else 0
        addr_fraud_cards_count = sum(1 for card in addr_cards[a] if card_stats[card]['fraud'] > 0) if valid_addr else 0
        
        is_dev_r5 = valid_dev and (len(device_cards[d]) >= 3) and (dev_fraud_cards_count >= 1)
        is_addr_r5 = valid_addr and (len(addr_cards[a]) >= 3) and (addr_fraud_cards_count >= 1)
        
        is_ring_abuse[i] = 1.0 if (is_dev_r5 and is_addr_r5) else 0.0

        # ---------------------------------------------------------
        # Update states (after reading stats)
        # ---------------------------------------------------------
        card_stats[c]['fraud'] += y
        
        if valid_dev:
            device_cards[d].add(c)
            device_obs[d] += 1
            device_fraud[d] += y
            device_history[d].append((t, c))
            
        if valid_addr:
            addr_cards[a].add(c)
            addr_obs[a] += 1
            addr_fraud[a] += y
            addr_history[a].append((t, c))
            
        if valid_email:
            email_cards[e].add(c)

    print(f"Simulation completed in {time.time() - t0:.2f} seconds.")

    # 3. Add to DataFrame
    print("\nAdding features and targets to DataFrame...")
    df['device_unique_card_count'] = device_unique_card_count
    df['addr_unique_card_count'] = addr_unique_card_count
    df['email_unique_card_count'] = email_unique_card_count
    
    df['device_connected_fraud_rate'] = device_connected_fraud_rate
    df['addr_connected_fraud_rate'] = addr_connected_fraud_rate
    
    df['rapid_card_convergence'] = rapid_card_convergence
    df['cross_entity_convergence'] = cross_entity_convergence
    df['ring_fraud_density'] = ring_fraud_density
    df['is_ring_abuse'] = is_ring_abuse

    print(f"Abuse ring proxy distribution: {df['is_ring_abuse'].value_counts().to_dict()}")

    # 4. Save Parquet
    print(f"\nSaving abuse ring features Parquet to: {output_path}")
    t0 = time.time()
    df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Parquet saved in {time.time() - t0:.2f} seconds.")

    print("\n" + "=" * 70)
    print("ABUSE-RING FEATURE GENERATION COMPLETED")
    print("=" * 70)
    print(f"Output Shape:     {df.shape}")
    print(f"Columns Count:    {df.shape[1]}")
    print(f"Total Time:       {time.time() - start_time:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    main()
