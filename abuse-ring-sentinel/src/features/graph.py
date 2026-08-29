import os
import sys
import time
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))

from src.data.connection import get_connection
from src.graph.builder import build_heterogeneous_graph

def compute_graph_features(df):
    """
    Computes chronological graph degree, sharing, and connected risk features, 
    supplemented by training-graph PageRank and clustering coefficients.
    """
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)
    
    # Ratios for split definitions
    split_idx = int(total_rows * 0.70)
    
    # Calculate training fraud rate prior for Bayes smoothing
    prior = df.iloc[:split_idx]['isFraud'].mean()
    if pd.isna(prior) or prior == 0:
        prior = 0.015 # default fallback
    m = 10 # smoothing factor
    
    print(f"Global training fraud rate prior: {prior:.5f} (using smoothing weight m={m})")
    
    # 1. Chronological State Map Loop
    card_devices = defaultdict(set)
    card_addrs = defaultdict(set)
    device_cards = defaultdict(set)
    addr_cards = defaultdict(set)
    card_stats = defaultdict(lambda: {'total': 0, 'fraud': 0})
    
    # Pre-allocate numpy arrays
    card_device_degree = np.zeros(total_rows)
    card_addr_degree = np.zeros(total_rows)
    device_card_degree = np.zeros(total_rows)
    addr_card_degree = np.zeros(total_rows)
    shared_device_card_count = np.zeros(total_rows)
    shared_addr_card_count = np.zeros(total_rows)
    device_connected_fraud_rate = np.full(total_rows, prior)
    addr_connected_fraud_rate = np.full(total_rows, prior)
    
    # Refined network features
    network_risk_mean = np.full(total_rows, prior)
    network_risk_max = np.full(total_rows, prior)
    network_risk_gap = np.zeros(total_rows)
    network_risk_product = np.full(total_rows, prior * prior)
    device_card_novelty = np.zeros(total_rows)
    addr_card_novelty = np.zeros(total_rows)
    
    cards = df['card1'].values
    devices = df['device_id'].values
    addrs = df['addr1'].values
    targets = df['isFraud'].values
    
    print("Running chronological graph state simulation...")
    for i in range(total_rows):
        c = cards[i]
        d = devices[i]
        a = addrs[i]
        
        # A. Degrees
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
                
        # D. Refined Network aggregates
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
                
        # E. Update State (after calculating features for current transaction)
        if not pd.isna(d) and d != 'UNKNOWN' and d != '':
            device_cards[d].add(c)
            card_devices[c].add(d)
            
        if not pd.isna(a) and a != -999 and a != -999.0:
            addr_cards[a].add(c)
            card_addrs[c].add(a)
            
        card_stats[c]['total'] += 1
        card_stats[c]['fraud'] += targets[i]
        
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
    
    # 2. Build training graph topological centralities (PageRank / Clustering)
    print("Building training graph for centrality calculations...")
    train_df = df.iloc[:split_idx]
    
    # Connect to SQLite to retrieve training graph relations (using end of training timestamp limit)
    limit_ts = df.iloc[split_idx]['timestamp'] if split_idx < len(df) else None
    
    G_train = build_heterogeneous_graph(timestamp_limit=limit_ts)
    
    print("Computing PageRank centrality on training graph...")
    try:
        pr = nx.pagerank(G_train, alpha=0.85, max_iter=100)
    except Exception as e:
        print(f"PageRank computation failed: {e}. Falling back to degree mapping.")
        pr = dict(G_train.degree())
        
    print("Computing Clustering Coefficient on training graph...")
    try:
        clustering = nx.clustering(G_train)
    except Exception as e:
        print(f"Clustering computation failed: {e}. Falling back to zero.")
        clustering = {}
        
    # Map PageRank and Clustering back to df (representing User PageRank / User Clustering)
    df['pagerank_centrality'] = df['user_id'].map(pr).fillna(0.0)
    df['clustering_coefficient'] = df['user_id'].map(clustering).fillna(0.0)
    
    return df
