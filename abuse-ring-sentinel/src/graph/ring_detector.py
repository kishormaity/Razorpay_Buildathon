import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))

from src.graph.builder import build_heterogeneous_graph

def get_timestamp_hours(ts_str):
    # Convert '2026-01-01T10:00:00Z' to floating point hours
    try:
        from datetime import datetime
        dt = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%SZ')
        return dt.timestamp() / 3600.0
    except:
        return 0.0

def evaluate_rings():
    print("=" * 70)
    print("           ABUSE-RING SENTINEL RING RISK ENGINE")
    print("=" * 70)
    
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    
    # 1. Load configuration and weights
    policy_path = os.path.join(proj_root, "configs", "risk_policy.yaml")
    with open(policy_path, "r") as f:
        config = yaml.safe_load(f)["ring_scoring"]
        
    weights = config["weights"]
    w_struct = weights.get("w_relationship", 0.35)
    w_temp = weights.get("w_temporal", 0.25)
    w_behav = weights.get("w_behavioral", 0.20)
    w_fin = weights.get("w_financial", 0.20)
    abuse_threshold = config.get("abuse_threshold", 0.70)
    
    # 2. Load champion partitions configuration
    comm_json_path = os.path.join(proj_root, "data", "processed", "champion_communities.json")
    if not os.path.exists(comm_json_path):
        print(f"[ERROR] Partition configuration not found at: {comm_json_path}")
        sys.exit(1)
        
    with open(comm_json_path, "r") as f:
        comm_data = json.load(f)
        
    comms = comm_data["comms"]
    node_to_comm = comm_data["node_to_comm"]
    
    # 3. Load transactional records from SQLite for profiling
    db_path = os.path.join(proj_root, "data", "processed", "risk_sentinel.db")
    conn = sqlite3.connect(db_path)
    tx_df = pd.read_sql_query("SELECT user_id, amount, timestamp, is_abuse FROM transactions;", conn)
    user_created_df = pd.read_sql_query("SELECT user_id, created_at FROM users;", conn)
    conn.close()
    
    user_tx_map = tx_df.groupby('user_id')
    user_created_map = dict(zip(user_created_df['user_id'], user_created_df['created_at']))
    
    # Load global graph for structural sharing
    G = build_heterogeneous_graph()
    
    ring_risk_profiles = {}
    
    print(f"Scoring {len(comms)} modularity community partitions...")
    for comm_id, info in comms.items():
        users = info["users"]
        size = info["size"]
        
        if size <= 1:
            ring_risk_profiles[comm_id] = {
                'score': 0.05,
                'structural': 0.0,
                'temporal': 0.0,
                'behavioral': 0.0,
                'financial': 0.0,
                'is_abuse_ring': False,
                'users': users
            }
            continue
            
        # A. Structural Score (Entity Sharing & Density)
        shared_entities = set()
        for u in users:
            if G.has_node(u):
                for neighbor in G.neighbors(u):
                    neighbor_type = G.nodes[neighbor].get('type')
                    if neighbor_type in ('DEVICE', 'IP', 'PAYMENT'):
                        shared_entities.add(neighbor)
                        
        sharing_density = len(shared_entities) / size
        s_struct = min(1.0, float(1.0 - np.exp(-sharing_density)))
        
        # B. Temporal Score (User Creation Synchronization)
        creation_hours = [get_timestamp_from_str(user_created_map.get(u, '')) for u in users]
        creation_hours = [h for h in creation_hours if h > 0]
        
        if len(creation_hours) > 1:
            std_dev = np.std(creation_hours)
            s_temp = float(np.exp(-std_dev / 24.0)) # Higher score for tightly clustered creation
        else:
            s_temp = 0.10
            
        # C. Behavioral Score (Transaction Amount similarities)
        tx_amounts = []
        for u in users:
            if u in user_tx_map.groups:
                tx_amounts.extend(user_tx_map.get_group(u)['amount'].values)
                
        if len(tx_amounts) > 1:
            std_amt = np.std(tx_amounts)
            s_behav = float(np.exp(-std_amt / 200.0)) # Higher score for highly similar amount profiles
        else:
            s_behav = 0.10
            
        # D. Financial Score (Fraud Density - Disabled when w_fin == 0.0 to prevent target leakage)
        if w_fin > 0.0:
            fraud_labels = []
            for u in users:
                if u in user_tx_map.groups:
                    fraud_labels.extend(user_tx_map.get_group(u)['is_abuse'].values)
            s_fin = float(np.mean(fraud_labels)) if fraud_labels else 0.0
        else:
            s_fin = 0.0
        
        # Blended final risk calculation
        r_final = w_struct * s_struct + w_temp * s_temp + w_behav * s_behav + w_fin * s_fin
        is_abuse_ring = bool(r_final >= abuse_threshold)
        
        ring_risk_profiles[comm_id] = {
            'score': float(r_final),
            'structural': float(s_struct),
            'temporal': float(s_temp),
            'behavioral': float(s_behav),
            'financial': float(s_fin),
            'is_abuse_ring': is_abuse_ring,
            'users': users
        }
        
    # Write ring risk profiles json
    out_path = os.path.join(proj_root, "data", "processed", "ring_risk_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ring_risk_profiles, f, indent=2)
        
    # Log top rings discovered
    ranked_rings = sorted(
        [(cid, r['score'], len(r['users']), r['is_abuse_ring']) for cid, r in ring_risk_profiles.items()],
        key=lambda x: x[1], reverse=True
    )
    
    print("\n" + "=" * 70)
    print("                    TOP RING SCORER FINDINGS")
    print("=" * 70)
    print(f"Community ID | Blended Risk Score | Member Size | Labeled Abuse Tag")
    print("-" * 70)
    for cid, score, size, is_abuse in ranked_rings[:5]:
        print(f"R-COMM-{cid:<5} | {score:.4f}             | {size:<11} | {str(is_abuse):<17}")
    print("=" * 70)
    print(f"Ring scorer profiles written to: {out_path}")

def get_timestamp_from_str(ts_str):
    # Convert '2026-01-01T10:00:00Z' to epoch hours
    try:
        from datetime import datetime
        dt = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%SZ')
        return dt.timestamp() / 3600.0
    except:
        return 0.0

if __name__ == "__main__":
    evaluate_rings()
