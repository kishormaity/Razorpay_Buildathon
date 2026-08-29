import os
import sys
import time
import networkx as nx
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))

from src.graph.builder import build_heterogeneous_graph

def evaluate_partitions():
    print("=" * 70)
    print("          ABUSE-RING SENTINEL COMMUNITY ALGORITHM BENCHMARK")
    print("=" * 70)
    
    # 1. Build projected user graph (users connected if they sharedevice, IP, or payment)
    print("Building heterogeneous graph...")
    G = build_heterogeneous_graph()
    
    # Extract projected user sharing graph
    users = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'USER']
    if not users:
        print("[ERROR] No user nodes found in graph.")
        return
        
    print(f"Projecting graph for {len(users)} users...")
    # Project: create link between users sharing device, IP or payment
    U = nx.Graph()
    U.add_nodes_from(users)
    
    # Map entities to users
    entity_to_users = {}
    for n, attr in G.nodes(data=True):
        ntype = attr.get('type')
        if ntype in ('DEVICE', 'IP', 'PAYMENT'):
            entity_to_users[n] = [u for u in G.neighbors(n) if G.nodes[u].get('type') == 'USER']
            
    # Add sharing edges
    for entity, usrs in entity_to_users.items():
        if len(usrs) > 1:
            for i in range(len(usrs)):
                for j in range(i + 1, len(usrs)):
                    U.add_edge(usrs[i], usrs[j])
                    
    print(f"Projected User Graph: {U.number_of_nodes()} nodes, {U.number_of_edges()} edges")
    
    # If projected graph has no edges, use the base graph projected subset
    if U.number_of_edges() == 0:
        print("[WARNING] Projected user graph has 0 edges. Using base graph connected components.")
        
    # Query fraud targets for purity calculation
    conn = sqlite3_connect_helper()
    tx_df = pd.read_sql_query("SELECT user_id, is_abuse FROM transactions;", conn)
    conn.close()
    
    user_fraud_map = tx_df.groupby('user_id')['is_abuse'].max().to_dict()
    
    # Algorithms to evaluate
    algorithms = {}
    
    # A. Louvain Modularity
    print("\nRunning Louvain Modularity partition...")
    t0 = time.time()
    try:
        louvain_comm = list(nx.community.louvain_communities(U))
        algorithms['Louvain'] = {
            'comms': louvain_comm,
            'time': time.time() - t0
        }
    except Exception as e:
        print(f"Louvain failed: {e}")
        
    # B. Greedy Modularity (Clauset-Newman-Moore)
    print("Running Greedy Modularity partition...")
    t0 = time.time()
    try:
        greedy_comm = list(nx.community.greedy_modularity_communities(U))
        # Convert set objects
        greedy_comm = [set(c) for c in greedy_comm]
        algorithms['Greedy Modularity'] = {
            'comms': greedy_comm,
            'time': time.time() - t0
        }
    except Exception as e:
        print(f"Greedy failed: {e}")
        
    # C. Label Propagation
    print("Running Label Propagation partition...")
    t0 = time.time()
    try:
        lp_comm = list(nx.community.label_propagation_communities(U))
        algorithms['Label Propagation'] = {
            'comms': lp_comm,
            'time': time.time() - t0
        }
    except Exception as e:
        print(f"Label Propagation failed: {e}")
        
    # 2. Compile metrics
    results = []
    for name, data in algorithms.items():
        comms = data['comms']
        num_comms = len(comms)
        
        # Modularity
        try:
            mod_val = nx.community.modularity(U, comms)
        except:
            mod_val = -1.0
            
        # Community sizes
        sizes = [len(c) for c in comms]
        avg_size = np.mean(sizes) if sizes else 0
        max_size = np.max(sizes) if sizes else 0
        
        # Purity of high-risk rings (communities with size > 3)
        purity_list = []
        high_risk_rings_count = 0
        for c in comms:
            if len(c) > 3:
                high_risk_rings_count += 1
                fraud_count = sum(1 for u in c if user_fraud_map.get(u, 0) == 1)
                purity_list.append(fraud_count / len(c))
                
        avg_purity = np.mean(purity_list) if purity_list else 0.0
        
        results.append({
            'Algorithm': name,
            'Partitions': num_comms,
            'Modularity': mod_val,
            'Avg Size': avg_size,
            'Max Size': max_size,
            'High-Risk Rings (>3)': high_risk_rings_count,
            'Avg Purity': avg_purity,
            'Latency (s)': data['time']
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("                COMMUNITY DETECTION BENCHMARK MATRIX")
    print("=" * 70)
    print(res_df.to_string(index=False))
    print("=" * 70)
    
    # Save the best community partitioning metadata for downstream scorer
    # Default to Louvain as standard
    best_comms = algorithms['Louvain']['comms'] if 'Louvain' in algorithms else list(nx.connected_components(U))
    
    # Write champion partitions output
    champion_metadata = {}
    node_to_comm = {}
    
    for c_idx, c in enumerate(best_comms):
        comm_users = list(c)
        champion_metadata[str(c_idx)] = {
            'users': comm_users,
            'size': len(comm_users)
        }
        for u in comm_users:
            node_to_comm[u] = c_idx
            
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    out_json = os.path.join(proj_root, "data", "processed", "champion_communities.json")
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({'node_to_comm': node_to_comm, 'comms': champion_metadata}, f, indent=2)
        
    print(f"Champion partitions configuration saved to: {out_json}")

def sqlite3_connect_helper():
    # Helper to resolve SQLite file
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    db_path = os.path.join(proj_root, "data", "processed", "risk_sentinel.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

if __name__ == "__main__":
    import json
    import sqlite3
    evaluate_partitions()
