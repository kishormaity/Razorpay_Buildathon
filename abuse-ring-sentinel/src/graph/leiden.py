import os
import sys
import time
import networkx as nx
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from data.connection import get_connection
from graph.builder import build_heterogeneous_graph

def partition_communities(G):
    """
    Finds coordinated communities in G using modularity-maximization Louvain/greedy partition.
    Returns:
      - node_to_community: dict mapping node -> community_id
      - communities_meta: dict mapping community_id -> metadata dictionary
    """
    print("Partitioning graph nodes into communities...")
    start_time = time.time()
    
    # We run modularity communities on the projected user-sharing graph
    # or directly on G. Running on G directly works well since modularity will separate
    # disconnected components and coordinate clusters.
    try:
        # Louvain is built-in to NetworkX 2.8+
        communities_list = nx.community.louvain_communities(G, seed=42)
    except Exception as e:
        print(f"Louvain communities failed: {e}. Falling back to greedy modularity...")
        communities_list = nx.community.greedy_modularity_communities(G)
        
    print(f"Modularity partitioning completed in {time.time() - start_time:.2f} seconds. Found {len(communities_list)} communities.")
    
    node_to_community = {}
    communities_meta = {}
    
    for comm_idx, comm_nodes in enumerate(communities_list):
        for node in comm_nodes:
            node_to_community[node] = comm_idx
            
        # Compile attributes of this community
        # Count types of nodes
        users_in_comm = [n for n in comm_nodes if G.nodes[n].get('type') == 'USER']
        txns_in_comm = [n for n in comm_nodes if G.nodes[n].get('type') == 'TRANSACTION']
        devices_in_comm = [n for n in comm_nodes if G.nodes[n].get('type') == 'DEVICE']
        payments_in_comm = [n for n in comm_nodes if G.nodes[n].get('type') == 'PAYMENT']
        ips_in_comm = [n for n in comm_nodes if G.nodes[n].get('type') == 'IP']
        
        # Calculate fraud statistics
        # Fraud rate based on TRANSACTION nodes that have is_abuse == 1
        fraud_txns = sum(1 for n in txns_in_comm if G.nodes[n].get('is_abuse', 0) == 1)
        total_txns = len(txns_in_comm)
        fraud_rate = (fraud_txns / total_txns) if total_txns > 0 else 0.0
        
        # Coordinated convergence
        # If multiple users share the same device or payment method in this community
        sub_g = G.subgraph(comm_nodes)
        density = nx.density(sub_g) if len(comm_nodes) > 1 else 0.0
        
        communities_meta[comm_idx] = {
            'community_id': comm_idx,
            'nodes_count': len(comm_nodes),
            'users_count': len(users_in_comm),
            'transactions_count': total_txns,
            'devices_count': len(devices_in_comm),
            'payments_count': len(payments_in_comm),
            'ips_count': len(ips_in_comm),
            'fraud_transactions': fraud_txns,
            'fraud_rate': fraud_rate,
            'density': density,
            'is_ring_suspicious': (total_txns >= 3 and fraud_rate >= 0.20) or (len(users_in_comm) >= 2 and len(devices_in_comm) >= 1 and fraud_rate >= 0.10)
        }
        
    return node_to_community, communities_meta

if __name__ == "__main__":
    print("Building global graph...")
    G = build_heterogeneous_graph()
    
    node_to_comm, comm_meta = partition_communities(G)
    
    # Print sample metadata for suspicious communities
    suspicious_comms = [meta for meta in comm_meta.values() if meta['is_ring_suspicious']]
    print(f"\nSuspicious Coordinated Communities: {len(suspicious_comms)} / {len(comm_meta)}")
    
    for comm in suspicious_comms[:5]:
        print(f"Community ID {comm['community_id']}:")
        print(f"  * Nodes:        {comm['nodes_count']}")
        print(f"  * Users:        {comm['users_count']}")
        print(f"  * Transactions: {comm['transactions_count']} (Fraud rate: {comm['fraud_rate']:.2f})")
        print(f"  * Devices:      {comm['devices_count']}")
        print(f"  * Payments:     {comm['payments_count']}")
