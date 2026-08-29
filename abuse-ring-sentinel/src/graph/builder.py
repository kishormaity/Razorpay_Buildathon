import os
import sys
import networkx as nx
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from data.connection import get_connection

def build_heterogeneous_graph(timestamp_limit=None):
    """
    Queries relations from SQLite and builds a heterogeneous NetworkX graph.
    If timestamp_limit is specified, only connections created before the limit are included.
    """
    conn = get_connection()
    G = nx.Graph()
    
    # 1. Query and insert nodes (Users, Devices, IPs, Payments, Merchants)
    # We will load nodes dynamically based on relations rather than inserting empty nodes,
    # but we can query them to get their attributes.
    
    # Define node queries
    user_query = "SELECT user_id, country FROM users;"
    device_query = "SELECT device_id, device_type, os FROM devices;"
    ip_query = "SELECT ip_id, country FROM ips;"
    pmt_query = "SELECT payment_id, payment_type FROM payment_methods;"
    merchant_query = "SELECT merchant_id, category FROM merchants;"
    
    users = pd.read_sql_query(user_query, conn)
    devices = pd.read_sql_query(device_query, conn)
    ips = pd.read_sql_query(ip_query, conn)
    payments = pd.read_sql_query(pmt_query, conn)
    merchants = pd.read_sql_query(merchant_query, conn)
    
    # Add nodes with attributes
    for row in users.itertuples(index=False):
        G.add_node(row.user_id, type='USER', country=row.country)
    for row in devices.itertuples(index=False):
        G.add_node(row.device_id, type='DEVICE', os=row.os, device_type=row.device_type)
    for row in ips.itertuples(index=False):
        G.add_node(row.ip_id, type='IP', country=row.country)
    for row in payments.itertuples(index=False):
        G.add_node(row.payment_id, type='PAYMENT', payment_type=row.payment_type)
    for row in merchants.itertuples(index=False):
        G.add_node(row.merchant_id, type='MERCHANT', category=row.category)
        
    # 2. Query and insert edges based on temporal constraint
    time_filter = ""
    params = []
    if timestamp_limit is not None:
        time_filter = "WHERE first_seen_at < ?"
        params = [timestamp_limit]
        
    # A. User - Device edges
    ud_query = f"SELECT user_id, device_id, usage_count FROM user_devices {time_filter};"
    ud_df = pd.read_sql_query(ud_query, conn, params=params)
    for row in ud_df.itertuples(index=False):
        if G.has_node(row.user_id) and G.has_node(row.device_id):
            G.add_edge(row.user_id, row.device_id, type='USER_DEVICE', weight=float(row.usage_count))
            
    # B. User - IP edges
    ui_query = f"SELECT user_id, ip_id, usage_count FROM user_ips {time_filter};"
    ui_df = pd.read_sql_query(ui_query, conn, params=params)
    for row in ui_df.itertuples(index=False):
        if G.has_node(row.user_id) and G.has_node(row.ip_id):
            G.add_edge(row.user_id, row.ip_id, type='USER_IP', weight=float(row.usage_count))
            
    # C. User - Payment edges
    up_query = f"SELECT user_id, payment_id, usage_count FROM user_payments {time_filter};"
    up_df = pd.read_sql_query(up_query, conn, params=params)
    for row in up_df.itertuples(index=False):
        if G.has_node(row.user_id) and G.has_node(row.payment_id):
            G.add_edge(row.user_id, row.payment_id, type='USER_PAYMENT', weight=float(row.usage_count))
            
    # D. User - Transaction - Merchant edges
    # We retrieve transactions and connect USER -> TRANSACTION -> MERCHANT
    tx_time_filter = ""
    tx_params = []
    if timestamp_limit is not None:
        tx_time_filter = "WHERE timestamp < ?"
        tx_params = [timestamp_limit]
        
    tx_query = f"SELECT transaction_id, user_id, merchant_id, payment_id, amount, is_abuse FROM transactions {tx_time_filter};"
    tx_df = pd.read_sql_query(tx_query, conn, params=tx_params)
    
    for row in tx_df.itertuples(index=False):
        # Add transaction node dynamically
        G.add_node(row.transaction_id, type='TRANSACTION', amount=float(row.amount), is_abuse=int(row.is_abuse))
        
        # Connect USER -> TRANSACTION
        if G.has_node(row.user_id):
            G.add_edge(row.user_id, row.transaction_id, type='USER_TRANSACTION', weight=1.0)
            
        # Connect TRANSACTION -> MERCHANT
        if row.merchant_id and G.has_node(row.merchant_id):
            G.add_edge(row.transaction_id, row.merchant_id, type='TRANSACTION_MERCHANT', weight=1.0)
            
        # Connect TRANSACTION -> PAYMENT
        if row.payment_id and G.has_node(row.payment_id):
            G.add_edge(row.transaction_id, row.payment_id, type='TRANSACTION_PAYMENT', weight=1.0)
            
    conn.close()
    return G

if __name__ == "__main__":
    print("Building global graph...")
    G = build_heterogeneous_graph()
    print(f"Graph built successfully:")
    print(f"  * Nodes count: {G.number_of_nodes()}")
    print(f"  * Edges count: {G.number_of_edges()}")
    
    # Print node type distributions
    types = {}
    for node, attrs in G.nodes(data=True):
        ntype = attrs.get('type', 'UNKNOWN')
        types[ntype] = types.get(ntype, 0) + 1
    print("Node types count:")
    for ntype, count in types.items():
        print(f"  * {ntype}: {count}")
