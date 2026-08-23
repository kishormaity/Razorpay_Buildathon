import os
import sqlite3
import json
import random
from datetime import datetime, timedelta

def setup_db(db_path, schema_path):
    print(f"Initializing SQLite database at {db_path}...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Read and execute schema
    with open(schema_path, 'r') as f:
        schema = f.read()
    cursor.executescript(schema)
    
    conn.commit()
    return conn

def process_real_dataset(raw_dir, conn):
    print("Parsing raw IEEE-CIS CSV files...")
    # Load raw CSVs dynamically with Pandas
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas is required to parse raw CSVs. Run: pip install pandas")
        return False
        
    txn_path = os.path.join(raw_dir, "train_transaction.csv")
    id_path = os.path.join(raw_dir, "train_identity.csv")
    
    if not (os.path.exists(txn_path) and os.path.exists(id_path)):
        print(f"[ERROR] Raw CSV files not found. Ensure both '{txn_path}' and '{id_path}' exist in '{raw_dir}'.")
        return False

    print("Loading datasets (this may take a minute)...")
    # Load subsets of the datasets to keep memory utilization low
    tx_df = pd.read_csv(txn_path, nrows=50000)
    id_df = pd.read_csv(id_path, nrows=50000)
    
    print("Merging transaction and identity on TransactionID...")
    merged = pd.merge(tx_df, id_df, on='TransactionID', how='left')
    
    # Isolate all fraud entries and sample normal entries for balance
    fraud_df = merged[merged['isFraud'] == 1]
    normal_df = merged[merged['isFraud'] == 0].sample(n=min(len(merged[merged['isFraud'] == 0]), 1000), random_state=42)
    sample_df = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42)
    
    print(f"Sampled {len(sample_df)} records (Fraud: {len(fraud_df)}, Normal: {len(normal_df)}).")
    
    cursor = conn.cursor()
    
    print("Mapping raw data into SQL Entity-Relationship schema...")
    
    # Helper to generate entity details
    def get_details(row, entity_type):
        details = {}
        if entity_type == 'DEVICE':
            if not pd.isna(row.get('DeviceInfo')):
                details['DeviceInfo'] = str(row['DeviceInfo'])
            if not pd.isna(row.get('id_30')):
                details['OS'] = str(row['id_30'])
            if not pd.isna(row.get('id_31')):
                details['Browser'] = str(row['id_31'])
        elif entity_type == 'IP':
            if not pd.isna(row.get('addr2')):
                details['Country'] = f"Country-{int(row['addr2'])}"
        elif entity_type == 'PAYMENT':
            if not pd.isna(row.get('card4')):
                details['CardBrand'] = str(row['card4'])
            if not pd.isna(row.get('card6')):
                details['CardType'] = str(row['card6'])
        return json.dumps(details)

    # Loop through sampled entries and insert
    for _, row in sample_df.iterrows():
        txn_id = f"TXN-{row['TransactionID']}"
        cust_id = f"CUS-{int(row['card1'])}" if not pd.isna(row.get('card1')) else f"CUS-{row['TransactionID']}"
        
        # Only assign device_id if there is raw hardware telemetry
        device_id = None
        if not pd.isna(row.get('id_02')):
            device_id = f"DEV-{int(row['id_02'])}"
            
        # Only assign region/IP if location code exists
        ip_addr = None
        if not pd.isna(row.get('addr1')):
            ip_addr = f"REG-{int(row['addr1'])}"
            
        is_fraud = int(row['isFraud'])
        risk_score = float(is_fraud)
        action = 'BLOCK' if is_fraud == 1 else 'ALLOW'
        status = 'CONFIRMED_ABUSE' if is_fraud == 1 else 'REVIEW_COMPLETED'
        
        # Derive timestamp dynamically based on TransactionDT (offset in seconds from reference date)
        ref_date = datetime(2026, 1, 1)
        txn_time_str = (ref_date + timedelta(seconds=int(row['TransactionDT']))).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Insert user entity
        cursor.execute("INSERT OR IGNORE INTO entities VALUES (?, 'USER', ?, ?, ?, ?, ?)", 
                       (cust_id, f"User Profile {cust_id}", risk_score, txn_time_str, txn_time_str, get_details(row, 'USER')))
        
        # Insert device entity and relationships conditionally
        if device_id:
            cursor.execute("INSERT OR IGNORE INTO entities VALUES (?, 'DEVICE', ?, ?, ?, ?, ?)", 
                           (device_id, f"Hardware identifier {device_id}", risk_score, txn_time_str, txn_time_str, get_details(row, 'DEVICE')))
            cursor.execute("INSERT OR IGNORE INTO relationships VALUES (?, ?, ?, 'USED_BY', 1.0, ?, ?)",
                           (f"R-{cust_id}-{device_id}", cust_id, device_id, txn_time_str, txn_time_str))

        # Insert IP entity and relationships conditionally
        if ip_addr:
            cursor.execute("INSERT OR IGNORE INTO entities VALUES (?, 'IP', ?, ?, ?, ?, ?)", 
                           (ip_addr, f"IP Subnet {ip_addr}", risk_score, txn_time_str, txn_time_str, get_details(row, 'IP')))
            cursor.execute("INSERT OR IGNORE INTO relationships VALUES (?, ?, ?, 'SHARED_IP', 1.0, ?, ?)",
                           (f"R-{cust_id}-{ip_addr}", cust_id, ip_addr, txn_time_str, txn_time_str))
            
        # Insert transaction entity and relationships
        cursor.execute("INSERT OR IGNORE INTO entities VALUES (?, 'TRANSACTION', ?, ?, ?, ?, ?)", 
                       (txn_id, f"Amount {row['TransactionAmt']}", risk_score, txn_time_str, txn_time_str, json.dumps({})))
        cursor.execute("INSERT OR IGNORE INTO relationships VALUES (?, ?, ?, 'MADE_TRANSACTION', 1.0, ?, ?)",
                       (f"R-{cust_id}-{txn_id}", cust_id, txn_id, txn_time_str, txn_time_str))

        # Dynamically build payment method, merchant, channel and location from dataset
        brand = str(row['card4']) if not pd.isna(row.get('card4')) else ""
        card_type = str(row['card6']) if not pd.isna(row.get('card6')) else ""
        payment_method = f"{brand} {card_type}".strip() or None
        
        merchant_name = str(row['P_emaildomain']) if not pd.isna(row.get('P_emaildomain')) else None
        channel = str(row['ProductCD']) if not pd.isna(row.get('ProductCD')) else None
        location = f"Country-{int(row['addr2'])}" if not pd.isna(row.get('addr2')) else None

        # Insert transaction details
        cursor.execute("""
            INSERT OR IGNORE INTO transactions VALUES 
            (?, ?, ?, 'USD', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Parsed from raw dataset.')
        """, (txn_id, txn_time_str, float(row['TransactionAmt']), merchant_name, payment_method, 
              channel, location, risk_score, action, cust_id, device_id, ip_addr, 
              f"Anomaly patterns flagged from raw IEEE-CIS ID {row['TransactionID']}", status))

    conn.commit()
    print("Real data processing successfully completed!")
    return True

def preprocess():
    # For data paths, look one level up since this script is in dataset/pipeline/
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    raw_dir = os.path.join(parent_dir, 'data', 'raw')
    db_path = os.path.join(parent_dir, 'data', 'processed', 'risk_sentinel.db')
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schema.sql')
    
    # Initialize DB
    conn = setup_db(db_path, schema_path)
    
    # Attempt to read raw IEEE-CIS CSVs
    processed = process_real_dataset(raw_dir, conn)
    
    if not processed:
        conn.close()
        # Clean up the DB file if we generated it during setup_db
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
        raise RuntimeError("Preprocessing failed. Synthetic fallback is disabled.")
        
    conn.close()
    print(f"\nPreprocessing finished. Active database available at: {db_path}")

if __name__ == "__main__":
    preprocess()
