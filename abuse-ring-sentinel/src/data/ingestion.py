import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from data.connection import get_connection, init_db

def load_config():
    config_path = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "configs", "data.yaml"))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def verify_raw_files():
    config = load_config()
    raw_dir_rel = config["paths"]["raw_dir"]
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    raw_dir = os.path.abspath(os.path.join(proj_root, raw_dir_rel))
    
    required_files = [
        "train_transaction.csv",
        "train_identity.csv"
    ]
    missing_files = []
    for f in required_files:
        path = os.path.join(raw_dir, f)
        if not os.path.exists(path):
            missing_files.append(path)
            
    if missing_files:
        raise FileNotFoundError(
            f"Required raw dataset files are missing: {', '.join(missing_files)}. "
            "Please check the shared raw directory."
        )
    return raw_dir

def read_raw_data(nrows=20000):
    raw_dir = verify_raw_files()
    tx_path = os.path.join(raw_dir, "train_transaction.csv")
    id_path = os.path.join(raw_dir, "train_identity.csv")
    
    print(f"Reading raw transactions (limit {nrows} rows)...")
    tx_df = pd.read_csv(tx_path, nrows=nrows)
    
    print(f"Reading raw identity (limit {nrows} rows)...")
    id_df = pd.read_csv(id_path, nrows=nrows)
    
    return tx_df, id_df

def clean_and_merge(tx_df, id_df):
    tx_df['TransactionID'] = tx_df['TransactionID'].astype(int)
    id_df['TransactionID'] = id_df['TransactionID'].astype(int)
    
    tx_df = tx_df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    merged = pd.merge(tx_df, id_df, on='TransactionID', how='left')
    merged = merged.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    merged['DeviceInfo'] = merged['DeviceInfo'].fillna('UNKNOWN').astype(str).str.strip()
    merged['id_30'] = merged['id_30'].fillna('UNKNOWN').astype(str).str.strip()
    merged['P_emaildomain'] = merged['P_emaildomain'].fillna('UNKNOWN').astype(str).str.strip()
    merged['isFraud'] = merged['isFraud'].fillna(0).astype(int)
    
    return merged

def get_timestamp_from_dt(dt_seconds):
    ref_date = datetime(2026, 1, 1)
    return (ref_date + timedelta(seconds=int(dt_seconds))).strftime('%Y-%m-%dT%H:%M:%SZ')

def seed_database(df):
    print("Truncating tables for V2 fresh start...")
    conn = get_connection()
    cursor = conn.cursor()
    
    tables = [
        "investigations", "user_payments", "user_ips", "user_devices", 
        "logins", "transactions", "merchants", "payment_methods", 
        "ips", "devices", "users"
    ]
    for table in tables:
        cursor.execute(f"DELETE FROM {table};")
    conn.commit()
    
    users_set = set()
    devices_set = set()
    ips_set = set()
    payments_set = set()
    merchants_set = set()
    
    user_devices_dict = {}
    user_ips_dict = {}
    user_payments_dict = {}
    
    print(f"Inserting {len(df)} records into SQLite tables...")
    
    for row in df.itertuples(index=False):
        txn_id = f"TXN-{row.TransactionID}"
        dt_sec = int(row.TransactionDT)
        timestamp = get_timestamp_from_dt(dt_sec)
        amount = float(row.TransactionAmt)
        
        card1_val = int(row.card1) if not pd.isna(row.card1) else 0
        user_id = f"CUS-{card1_val}"
        
        # Device details
        device_id = None
        device_type = str(row.DeviceType) if not pd.isna(row.DeviceType) else "UNKNOWN"
        device_os = str(row.id_30) if not pd.isna(row.id_30) else "UNKNOWN"
        device_name = str(row.DeviceInfo) if not pd.isna(row.DeviceInfo) else "UNKNOWN"
        
        if device_name != "UNKNOWN" or device_os != "UNKNOWN" or device_type != "UNKNOWN":
            dev_hash = int(abs(hash(device_name + device_os))) % 100000
            device_id = f"DEV-{dev_hash}"
            
        # IP / Region details
        ip_id = None
        addr1_val = int(float(row.addr1)) if not pd.isna(row.addr1) and row.addr1 != -999 else None
        if addr1_val is not None:
            ip_id = f"IP-{addr1_val}"
            
        # Payment Card
        payment_id = None
        card2_val = int(row.card2) if not pd.isna(row.card2) else 0
        
        if card1_val > 0:
            payment_id = f"PMT-{card1_val}-{card2_val}"
            
        # Merchant details
        merchant_domain = str(row.P_emaildomain) if not pd.isna(row.P_emaildomain) else "UNKNOWN"
        merchant_id = f"MER-{merchant_domain}"
        
        # isFraud mapping
        is_fraud = int(row.isFraud)
        
        # 1. Insert Core Entities
        if user_id not in users_set:
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id, created_at, country) VALUES (?, ?, ?);",
                (user_id, timestamp, f"Country-{int(row.addr2)}" if not pd.isna(row.addr2) else "Country-UNKNOWN")
            )
            users_set.add(user_id)
            
        if device_id and device_id not in devices_set:
            cursor.execute(
                "INSERT OR IGNORE INTO devices (device_id, first_seen_at, device_type, os) VALUES (?, ?, ?, ?);",
                (device_id, timestamp, device_name, device_os)
            )
            devices_set.add(device_id)
            
        if ip_id and ip_id not in ips_set:
            cursor.execute(
                "INSERT OR IGNORE INTO ips (ip_id, ip_hash, country, first_seen_at) VALUES (?, ?, ?, ?);",
                (ip_id, f"IP_HASH_{addr1_val}", f"Country-{int(row.addr2)}" if not pd.isna(row.addr2) else "Country-UNKNOWN", timestamp)
            )
            ips_set.add(ip_id)
            
        if payment_id and payment_id not in payments_set:
            cursor.execute(
                "INSERT OR IGNORE INTO payment_methods (payment_id, payment_type, fingerprint_hash, first_seen_at) VALUES (?, ?, ?, ?);",
                (payment_id, str(row.card6) if not pd.isna(row.card6) else "UNKNOWN", f"FINGERPRINT_{card1_val}_{card2_val}", timestamp)
            )
            payments_set.add(payment_id)
            
        if merchant_id not in merchants_set:
            cursor.execute(
                "INSERT OR IGNORE INTO merchants (merchant_id, category, country) VALUES (?, ?, ?);",
                (merchant_id, "E-COMMERCE", "India")
            )
            merchants_set.add(merchant_id)
            
        # 2. Insert Transaction Record
        cursor.execute(
            """
            INSERT INTO transactions (transaction_id, user_id, merchant_id, payment_id, amount, timestamp, status, is_abuse)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (txn_id, user_id, merchant_id, payment_id, amount, timestamp, "MONITOR", is_fraud)
        )
        
        # 3. Logins
        login_id = f"LOG-{row.TransactionID}"
        cursor.execute(
            """
            INSERT INTO logins (login_id, user_id, device_id, ip_id, timestamp, success)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (login_id, user_id, device_id, ip_id, timestamp, 1)
        )
        
        # 4. Aggregate Bipartite Edge Statistics (In-Memory Tracking)
        if device_id:
            key = (user_id, device_id)
            if key not in user_devices_dict:
                user_devices_dict[key] = {'first': timestamp, 'last': timestamp, 'count': 0}
            user_devices_dict[key]['last'] = timestamp
            user_devices_dict[key]['count'] += 1
            
        if ip_id:
            key = (user_id, ip_id)
            if key not in user_ips_dict:
                user_ips_dict[key] = {'first': timestamp, 'last': timestamp, 'count': 0}
            user_ips_dict[key]['last'] = timestamp
            user_ips_dict[key]['count'] += 1
            
        if payment_id:
            key = (user_id, payment_id)
            if key not in user_payments_dict:
                user_payments_dict[key] = {'first': timestamp, 'last': timestamp, 'count': 0}
            user_payments_dict[key]['last'] = timestamp
            user_payments_dict[key]['count'] += 1
            
    # Batch write bipartite edge records
    print("Writing bipartite relationship tables...")
    for (u_id, d_id), stats in user_devices_dict.items():
        cursor.execute(
            """
            INSERT INTO user_devices (user_id, device_id, first_seen_at, last_seen_at, usage_count)
            VALUES (?, ?, ?, ?, ?);
            """,
            (u_id, d_id, stats['first'], stats['last'], stats['count'])
        )
        
    for (u_id, i_id), stats in user_ips_dict.items():
        cursor.execute(
            """
            INSERT INTO user_ips (user_id, ip_id, first_seen_at, last_seen_at, usage_count)
            VALUES (?, ?, ?, ?, ?);
            """,
            (u_id, i_id, stats['first'], stats['last'], stats['count'])
        )
        
    for (u_id, p_id), stats in user_payments_dict.items():
        cursor.execute(
            """
            INSERT INTO user_payments (user_id, payment_id, first_seen_at, last_seen_at, usage_count)
            VALUES (?, ?, ?, ?, ?);
            """,
            (u_id, p_id, stats['first'], stats['last'], stats['count'])
        )
        
    # 5. Populate Analyst Investigation Tickets for Fraud Alerts
    print("Generating investigation queues...")
    cursor.execute(
        """
        SELECT transaction_id, timestamp, is_abuse FROM transactions 
        WHERE is_abuse = 1 OR amount > 150.0;
        """
    )
    flagged_txns = cursor.fetchall()
    
    for idx, t_row in enumerate(flagged_txns):
        inv_id = f"INV-{1000 + idx}"
        t_id = t_row['transaction_id']
        t_ts = t_row['timestamp']
        is_abuse = t_row['is_abuse']
        
        status = "PENDING_REVIEW"
        analyst_dec = None
        resolved_at = None
        notes = None
        
        # Chronologically resolve older alerts as a reference dataset
        if idx % 3 != 0:
            status = "CONFIRMED_ABUSE" if is_abuse == 1 else "FALSE_POSITIVE"
            analyst_dec = status
            resolved_at = t_ts
            notes = "Auto-resolved via historical batch baseline check."
            
        cursor.execute(
            """
            INSERT INTO investigations (investigation_id, alert_id, status, analyst_decision, notes, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (inv_id, t_id, status, analyst_dec, notes, t_ts, resolved_at)
        )
        
    conn.commit()
    conn.close()
    print("SQLite Relational database seeding complete.")

if __name__ == "__main__":
    init_db()
    tx_df, id_df = read_raw_data()
    merged = clean_and_merge(tx_df, id_df)
    seed_database(merged)
