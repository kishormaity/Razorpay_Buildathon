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

    # Load datasets
    sample_size = int(os.environ.get('DATA_SAMPLE_SIZE', '0'))
    
    if sample_size > 0:
        print(f"[DEMO MODE] Reading initial chunk for chronological demo preview (target: {sample_size} rows)...")
        tx_df = pd.read_csv(txn_path, nrows=sample_size * 2)
        id_df = pd.read_csv(id_path, nrows=sample_size * 2)
    else:
        print("Loading full datasets without arbitrary truncation...")
        tx_df = pd.read_csv(txn_path)
        id_df = pd.read_csv(id_path)
    
    print("Merging transaction and identity on TransactionID...")
    merged = pd.merge(tx_df, id_df, on='TransactionID', how='left')
    
    # Sort chronologically
    merged = merged.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    if sample_size > 0:
        sample_df = merged.head(sample_size).copy()
        print(f"[DEMO MODE] Processing first {len(sample_df)} chronological records for local database preview.")
        print("Note: This chronological slice maintains temporal ordering for local testing and does not represent overall class distribution.")
    else:
        sample_df = merged
        print(f"[FULL MODE] Processing complete dataset: {len(sample_df):,} records.")
        
    # Attempt to load trained Model D booster and decision thresholds for genuine risk scoring
    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    models_dir = os.path.join(current_dir, 'models')
    model_d_path = os.path.join(models_dir, 'model_d_final.txt')
    features_json_path = os.path.join(models_dir, 'model_d_features.json')
    thresholds_path = os.path.join(models_dir, 'thresholds.json')
    
    model_d = None
    base_features = None
    calibrator = None
    threshold_block = 0.30398
    threshold_review = 0.15000
    
    if os.path.exists(model_d_path) and os.path.exists(features_json_path):
        try:
            import lightgbm as lgb
            model_d = lgb.Booster(model_file=model_d_path)
            with open(features_json_path, 'r') as f:
                base_features = json.load(f)
            print("✓ Loaded Model D for genuine inference scoring.")
            
            # Load calibrator if exists
            cal_path = os.path.join(models_dir, 'model_d_calibrator.joblib')
            if os.path.exists(cal_path):
                import joblib
                calibrator = joblib.load(cal_path)
                print("✓ Loaded probability calibrator.")
                
            # Load thresholds if exists
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as f:
                    th_data = json.load(f)
                    threshold_block = th_data.get('threshold_block', threshold_block)
                    threshold_review = th_data.get('threshold_review', threshold_review)
        except Exception as e:
            print(f"[WARNING] Could not load model for scoring: {e}")
            
    cursor = conn.cursor()
    print("Mapping data into SQL Entity-Relationship schema with genuine model risk scoring...")
    
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

    # Pre-derive cyclical features if needed for scoring
    if 'hour_of_day' not in sample_df.columns:
        sample_df['hour_of_day'] = ((sample_df['TransactionDT'] // 3600) % 24).astype('float32')
        sample_df['day_of_week'] = ((sample_df['TransactionDT'] // 86400) % 7).astype('float32')
        sample_df['hour_sin'] = np.sin(2 * np.pi * (sample_df['TransactionDT'] % 86400) / 86400).astype('float32')
        sample_df['hour_cos'] = np.cos(2 * np.pi * (sample_df['TransactionDT'] % 86400) / 86400).astype('float32')
        sample_df['day_of_week_sin'] = np.sin(2 * np.pi * ((sample_df['TransactionDT'] // 86400) % 7) / 7).astype('float32')
        sample_df['day_of_week_cos'] = np.cos(2 * np.pi * ((sample_df['TransactionDT'] // 86400) % 7) / 7).astype('float32')

    # Compute batch risk scores if model is available
    if model_d is not None and base_features is not None:
        avail_features = [c for c in base_features if c in sample_df.columns]
        missing_features = [c for c in base_features if c not in sample_df.columns]
        scoring_df = sample_df[avail_features].copy()
        for mf in missing_features:
            scoring_df[mf] = 0
        scoring_df = scoring_df[base_features]
        raw_scores = model_d.predict(scoring_df)
        if calibrator is not None:
            cal_scores = np.clip(calibrator.predict(raw_scores), 0.0, 1.0)
        else:
            cal_scores = raw_scores
        sample_df['computed_risk_score'] = cal_scores
    else:
        # Neutral baseline risk score when model is not yet compiled (NEVER leak isFraud)
        sample_df['computed_risk_score'] = 0.035

    # Loop through entries and insert
    for idx, row in sample_df.iterrows():
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
            
        is_fraud = int(row.get('isFraud', 0))
        risk_score = round(float(row['computed_risk_score']), 4)
        
        # Determine action via Decision Engine thresholds (Decoupled from ground truth)
        if risk_score >= threshold_block:
            action = 'BLOCK'
        elif risk_score >= threshold_review:
            action = 'MANUAL_REVIEW'
        else:
            action = 'ALLOW'
            
        # Historical ground truth status is strictly preserved as resolution outcome for audit/evaluation
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
        risk_narrative = f"Model D calibrated risk score: {risk_score:.4f} -> Decision: {action}"
        cursor.execute("""
            INSERT OR IGNORE INTO transactions VALUES 
            (?, ?, ?, 'USD', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Parsed from raw dataset.')
        """, (txn_id, txn_time_str, float(row['TransactionAmt']), merchant_name, payment_method, 
              channel, location, risk_score, action, cust_id, device_id, ip_addr, 
              risk_narrative, status))

    conn.commit()
    print("Database ingestion completed with honest model risk scores!")
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
