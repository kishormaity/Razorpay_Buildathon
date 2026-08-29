import os
import pandas as pd
import numpy as np

def extract_transaction_features(conn):
    """
    Extracts baseline transaction features directly from the SQLite database.
    """
    query = """
    SELECT 
        t.transaction_id as TransactionID,
        t.amount as TransactionAmt,
        t.timestamp,
        t.status,
        t.is_abuse as isFraud,
        u.user_id,
        u.country as user_country,
        u.email_hash,
        l.device_id,
        d.device_type,
        d.os,
        l.ip_id,
        i.country as ip_country,
        p.payment_id,
        p.payment_type,
        p.fingerprint_hash,
        m.merchant_id,
        m.category as merchant_category,
        m.country as merchant_country
    FROM transactions t
    LEFT JOIN users u ON t.user_id = u.user_id
    LEFT JOIN logins l ON t.user_id = l.user_id AND t.timestamp = l.timestamp
    LEFT JOIN devices d ON l.device_id = d.device_id
    LEFT JOIN ips i ON l.ip_id = i.ip_id
    LEFT JOIN payment_methods p ON t.payment_id = p.payment_id
    LEFT JOIN merchants m ON t.merchant_id = m.merchant_id;
    """
    df = pd.read_sql_query(query, conn)
    
    # Parse relative TransactionDT in seconds from timestamp (Jan 1, 2026 reference)
    from datetime import datetime
    ref_date = datetime(2026, 1, 1)
    
    def get_dt_seconds(ts_str):
        try:
            dt = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%SZ')
            return int((dt - ref_date).total_seconds())
        except (ValueError, TypeError):
            return 0
            
    df['TransactionDT'] = df['timestamp'].apply(get_dt_seconds)
    
    # Parse numeric identifiers for GBDTs
    # Extrapolate numeric indices from user / device / ip keys
    df['card1'] = df['user_id'].str.replace('CUS-', '', regex=False).fillna(0).astype(int)
    
    # Parse card2, card3, etc from payment methods
    def parse_card2(pmt_id):
        if pd.isna(pmt_id):
            return 0
        parts = pmt_id.replace('PMT-', '').split('-')
        return int(parts[1]) if len(parts) > 1 else 0
        
    df['card2'] = df['payment_id'].apply(parse_card2)
    
    # Set default/missing values for categories
    df['ProductCD'] = df['merchant_category'].apply(lambda x: 'W' if x == 'E-Commerce' else 'H')
    df['addr1'] = df['ip_id'].str.replace('IP-', '', regex=False).fillna(-999).astype(float)
    df['addr2'] = df['user_country'].str.replace('Country-', '', regex=False).fillna('UNKNOWN')
    df['P_emaildomain'] = df['merchant_id'].str.replace('MER-', '', regex=False).fillna('UNKNOWN')
    
    # Drop raw timestamp to prevent label leaks / parsing complexity
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    return df
