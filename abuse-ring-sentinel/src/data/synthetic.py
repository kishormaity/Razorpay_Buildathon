import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from data.connection import get_config

def create_synthetic_db():
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    synthetic_db_path = os.path.join(proj_root, "data", "synthetic", "synthetic_benchmark.db")
    
    if os.path.exists(synthetic_db_path):
        try:
            os.remove(synthetic_db_path)
        except Exception as e:
            print(f"Could not remove existing database: {e}")
            
    os.makedirs(os.path.dirname(synthetic_db_path), exist_ok=True)
    conn = sqlite3.connect(synthetic_db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    
    # Read and apply schema
    schema_path = os.path.join(proj_root, "src", "data", "schema.sql")
    if not os.path.exists(schema_path):
        schema_path = os.path.join(proj_root, "src", "data", "schema.sql")
        
    with open(schema_path, "r") as f:
        schema_sql = f.read()
        
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()
    return synthetic_db_path

def generate_benchmark():
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    db_path = create_synthetic_db()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    
    ref_time = datetime(2026, 6, 1, 10, 0, 0)
    
    scenarios_metadata = []
    
    # ----------------------------------------------------
    # Scenario A: Shared Device (20 Users -> 1 Device)
    # ----------------------------------------------------
    a_users = [f"SYN-A-USR-{i:02d}" for i in range(1, 21)]
    a_device = "SYN-A-DEV-99"
    a_ip = "SYN-A-IP-99"
    a_pmt = "SYN-A-PMT-99"
    
    # Insert device
    cursor.execute("INSERT INTO devices (device_id, first_seen_at, device_type, os) VALUES (?, ?, ?, ?);",
                   (a_device, ref_time.strftime('%Y-%m-%dT%H:%M:%SZ'), "Mobile", "Android"))
    # Insert IP
    cursor.execute("INSERT INTO ips (ip_id, ip_hash, country, first_seen_at) VALUES (?, ?, ?, ?);",
                   (a_ip, "HASH_A_IP", "India", ref_time.strftime('%Y-%m-%dT%H:%M:%SZ')))
    # Insert Merchant
    cursor.execute("INSERT OR IGNORE INTO merchants (merchant_id, category, country) VALUES (?, ?, ?);",
                   ("MER-digital.com", "E-COMMERCE", "India"))
                   
    for idx, usr in enumerate(a_users):
        timestamp = (ref_time + timedelta(minutes=idx * 2)).strftime('%Y-%m-%dT%H:%M:%SZ')
        cursor.execute("INSERT INTO users (user_id, created_at, country) VALUES (?, ?, ?);",
                       (usr, timestamp, "India"))
                       
        pmt_id = f"{a_pmt}-{idx}"
        cursor.execute("INSERT INTO payment_methods (payment_id, payment_type, fingerprint_hash, first_seen_at) VALUES (?, ?, ?, ?);",
                       (pmt_id, "CREDIT", f"FP_A_{idx}", timestamp))
                       
        # Insert transaction
        cursor.execute("INSERT INTO transactions (transaction_id, user_id, merchant_id, payment_id, amount, timestamp, status, is_abuse) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                       (f"SYN-A-TXN-{idx:03d}", usr, "MER-digital.com", pmt_id, 1000.0, timestamp, "MONITOR", 1))
                       
        # Insert login
        cursor.execute("INSERT INTO logins (login_id, user_id, device_id, ip_id, timestamp, success) VALUES (?, ?, ?, ?, ?, ?);",
                       (f"SYN-A-LOG-{idx:03d}", usr, a_device, a_ip, timestamp, 1))
                       
        # Aggregate Edges
        cursor.execute("INSERT INTO user_devices (user_id, device_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, a_device, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_ips (user_id, ip_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, a_ip, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_payments (user_id, payment_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, pmt_id, timestamp, timestamp, 1))
                       
    scenarios_metadata.append({
        "scenario_id": "RING_A_DEVICE_SHARE",
        "scenario_type": "SHARED_DEVICE",
        "member_accounts": a_users,
        "shared_devices": [a_device],
        "shared_ips": [a_ip],
        "shared_payment_instruments": [],
        "start_time": ref_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "end_time": (ref_time + timedelta(minutes=40)).strftime('%Y-%m-%dT%H:%M:%SZ')
    })
    
    # ----------------------------------------------------
    # Scenario B: Shared IP (15 Users -> 1 IP)
    # ----------------------------------------------------
    b_users = [f"SYN-B-USR-{i:02d}" for i in range(1, 16)]
    b_ip = "SYN-B-IP-88"
    
    cursor.execute("INSERT INTO ips (ip_id, ip_hash, country, first_seen_at) VALUES (?, ?, ?, ?);",
                   (b_ip, "HASH_B_IP", "India", ref_time.strftime('%Y-%m-%dT%H:%M:%SZ')))
                   
    for idx, usr in enumerate(b_users):
        timestamp = (ref_time + timedelta(minutes=idx * 3)).strftime('%Y-%m-%dT%H:%M:%SZ')
        cursor.execute("INSERT INTO users (user_id, created_at, country) VALUES (?, ?, ?);",
                       (usr, timestamp, "India"))
                       
        pmt_id = f"SYN-B-PMT-{idx}"
        cursor.execute("INSERT INTO payment_methods (payment_id, payment_type, fingerprint_hash, first_seen_at) VALUES (?, ?, ?, ?);",
                       (pmt_id, "CREDIT", f"FP_B_{idx}", timestamp))
                       
        dev_id = f"SYN-B-DEV-{idx}"
        cursor.execute("INSERT INTO devices (device_id, first_seen_at, device_type, os) VALUES (?, ?, ?, ?);",
                       (dev_id, timestamp, "Desktop", "Windows"))
                       
        # Insert transaction
        cursor.execute("INSERT INTO transactions (transaction_id, user_id, merchant_id, payment_id, amount, timestamp, status, is_abuse) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                       (f"SYN-B-TXN-{idx:03d}", usr, "MER-digital.com", pmt_id, 1500.0, timestamp, "MONITOR", 1))
                       
        # Insert login
        cursor.execute("INSERT INTO logins (login_id, user_id, device_id, ip_id, timestamp, success) VALUES (?, ?, ?, ?, ?, ?);",
                       (f"SYN-B-LOG-{idx:03d}", usr, dev_id, b_ip, timestamp, 1))
                       
        # Aggregate Edges
        cursor.execute("INSERT INTO user_devices (user_id, device_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, dev_id, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_ips (user_id, ip_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, b_ip, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_payments (user_id, payment_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, pmt_id, timestamp, timestamp, 1))
                       
    scenarios_metadata.append({
        "scenario_id": "RING_B_IP_SHARE",
        "scenario_type": "SHARED_IP",
        "member_accounts": b_users,
        "shared_devices": [],
        "shared_ips": [b_ip],
        "shared_payment_instruments": [],
        "start_time": ref_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "end_time": (ref_time + timedelta(minutes=45)).strftime('%Y-%m-%dT%H:%M:%SZ')
    })
    
    # ----------------------------------------------------
    # Scenario C: Payment Instrument Reuse (10 Users -> 1 Payment ID)
    # ----------------------------------------------------
    c_users = [f"SYN-C-USR-{i:02d}" for i in range(1, 11)]
    c_pmt = "SYN-C-PMT-77"
    
    cursor.execute("INSERT INTO payment_methods (payment_id, payment_type, fingerprint_hash, first_seen_at) VALUES (?, ?, ?, ?);",
                   (c_pmt, "DEBIT", "HASH_C_PMT", ref_time.strftime('%Y-%m-%dT%H:%M:%SZ')))
                   
    for idx, usr in enumerate(c_users):
        timestamp = (ref_time + timedelta(minutes=idx * 5)).strftime('%Y-%m-%dT%H:%M:%SZ')
        cursor.execute("INSERT INTO users (user_id, created_at, country) VALUES (?, ?, ?);",
                       (usr, timestamp, "India"))
                       
        dev_id = f"SYN-C-DEV-{idx}"
        cursor.execute("INSERT INTO devices (device_id, first_seen_at, device_type, os) VALUES (?, ?, ?, ?);",
                       (dev_id, timestamp, "Mobile", "iOS"))
                       
        ip_id = f"SYN-C-IP-{idx}"
        cursor.execute("INSERT INTO ips (ip_id, ip_hash, country, first_seen_at) VALUES (?, ?, ?, ?);",
                       (ip_id, f"HASH_C_IP_{idx}", "India", timestamp))
                       
        # Insert transaction
        cursor.execute("INSERT INTO transactions (transaction_id, user_id, merchant_id, payment_id, amount, timestamp, status, is_abuse) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                       (f"SYN-C-TXN-{idx:03d}", usr, "MER-digital.com", c_pmt, 2500.0, timestamp, "MONITOR", 1))
                       
        # Insert login
        cursor.execute("INSERT INTO logins (login_id, user_id, device_id, ip_id, timestamp, success) VALUES (?, ?, ?, ?, ?, ?);",
                       (f"SYN-C-LOG-{idx:03d}", usr, dev_id, ip_id, timestamp, 1))
                       
        # Aggregate Edges
        cursor.execute("INSERT INTO user_devices (user_id, device_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, dev_id, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_ips (user_id, ip_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, ip_id, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_payments (user_id, payment_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, c_pmt, timestamp, timestamp, 1))
                       
    scenarios_metadata.append({
        "scenario_id": "RING_C_PAYMENT_REUSE",
        "scenario_type": "PAYMENT_REUSE",
        "member_accounts": c_users,
        "shared_devices": [],
        "shared_ips": [],
        "shared_payment_instruments": [c_pmt],
        "start_time": ref_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "end_time": (ref_time + timedelta(minutes=50)).strftime('%Y-%m-%dT%H:%M:%SZ')
    })
    
    # ----------------------------------------------------
    # Scenario D: Temporal Burst (5 Users created and transacting in 30 seconds)
    # ----------------------------------------------------
    d_users = [f"SYN-D-USR-{i:02d}" for i in range(1, 6)]
    
    for idx, usr in enumerate(d_users):
        timestamp = (ref_time + timedelta(seconds=idx * 6)).strftime('%Y-%m-%dT%H:%M:%SZ')
        cursor.execute("INSERT INTO users (user_id, created_at, country) VALUES (?, ?, ?);",
                       (usr, timestamp, "India"))
                       
        pmt_id = f"SYN-D-PMT-{idx}"
        cursor.execute("INSERT INTO payment_methods (payment_id, payment_type, fingerprint_hash, first_seen_at) VALUES (?, ?, ?, ?);",
                       (pmt_id, "CREDIT", f"FP_D_{idx}", timestamp))
                       
        dev_id = f"SYN-D-DEV-{idx}"
        cursor.execute("INSERT INTO devices (device_id, first_seen_at, device_type, os) VALUES (?, ?, ?, ?);",
                       (dev_id, timestamp, "Mobile", "Android"))
                       
        ip_id = f"SYN-D-IP-{idx}"
        cursor.execute("INSERT INTO ips (ip_id, ip_hash, country, first_seen_at) VALUES (?, ?, ?, ?);",
                       (ip_id, f"HASH_D_IP_{idx}", "India", timestamp))
                       
        # Insert transaction
        cursor.execute("INSERT INTO transactions (transaction_id, user_id, merchant_id, payment_id, amount, timestamp, status, is_abuse) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                       (f"SYN-D-TXN-{idx:03d}", usr, "MER-digital.com", pmt_id, 300.0, timestamp, "MONITOR", 1))
                       
        # Insert login
        cursor.execute("INSERT INTO logins (login_id, user_id, device_id, ip_id, timestamp, success) VALUES (?, ?, ?, ?, ?, ?);",
                       (f"SYN-D-LOG-{idx:03d}", usr, dev_id, ip_id, timestamp, 1))
                       
        # Aggregate Edges
        cursor.execute("INSERT INTO user_devices (user_id, device_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, dev_id, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_ips (user_id, ip_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, ip_id, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_payments (user_id, payment_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, pmt_id, timestamp, timestamp, 1))
                       
    scenarios_metadata.append({
        "scenario_id": "RING_D_TEMPORAL_BURST",
        "scenario_type": "TEMPORAL_BURST",
        "member_accounts": d_users,
        "shared_devices": [],
        "shared_ips": [],
        "shared_payment_instruments": [],
        "start_time": ref_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "end_time": (ref_time + timedelta(seconds=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    })
    
    # ----------------------------------------------------
    # Scenario E: Hybrid Overlapping Ring (12 Users with overlapping elements)
    # ----------------------------------------------------
    e_users = [f"SYN-E-USR-{i:02d}" for i in range(1, 13)]
    e_devices = ["SYN-E-DEV-1", "SYN-E-DEV-2"]
    e_ips = ["SYN-E-IP-1", "SYN-E-IP-2"]
    e_pmt = "SYN-E-PMT-SHARED"
    
    # Insert device 1 & 2
    for d in e_devices:
        cursor.execute("INSERT INTO devices (device_id, first_seen_at, device_type, os) VALUES (?, ?, ?, ?);",
                       (d, ref_time.strftime('%Y-%m-%dT%H:%M:%SZ'), "Desktop", "macOS"))
    # Insert IP 1 & 2
    for ip in e_ips:
        cursor.execute("INSERT INTO ips (ip_id, ip_hash, country, first_seen_at) VALUES (?, ?, ?, ?);",
                       (ip, f"HASH_{ip}", "India", ref_time.strftime('%Y-%m-%dT%H:%M:%SZ')))
    # Insert Payment Instrument
    cursor.execute("INSERT INTO payment_methods (payment_id, payment_type, fingerprint_hash, first_seen_at) VALUES (?, ?, ?, ?);",
                   (e_pmt, "CREDIT", "HASH_E_PMT_SHARED", ref_time.strftime('%Y-%m-%dT%H:%M:%SZ')))
                   
    for idx, usr in enumerate(e_users):
        timestamp = (ref_time + timedelta(minutes=idx * 4)).strftime('%Y-%m-%dT%H:%M:%SZ')
        cursor.execute("INSERT INTO users (user_id, created_at, country) VALUES (?, ?, ?);",
                       (usr, timestamp, "India"))
                       
        # Rotate devices and IPs to simulate coordination overlap
        dev_id = e_devices[idx % 2]
        ip_id = e_ips[idx % 2]
        
        # Insert transaction
        cursor.execute("INSERT INTO transactions (transaction_id, user_id, merchant_id, payment_id, amount, timestamp, status, is_abuse) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                       (f"SYN-E-TXN-{idx:03d}", usr, "MER-digital.com", e_pmt, 180.0, timestamp, "MONITOR", 1))
                       
        # Insert login
        cursor.execute("INSERT INTO logins (login_id, user_id, device_id, ip_id, timestamp, success) VALUES (?, ?, ?, ?, ?, ?);",
                       (f"SYN-E-LOG-{idx:03d}", usr, dev_id, ip_id, timestamp, 1))
                       
        # Aggregate Edges
        cursor.execute("INSERT INTO user_devices (user_id, device_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, dev_id, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_ips (user_id, ip_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, ip_id, timestamp, timestamp, 1))
        cursor.execute("INSERT INTO user_payments (user_id, payment_id, first_seen_at, last_seen_at, usage_count) VALUES (?, ?, ?, ?, ?);",
                       (usr, e_pmt, timestamp, timestamp, 1))
                       
    scenarios_metadata.append({
        "scenario_id": "RING_E_HYBRID_OVERLAP",
        "scenario_type": "HYBRID",
        "member_accounts": e_users,
        "shared_devices": e_devices,
        "shared_ips": e_ips,
        "shared_payment_instruments": [e_pmt],
        "start_time": ref_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "end_time": (ref_time + timedelta(minutes=48)).strftime('%Y-%m-%dT%H:%M:%SZ')
    })
    
    conn.commit()
    conn.close()
    
    # Save metadata descriptor file
    metadata_path = os.path.join(proj_root, "data", "synthetic", "scenario_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(scenarios_metadata, f, indent=2)
        
    print(f"Track B synthetic benchmark database seeded at: {db_path}")
    print(f"Scenario metadata descriptor file saved to: {metadata_path}")

if __name__ == "__main__":
    generate_benchmark()
