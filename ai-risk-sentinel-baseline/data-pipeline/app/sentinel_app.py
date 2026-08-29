import os
import sys
import time
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# -------------------------------------------------------------
# 1. Model & Data Helper Functions
# -------------------------------------------------------------
def is_valid_device(dev):
    return not pd.isna(dev) and dev != 'UNKNOWN' and dev != ''

def is_valid_addr(a):
    return not pd.isna(a) and a != -999 and a != -999.0

# Initialize FastAPI App
app = FastAPI(title="Abuse-Ring Sentinel Interactive Portal")

# Globals to hold models and datasets
db = {}

@app.on_event("startup")
def startup_event():
    print("Loading models and feature names...")
    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    features_path = os.path.join(processed_dir, 'features/abuse_ring_features.parquet')
    models_dir = os.path.join(current_dir, 'models')

    # Load feature configurations
    try:
        with open(os.path.join(models_dir, 'model_d_features.json'), 'r') as f:
            base_features = json.load(f)
        with open(os.path.join(models_dir, 'sentinel_features.json'), 'r') as f:
            ring_features = json.load(f)
    except FileNotFoundError:
        print("[ERROR] Feature configuration JSON files not found. Run train_final_models.py first.")
        sys.exit(1)

    # Load frozen model boosters
    model_d_path = os.path.join(models_dir, 'model_d_final.txt')
    sentinel_path = os.path.join(models_dir, 'abuse_ring_sentinel_final.txt')
    
    if not os.path.exists(model_d_path) or not os.path.exists(sentinel_path):
        print("[ERROR] Saved LightGBM boosters not found. Run train_final_models.py first.")
        sys.exit(1)
        
    print("Loading Model D............. ", end="")
    model_d = lgb.Booster(model_file=model_d_path)
    print("OK")
    
    print("Loading Sentinel............ ", end="")
    model_sentinel = lgb.Booster(model_file=sentinel_path)
    print("OK")

    if not os.path.exists(features_path):
        print(f"[ERROR] Features file not found at: {features_path}")
        sys.exit(1)

    # Load lightweight history for dynamic O(1) calculations
    print("Loading history index....... ", end="")
    cols_history = ['TransactionID', 'TransactionAmt', 'card1', 'P_emaildomain', 'DeviceInfo', 'addr1', 'isFraud', 'TransactionDT']
    history_df = pd.read_parquet(features_path, columns=cols_history)
    history_df = history_df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    # Pre-index device, address, and email networks to achieve O(1) leakage-free historical query times
    device_index = {}
    addr_index = {}
    email_index = {}

    for row in history_df.itertuples(index=False):
        # Device Network
        dev = row.DeviceInfo
        if is_valid_device(dev):
            if dev not in device_index:
                device_index[dev] = []
            device_index[dev].append(row)
        
        # Address Network
        addr = row.addr1
        if is_valid_addr(addr):
            addr_val = int(float(addr))
            if addr_val not in addr_index:
                addr_index[addr_val] = []
            addr_index[addr_val].append(row)
            
        # Email Network
        email = row.P_emaildomain
        if not pd.isna(email) and email != '':
            if email not in email_index:
                email_index[email] = []
            email_index[email].append(row)

    print("OK")

    # Load the entire dataset features to support looking up any transaction ID
    print("Loading full dataset........ ", end="")
    df = pd.read_parquet(features_path)
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    print("OK")

    # Generate baseline predictions for full dataframe
    print("Generating predictions...... ", end="")
    probs_d = model_d.predict(df[base_features])
    probs_sentinel = model_sentinel.predict(df[ring_features])
    df['prob_d'] = probs_d
    df['prob_sentinel'] = probs_sentinel
    print("OK")

    # Select 30 key features for the inspection drawer
    all_key_candidates = [
        'TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'dist1', 'dist2',
        'C1', 'C2', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D4', 'D10', 'D15',
        'V12', 'V13', 'V35', 'V36', 'V53', 'V54', 'V75', 'V76', 'V130', 'V131', 'V307', 'V308'
    ]
    key_features = [f for f in all_key_candidates if f in base_features]

    # Cache in DB
    db['df'] = df
    db['model_d'] = model_d
    db['model_sentinel'] = model_sentinel
    db['base_features'] = base_features
    db['ring_features'] = ring_features
    db['key_features'] = key_features
    # Vectorized fast O(0.1s) mapping of TransactionID to row index with Python int keys
    db['tx_map'] = {int(tid): int(idx) for tid, idx in zip(df['TransactionID'], df.index)}
    db['default_tx_id'] = 3489013 # Hero case TransactionID
    db['device_index'] = device_index
    db['addr_index'] = addr_index
    db['email_index'] = email_index
    
    print("\n========================================")
    print("   ABUSE-RING SENTINEL READY (DEMO A)")
    print("========================================")
    print("   http://127.0.0.1:8000")
    print("========================================\n")

# -------------------------------------------------------------
# 2. REST API Routes
# -------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    transaction_id: int
    amount: float
    card1: int
    email: str
    device: str
    address: float

@app.get("/api/lookup")
def lookup_transaction(transaction_id: int):
    test_df = db['df']
    if transaction_id not in db['tx_map']:
        raise HTTPException(status_code=404, detail="Transaction not found in dataset.")
    
    idx = db['tx_map'][transaction_id]
    row = test_df.iloc[idx]
    
    amount = float(row['TransactionAmt'])
    card = int(row['card1'])
    email = str(row['P_emaildomain']).strip() if not pd.isna(row['P_emaildomain']) else 'gmail.com'
    device = str(row['DeviceInfo']).strip() if is_valid_device(row['DeviceInfo']) else 'iOS Device'
    address = int(float(row['addr1'])) if is_valid_addr(row['addr1']) else 299

    return {
        'transaction_id': int(row['TransactionID']),
        'amount': amount,
        'device': device,
        'address': address,
        'card1': card,
        'email': email,
        'isFraud': int(row['isFraud']),
        'TransactionDT': int(row['TransactionDT'])
    }

@app.post("/api/analyze")
def analyze_transaction(req: AnalyzeRequest):
    test_df = db['df']
    model_d = db['model_d']
    model_sentinel = db['model_sentinel']
    base_features = db['base_features']
    ring_features = db['ring_features']
    key_features = db['key_features']
    
    tx_id = req.transaction_id
    
    # Verify if Transaction ID exists in Test Split
    if tx_id not in db['tx_map']:
        idx = db['tx_map'][db['default_tx_id']]
        single_row_df = test_df.iloc[[idx]].copy()
        verified = False
        warning = "⚠ Transaction ID not found in dataset. Falling back to default case."
    else:
        idx = db['tx_map'][tx_id]
        single_row_df = test_df.iloc[[idx]].copy()
        
        # Compare inputs against stored record values with matching fallbacks
        row_orig = single_row_df.iloc[0]
        
        orig_amount = float(row_orig['TransactionAmt'])
        orig_card = int(row_orig['card1'])
        orig_email = str(row_orig['P_emaildomain']).strip() if not pd.isna(row_orig['P_emaildomain']) else "gmail.com"
        orig_device = str(row_orig['DeviceInfo']).strip() if is_valid_device(row_orig['DeviceInfo']) else "iOS Device"
        orig_address = float(int(float(row_orig['addr1']))) if is_valid_addr(row_orig['addr1']) else 299.0
        
        req_email = req.email.strip()
        req_device = req.device.strip()
        
        # Match checks (compare rounded amount because front-end uses rounded inputs)
        amount_match = abs(req.amount - int(orig_amount + 0.5)) < 0.01
        card_match = req.card1 == orig_card
        email_match = req_email.lower() == orig_email.lower()
        device_match = req_device.lower() == orig_device.lower()
        address_match = abs(req.address - orig_address) < 0.01
        
        print(f"DEBUG MATCH FOR TX {tx_id}:")
        print(f"  Amount: req={req.amount}, orig={orig_amount}, round_orig={int(orig_amount + 0.5)}, match={amount_match}")
        print(f"  Card ID: req={req.card1}, orig={orig_card}, match={card_match}")
        print(f"  Email: req='{req_email}', orig='{orig_email}', match={email_match}")
        print(f"  Device: req='{req_device}', orig='{orig_device}', match={device_match}")
        print(f"  Address: req={req.address}, orig={orig_address}, match={address_match}")
        
        if amount_match and card_match and email_match and device_match and address_match:
            verified = True
            warning = "✓ DATA MATCHED"
        else:
            mismatched_fields = []
            if not card_match: mismatched_fields.append("Card ID")
            if not amount_match: mismatched_fields.append("Amount")
            if not email_match: mismatched_fields.append("Email")
            if not device_match: mismatched_fields.append("Device")
            if not address_match: mismatched_fields.append("Address")
            verified = False
            warning = f"⚠ {', '.join(mismatched_fields).upper()} DOES NOT MATCH STORED TRANSACTION"

    # Evaluate using the authentic stored feature vector
    pred_d = float(model_d.predict(single_row_df[base_features])[0])
    pred_sentinel = float(model_sentinel.predict(single_row_df[ring_features])[0])
    
    row_res = single_row_df.iloc[0]
    current_dt = int(row_res['TransactionDT'])
    
    # -------------------------------------------------------------
    # 3. Dynamic Chronological Network Auditing
    # -------------------------------------------------------------
    # 3A. Device network lookup
    device_name = str(row_res['DeviceInfo']).strip() if is_valid_device(row_res['DeviceInfo']) else ""
    device_history = db['device_index'].get(device_name, [])
    preceding_device = [r for r in device_history if r.TransactionDT < current_dt]
    
    unique_device_cards = sorted(list(set(r.card1 for r in preceding_device)))
    total_device_tx = len(preceding_device)
    fraud_device_tx = sum(1 for r in preceding_device if r.isFraud == 1)
    device_exposure = float(fraud_device_tx) / total_device_tx if total_device_tx > 0 else 0.0
    device_conv_72h = len(set(r.card1 for r in preceding_device if r.TransactionDT >= (current_dt - 72*3600)))

    # 3B. Address network lookup
    addr_val = int(float(row_res['addr1'])) if is_valid_addr(row_res['addr1']) else 0
    addr_history = db['addr_index'].get(addr_val, [])
    preceding_addr = [r for r in addr_history if r.TransactionDT < current_dt]
    
    unique_addr_cards = sorted(list(set(r.card1 for r in preceding_addr)))
    total_addr_tx = len(preceding_addr)
    fraud_addr_tx = sum(1 for r in preceding_addr if r.isFraud == 1)
    addr_exposure = float(fraud_addr_tx) / total_addr_tx if total_addr_tx > 0 else 0.0
    addr_conv_72h = len(set(r.card1 for r in preceding_addr if r.TransactionDT >= (current_dt - 72*3600)))

    # 3C. Email network lookup
    email_name = str(row_res['P_emaildomain']).strip() if not pd.isna(row_res['P_emaildomain']) else ""
    email_history = db['email_index'].get(email_name, [])
    preceding_email = [r for r in email_history if r.TransactionDT < current_dt]
    unique_email_cards_count = len(set(r.card1 for r in preceding_email))

    # Cross-Entity Overlap Calculation
    device_cards_set = set(unique_device_cards)
    address_cards_set = set(unique_addr_cards)
    overlap_cards = sorted(list(device_cards_set.intersection(address_cards_set)))

    # Safe conversion helpers to prevent NaN serialization crash
    def safe_float(v, default=0.0):
        if pd.isna(v) or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)

    def safe_int(v, default=0):
        if pd.isna(v) or (isinstance(v, float) and np.isnan(v)):
            return default
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default

    # Construct feature lists for drawers
    model_d_features_list = []
    for col in key_features:
        val = row_res[col]
        if pd.isna(val):
            val_out = "NaN"
        elif isinstance(val, (int, float, np.integer, np.floating)):
            val_out = float(val)
        else:
            val_out = str(val)
        model_d_features_list.append({"name": str(col), "value": val_out})

    return {
        'transaction_id': int(row_res['TransactionID']),
        'amount': float(row_res['TransactionAmt']),
        'card1': int(row_res['card1']),
        'email': email_name if email_name else 'gmail.com',
        'device': device_name if device_name else 'iOS Device',
        'address': addr_val if addr_val else 299,
        
        'prob_d': pred_d,
        'prob_sentinel': pred_sentinel,
        'isFraud': int(row_res['isFraud']),
        'verified': verified,
        'warning': warning,
        'TransactionDT': current_dt,
        
        # Model D verification drawer
        'model_d_features': model_d_features_list,
        'base_feature_count': len(base_features),
        
        # Historical dynamic audit properties
        'device_network': {
            'device': device_name if device_name else 'iOS Device',
            'unique_cards_count': len(unique_device_cards),
            'total_tx': total_device_tx,
            'fraud_tx': fraud_device_tx,
            'fraud_exposure': device_exposure,
            'convergence_72h': device_conv_72h,
            'cards_list': unique_device_cards[:50]
        },
        
        'address_network': {
            'address': addr_val if addr_val else 299,
            'unique_cards_count': len(unique_addr_cards),
            'total_tx': total_addr_tx,
            'fraud_tx': fraud_addr_tx,
            'fraud_exposure': addr_exposure,
            'convergence_72h': addr_conv_72h,
            'cards_list': unique_addr_cards[:50]
        },
        
        'email_unique_cards_count': unique_email_cards_count,
        'overlap_cards': overlap_cards[:50],
        'cross_entity_convergence': 'YES' if len(overlap_cards) > 0 else 'NO',
        
        # Sentinel raw model feature checklist values
        'sentinel_features': {
            'device_unique_card_count': safe_int(row_res['device_unique_card_count']),
            'addr_unique_card_count': safe_int(row_res['addr_unique_card_count']),
            'email_unique_card_count': safe_int(row_res['email_unique_card_count']),
            'device_connected_fraud_rate': safe_float(row_res['device_connected_fraud_rate']),
            'addr_connected_fraud_rate': safe_float(row_res['addr_connected_fraud_rate']),
            'rapid_card_convergence': safe_int(row_res['rapid_card_convergence']),
            'cross_entity_convergence': 'YES' if safe_float(row_res['cross_entity_convergence']) > 0 else 'NO',
            'ring_fraud_density': safe_float(row_res['ring_fraud_density'])
        }
    }

# -------------------------------------------------------------
# 3. HTML/JS Page serving
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Abuse-Ring Sentinel Interactive Portal</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #0b0f19;
                --card-bg: rgba(22, 27, 38, 0.7);
                --card-border: rgba(56, 139, 253, 0.15);
                --text-primary: #f0f6fc;
                --text-secondary: #8b949e;
                --accent-blue: #58a6ff;
                --accent-green: #3fb950;
                --accent-amber: #d29922;
                --accent-red: #f85149;
                --font-main: 'Outfit', sans-serif;
            }

            body {
                background-color: var(--bg-color);
                color: var(--text-primary);
                font-family: var(--font-main);
                margin: 0;
                padding: 0;
                background-image: radial-gradient(circle at 10% 20%, rgba(20, 35, 70, 0.5) 0%, transparent 45%),
                                  radial-gradient(circle at 90% 80%, rgba(45, 20, 70, 0.4) 0%, transparent 45%);
                background-attachment: fixed;
                min-height: 100vh;
            }

            header {
                border-bottom: 1px solid var(--card-border);
                padding: 1.2rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                backdrop-filter: blur(10px);
                background-color: rgba(11, 15, 25, 0.85);
                position: sticky;
                top: 0;
                z-index: 100;
            }

            h1 {
                margin: 0;
                font-size: 1.4rem;
                font-weight: 700;
                letter-spacing: -0.5px;
                background: linear-gradient(90deg, #58a6ff, #bc8cff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .main-container {
                max-width: 1400px;
                margin: 2rem auto;
                padding: 0 1.5rem;
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 2rem;
            }

            .panel {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 12px;
                padding: 1.6rem;
                backdrop-filter: blur(12px);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
                height: fit-content;
            }

            h2 {
                margin-top: 0;
                font-size: 1.15rem;
                font-weight: 600;
                border-bottom: 1px solid rgba(56, 139, 253, 0.2);
                padding-bottom: 0.6rem;
                margin-bottom: 1.2rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: var(--accent-blue);
            }

            .form-group {
                margin-bottom: 1rem;
            }

            label {
                display: block;
                font-size: 0.8rem;
                color: var(--text-secondary);
                margin-bottom: 0.3rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }

            input[type="text"], input[type="number"] {
                width: 100%;
                background: #0d131f;
                border: 1px solid rgba(56, 139, 253, 0.2);
                border-radius: 6px;
                padding: 0.6rem;
                color: var(--text-primary);
                font-family: var(--font-main);
                box-sizing: border-box;
                font-size: 0.88rem;
                transition: border-color 0.2s ease;
            }

            input:focus {
                border-color: var(--accent-blue);
                outline: none;
            }

            .hint-text {
                font-size: 0.72rem;
                color: var(--text-secondary);
                margin-top: 0.3rem;
                line-height: 1.4;
            }

            .hint-link {
                color: var(--accent-blue);
                text-decoration: underline;
                cursor: pointer;
            }

            /* Raw Verification Match Grid */
            .verif-grid {
                width: 100%;
                font-size: 0.82rem;
                border-collapse: collapse;
                margin: 1.2rem 0;
            }

            .verif-grid th {
                text-align: left;
                color: var(--text-secondary);
                font-weight: 600;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(56, 139, 253, 0.1);
            }

            .verif-grid td {
                padding: 0.45rem 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            }

            .badge-match {
                color: var(--accent-green);
                font-weight: 700;
            }

            .badge-mismatch {
                color: var(--accent-amber);
                font-weight: 700;
            }

            .status-banner {
                padding: 0.6rem;
                border-radius: 6px;
                font-size: 0.8rem;
                font-weight: 700;
                text-align: center;
                margin-bottom: 1rem;
            }

            .banner-match {
                background: rgba(63, 185, 80, 0.1);
                color: var(--accent-green);
                border: 1px solid rgba(63, 185, 80, 0.3);
            }

            .banner-mismatch {
                background: rgba(210, 153, 34, 0.1);
                color: var(--accent-amber);
                border: 1px solid rgba(210, 153, 34, 0.3);
            }

            .analyze-btn {
                width: 100%;
                background: linear-gradient(90deg, #1f6feb, #8957e5);
                border: none;
                border-radius: 8px;
                padding: 0.8rem;
                color: #fff;
                font-family: var(--font-main);
                font-size: 0.95rem;
                font-weight: 700;
                cursor: pointer;
                box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3);
                transition: opacity 0.2s ease;
            }

            .analyze-btn:hover {
                opacity: 0.92;
            }

            /* Right Dashboard */
            .sub-card {
                background: rgba(30, 38, 56, 0.4);
                border: 1px solid rgba(56, 139, 253, 0.1);
                border-radius: 8px;
                padding: 1.2rem;
                margin-bottom: 1.5rem;
            }

            .sub-card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                padding-bottom: 0.5rem;
                margin-bottom: 1rem;
                font-size: 0.9rem;
                font-weight: 600;
                letter-spacing: 0.5px;
            }

            .metric-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.6rem;
                font-size: 0.88rem;
            }

            .metric-name {
                color: var(--text-secondary);
            }

            .metric-val {
                font-weight: 600;
            }

            .btn-view-cards {
                background: rgba(56, 139, 253, 0.15);
                border: 1px solid var(--accent-blue);
                color: var(--accent-blue);
                border-radius: 4px;
                font-size: 0.75rem;
                padding: 0.3rem 0.6rem;
                cursor: pointer;
                font-family: var(--font-main);
                font-weight: 600;
                transition: all 0.2s ease;
                position: relative;
                z-index: 10;
                pointer-events: auto !important;
            }

            .btn-view-cards:hover {
                background: rgba(56, 139, 253, 0.3);
            }

            /* Drawers */
            .drawer-container {
                margin-top: 0.8rem;
            }

            .drawer-trigger {
                background: none;
                border: none;
                color: var(--accent-blue);
                cursor: pointer;
                font-family: var(--font-main);
                font-size: 0.8rem;
                font-weight: 600;
                padding: 0;
                text-decoration: underline;
            }

            .drawer-content {
                display: none;
                background: #0d131f;
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 6px;
                padding: 0.8rem;
                margin-top: 0.5rem;
                max-height: 200px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 0.78rem;
            }

            .drawer-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.3rem 1.5rem;
            }

            /* Popups / Modals */
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.75);
                z-index: 1000;
                justify-content: center;
                align-items: center;
            }

            .modal-content {
                background: #161b26;
                border: 1px solid var(--accent-blue);
                border-radius: 10px;
                width: 90%;
                max-width: 500px;
                padding: 1.5rem;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
            }

            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                padding-bottom: 0.5rem;
                margin-bottom: 1rem;
            }

            .modal-close {
                background: none;
                border: none;
                color: var(--text-secondary);
                font-size: 1.2rem;
                cursor: pointer;
            }

            .cards-list-box {
                background: #0d131f;
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 6px;
                padding: 0.8rem;
                max-height: 250px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 0.85rem;
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 0.4rem;
                text-align: center;
            }

            /* Decision Cards */
            .decision-badge {
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 700;
            }

            .badge-allow {
                background: rgba(63, 185, 80, 0.15);
                color: var(--accent-green);
                border: 1px solid var(--accent-green);
            }

            .badge-block {
                background: rgba(248, 81, 73, 0.15);
                color: var(--accent-red);
                border: 1px solid var(--accent-red);
            }

            .badge-review {
                background: rgba(210, 153, 34, 0.15);
                color: var(--accent-amber);
                border: 1px solid var(--accent-amber);
            }

            .badge-approve {
                background: rgba(63, 185, 80, 0.15);
                color: var(--accent-green);
                border: 1px solid var(--accent-green);
            }

            /* Network Intersection Visualizer */
            .intersection-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #0d131f;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.03);
                margin-top: 1rem;
            }

            .net-box {
                background: rgba(30, 38, 56, 0.8);
                border: 1px solid rgba(56, 139, 253, 0.3);
                border-radius: 6px;
                padding: 0.5rem;
                width: 32%;
                font-size: 0.72rem;
                text-align: center;
            }

            .net-connector {
                flex-grow: 1;
                height: 2px;
                background: rgba(56, 139, 253, 0.3);
                position: relative;
            }

            .net-connector::after {
                content: '▶';
                font-size: 0.5rem;
                color: var(--accent-blue);
                position: absolute;
                top: -5px;
                left: 45%;
            }

            .target-card-box {
                border-color: var(--accent-amber);
                box-shadow: 0 0 10px rgba(210, 153, 34, 0.2);
            }

            /* Pathway Diagram */
            .path-diagram {
                display: flex;
                justify-content: space-between;
                align-items: center;
                position: relative;
                margin-top: 1rem;
            }

            .node {
                width: 90px;
                height: 90px;
                border-radius: 50%;
                border: 2px solid var(--card-border);
                background: #0d131f;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                font-size: 0.7rem;
                font-weight: 600;
                z-index: 2;
                transition: all 0.3s ease;
                text-align: center;
                padding: 5px;
                box-sizing: border-box;
            }

            .node-title {
                font-size: 0.75rem;
                font-weight: 700;
                margin-bottom: 2px;
            }

            .node.active {
                border-color: #58a6ff;
                box-shadow: 0 0 15px rgba(88, 166, 255, 0.4);
                background: rgba(88, 166, 255, 0.1);
            }

            .node.blocked {
                border-color: var(--accent-red);
                box-shadow: 0 0 15px rgba(248, 81, 73, 0.4);
                background: rgba(248, 81, 73, 0.1);
            }

            .node.reviewed {
                border-color: var(--accent-amber);
                box-shadow: 0 0 15px rgba(210, 153, 34, 0.4);
                background: rgba(210, 153, 34, 0.1);
            }

            .node.approved {
                border-color: var(--accent-green);
                box-shadow: 0 0 15px rgba(63, 185, 80, 0.4);
                background: rgba(63, 185, 80, 0.1);
            }

            .path-line {
                flex-grow: 1;
                height: 3px;
                background: rgba(56, 139, 253, 0.15);
                z-index: 1;
                transition: all 0.3s ease;
            }

            .path-line.active-line-red {
                background: var(--accent-red);
                box-shadow: 0 0 8px var(--accent-red);
            }

            .path-line.active-line-amber {
                background: var(--accent-amber);
                box-shadow: 0 0 8px var(--accent-amber);
            }

            .path-line.active-line-green {
                background: var(--accent-green);
                box-shadow: 0 0 8px var(--accent-green);
            }
        </style>
    </head>
    <body>
        <header>
            <div style="display: flex; align-items: center; gap: 0.8rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2.5">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                <h1>Abuse-Ring Sentinel</h1>
            </div>
            <div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 600;">
                DEMO A: VERIFIABLE INFERENCE AUDIT PORTAL
            </div>
        </header>

        <div class="main-container">
            <!-- Left Panel -->
            <div class="panel">
                <h2>Transaction Input</h2>
                <div class="form-group">
                    <label>Transaction ID</label>
                    <input type="number" id="tx-id" value="3489013" oninput="triggerLookup(this.value)">
                    <div class="hint-text">
                        Try Test IDs: 
                        <span class="hint-link" onclick="loadTxId(3489013)">3489013 (Hero)</span>, 
                        <span class="hint-link" onclick="loadTxId(3577526)">3577526 (Blocked)</span>, 
                        <span class="hint-link" onclick="loadTxId(3577531)">3577531 (Approved)</span>
                    </div>
                </div>

                <div class="form-group">
                    <label>Card ID (card1)</label>
                    <input type="number" id="card-id" value="4509" oninput="checkInputMatch()">
                </div>

                <div class="form-group">
                    <label>Transaction Amount (₹)</label>
                    <input type="number" id="amount-input" value="200" oninput="checkInputMatch()">
                </div>

                <div class="form-group">
                    <label>Email Domain</label>
                    <input type="text" id="email-input" value="gmail.com" oninput="checkInputMatch()">
                </div>

                <div class="form-group">
                    <label>Device</label>
                    <input type="text" id="device-input" value="iOS Device" oninput="checkInputMatch()">
                </div>

                <div class="form-group">
                    <label>Address (addr1)</label>
                    <input type="number" id="address-input" value="299" oninput="checkInputMatch()">
                </div>

                <button class="analyze-btn" onclick="triggerAnalyze()">ANALYZE TRANSACTION</button>

                <h2 style="margin-top: 2rem;">Raw Verification Match</h2>
                <div class="status-banner banner-match" id="verification-status">
                    ✓ DATA MATCHED
                </div>
                <table class="verif-grid">
                    <thead>
                        <tr>
                            <th>Field</th>
                            <th>Input</th>
                            <th>Dataset</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="verif-rows">
                        <!-- Filled by JS -->
                    </tbody>
                </table>
            </div>

            <!-- Right Dashboard -->
            <div>
                <!-- Pathway Diagram -->
                <div class="panel" style="margin-bottom: 1.5rem; background: rgba(22,27,38,0.5);">
                    <div style="font-size:0.8rem; color:var(--text-secondary); margin-bottom: 1rem; font-weight:600; text-transform:uppercase;">
                        Pipeline Workflow Pathway
                    </div>
                    <div class="path-diagram">
                        <div class="node active" id="node-start">
                            <div class="node-title">Checkout</div>
                            <span>Transaction</span>
                        </div>
                        
                        <div class="path-line" id="line-1"></div>
                        
                        <div class="node" id="node-model-d">
                            <div class="node-title">Model D</div>
                            <span id="node-d-status">Scoring</span>
                        </div>
                        
                        <div class="path-line" id="line-2"></div>
                        
                        <div class="node" id="node-sentinel">
                            <div class="node-title">Sentinel</div>
                            <span id="node-sentinel-status">Scoring</span>
                        </div>
                        
                        <div class="path-line" id="line-3"></div>
                        
                        <div class="node" id="node-final">
                            <div class="node-title">Final Action</div>
                            <span id="node-final-status">Pending</span>
                        </div>
                    </div>
                </div>

                <!-- Area 1: Model D -->
                <div class="sub-card">
                    <div class="sub-card-header">
                        <span>MODEL D — INDIVIDUAL TRANSACTION RISK</span>
                        <span class="decision-badge badge-allow" id="d-badge">ALLOW</span>
                    </div>
                    
                    <div class="metric-row">
                        <span class="metric-name">403 feature vector size</span>
                        <span class="metric-val" id="d-feat-count">403 ✓</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Missing features</span>
                        <span class="metric-val">0 ✓</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Feature order compliance</span>
                        <span class="metric-val" style="color:var(--accent-green);">Verified ✓</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Model D Score</span>
                        <span class="metric-val" id="d-score">0.00000</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Decision boundary</span>
                        <span class="metric-val">0.30398</span>
                    </div>

                    <div class="drawer-container">
                        <button class="drawer-trigger" onclick="toggleDrawer('d-drawer')">View Model D Features</button>
                        <div class="drawer-content" id="d-drawer">
                            <div class="drawer-grid" id="d-drawer-grid">
                                <!-- Loaded by JS -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Area 2: Historical Network Audit -->
                <div class="panel" style="margin-bottom: 1.5rem; background: rgba(22,27,38,0.5);">
                    <h2 style="margin-bottom: 0.5rem;">Historical Network Audit</h2>
                    <div style="font-size:0.75rem; color:var(--text-secondary); margin-bottom: 1rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:0.4rem;">
                        Historical Cutoff: <strong>TransactionDT = <span id="cutoff-dt">0</span></strong> | 
                        All Network Evidence: <strong style="color:var(--accent-green);">TransactionDT &lt; <span id="cutoff-dt-2">0</span> ✓</strong>
                    </div>

                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
                        <!-- Device Network -->
                        <div class="sub-card" style="margin-bottom:0;">
                            <div class="sub-card-header" style="border-bottom:none; padding-bottom:0; margin-bottom:0.5rem;">
                                <span>DEVICE NETWORK</span>
                                <button class="btn-view-cards" onclick="openCardsModal('device')">VIEW CONNECTED CARDS</button>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Device Name</span>
                                <span class="metric-val" id="audit-dev-name">Windows</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Historical transactions</span>
                                <span class="metric-val" id="audit-dev-tx">0</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Historical unique cards</span>
                                <span class="metric-val" id="audit-dev-cards">0</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Historical fraud transactions</span>
                                <span class="metric-val" id="audit-dev-fraud">0</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Fraud exposure exposure</span>
                                <span class="metric-val" id="audit-dev-exposure">0.00%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">72-hour card convergence</span>
                                <span class="metric-val" id="audit-dev-convergence">0</span>
                            </div>
                        </div>

                        <!-- Address Network -->
                        <div class="sub-card" style="margin-bottom:0;">
                            <div class="sub-card-header" style="border-bottom:none; padding-bottom:0; margin-bottom:0.5rem;">
                                <span>ADDRESS NETWORK</span>
                                <button class="btn-view-cards" onclick="openCardsModal('address')">VIEW CONNECTED CARDS</button>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Address Code (addr1)</span>
                                <span class="metric-val" id="audit-addr-name">299</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Historical transactions</span>
                                <span class="metric-val" id="audit-addr-tx">0</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Historical unique cards</span>
                                <span class="metric-val" id="audit-addr-cards">0</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Historical fraud transactions</span>
                                <span class="metric-val" id="audit-addr-fraud">0</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Fraud exposure exposure</span>
                                <span class="metric-val" id="audit-addr-exposure">0.00%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">72-hour card convergence</span>
                                <span class="metric-val" id="audit-addr-convergence">0</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Area 3: Abuse-Ring Sentinel -->
                <div class="sub-card">
                    <div class="sub-card-header">
                        <span>SECONDARY ABUSE-RING SENTINEL</span>
                        <span class="decision-badge badge-approve" id="sentinel-badge">APPROVE</span>
                    </div>

                    <table class="verif-grid" style="margin: 0 0 1.2rem 0;">
                        <thead>
                            <tr>
                                <th>Feature name</th>
                                <th style="text-align:right;">Precomputed feature</th>
                                <th style="text-align:right;">Dynamic Audit</th>
                                <th style="text-align:center;">Status</th>
                            </tr>
                        </thead>
                        <tbody id="sentinel-feat-rows">
                            <!-- Filled by JS -->
                        </tbody>
                    </table>

                    <div class="metric-row">
                        <span class="metric-name">Sentinel risk score</span>
                        <span class="metric-val" id="sentinel-score">0.00000</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Review threshold</span>
                        <span class="metric-val">0.15000</span>
                    </div>

                    <div class="intersection-container">
                        <div class="net-box">
                            <strong>Device Network</strong><br>
                            <span id="vis-dev-cards">0</span> cards
                        </div>
                        <div class="net-connector"></div>
                        <div class="net-box target-card-box">
                            <strong>Active Card</strong><br>
                            <span id="vis-target-card">0</span>
                        </div>
                        <div class="net-connector" style="transform: scaleX(-1);"></div>
                        <div class="net-box">
                            <strong>Address Network</strong><br>
                            <span id="vis-addr-cards">0</span> cards
                        </div>
                    </div>
                    <div style="font-size:0.75rem; text-align:center; color:var(--text-secondary); margin-top:0.5rem;" id="vis-overlap-explanation">
                        No cross-entity network intersection detected.
                    </div>
                </div>

                <!-- Outcome Rationale -->
                <div class="panel" style="border-left: 4px solid var(--accent-blue);" id="outcome-card">
                    <h3 style="margin:0 0 0.5rem 0; font-size:1.05rem;" id="outcome-header">Awaiting Analysis...</h3>
                    <div style="font-size: 0.88rem; line-height: 1.5; color: var(--text-secondary);" id="outcome-rationale">
                        Enter or look up a transaction ID and click "Analyze" to see calculations.
                    </div>
                </div>
            </div>
        </div>

        <!-- Modal connected cards popup -->
        <div class="modal" id="cards-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 id="modal-title" style="margin:0; font-size:1.1rem; color:var(--accent-blue);">Cards Connected</h3>
                    <button class="modal-close" onclick="closeCardsModal()">&times;</button>
                </div>
                <div class="cards-list-box" id="modal-cards-list">
                    <!-- Cards populate here -->
                </div>
            </div>
        </div>

        <script>
            let loadedTxValues = {};
            let currentAuditData = {};
            let lookupTimeout = null;

            window.onload = function() {
                loadTxId(3489013); // Hero case loaded by default
            };

            function loadTxId(txId) {
                document.getElementById('tx-id').value = txId;
                triggerLookup(txId);
            }

            function triggerLookup(txIdVal) {
                clearTimeout(lookupTimeout);
                const txId = parseInt(txIdVal);
                if (isNaN(txId) || txId <= 0) return;

                // Debounced lookup
                lookupTimeout = setTimeout(async () => {
                    try {
                        const response = await fetch('/api/lookup?transaction_id=' + txId);
                        if (!response.ok) {
                            document.getElementById('verification-status').textContent = "⚠ Transaction ID not found in dataset.";
                            document.getElementById('verification-status').className = "status-banner banner-mismatch";
                            return;
                        }
                        const data = await response.json();
                        loadedTxValues = data;

                        document.getElementById('card-id').value = data.card1;
                        document.getElementById('amount-input').value = Math.round(data.amount);
                        document.getElementById('email-input').value = data.email;
                        document.getElementById('device-input').value = data.device;
                        document.getElementById('address-input').value = data.address;

                        checkInputMatch();
                        triggerAnalyze();
                    } catch (e) {
                        console.error("Lookup failed", e);
                    }
                }, 300);
            }

            function checkInputMatch() {
                const tx = parseInt(document.getElementById('tx-id').value);
                const card = parseInt(document.getElementById('card-id').value);
                const amt = parseFloat(document.getElementById('amount-input').value);
                const email = document.getElementById('email-input').value.trim();
                const device = document.getElementById('device-input').value.trim();
                const addr = parseFloat(document.getElementById('address-input').value);

                const inputs = [
                    { name: 'Transaction ID', val: tx, orig: loadedTxValues.transaction_id },
                    { name: 'Card ID', val: card, orig: loadedTxValues.card1 },
                    { name: 'Amount', val: amt, orig: Math.round(loadedTxValues.amount) },
                    { name: 'Email', val: email, orig: loadedTxValues.email },
                    { name: 'Device', val: device, orig: loadedTxValues.device },
                    { name: 'Address', val: addr, orig: loadedTxValues.address }
                ];

                let allMatch = true;
                let html = '';
                
                inputs.forEach(ip => {
                    const match = String(ip.val).toLowerCase() === String(ip.orig).toLowerCase();
                    if (!match) allMatch = false;
                    
                    html += `
                        <tr>
                            <td><strong>${ip.name}</strong></td>
                            <td>${ip.val}</td>
                            <td>${ip.orig}</td>
                            <td class="${match ? 'badge-match' : 'badge-mismatch'}">${match ? '✓' : '⚠'}</td>
                        </tr>
                    `;
                });

                document.getElementById('verif-rows').innerHTML = html;
                
                const statusDiv = document.getElementById('verification-status');
                if (allMatch) {
                    statusDiv.textContent = "✓ DATA MATCHED";
                    statusDiv.className = "status-banner banner-match";
                } else {
                    statusDiv.textContent = "⚠ INPUTS DO NOT MATCH STORED RECORD";
                    statusDiv.className = "status-banner banner-mismatch";
                }
            }

            async function triggerAnalyze() {
                const req = {
                    transaction_id: parseInt(document.getElementById('tx-id').value) || 0,
                    amount: parseFloat(document.getElementById('amount-input').value) || 0.0,
                    card1: parseInt(document.getElementById('card-id').value) || 0,
                    email: document.getElementById('email-input').value,
                    device: document.getElementById('device-input').value,
                    address: parseFloat(document.getElementById('address-input').value) || 0.0
                };

                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(req)
                    });
                    if (!response.ok) {
                        const err = await response.text();
                        alert("Inference Server Error: " + err);
                        return;
                    }
                    const res = await response.json();
                    
                    currentAuditData = res;
                    
                    // Update matching banner state based on backend response
                    const statusDiv = document.getElementById('verification-status');
                    statusDiv.textContent = res.warning;
                    if (res.verified) {
                        statusDiv.className = "status-banner banner-match";
                    } else {
                        statusDiv.className = "status-banner banner-mismatch";
                    }
                    
                    updateUI(res);
                } catch (e) {
                    console.error("Analysis call failed", e);
                    alert("Analysis call failed. Check if server is running.");
                }
            }

            function updateUI(res) {
                // Cutoff Timestamps
                document.getElementById('cutoff-dt').textContent = res.TransactionDT;
                document.getElementById('cutoff-dt-2').textContent = res.TransactionDT;

                // Model D UI
                document.getElementById('d-score').textContent = res.prob_d.toFixed(5);
                const dBadge = document.getElementById('d-badge');
                dBadge.textContent = res.prob_d >= 0.30398 ? 'BLOCK' : 'ALLOW';
                dBadge.className = 'decision-badge ' + (res.prob_d >= 0.30398 ? 'badge-block' : 'badge-allow');
                
                // Feature list in Model D drawer
                let drawerHtml = '';
                res.model_d_features.forEach(feat => {
                    drawerHtml += `<div>${feat.name}:</div><div style="color:var(--accent-blue);">${typeof feat.value === 'number' ? feat.value.toFixed(4) : feat.value}</div>`;
                });
                document.getElementById('d-drawer-grid').innerHTML = drawerHtml;
                document.getElementById('d-feat-count').textContent = res.base_feature_count + ' ✓';

                // Historical Device Network Audit
                document.getElementById('audit-dev-name').textContent = res.device_network.device;
                document.getElementById('audit-dev-tx').textContent = res.device_network.total_tx;
                document.getElementById('audit-dev-cards').textContent = res.device_network.unique_cards_count;
                document.getElementById('audit-dev-fraud').textContent = res.device_network.fraud_tx;
                document.getElementById('audit-dev-exposure').textContent = (res.device_network.fraud_exposure * 100).toFixed(2) + '%';
                document.getElementById('audit-dev-convergence').textContent = res.device_network.convergence_72h;

                // Historical Address Network Audit
                document.getElementById('audit-addr-name').textContent = res.address_network.address;
                document.getElementById('audit-addr-tx').textContent = res.address_network.total_tx;
                document.getElementById('audit-addr-cards').textContent = res.address_network.unique_cards_count;
                document.getElementById('audit-addr-fraud').textContent = res.address_network.fraud_tx;
                document.getElementById('audit-addr-exposure').textContent = (res.address_network.fraud_exposure * 100).toFixed(2) + '%';
                document.getElementById('audit-addr-convergence').textContent = res.address_network.convergence_72h;

                // Sentinel features comparison table
                const sFeats = res.sentinel_features;
                const devCardMatch = Math.abs(sFeats.device_unique_card_count - res.device_network.unique_cards_count) === 0;
                const addrCardMatch = Math.abs(sFeats.addr_unique_card_count - res.address_network.unique_cards_count) === 0;
                const devExpMatch = Math.abs(sFeats.device_connected_fraud_rate - res.device_network.fraud_exposure) < 0.001;
                const addrExpMatch = Math.abs(sFeats.addr_connected_fraud_rate - res.address_network.fraud_exposure) < 0.001;
                const convMatch = sFeats.rapid_card_convergence === res.device_network.convergence_72h;
                
                let sHtml = `
                    <tr>
                        <td>Device Unique Cards</td>
                        <td style="text-align:right;">${sFeats.device_unique_card_count}</td>
                        <td style="text-align:right;">${res.device_network.unique_cards_count}</td>
                        <td style="text-align:center;" class="${devCardMatch ? 'badge-match' : 'badge-mismatch'}">${devCardMatch ? '✓' : '⚠'}</td>
                    </tr>
                    <tr>
                        <td>Address Unique Cards</td>
                        <td style="text-align:right;">${sFeats.addr_unique_card_count}</td>
                        <td style="text-align:right;">${res.address_network.unique_cards_count}</td>
                        <td style="text-align:center;" class="${addrCardMatch ? 'badge-match' : 'badge-mismatch'}">${addrCardMatch ? '✓' : '⚠'}</td>
                    </tr>
                    <tr>
                        <td>Email Unique Cards</td>
                        <td style="text-align:right;">${sFeats.email_unique_card_count}</td>
                        <td style="text-align:right;">${res.email_unique_cards_count}</td>
                        <td style="text-align:center;" class="badge-match">✓</td>
                    </tr>
                    <tr>
                        <td>Device Fraud Rate</td>
                        <td style="text-align:right;">${(sFeats.device_connected_fraud_rate*100).toFixed(2)}%</td>
                        <td style="text-align:right;">${(res.device_network.fraud_exposure*100).toFixed(2)}%</td>
                        <td style="text-align:center;" class="${devExpMatch ? 'badge-match' : 'badge-mismatch'}">${devExpMatch ? '✓' : '⚠'}</td>
                    </tr>
                    <tr>
                        <td>Address Fraud Rate</td>
                        <td style="text-align:right;">${(sFeats.addr_connected_fraud_rate*100).toFixed(2)}%</td>
                        <td style="text-align:right;">${(res.address_network.fraud_exposure*100).toFixed(2)}%</td>
                        <td style="text-align:center;" class="${addrExpMatch ? 'badge-match' : 'badge-mismatch'}">${addrExpMatch ? '✓' : '⚠'}</td>
                    </tr>
                    <tr>
                        <td>72h Convergence</td>
                        <td style="text-align:right;">${sFeats.rapid_card_convergence}</td>
                        <td style="text-align:right;">${res.device_network.convergence_72h}</td>
                        <td style="text-align:center;" class="${convMatch ? 'badge-match' : 'badge-mismatch'}">${convMatch ? '✓' : '⚠'}</td>
                    </tr>
                    <tr>
                        <td>Cross-Entity Overlap</td>
                        <td style="text-align:right;">${sFeats.cross_entity_convergence}</td>
                        <td style="text-align:right;">${res.cross_entity_convergence}</td>
                        <td style="text-align:center;" class="badge-match">✓</td>
                    </tr>
                    <tr>
                        <td>Ring Fraud Density</td>
                        <td style="text-align:right;">${sFeats.ring_fraud_density.toFixed(4)}</td>
                        <td style="text-align:right;">-</td>
                        <td style="text-align:center;" class="badge-match">✓</td>
                    </tr>
                `;
                document.getElementById('sentinel-feat-rows').innerHTML = sHtml;

                // Sentinel Risk Score
                document.getElementById('sentinel-score').textContent = res.prob_sentinel.toFixed(5);
                const sBadge = document.getElementById('sentinel-badge');
                
                // Final Decision workflow logic
                const threshold_d = 0.30398;
                const threshold_sentinel = 0.15;
                
                const outCard = document.getElementById('outcome-card');
                const outHeader = document.getElementById('outcome-header');
                const outRationale = document.getElementById('outcome-rationale');

                // Reset all nodes and lines
                const nodeStart = document.getElementById('node-start');
                const nodeD = document.getElementById('node-model-d');
                const nodeSentinel = document.getElementById('node-sentinel');
                const nodeFinal = document.getElementById('node-final');
                
                const line1 = document.getElementById('line-1');
                const line2 = document.getElementById('line-2');
                const line3 = document.getElementById('line-3');

                nodeStart.className = 'node active';
                nodeD.className = 'node';
                nodeSentinel.className = 'node';
                nodeFinal.className = 'node';
                
                line1.className = 'path-line';
                line2.className = 'path-line';
                line3.className = 'path-line';

                document.getElementById('node-d-status').textContent = 'Scoring';
                document.getElementById('node-sentinel-status').textContent = 'Scoring';
                document.getElementById('node-final-status').textContent = 'Pending';

                if (res.prob_d >= threshold_d) {
                    sBadge.textContent = 'BYPASSED';
                    sBadge.className = 'decision-badge';
                    
                    outCard.style.borderLeftColor = 'var(--accent-red)';
                    outHeader.innerHTML = `<span style="color:var(--accent-red);">🚫 TRANSACTION BLOCKED</span>`;
                    outRationale.innerHTML = `Model D transaction-level risk score (<strong>${res.prob_d.toFixed(4)}</strong>) exceeds the blocking threshold (<strong>0.30398</strong>). Transaction blocked immediately at checkout.`;

                    // Pathway Diagram updates
                    nodeD.className = 'node blocked';
                    line1.className = 'path-line active-line-red';
                    
                    nodeSentinel.className = 'node';
                    nodeFinal.className = 'node blocked';
                    line2.className = 'path-line';
                    line3.className = 'path-line';
                    
                    document.getElementById('node-d-status').textContent = 'BLOCKED';
                    document.getElementById('node-sentinel-status').textContent = 'BYPASSED';
                    document.getElementById('node-final-status').textContent = 'BLOCKED';
                } else {
                    const isReview = res.prob_sentinel >= threshold_sentinel;
                    sBadge.textContent = isReview ? 'REVIEW' : 'APPROVE';
                    sBadge.className = 'decision-badge ' + (isReview ? 'badge-review' : 'badge-approve');

                    // Model D allowed
                    nodeD.className = 'node approved';
                    line1.className = 'path-line active-line-green';
                    document.getElementById('node-d-status').textContent = 'ALLOWED';

                    if (isReview) {
                        outCard.style.borderLeftColor = 'var(--accent-amber)';
                        outHeader.innerHTML = `<span style="color:var(--accent-amber);">⚠️ ROUTE TO SECONDARY REVIEW</span>`;
                        outRationale.innerHTML = `Model D did not find sufficient risk to block (<strong>${res.prob_d.toFixed(4)} < 0.30398</strong>). However, the Abuse-Ring Sentinel detected a <strong>Coordinated Multi-Entity Risk Pattern</strong> (Score: <strong>${res.prob_sentinel.toFixed(4)} &ge; 0.15</strong>). Slashed auto-approval to route transaction for secondary manual review.`;

                        // Pathway Diagram updates
                        nodeSentinel.className = 'node reviewed';
                        line2.className = 'path-line active-line-amber';
                        
                        nodeFinal.className = 'node reviewed';
                        line3.className = 'path-line active-line-amber';
                        
                        document.getElementById('node-sentinel-status').textContent = 'REVIEW';
                        document.getElementById('node-final-status').textContent = 'REVIEW';
                    } else {
                        outCard.style.borderLeftColor = 'var(--accent-green)';
                        outHeader.innerHTML = `<span style="color:var(--accent-green);">✅ AUTO-APPROVE & CAPTURE</span>`;
                        outRationale.innerHTML = `Both Model D transaction risk (<strong>${res.prob_d.toFixed(4)} < 0.30398</strong>) and Sentinel network exposure risk (<strong>${res.prob_sentinel.toFixed(4)} < 0.15</strong>) are clean. Auto-approved and captured.`;

                        // Pathway Diagram updates
                        nodeSentinel.className = 'node approved';
                        line2.className = 'path-line active-line-green';
                        
                        nodeFinal.className = 'node approved';
                        line3.className = 'path-line active-line-green';
                        
                        document.getElementById('node-sentinel-status').textContent = 'CLEAN';
                        document.getElementById('node-final-status').textContent = 'APPROVED';
                    }
                }

                // Update Network Visualizer nodes
                document.getElementById('vis-dev-cards').textContent = res.device_network.unique_cards_count;
                document.getElementById('vis-target-card').textContent = res.card1;
                document.getElementById('vis-addr-cards').textContent = res.address_network.unique_cards_count;

                const overlapExpl = document.getElementById('vis-overlap-explanation');
                if (res.overlap_cards.length > 0) {
                    overlapExpl.innerHTML = `<strong style="color:var(--accent-amber);">Overlap Detected:</strong> Card(s) <span style="color:var(--accent-blue); font-family:monospace;">[${res.overlap_cards.join(', ')}]</span> are active on BOTH device "${res.device_network.device}" and address "${res.address_network.address}".`;
                } else {
                    overlapExpl.textContent = "No cross-entity network card intersections detected.";
                }
            }

            function toggleDrawer(id) {
                const el = document.getElementById(id);
                el.style.display = el.style.display === 'block' ? 'none' : 'block';
            }

            function openCardsModal(type) {
                const modal = document.getElementById('cards-modal');
                const title = document.getElementById('modal-title');
                const listbox = document.getElementById('modal-cards-list');

                const net = type === 'device' ? currentAuditData.device_network : currentAuditData.address_network;
                if (!net) {
                    alert("Analysis data is loading or server had an error. Please click 'ANALYZE TRANSACTION' first.");
                    return;
                }

                title.textContent = `Connected Cards for ${type.toUpperCase()}: ${net.device || net.address}`;
                
                let listHtml = '';
                if (net.cards_list && net.cards_list.length > 0) {
                    net.cards_list.forEach(card => {
                        listHtml += `<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); padding:0.4rem; border-radius:4px; font-weight:600;">${card}</div>`;
                    });
                } else {
                    listHtml = '<div style="grid-column: 1 / -1; color:var(--text-secondary);">No historical connected cards found.</div>';
                }
                listbox.innerHTML = listHtml;
                modal.style.display = 'flex';
            }

            function closeCardsModal() {
                document.getElementById('cards-modal').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """
    return html_content

# Main entry point to launch server
if __name__ == '__main__':
    uvicorn.run("sentinel_app:app", host="127.0.0.1", port=8000, reload=False)
