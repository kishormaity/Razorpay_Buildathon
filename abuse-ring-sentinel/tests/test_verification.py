import os
import sys
import unittest
import json
import sqlite3
import pandas as pd
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJ_ROOT)

class TestAbuseRingSentinelV2(unittest.TestCase):
    def test_database_integrity(self):
        db_path = os.path.join(PROJ_ROOT, "data", "processed", "risk_sentinel.db")
        self.assertTrue(os.path.exists(db_path), "SQLite processed DB does not exist.")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT count(*) FROM transactions;")
        txn_count = cursor.fetchone()[0]
        self.assertGreater(txn_count, 0, "No transactions found in database.")
        
        cursor.execute("SELECT count(*) FROM users;")
        user_count = cursor.fetchone()[0]
        self.assertGreater(user_count, 0, "No users found in database.")
        
        conn.close()
        print(f"[SUCCESS] Database integrity passed: {txn_count} transactions, {user_count} users verified.")
        
    def test_compiled_features(self):
        features_path = os.path.join(PROJ_ROOT, "data", "processed", "features", "features.parquet")
        self.assertTrue(os.path.exists(features_path), "Parquet features matrix does not exist.")
        
        df = pd.read_parquet(features_path)
        self.assertEqual(df.shape[0], 20020, "Features matrix row count is not 20020.")
        self.assertGreater(df.shape[1], 40, "Features count is less than 40.")
        self.assertIn("isFraud", df.columns, "isFraud label column missing.")
        print(f"[SUCCESS] Features store parquet passed: {df.shape[0]} rows, {df.shape[1]} columns verified.")
        
    def test_model_outputs(self):
        # 1. LightGBM
        gbm_booster = os.path.join(PROJ_ROOT, "models", "lightgbm", "sentinel_gbm_booster.txt")
        self.assertTrue(os.path.exists(gbm_booster), "LightGBM booster file missing.")
        
        # 2. Node2Vec
        n2v_json = os.path.join(PROJ_ROOT, "models", "node2vec", "node2vec_user_risks.json")
        self.assertTrue(os.path.exists(n2v_json), "Node2Vec user risks JSON missing.")
        
        # 3. GraphSAGE
        sage_weights = os.path.join(PROJ_ROOT, "models", "graphsage", "graphsage_weights.pt")
        self.assertTrue(os.path.exists(sage_weights), "GraphSAGE model weights missing.")
        
        # 4. Fusion Stacker
        stacker_path = os.path.join(PROJ_ROOT, "models", "fusion", "fusion_stacker.pkl")
        self.assertTrue(os.path.exists(stacker_path), "Fusion stacker model missing.")
        
        print("[SUCCESS] All model checkpoint outputs verified.")
        
    def test_api_endpoints(self):
        url = "http://127.0.0.1:8001/api/dashboard/stats"
        try:
            res = requests.get(url, timeout=0.5)
            self.assertEqual(res.status_code, 200, "API dashboard endpoint returned non-200.")
            data = res.json()
            self.assertIn("total_transactions", data, "Stats key missing.")
            print(f"[SUCCESS] Live API server verified on port 8001. Transactions scored count: {data['total_transactions']}.")
        except Exception:
            from fastapi.testclient import TestClient
            from api.main import app, startup_event
            startup_event()
            client = TestClient(app)
            res = client.get("/api/dashboard/stats")
            self.assertEqual(res.status_code, 200, "In-memory API endpoint returned non-200.")
            data = res.json()
            self.assertIn("total_transactions", data, "Stats key missing.")
            print(f"[SUCCESS] In-memory FastAPI server verified. Transactions scored count: {data['total_transactions']}.")

if __name__ == "__main__":
    unittest.main()
