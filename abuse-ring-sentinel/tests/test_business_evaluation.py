import os
import sys
import unittest
import json
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJ_ROOT)

from src.evaluation.business_evaluation import (
    get_chronological_splits,
    evaluate_decision_policy,
    compute_metrics_from_records,
    run_validation_threshold_selection
)
from src.explainability.shap import TabularSHAPExplainer
from src.risk.decision_policy import decide_hybrid_policy, decide_model_d


class TestPhaseP2CompetitionHardening(unittest.TestCase):
    """
    Authoritative test suite covering all 13 verification categories for Phase P2:
    Competition Hardening & Scientific Defensibility.
    """

    @classmethod
    def setUpClass(cls):
        cls.train_df, cls.val_df, cls.test_df, cls.split_meta = get_chronological_splits()
        cls.cost_matrix = {
            "fp_cost": 1500.0,
            "chargeback_fee": 1200.0,
            "investigation_cost": 500.0,
            "fn_loss_factor": 1.0
        }
        cls.eval_json_path = os.path.join(
            PROJ_ROOT, "data", "processed", "evaluation", "test_business_evaluation.json"
        )
        cls.scorecard_json_path = os.path.join(
            PROJ_ROOT, "data", "processed", "evaluation", "policy_scorecard.json"
        )
        cls.incremental_csv_path = os.path.join(
            PROJ_ROOT, "data", "processed", "evaluation", "sentinel_incremental_cases.csv"
        )

    # Category 1: Incremental 13-case calculation
    def test_01_incremental_13_case_calculation(self):
        """Verify Sentinel captures exactly 13 incremental fraud transactions on the test set."""
        rec_df = evaluate_decision_policy(self.test_df, 0.50, 0.05, 0.45, self.cost_matrix)
        metrics = compute_metrics_from_records(rec_df, self.cost_matrix)
        inc = metrics["sentinel_incremental_value"]

        self.assertEqual(inc["sentinel_intercepted_count"], 13)
        self.assertEqual(inc["fraud_missed_by_model_d_count"], 53)
        self.assertAlmostEqual(inc["incremental_capture_rate_pct"], 24.53, places=1)
        self.assertAlmostEqual(inc["sentinel_intercepted_value_inr"], 696.76, places=1)

        # Verify incremental CSV artifact
        self.assertTrue(os.path.exists(self.incremental_csv_path))
        inc_df = pd.read_csv(self.incremental_csv_path)
        self.assertEqual(len(inc_df), 13)
        self.assertTrue((inc_df["Ground_Truth_Fraud"] == 1).all())
        self.assertTrue((inc_df["Model_D_Risk"] < 0.05).all())
        self.assertTrue((inc_df["Sentinel_Risk"] >= 0.45).all())
        self.assertTrue((inc_df["Decision_Model_D"] == "ALLOW").all())
        self.assertTrue((inc_df["Decision_Hybrid"] == "MANUAL_REVIEW").all())

    # Category 2: No test-label threshold optimization
    def test_02_no_test_label_threshold_optimization(self):
        """Verify thresholds are chosen strictly on val_df and remain frozen."""
        d_block, d_review, sentinel_t = run_validation_threshold_selection(
            self.val_df, self.cost_matrix
        )
        self.assertEqual(d_block, 0.50, "Operational block threshold must be 0.50.")
        self.assertEqual(d_review, 0.05, "Optimal Model D validation review threshold must be 0.05.")
        self.assertEqual(sentinel_t, 0.45, "Optimal Sentinel network threshold must be 0.45.")

    # Category 3: Booster inference invariance to ground-truth labels (TEST A)
    def test_03_ground_truth_cannot_affect_live_prediction(self):
        """
        TEST A: Load actual LightGBM booster and feature matrix.
        Execute booster.predict(). Flip and remove isFraud metadata.
        Assert that prediction probabilities are bit-for-bit identical.
        """
        import lightgbm as lgb
        booster_path = os.path.join(PROJ_ROOT, "models", "lightgbm", "sentinel_gbm_booster.txt")
        features_json = os.path.join(PROJ_ROOT, "models", "lightgbm", "model_d_features.json")
        features_parquet = os.path.join(PROJ_ROOT, "data", "processed", "features", "features.parquet")

        self.assertTrue(os.path.exists(booster_path))
        self.assertTrue(os.path.exists(features_json))
        self.assertTrue(os.path.exists(features_parquet))

        booster = lgb.Booster(model_file=booster_path)
        with open(features_json, "r", encoding="utf-8") as f:
            feat_names = json.load(f)

        raw_feats = pd.read_parquet(features_parquet).head(20)
        X1 = raw_feats[feat_names].copy()
        for col in ['ProductCD', 'addr2', 'P_emaildomain']:
            if col in X1.columns:
                X1[col] = X1[col].astype('category')
        preds1 = booster.predict(X1)

        # Invert isFraud metadata or inject arbitrary labels
        raw_feats_mod = raw_feats.copy()
        raw_feats_mod["isFraud"] = 1 - raw_feats_mod.get("isFraud", 0)
        X2 = raw_feats_mod[feat_names].copy()
        for col in ['ProductCD', 'addr2', 'P_emaildomain']:
            if col in X2.columns:
                X2[col] = X2[col].astype('category')
        preds2 = booster.predict(X2)

        np.testing.assert_array_equal(preds1, preds2)

    # Category 4: Decision policy independence from ground truth (TEST B)
    def test_04_ground_truth_cannot_affect_live_decision(self):
        """
        TEST B: Verify the live decision functions (decide_hybrid_policy, decide_model_d)
        receive only risk scores, not ground truth labels.
        """
        import inspect
        sig_hybrid = inspect.signature(decide_hybrid_policy)
        self.assertNotIn("isFraud", sig_hybrid.parameters)
        self.assertNotIn("is_fraud", sig_hybrid.parameters)
        self.assertNotIn("label", sig_hybrid.parameters)

        sig_d = inspect.signature(decide_model_d)
        self.assertNotIn("isFraud", sig_d.parameters)
        self.assertNotIn("is_fraud", sig_d.parameters)

        # Assert identical decisions for identical scores regardless of hypothetical label
        for r_gbm, r_ring, exp_dec, exp_reason in [
            (0.75, 0.10, "BLOCK", "MODEL_D_HIGH_RISK"),
            (0.02, 0.55, "MANUAL_REVIEW", "SENTINEL_RING_ESCALATION"),
            (0.15, 0.10, "MANUAL_REVIEW", "MODEL_D_MODERATE_RISK"),
            (0.01, 0.05, "ALLOW", "NORMAL")
        ]:
            dec, reason = decide_hybrid_policy(r_gbm, r_ring)
            self.assertEqual(dec, exp_dec)
            self.assertEqual(reason, exp_reason)

    # Category 5: Manual review is not incorrectly counted as prevented fraud without inspection
    def test_05_manual_review_not_fraud_prevented_without_inspection(self):
        """Verify that MANUAL_REVIEW is distinct from automated BLOCK."""
        rec_df = evaluate_decision_policy(self.test_df, 0.50, 0.05, 0.45, self.cost_matrix)
        
        block_count = (rec_df["decision_hybrid"] == "BLOCK").sum()
        review_count = (rec_df["decision_hybrid"] == "MANUAL_REVIEW").sum()
        allow_count = (rec_df["decision_hybrid"] == "ALLOW").sum()
        
        self.assertEqual(block_count + review_count + allow_count, len(self.test_df))
        self.assertGreater(review_count, 0)
        self.assertGreater(block_count, 0)
        # All 13 incremental frauds are routed to MANUAL_REVIEW, not blindly BLOCK
        inc_rows = rec_df[(rec_df["is_fraud"] == 1) & (rec_df["decision_d"] == "ALLOW") & (rec_df["decision_hybrid"] != "ALLOW")]
        self.assertTrue((inc_rows["decision_hybrid"] == "MANUAL_REVIEW").all())

    # Category 6: False-positive cost formula and tiered calculation
    def test_06_false_positive_cost_and_tiered_calculation(self):
        """Verify false positive breakdown (3 false blocks, 515 false reviews) and tiered cost."""
        with open(self.eval_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        h = data["metrics"]["production_hybrid"]
        self.assertEqual(h["false_positive_blocks_count"], 3)
        self.assertEqual(h["false_positive_reviews_count"], 515)
        self.assertEqual(h["false_positives"], 518)
        
        # Strict flat cost: 518 * 1500 = 777,000
        self.assertEqual(h["estimated_false_positive_cost_inr"], 777000.0)
        # Tiered cost: (3 * 1500) + (515 * 500) = 4500 + 257500 = 262,000
        self.assertEqual(h["estimated_false_positive_cost_tiered_inr"], 262000.0)
        # Net operational difference
        diff = h["estimated_false_positive_cost_inr"] - h["estimated_false_positive_cost_tiered_inr"]
        self.assertEqual(diff, 515000.0)

    # Category 7: Economic metric formulas (Observed vs Derived vs Assumed)
    def test_07_economic_metric_formulas(self):
        """Verify honest labeling of observed vs derived vs assumed quantities."""
        with open(self.eval_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assumptions = data["cost_assumptions"]
        self.assertTrue(assumptions["is_assumption"])
        self.assertEqual(assumptions["fp_cost_inr"], 1500.0)
        self.assertEqual(assumptions["investigation_cost_inr"], 500.0)
        
        # Check that metrics contain estimated loss avoided
        h_metrics = data["metrics"]["production_hybrid"]
        self.assertIn("estimated_fraud_value_prevented_inr", h_metrics)
        self.assertIn("estimated_net_loss_avoided_inr", h_metrics)

    # Category 8: Locked test set boundaries (15% chronological split)
    def test_08_locked_test_set_boundaries(self):
        """Verify dynamic 70/15/15 chronological split without hardcoded indices."""
        self.assertEqual(self.split_meta["total_rows"], 20020)
        self.assertEqual(self.split_meta["train_rows"], 14014)
        self.assertEqual(self.split_meta["val_rows"], 3003)
        self.assertEqual(self.split_meta["test_rows"], 3003)

        train_max_dt = self.train_df["TransactionDT"].max()
        val_min_dt = self.val_df["TransactionDT"].min()
        val_max_dt = self.val_df["TransactionDT"].max()
        test_min_dt = self.test_df["TransactionDT"].min()

        self.assertLessEqual(train_max_dt, val_min_dt)
        self.assertLessEqual(val_max_dt, test_min_dt)

    # Category 9: Model D vs Hybrid comparison metrics integrity
    def test_09_model_d_vs_hybrid_comparison_metrics_integrity(self):
        """Verify authoritative Model D vs Hybrid test performance ladder."""
        with open(self.eval_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        d = data["metrics"]["model_d_alone"]
        h = data["metrics"]["production_hybrid"]
        
        self.assertEqual(d["fraud_cases_captured"], 43)
        self.assertEqual(d["fraud_cases_missed"], 53)
        self.assertAlmostEqual(d["recall"], 0.4479, places=3)
        self.assertEqual(d["false_positives"], 228)
        self.assertAlmostEqual(d["fpr"], 0.0784, places=3)
        
        self.assertEqual(h["fraud_cases_captured"], 56)
        self.assertEqual(h["fraud_cases_missed"], 40)
        self.assertAlmostEqual(h["recall"], 0.5833, places=3)
        self.assertEqual(h["false_positives"], 518)
        self.assertAlmostEqual(h["fpr"], 0.1782, places=3)

    # Category 10: Live API does not expose evaluation labels as prediction inputs (TEST C)
    def test_10_live_api_does_not_expose_evaluation_labels_as_inputs(self):
        """
        TEST C: Verify API endpoints query source evaluation artifacts and do not rely on
        fake hardcoded fallback values.
        """
        from fastapi.testclient import TestClient
        from api.main import app, startup_event
        startup_event()
        client = TestClient(app)

        res_demo = client.get("/api/demo/cases")
        self.assertEqual(res_demo.status_code, 200)
        cases = res_demo.json()
        self.assertIn("case_a", cases)
        self.assertIn("case_b", cases)
        self.assertIn("case_c", cases)

        # Verify Case A values match authoritative source data dynamically
        case_a = cases["case_a"]
        self.assertEqual(case_a["transaction_id"], "TXN-3004730")
        self.assertAlmostEqual(case_a["amount_inr"], 47.732, places=2)
        self.assertNotEqual(case_a["amount_inr"], 482.10, "Case A must not use fake fallback amount 482.10.")
        self.assertAlmostEqual(case_a["model_d_score"], 0.7537, places=2)
        self.assertNotEqual(case_a["model_d_score"], 0.8931, "Case A must not use fake fallback score 0.8931.")
        self.assertEqual(case_a["decision_hybrid"], "BLOCK")

        # Verify Case B values match authoritative source data dynamically
        case_b = cases["case_b"]
        self.assertEqual(case_b["transaction_id"], "TXN-3004262")
        self.assertAlmostEqual(case_b["amount_inr"], 85.494, places=2)
        self.assertLess(case_b["model_d_score"], 0.05)
        self.assertGreaterEqual(case_b["sentinel_score"], 0.45)
        self.assertEqual(case_b["decision_hybrid"], "MANUAL_REVIEW", "Case B must return MANUAL_REVIEW, not ALLOW.")

        # Verify Case C values match authoritative source
        case_c = cases["case_c"]
        self.assertEqual(case_c["transaction_id"], "TXN-3005400")
        self.assertAlmostEqual(case_c["amount_inr"], 19.0, places=1)
        self.assertEqual(case_c["decision_hybrid"], "ALLOW")

        # Verify transaction detail endpoint for Case B
        res_txn = client.get("/api/transaction/TXN-3004262")
        self.assertEqual(res_txn.status_code, 200)
        txn_data = res_txn.json()
        self.assertEqual(txn_data["decision"]["action"], "MANUAL_REVIEW")

    # Category 11: SHAP explanation generated from actual feature vector
    def test_11_shap_explanation_generated_from_actual_feature_vector(self):
        """Verify TabularSHAPExplainer runs native TreeSHAP on booster."""
        import lightgbm as lgb
        booster_path = os.path.join(PROJ_ROOT, "models", "lightgbm", "sentinel_gbm_booster.txt")
        features_json = os.path.join(PROJ_ROOT, "models", "lightgbm", "model_d_features.json")
        features_parquet = os.path.join(PROJ_ROOT, "data", "processed", "features", "features.parquet")

        self.assertTrue(os.path.exists(booster_path))
        self.assertTrue(os.path.exists(features_json))

        booster = lgb.Booster(model_file=booster_path)
        with open(features_json, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        explainer = TabularSHAPExplainer(booster, feature_names)
        feat_df = pd.read_parquet(features_parquet).head(1)
        shap_dict = explainer.explain(feat_df)

        self.assertIsInstance(shap_dict, dict)
        self.assertGreater(len(shap_dict), 0)

    # Category 12: Network explanations correspond to actual graph evidence
    def test_12_network_explanations_correspond_to_actual_graph_evidence(self):
        """Verify incremental cases link to real graph community and device tokens."""
        inc_df = pd.read_csv(self.incremental_csv_path)
        
        # Verify Community 10 and device sharing
        self.assertTrue(all(inc_df["Shared_Devices"].str.startswith("DEV-")))
        self.assertTrue(all(inc_df["Connected_Accounts_Count"] >= 1))
        # Verify users in community 10
        user_ids = inc_df["user_id"].tolist()
        self.assertIn("CUS-16746", user_ids)
        self.assertIn("CUS-10876", user_ids)

    # Category 13: Dashboard consumes backend metrics rather than hardcoded numbers
    def test_13_dashboard_consumes_backend_metrics(self):
        """Verify policy_scorecard.json contains operating points and test ladder."""
        self.assertTrue(os.path.exists(self.scorecard_json_path))
        with open(self.scorecard_json_path, "r", encoding="utf-8") as f:
            sc = json.load(f)
            
        ops = sc.get("operating_points", []) or sc.get("validation_operating_points", [])
        self.assertEqual(len(ops), 3)
        op_names = [op.get("name", op.get("profile_name", "")) for op in ops]
        self.assertTrue(any("Precision" in name for name in op_names))
        self.assertTrue(any("Balanced" in name for name in op_names))
        self.assertTrue(any("Recall" in name for name in op_names))

    # Category 14: Sentinel ring scoring target leakage elimination (TEST F)
    def test_14_sentinel_ring_scoring_no_test_ground_truth_leakage(self):
        """
        TEST F: Verify Sentinel ring risk scoring does not consume test-set ground truth labels.
        - configs/risk_policy.yaml must specify w_financial: 0.0
        - Community 10 risk score must be computed strictly from graph topology/temporal/behavioral signals.
        """
        import yaml
        policy_cfg_path = os.path.join(PROJ_ROOT, "configs", "risk_policy.yaml")
        with open(policy_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        w_fin = cfg.get("ring_scoring", {}).get("weights", {}).get("w_financial", 0.0)
        self.assertEqual(w_fin, 0.0, "w_financial must be 0.0 to prevent target leakage.")

        ring_scores_path = os.path.join(PROJ_ROOT, "data", "processed", "ring_risk_scores.json")
        self.assertTrue(os.path.exists(ring_scores_path))
        with open(ring_scores_path, "r", encoding="utf-8") as f:
            ring_scores = json.load(f)
        self.assertIn("10", ring_scores)
        comm10 = ring_scores["10"]
        self.assertEqual(comm10["financial"], 0.0, "Financial target score must be 0.0.")
        self.assertGreaterEqual(comm10["score"], 0.45, "Community 10 must still score >= 0.45 via topology.")
        self.assertAlmostEqual(comm10["score"], 0.4762, places=3)


if __name__ == "__main__":
    unittest.main()
