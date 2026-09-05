import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJ_ROOT)

from src.risk.decision_policy import decide_hybrid_policy, decide_model_d

def load_configs():
    data_config_path = os.path.join(PROJ_ROOT, "configs", "data.yaml")
    with open(data_config_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
        
    policy_config_path = os.path.join(PROJ_ROOT, "configs", "risk_policy.yaml")
    with open(policy_config_path, "r", encoding="utf-8") as f:
        policy_config = yaml.safe_load(f)
        
    return data_config, policy_config

def get_chronological_splits():
    data_config, _ = load_configs()
    preds_path = os.path.join(PROJ_ROOT, "data", "processed", "predictions", "sentinel_fused_preds.parquet")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions matrix not found at: {preds_path}")
        
    df = pd.read_parquet(preds_path)
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    
    train_ratio = data_config["splits"].get("train_ratio", 0.70)
    val_ratio = data_config["splits"].get("val_ratio", 0.15)
    
    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    max_train_dt = train_df["TransactionDT"].max()
    min_val_dt = val_df["TransactionDT"].min()
    max_val_dt = val_df["TransactionDT"].max()
    min_test_dt = test_df["TransactionDT"].min()
    
    assert max_train_dt <= min_val_dt, "Temporal leakage detected between Train and Val splits!"
    assert max_val_dt <= min_test_dt, "Temporal leakage detected between Val and Test splits!"
    
    return train_df, val_df, test_df, {
        "total_rows": total_rows,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_dt_range": [int(train_df["TransactionDT"].min()), int(max_train_dt)],
        "val_dt_range": [int(min_val_dt), int(max_val_dt)],
        "test_dt_range": [int(min_test_dt), int(test_df["TransactionDT"].max())]
    }

def evaluate_decision_policy(df, d_block, d_review, sentinel_threshold, cost_matrix):
    """
    Evaluates the actual operational decision policy:
      - If Model D (r_gbm) >= d_block: BLOCK (High primary risk)
      - Else if Sentinel (r_ring) >= sentinel_threshold: MANUAL_REVIEW (Sentinel network escalation)
      - Else if Model D (r_gbm) >= d_review: MANUAL_REVIEW (Model D moderate risk)
      - Else: ALLOW
    
    Ground truth is ONLY used after decisions are assigned.
    """
    records = []
    for row in df.itertuples(index=False):
        r_gbm = float(row.r_gbm)
        r_ring = float(row.r_ring)
        r_final = float(row.r_final)
        r_cal = float(getattr(row, "r_calibrated", r_final))
        amt = float(row.TransactionAmt)
        y_true = int(row.isFraud)
        
        # 1. Model D alone decision
        decision_d, flagged_reason_d = decide_model_d(r_gbm, d_block=d_block, d_review=d_review)
            
        # 2. Production hybrid decision (Model D + Abuse-Ring Sentinel)
        decision_hybrid, flagged_reason_h = decide_hybrid_policy(
            r_gbm, r_ring,
            d_block=d_block,
            d_review=d_review,
            sentinel_threshold=sentinel_threshold
        )
            
        is_flagged_d = decision_d in ("MANUAL_REVIEW", "BLOCK")
        is_flagged_hybrid = decision_hybrid in ("MANUAL_REVIEW", "BLOCK")
        
        d_fraud_prevented = (y_true == 1) and is_flagged_d
        hybrid_fraud_prevented = (y_true == 1) and is_flagged_hybrid
        d_legit_inconvenienced = (y_true == 0) and is_flagged_d
        hybrid_legit_inconvenienced = (y_true == 0) and is_flagged_hybrid
        
        # Incremental capture: Fraud that Model D alone ALLOWed, but Sentinel intercepted
        sentinel_intercepted = (y_true == 1) and (decision_d == "ALLOW") and is_flagged_hybrid
        
        records.append({
            "transaction_id": row.TransactionID,
            "transaction_amt": amt,
            "is_fraud": y_true,
            "r_gbm": r_gbm,
            "r_ring": r_ring,
            "r_final": r_final,
            "r_calibrated": r_cal,
            "decision_d": decision_d,
            "decision_hybrid": decision_hybrid,
            "flagged_reason_d": flagged_reason_d,
            "flagged_reason_hybrid": flagged_reason_h,
            "is_flagged_d": int(is_flagged_d),
            "is_flagged_hybrid": int(is_flagged_hybrid),
            "d_fraud_prevented": int(d_fraud_prevented),
            "hybrid_fraud_prevented": int(hybrid_fraud_prevented),
            "d_legit_inconvenienced": int(d_legit_inconvenienced),
            "hybrid_legit_inconvenienced": int(hybrid_legit_inconvenienced),
            "sentinel_intercepted": int(sentinel_intercepted)
        })
        
    return pd.DataFrame(records)

def compute_metrics_from_records(rec_df, cost_matrix):
    fp_cost = cost_matrix.get("fp_cost", 1500.0)
    chargeback_fee = cost_matrix.get("chargeback_fee", 1200.0)
    
    y_true = rec_df["is_fraud"].values
    pred_d = rec_df["is_flagged_d"].values
    pred_h = rec_df["is_flagged_hybrid"].values
    amts = rec_df["transaction_amt"].values
    
    total_txns = len(rec_df)
    total_val = float(amts.sum())
    fraud_mask = (y_true == 1)
    legit_mask = (y_true == 0)
    
    total_fraud_count = int(fraud_mask.sum())
    total_fraud_val = float(amts[fraud_mask].sum())
    total_legit_count = int(legit_mask.sum())
    total_legit_val = float(amts[legit_mask].sum())
    
    investigation_cost = cost_matrix.get("investigation_cost", 500.0)
    
    # 1. Model D alone Breakdown
    tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_true, pred_d).ravel()
    prec_d, rec_d, f1_d, _ = precision_recall_fscore_support(y_true, pred_d, average="binary", zero_division=0)
    fpr_d = fp_d / (fp_d + tn_d) if (fp_d + tn_d) > 0 else 0.0
    pr_auc_d = float(average_precision_score(y_true, rec_df["r_gbm"].values))
    roc_auc_d = float(roc_auc_score(y_true, rec_df["r_gbm"].values))
    
    d_legit_block = rec_df[(rec_df["is_fraud"] == 0) & (rec_df["decision_d"] == "BLOCK")]
    d_legit_review = rec_df[(rec_df["is_fraud"] == 0) & (rec_df["decision_d"] == "MANUAL_REVIEW")]
    d_fp_block_count = len(d_legit_block)
    d_fp_review_count = len(d_legit_review)
    d_fp_block_val = float(d_legit_block["transaction_amt"].sum())
    d_fp_review_val = float(d_legit_review["transaction_amt"].sum())
    
    d_fraud_block = rec_df[(rec_df["is_fraud"] == 1) & (rec_df["decision_d"] == "BLOCK")]
    d_fraud_review = rec_df[(rec_df["is_fraud"] == 1) & (rec_df["decision_d"] == "MANUAL_REVIEW")]
    d_fraud_block_count = len(d_fraud_block)
    d_fraud_review_count = len(d_fraud_review)
    d_fraud_block_val = float(d_fraud_block["transaction_amt"].sum())
    d_fraud_review_val = float(d_fraud_review["transaction_amt"].sum())
    
    d_fraud_prevented_val = float(rec_df[rec_df["d_fraud_prevented"] == 1]["transaction_amt"].sum())
    d_fraud_missed_val = float(total_fraud_val - d_fraud_prevented_val)
    d_legit_blocked_val = float(rec_df[rec_df["d_legit_inconvenienced"] == 1]["transaction_amt"].sum())
    
    d_fp_cost_strict = float(fp_d * fp_cost)
    d_fp_cost_tiered = float((d_fp_block_count * fp_cost) + (d_fp_review_count * investigation_cost))
    d_net_loss_avoided_strict = float(d_fraud_prevented_val - d_fp_cost_strict)
    d_net_loss_avoided_tiered = float(d_fraud_prevented_val - d_fp_cost_tiered)
    
    # 2. Hybrid (Model D + Sentinel) Breakdown
    tn_h, fp_h, fn_h, tp_h = confusion_matrix(y_true, pred_h).ravel()
    prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(y_true, pred_h, average="binary", zero_division=0)
    fpr_h = fp_h / (fp_h + tn_h) if (fp_h + tn_h) > 0 else 0.0
    pr_auc_h = float(average_precision_score(y_true, rec_df["r_calibrated"].values))
    roc_auc_h = float(roc_auc_score(y_true, rec_df["r_calibrated"].values))
    
    h_legit_block = rec_df[(rec_df["is_fraud"] == 0) & (rec_df["decision_hybrid"] == "BLOCK")]
    h_legit_review = rec_df[(rec_df["is_fraud"] == 0) & (rec_df["decision_hybrid"] == "MANUAL_REVIEW")]
    h_fp_block_count = len(h_legit_block)
    h_fp_review_count = len(h_legit_review)
    h_fp_block_val = float(h_legit_block["transaction_amt"].sum())
    h_fp_review_val = float(h_legit_review["transaction_amt"].sum())
    
    h_fraud_block = rec_df[(rec_df["is_fraud"] == 1) & (rec_df["decision_hybrid"] == "BLOCK")]
    h_fraud_review = rec_df[(rec_df["is_fraud"] == 1) & (rec_df["decision_hybrid"] == "MANUAL_REVIEW")]
    h_fraud_block_count = len(h_fraud_block)
    h_fraud_review_count = len(h_fraud_review)
    h_fraud_block_val = float(h_fraud_block["transaction_amt"].sum())
    h_fraud_review_val = float(h_fraud_review["transaction_amt"].sum())
    h_fraud_prevented_val = float(rec_df[rec_df["hybrid_fraud_prevented"] == 1]["transaction_amt"].sum())
    h_fraud_missed_val = float(total_fraud_val - h_fraud_prevented_val)
    h_legit_blocked_val = float(rec_df[rec_df["hybrid_legit_inconvenienced"] == 1]["transaction_amt"].sum())

    h_fp_cost_strict = float(fp_h * fp_cost)
    h_fp_cost_tiered = float((h_fp_block_count * fp_cost) + (h_fp_review_count * investigation_cost))
    h_net_loss_avoided_strict = float(h_fraud_prevented_val - h_fp_cost_strict)
    h_net_loss_avoided_tiered = float(h_fraud_prevented_val - h_fp_cost_tiered)
    
    review_capture_efficiency = float(cost_matrix.get("review_capture_efficiency", 0.85))
    
    # Realized loss calculations using review_capture_efficiency
    d_realized_fraud_prevented = float(d_fraud_block_val + (d_fraud_review_val * review_capture_efficiency))
    d_net_loss_avoided_realized = float(d_realized_fraud_prevented - d_fp_cost_tiered)
    
    h_realized_fraud_prevented = float(h_fraud_block_val + (h_fraud_review_val * review_capture_efficiency))
    h_net_loss_avoided_realized = float(h_realized_fraud_prevented - h_fp_cost_tiered)
    
    # 3. Sentinel Incremental Value & Operational Trade-off
    missed_by_d = rec_df[(rec_df["is_fraud"] == 1) & (rec_df["is_flagged_d"] == 0)]
    missed_by_d_count = len(missed_by_d)
    missed_by_d_val = float(missed_by_d["transaction_amt"].sum())
    
    sentinel_intercepted = missed_by_d[missed_by_d["sentinel_intercepted"] == 1]
    sentinel_intercepted_count = len(sentinel_intercepted)
    sentinel_intercepted_val = float(sentinel_intercepted["transaction_amt"].sum())
    sentinel_realized_val = float(sentinel_intercepted_val * review_capture_efficiency)
    
    incremental_capture_rate = (sentinel_intercepted_count / missed_by_d_count * 100.0) if missed_by_d_count > 0 else 0.0
    additional_legit_escalations = fp_h - fp_d
    escalations_per_fraud_caught = (additional_legit_escalations / sentinel_intercepted_count) if sentinel_intercepted_count > 0 else 0.0
    
    return {
        "summary": {
            "total_transactions": total_txns,
            "total_transaction_value_inr": total_val,
            "total_fraud_cases": total_fraud_count,
            "total_fraud_value_inr": total_fraud_val,
            "total_legitimate_cases": total_legit_count,
            "total_legitimate_value_inr": total_legit_val,
            "fraud_prevalence_pct": (total_fraud_count / total_txns * 100.0)
        },
        "model_d_alone": {
            "true_positives": int(tp_d),
            "false_positives": int(fp_d),
            "true_negatives": int(tn_d),
            "false_negatives": int(fn_d),
            "precision": float(prec_d),
            "recall": float(rec_d),
            "f1_score": float(f1_d),
            "fpr": float(fpr_d),
            "pr_auc": float(pr_auc_d),
            "roc_auc": float(roc_auc_d),
            "fraud_cases_captured": int(tp_d),
            "fraud_cases_missed": int(fn_d),
            "fraud_direct_block_count": d_fraud_block_count,
            "fraud_review_count": d_fraud_review_count,
            "fraud_direct_block_val_inr": d_fraud_block_val,
            "fraud_review_val_inr": d_fraud_review_val,
            "direct_block_fraud_prevented_inr": d_fraud_block_val,
            "triaged_review_fraud_exposure_inr": d_fraud_review_val,
            "gross_fraud_exposure_intercepted_inr": d_fraud_prevented_val,
            "estimated_fraud_value_prevented_inr": d_fraud_prevented_val,
            "estimated_realized_fraud_prevented_inr": d_realized_fraud_prevented,
            "fraud_value_missed_inr": d_fraud_missed_val,
            "false_positive_blocks_count": d_fp_block_count,
            "false_positive_reviews_count": d_fp_review_count,
            "false_positive_blocks_val_inr": d_fp_block_val,
            "false_positive_reviews_val_inr": d_fp_review_val,
            "legitimate_value_inconvenienced_inr": d_legit_blocked_val,
            "estimated_false_positive_cost_inr": d_fp_cost_strict,
            "estimated_false_positive_cost_tiered_inr": d_fp_cost_tiered,
            "estimated_net_loss_avoided_inr": d_net_loss_avoided_strict,
            "estimated_net_loss_avoided_tiered_inr": d_net_loss_avoided_tiered,
            "estimated_realized_net_loss_avoided_tiered_inr": d_net_loss_avoided_realized,
            "review_capture_efficiency_assumption": review_capture_efficiency
        },
        "production_hybrid": {
            "true_positives": int(tp_h),
            "false_positives": int(fp_h),
            "true_negatives": int(tn_h),
            "false_negatives": int(fn_h),
            "precision": float(prec_h),
            "recall": float(rec_h),
            "f1_score": float(f1_h),
            "fpr": float(fpr_h),
            "pr_auc": float(pr_auc_h),
            "roc_auc": float(roc_auc_h),
            "fraud_cases_captured": int(tp_h),
            "fraud_cases_missed": int(fn_h),
            "fraud_direct_block_count": h_fraud_block_count,
            "fraud_review_count": h_fraud_review_count,
            "fraud_direct_block_val_inr": h_fraud_block_val,
            "fraud_review_val_inr": h_fraud_review_val,
            "direct_block_fraud_prevented_inr": h_fraud_block_val,
            "triaged_review_fraud_exposure_inr": h_fraud_review_val,
            "gross_fraud_exposure_intercepted_inr": h_fraud_prevented_val,
            "estimated_fraud_value_prevented_inr": h_fraud_prevented_val,
            "estimated_realized_fraud_prevented_inr": h_realized_fraud_prevented,
            "fraud_value_missed_inr": h_fraud_missed_val,
            "false_positive_blocks_count": h_fp_block_count,
            "false_positive_reviews_count": h_fp_review_count,
            "false_positive_blocks_val_inr": h_fp_block_val,
            "false_positive_reviews_val_inr": h_fp_review_val,
            "legitimate_value_inconvenienced_inr": h_legit_blocked_val,
            "estimated_false_positive_cost_inr": h_fp_cost_strict,
            "estimated_false_positive_cost_tiered_inr": h_fp_cost_tiered,
            "estimated_net_loss_avoided_inr": h_net_loss_avoided_strict,
            "estimated_net_loss_avoided_tiered_inr": h_net_loss_avoided_tiered,
            "estimated_realized_net_loss_avoided_tiered_inr": h_net_loss_avoided_realized,
            "review_capture_efficiency_assumption": review_capture_efficiency
        },
        "sentinel_incremental_value": {
            "fraud_missed_by_model_d_count": missed_by_d_count,
            "fraud_missed_by_model_d_value_inr": missed_by_d_val,
            "sentinel_intercepted_count": sentinel_intercepted_count,
            "sentinel_intercepted_value_inr": sentinel_intercepted_val,
            "incremental_fraud_exposure_intercepted_inr": sentinel_intercepted_val,
            "incremental_realized_loss_prevented_inr": sentinel_realized_val,
            "incremental_capture_rate_pct": float(incremental_capture_rate),
            "total_fraud_capture_increase_pct": float((tp_h - tp_d) / total_fraud_count * 100.0) if total_fraud_count > 0 else 0.0,
            "additional_legitimate_escalations": int(additional_legit_escalations),
            "escalations_per_additional_fraud_caught": float(escalations_per_fraud_caught),
            "operating_tradeoff": f"{escalations_per_fraud_caught:.1f} legitimate reviews per incremental fraud captured"
        }
    }

def run_validation_threshold_selection(val_df, cost_matrix, d_block=0.50):
    print("\n[P1-B] Selecting optimal thresholds on VALIDATION SPLIT ONLY...")
    y_val = val_df["isFraud"].values
    r_gbm_val = val_df["r_gbm"].values
    
    t_grid = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_d_review = 0.05
    for t in t_grid:
        preds = (r_gbm_val >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_d_review = float(t)
    print(f"  > Optimal Model D Validation Threshold (Review): {best_d_review:.2f} (Val F1: {best_f1:.4f})")
    print(f"  > Operational Model D Block Threshold: {d_block:.2f}")
    
    # Search for Sentinel threshold on validation
    best_sentinel_t = 0.45
    best_hybrid_score = -float("inf")
    # Search within operational community scoring range [0.40, 0.55]
    for s_t in np.linspace(0.40, 0.55, 31):
        test_rec = evaluate_decision_policy(val_df, d_block=d_block, d_review=best_d_review, sentinel_threshold=s_t, cost_matrix=cost_matrix)
        metrics = compute_metrics_from_records(test_rec, cost_matrix)
        inc_count = metrics["sentinel_incremental_value"]["sentinel_intercepted_count"]
        h_f1 = metrics["production_hybrid"]["f1_score"]
        h_fp = metrics["production_hybrid"]["false_positives"]
        
        # Objective: Reward incremental fraud detection while penalizing false positive volume
        # Tie-break preference around 0.45
        if inc_count > 0:
            score = (inc_count * 100.0) - h_fp + (h_f1 * 50.0) - abs(s_t - 0.45)
        else:
            score = (h_f1 * 50.0) - h_fp
            
        if score > best_hybrid_score:
            best_hybrid_score = score
            best_sentinel_t = float(s_t)
            
    print(f"  > Optimal Sentinel Network Threshold: {best_sentinel_t:.2f}")
    print("  > FREEZING THRESHOLDS FOR LOCKED TEST EVALUATION.")
    return d_block, best_d_review, best_sentinel_t

def run_business_evaluation():
    print("=" * 85)
    print("      ABUSE-RING SENTINEL: LOCKED CHRONOLOGICAL BUSINESS EVALUATION")
    print("=" * 85)
    
    data_config, policy_config = load_configs()
    cost_matrix = policy_config.get("cost_matrix", {
        "fp_cost": 1500.0,
        "chargeback_fee": 1200.0,
        "investigation_cost": 500.0,
        "fn_loss_factor": 1.0
    })
    
    train_df, val_df, test_df, split_meta = get_chronological_splits()
    print("\n[Split Integrity]")
    print(f"  Total Dataset: {split_meta['total_rows']:,} rows")
    print(f"  Train Split (70%): {split_meta['train_rows']:,} rows")
    print(f"  Validation Split (15%): {split_meta['val_rows']:,} rows")
    print(f"  LOCKED TEST Split (15%): {split_meta['test_rows']:,} rows")
    
    d_block, t_d_review, t_sentinel = run_validation_threshold_selection(val_df, cost_matrix)
    
    print("\n[P1-A] Executing Locked Test-Set Business Evaluation...")
    test_records_df = evaluate_decision_policy(
        test_df,
        d_block=d_block,
        d_review=t_d_review,
        sentinel_threshold=t_sentinel,
        cost_matrix=cost_matrix
    )
    metrics = compute_metrics_from_records(test_records_df, cost_matrix)
    
    evaluation_payload = {
        "evaluation_basis": "Locked Chronological Test Set",
        "timestamp_evaluated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits": split_meta,
        "frozen_thresholds": {
            "model_d_block_threshold": d_block,
            "model_d_review_threshold": t_d_review,
            "sentinel_threshold": t_sentinel,
            "threshold_selection_split": "VALIDATION ONLY (Frozen before test)"
        },
        "cost_assumptions": {
            "fp_cost_inr": cost_matrix.get("fp_cost", 1500.0),
            "chargeback_fee_inr": cost_matrix.get("chargeback_fee", 1200.0),
            "investigation_cost_inr": cost_matrix.get("investigation_cost", 500.0),
            "review_capture_efficiency": cost_matrix.get("review_capture_efficiency", 0.85),
            "is_assumption": True,
            "note": "Cost parameters and review capture efficiency (85%) are modeled operational assumptions, not observed transactional ledger losses."
        },
        "metrics": metrics
    }
    
    out_dir = os.path.join(PROJ_ROOT, "data", "processed", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    eval_json_path = os.path.join(out_dir, "test_business_evaluation.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)
        
    records_csv_path = os.path.join(out_dir, "test_decision_records.csv")
    test_records_df.to_csv(records_csv_path, index=False)
    
    print(f"\n[Saved Evaluation Artifacts]")
    print(f"  JSON Evaluation Summary: {eval_json_path}")
    print(f"  Transaction Decision Records: {records_csv_path}")
    
    s = metrics["summary"]
    d = metrics["model_d_alone"]
    h = metrics["production_hybrid"]
    inc = metrics["sentinel_incremental_value"]
    
    print("\n" + "=" * 85)
    print("                    FINAL LOCKED TEST SET PERFORMANCE LADDER")
    print("=" * 85)
    print(f"{'Metric':<36} | {'Model D Alone':<20} | {'Model D + Sentinel':<20} | {'Lift / Delta':<15}")
    print("-" * 85)
    print(f"{'Precision':<36} | {d['precision'] * 100:>18.2f}% | {h['precision'] * 100:>18.2f}% | {(h['precision'] - d['precision']) * 100:>+14.2f}%")
    print(f"{'Recall (Detection Rate)':<36} | {d['recall'] * 100:>18.2f}% | {h['recall'] * 100:>18.2f}% | {(h['recall'] - d['recall']) * 100:>+14.2f}%")
    print(f"{'F1-Score':<36} | {d['f1_score']:>19.4f} | {h['f1_score']:>19.4f} | {h['f1_score'] - d['f1_score']:>+14.4f}")
    print(f"{'PR-AUC':<36} | {d['pr_auc']:>19.4f} | {h['pr_auc']:>19.4f} | {h['pr_auc'] - d['pr_auc']:>+14.4f}")
    print(f"{'False Positive Rate (FPR)':<36} | {d['fpr'] * 100:>18.2f}% | {h['fpr'] * 100:>18.2f}% | {(h['fpr'] - d['fpr']) * 100:>+14.2f}%")
    print(f"{'Fraud Cases Captured / Total':<36} | {d['fraud_cases_captured']:>14} / {s['total_fraud_cases']} | {h['fraud_cases_captured']:>14} / {s['total_fraud_cases']} | {h['fraud_cases_captured'] - d['fraud_cases_captured']:>+14}")
    print("-" * 85)
    print(f"{'Est. Fraud Value Prevented':<36} | INR {d['estimated_fraud_value_prevented_inr']:>14,.2f} | INR {h['estimated_fraud_value_prevented_inr']:>14,.2f} | INR {h['estimated_fraud_value_prevented_inr'] - d['estimated_fraud_value_prevented_inr']:>+10,.2f}")
    print(f"{'Fraud Value Missed':<36} | INR {d['fraud_value_missed_inr']:>14,.2f} | INR {h['fraud_value_missed_inr']:>14,.2f} | INR {h['fraud_value_missed_inr'] - d['fraud_value_missed_inr']:>+10,.2f}")
    print(f"{'False Positives (Total)':<36} | {d['false_positives']:>19} | {h['false_positives']:>19} | {h['false_positives'] - d['false_positives']:>+14}")
    print(f"{'  - False Blocks (Friction)':<36} | {d['false_positive_blocks_count']:>19} | {h['false_positive_blocks_count']:>19} | {h['false_positive_blocks_count'] - d['false_positive_blocks_count']:>+14}")
    print(f"{'  - False Reviews (SLA Triage)':<36} | {d['false_positive_reviews_count']:>19} | {h['false_positive_reviews_count']:>19} | {h['false_positive_reviews_count'] - d['false_positive_reviews_count']:>+14}")
    print(f"{'Est. FP Cost (Strict INR 1500)':<36} | INR {d['estimated_false_positive_cost_inr']:>14,.2f} | INR {h['estimated_false_positive_cost_inr']:>14,.2f} | INR {h['estimated_false_positive_cost_inr'] - d['estimated_false_positive_cost_inr']:>+10,.2f}")
    print(f"{'Est. FP Cost (Tiered Operational)':<36} | INR {d['estimated_false_positive_cost_tiered_inr']:>14,.2f} | INR {h['estimated_false_positive_cost_tiered_inr']:>14,.2f} | INR {h['estimated_false_positive_cost_tiered_inr'] - d['estimated_false_positive_cost_tiered_inr']:>+10,.2f}")
    print(f"{'Est. Net Loss Avoided (Tiered)':<36} | INR {d['estimated_net_loss_avoided_tiered_inr']:>14,.2f} | INR {h['estimated_net_loss_avoided_tiered_inr']:>14,.2f} | INR {h['estimated_net_loss_avoided_tiered_inr'] - d['estimated_net_loss_avoided_tiered_inr']:>+10,.2f}")
    print("=" * 85)
    
    print("\n[Sentinel Incremental Fraud Capture & Operating Trade-off]")
    print(f"  Fraud missed by Model D alone: {inc['fraud_missed_by_model_d_count']} cases (INR {inc['fraud_missed_by_model_d_value_inr']:,.2f})")
    print(f"  Fraud intercepted by Sentinel: {inc['sentinel_intercepted_count']} cases (INR {inc['sentinel_intercepted_value_inr']:,.2f})")
    print(f"  Sentinel incremental capture rate: {inc['incremental_capture_rate_pct']:.2f}% of missed fraud")
    print(f"  Total fraud capture lift: {inc['total_fraud_capture_increase_pct']:+.2f}%")
    print(f"  Operational Trade-off: {inc['operating_tradeoff']}")
    print("=" * 85)
    
    return evaluation_payload

if __name__ == "__main__":
    run_business_evaluation()
