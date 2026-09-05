"""
Risk Decision Engine & Merchant Loss Evaluation Module
Part of the AI Risk Sentinel - Fraud Loss Prevention System

Provides:
1. Probability Calibration (Isotonic Regression)
2. Three-Way Decision Routing (ALLOW / MANUAL_REVIEW / BLOCK)
3. Merchant Loss & Financial Impact Evaluation (Fraud Value Prevented vs. FP Friction Cost)
4. Validation-Driven Threshold Optimization
5. Deterministic Risk Explanations ("Why was this transaction flagged?")
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import joblib

# Default business assumptions
DEFAULT_FP_COST_RATE = 0.15  # 15% friction/lost lifetime value cost on blocked legitimate transactions
DEFAULT_THRESHOLD_BLOCK = 0.30398
DEFAULT_THRESHOLD_REVIEW = 0.15000

# -------------------------------------------------------------------
# 1. Probability Calibration
# -------------------------------------------------------------------
def fit_probability_calibrator(y_val: np.ndarray, raw_scores: np.ndarray) -> IsotonicRegression:
    """
    Fits an isotonic regression model on validation data to calibrate raw model scores
    into true empirical fraud probabilities.
    """
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    ir.fit(raw_scores, y_val)
    return ir

def calibrate_probabilities(calibrator: IsotonicRegression, raw_scores: np.ndarray) -> np.ndarray:
    """
    Transforms raw GBDT scores into calibrated probabilities.
    """
    if calibrator is None:
        return np.clip(raw_scores, 0.0, 1.0)
    return np.clip(calibrator.predict(raw_scores), 0.0, 1.0)

def save_calibrator(calibrator: IsotonicRegression, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(calibrator, path)

def load_calibrator(path: str) -> IsotonicRegression:
    if os.path.exists(path):
        return joblib.load(path)
    return None

# -------------------------------------------------------------------
# 2. Three-Way Decision Routing
# -------------------------------------------------------------------
def make_decision(prob_d: float, prob_sentinel: float, threshold_block: float = DEFAULT_THRESHOLD_BLOCK, threshold_review: float = DEFAULT_THRESHOLD_REVIEW) -> str:
    """
    Server-side decision logic routing transactions to BLOCK, MANUAL_REVIEW, or ALLOW.
    Stage 1: If Model D primary risk >= threshold_block -> BLOCK immediately.
    Stage 2: If Model D allows, but Sentinel network coordination >= threshold_review -> MANUAL_REVIEW.
    Stage 3: Otherwise -> ALLOW.
    """
    if prob_d >= threshold_block:
        return "BLOCK"
    elif prob_sentinel >= threshold_review:
        return "MANUAL_REVIEW"
    else:
        return "ALLOW"

def make_decisions_batch(probs_d: np.ndarray, probs_sentinel: np.ndarray, threshold_block: float = DEFAULT_THRESHOLD_BLOCK, threshold_review: float = DEFAULT_THRESHOLD_REVIEW) -> np.ndarray:
    """
    Vectorized decision assignment for bulk validation/test evaluation.
    """
    n = len(probs_d)
    decisions = np.full(n, "ALLOW", dtype=object)
    
    review_mask = (probs_d < threshold_block) & (probs_sentinel >= threshold_review)
    block_mask = probs_d >= threshold_block
    
    decisions[review_mask] = "MANUAL_REVIEW"
    decisions[block_mask] = "BLOCK"
    return decisions

# -------------------------------------------------------------------
# 3. Financial Impact & Merchant Loss Evaluation
# -------------------------------------------------------------------
def compute_merchant_loss_metrics(
    y_true: np.ndarray,
    probs_d: np.ndarray,
    probs_sentinel: np.ndarray,
    amounts: np.ndarray,
    threshold_block: float,
    threshold_review: float,
    fp_cost_rate: float = DEFAULT_FP_COST_RATE
) -> dict:
    """
    Calculates comprehensive financial impact metrics on transaction values ($),
    contrasting fraud loss prevented against false-positive friction costs.
    """
    decisions = make_decisions_batch(probs_d, probs_sentinel, threshold_block, threshold_review)
    
    total_transactions = int(len(y_true))
    total_volume = float(np.sum(amounts))
    
    fraud_mask = (y_true == 1)
    legit_mask = (y_true == 0)
    
    total_fraud_count = int(np.sum(fraud_mask))
    total_fraud_volume = float(np.sum(amounts[fraud_mask]))
    
    total_legit_count = int(np.sum(legit_mask))
    total_legit_volume = float(np.sum(amounts[legit_mask]))
    
    # Decisions breakdown
    blocked_mask = (decisions == "BLOCK")
    reviewed_mask = (decisions == "MANUAL_REVIEW")
    allowed_mask = (decisions == "ALLOW")
    
    # 1. Fraud Prevented (Blocked Fraud)
    fraud_blocked_mask = fraud_mask & blocked_mask
    fraud_blocked_count = int(np.sum(fraud_blocked_mask))
    fraud_value_prevented = float(np.sum(amounts[fraud_blocked_mask]))
    
    # 2. Fraud Missed (Allowed Fraud)
    fraud_missed_mask = fraud_mask & allowed_mask
    fraud_missed_count = int(np.sum(fraud_missed_mask))
    fraud_value_missed = float(np.sum(amounts[fraud_missed_mask]))
    
    # 3. Fraud in Secondary Review
    fraud_reviewed_mask = fraud_mask & reviewed_mask
    fraud_reviewed_count = int(np.sum(fraud_reviewed_mask))
    fraud_value_reviewed = float(np.sum(amounts[fraud_reviewed_mask]))
    
    # 4. False Positives (Legitimate Blocked)
    fp_mask = legit_mask & blocked_mask
    fp_count = int(np.sum(fp_mask))
    fp_value = float(np.sum(amounts[fp_mask]))
    fp_cost = float(fp_value * fp_cost_rate)
    
    # 5. False Positives in Review (Legitimate Reviewed)
    fp_review_mask = legit_mask & reviewed_mask
    fp_review_count = int(np.sum(fp_review_mask))
    fp_review_value = float(np.sum(amounts[fp_review_mask]))
    
    # 6. Net Loss Avoided
    net_loss_avoided = float(fraud_value_prevented - fp_cost)
    
    # 7. Rates
    fraud_recall = float(fraud_blocked_count / total_fraud_count) if total_fraud_count > 0 else 0.0
    fraud_value_capture_rate = float(fraud_value_prevented / total_fraud_volume) if total_fraud_volume > 0 else 0.0
    
    precision = float(fraud_blocked_count / np.sum(blocked_mask)) if np.sum(blocked_mask) > 0 else 0.0
    fpr = float(fp_count / total_legit_count) if total_legit_count > 0 else 0.0
    f1 = float(2 * precision * fraud_recall / (precision + fraud_recall)) if (precision + fraud_recall) > 0 else 0.0
    
    # Model D standalone AUCs
    try:
        pr_auc = float(average_precision_score(y_true, probs_d))
        roc_auc = float(roc_auc_score(y_true, probs_d))
    except Exception:
        pr_auc = 0.0
        roc_auc = 0.0

    return {
        "threshold_block": float(threshold_block),
        "threshold_review": float(threshold_review),
        "fp_cost_rate": float(fp_cost_rate),
        
        # Volume & Counts
        "total_transactions": total_transactions,
        "total_volume": round(total_volume, 2),
        "total_fraud_count": total_fraud_count,
        "total_fraud_volume": round(total_fraud_volume, 2),
        "total_legit_count": total_legit_count,
        "total_legit_volume": round(total_legit_volume, 2),
        
        # Operational Actions
        "blocked_total": int(np.sum(blocked_mask)),
        "reviewed_total": int(np.sum(reviewed_mask)),
        "allowed_total": int(np.sum(allowed_mask)),
        
        # Financial Losses & Savings
        "fraud_loss_before": round(total_fraud_volume, 2),
        "fraud_loss_after": round(fraud_value_missed, 2),
        "fraud_value_prevented": round(fraud_value_prevented, 2),
        "fraud_value_missed": round(fraud_value_missed, 2),
        "fraud_value_in_review": round(fraud_value_reviewed, 2),
        "false_positive_value": round(fp_value, 2),
        "false_positive_cost": round(fp_cost, 2),
        "net_loss_avoided": round(net_loss_avoided, 2),
        
        # Model & Policy Rates
        "fraud_recall": round(fraud_recall, 4),
        "fraud_value_capture_rate": round(fraud_value_capture_rate, 4),
        "precision": round(precision, 4),
        "false_positive_rate": round(fpr, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        
        # Sentinel Capture Counts
        "sentinel_captured_fraud": fraud_reviewed_count,
        "sentinel_captured_fraud_value": round(fraud_value_reviewed, 2)
    }

# -------------------------------------------------------------------
# 4. Validation-Driven Threshold Optimization
# -------------------------------------------------------------------
def optimize_thresholds_validation(
    y_val: np.ndarray,
    probs_d_val: np.ndarray,
    probs_sentinel_val: np.ndarray,
    amounts_val: np.ndarray,
    fp_cost_rate: float = DEFAULT_FP_COST_RATE
) -> dict:
    """
    Grid sweep on the validation split to select thresholds that maximize net loss avoided
    subject to operational friction constraints.
    """
    best_net_loss = -float('inf')
    best_thresholds = (DEFAULT_THRESHOLD_BLOCK, DEFAULT_THRESHOLD_REVIEW)
    best_metrics = None
    
    d_candidates = [0.15, 0.20, 0.25, 0.30, 0.30398, 0.35, 0.40, 0.50]
    sentinel_candidates = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    
    sweep_results = []
    
    for tb in d_candidates:
        for tr in sentinel_candidates:
            metrics = compute_merchant_loss_metrics(
                y_val, probs_d_val, probs_sentinel_val, amounts_val,
                threshold_block=tb, threshold_review=tr, fp_cost_rate=fp_cost_rate
            )
            sweep_results.append({
                "threshold_block": tb,
                "threshold_review": tr,
                "net_loss_avoided": metrics["net_loss_avoided"],
                "fraud_value_prevented": metrics["fraud_value_prevented"],
                "fp_cost": metrics["false_positive_cost"],
                "f1_score": metrics["f1_score"],
                "precision": metrics["precision"],
                "recall": metrics["fraud_recall"]
            })
            
            # Criterion: Maximize Net Loss Avoided with FPR <= 3%
            if metrics["false_positive_rate"] <= 0.03:
                if metrics["net_loss_avoided"] > best_net_loss:
                    best_net_loss = metrics["net_loss_avoided"]
                    best_thresholds = (tb, tr)
                    best_metrics = metrics
                    
    if best_metrics is None:
        best_thresholds = (DEFAULT_THRESHOLD_BLOCK, DEFAULT_THRESHOLD_REVIEW)
        best_metrics = compute_merchant_loss_metrics(
            y_val, probs_d_val, probs_sentinel_val, amounts_val,
            threshold_block=best_thresholds[0], threshold_review=best_thresholds[1],
            fp_cost_rate=fp_cost_rate
        )

    return {
        "best_threshold_block": best_thresholds[0],
        "best_threshold_review": best_thresholds[1],
        "best_metrics": best_metrics,
        "sweep_results": sweep_results
    }

# -------------------------------------------------------------------
# 5. Deterministic Risk Explanations ("Why was this flagged?")
# -------------------------------------------------------------------
def explain_risk(row: dict, prob_d: float, prob_sentinel: float, threshold_block: float, threshold_review: float) -> list:
    """
    Generates human-readable evidence signals explaining why a transaction was blocked or reviewed.
    CRITICAL: Does NOT use ground-truth isFraud label. Uses strictly feature evidence available at runtime.
    """
    signals = []
    
    # 1. Model D Risk Trigger
    if prob_d >= threshold_block:
        signals.append(f"Model D transaction risk score ({prob_d:.4f}) exceeds blocking threshold ({threshold_block:.4f}).")
    elif prob_d >= 0.10:
        signals.append(f"Model D flagged elevated transaction anomaly patterns (Score: {prob_d:.4f}).")
        
    # 2. Sentinel Network Risk Trigger
    if prob_sentinel >= threshold_review:
        signals.append(f"Abuse-Ring Sentinel detected multi-entity coordination (Score: {prob_sentinel:.4f} >= {threshold_review:.4f}).")

    # 3. Device Entity Evidence
    dev_cards = row.get('device_unique_card_count', 0)
    if dev_cards is not None and not pd.isna(dev_cards) and int(dev_cards) >= 3:
        signals.append(f"Device associated with {int(dev_cards)} unique card identities in historical records.")
        
    dev_fraud_rate = row.get('device_connected_fraud_rate', 0.0)
    if dev_fraud_rate is not None and not pd.isna(dev_fraud_rate) and float(dev_fraud_rate) >= 0.05:
        signals.append(f"Connected device network exhibits elevated historical fraud exposure ({float(dev_fraud_rate)*100:.1f}%).")
        
    # 4. Address Entity Evidence
    addr_cards = row.get('addr_unique_card_count', 0)
    if addr_cards is not None and not pd.isna(addr_cards) and int(addr_cards) >= 3:
        signals.append(f"Billing address associated with {int(addr_cards)} distinct cards in network history.")
        
    addr_fraud_rate = row.get('addr_connected_fraud_rate', 0.0)
    if addr_fraud_rate is not None and not pd.isna(addr_fraud_rate) and float(addr_fraud_rate) >= 0.05:
        signals.append(f"Connected address network exhibits elevated historical fraud exposure ({float(addr_fraud_rate)*100:.1f}%).")

    # 5. Temporal Velocity & Convergence
    conv_72h = row.get('rapid_card_convergence', 0)
    if conv_72h is not None and not pd.isna(conv_72h) and int(conv_72h) >= 3:
        signals.append(f"Rapid entity convergence: {int(conv_72h)} cards linked to same entity cluster within trailing 72 hours.")
        
    cross_overlap = row.get('cross_entity_convergence', 0)
    if cross_overlap is not None and not pd.isna(cross_overlap) and float(cross_overlap) > 0:
        signals.append("Cross-entity hub overlap: Card shares both device hardware and location code with multiple distinct cards.")
        
    # 6. Past Card Velocity
    card_past_count = row.get('card1_past_count', 0)
    if card_past_count is not None and not pd.isna(card_past_count) and int(card_past_count) >= 20:
        signals.append(f"High transaction frequency: {int(card_past_count)} historical transactions on this payment instrument.")

    if not signals:
        signals.append("Transaction exhibits standard baseline characteristics with clean historical entity exposure.")
        
    return signals
