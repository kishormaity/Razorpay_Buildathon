import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "configs", "risk_policy.yaml"))

def load_policy_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    # Default fallbacks
    return {
        "cost_matrix": {
            "fp_cost": 1500.0,
            "chargeback_fee": 1200.0,
            "investigation_cost": 500.0,
            "fn_loss_factor": 1.0
        }
    }

def calculate_expected_loss(y_true, y_pred_action, amounts):
    """
    Computes financial expected loss based on action outcomes.
    y_pred_action: array of strings ('ALLOW', 'MONITOR', 'MANUAL_REVIEW', 'HOLD')
    amounts: transaction amounts
    """
    config = load_policy_config()
    fp_cost = config["cost_matrix"]["fp_cost"]
    chargeback_fee = config["cost_matrix"]["chargeback_fee"]
    invest_cost = config["cost_matrix"]["investigation_cost"]
    fn_factor = config["cost_matrix"]["fn_loss_factor"]
    
    total_loss = 0.0
    
    for i in range(len(y_true)):
        true_label = y_true[i]
        action = y_pred_action[i]
        amt = float(amounts[i])
        
        # Scenario A: ALLOW
        if action == "ALLOW":
            if true_label == 1: # False Negative ( Fraud )
                total_loss += (amt * fn_factor) + chargeback_fee
                
        # Scenario B: MONITOR
        elif action == "MONITOR":
            if true_label == 1:
                total_loss += (amt * fn_factor) + chargeback_fee
                
        # Scenario C: MANUAL_REVIEW
        elif action == "MANUAL_REVIEW":
            total_loss += invest_cost
            if true_label == 1: # Flagged, but held in review (prevented fraud loss)
                pass # Prevented fraud loss, but incurred review fee
            else: # False Positive friction
                total_loss += fp_cost
                
        # Scenario D: HOLD / BLOCK
        elif action in ("HOLD", "BLOCK"):
            if true_label == 0: # False Positive block
                total_loss += fp_cost
                
    return total_loss

def evaluate_predictions(y_true, y_prob, amounts=None):
    """
    Computes PR-AUC, ROC-AUC, optimal F1, best threshold, and FPR.
    """
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Safely compute F1 curve
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0
    )
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    preds_opt = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds_opt).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    results = {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'best_f1': float(best_f1),
        'best_threshold': float(best_threshold),
        'fpr': float(fpr),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }
    
    if amounts is not None:
        # Evaluate standard policy expected loss vs baseline
        # Let's map predictions using the best threshold: >= threshold => MANUAL_REVIEW else ALLOW
        actions = np.where(y_prob >= best_threshold, 'MANUAL_REVIEW', 'ALLOW')
        expected_loss = calculate_expected_loss(y_true, actions, amounts)
        results['expected_loss'] = expected_loss
        
    return results
