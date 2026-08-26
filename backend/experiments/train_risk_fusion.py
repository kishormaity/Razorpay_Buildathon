import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix
)

def evaluate_predictions(y_true, y_prob):
    """
    Computes standard evaluation metrics (PR-AUC, ROC-AUC, optimal F1, Precision, Recall, FPR)
    """
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Precision Recall Curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # F1 Score calculation
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
    p_opt, r_opt, f_opt, _ = precision_recall_fscore_support(y_true, preds_opt, average='binary')
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, preds_opt).ravel()
    fpr_opt = fp_opt / (fp_opt + tn_opt) if (fp_opt + tn_opt) > 0 else 0.0
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'precision': p_opt,
        'recall': r_opt,
        'fpr': fpr_opt,
        'recalls_curve': recalls,
        'precisions_curve': precisions
    }

def main():
    print("=" * 70)
    print("               IEEE-CIS RISK SCORE FUSION PIPELINE (5B)")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    baseline_preds_path = os.path.join(processed_dir, 'predictions/baseline_predictions.parquet')
    behav_preds_path = os.path.join(processed_dir, 'predictions/behavior_only_xgb_predictions.parquet')
    
    fused_preds_path = os.path.join(processed_dir, 'predictions/fused_predictions.parquet')
    report_path = os.path.join(processed_dir, 'reports/risk_fusion_report.md')
    curves_path = os.path.join(processed_dir, 'plots/fused_pr_curves.png')

    # 1. Load predictions
    print("\n[1/6] Loading predictions datasets...")
    if not os.path.exists(baseline_preds_path):
        print(f"[ERROR] Baseline predictions file not found at: {baseline_preds_path}")
        sys.exit(1)
        
    if not os.path.exists(behav_preds_path):
        print(f"[ERROR] Behavior-only predictions file not found at: {behav_preds_path}")
        sys.exit(1)

    start_time = time.time()
    baseline_df = pd.read_parquet(baseline_preds_path)
    behav_df = pd.read_parquet(behav_preds_path)
    print(f"Loaded baseline shape: {baseline_df.shape} and behavior shape: {behav_df.shape} in {time.time() - start_time:.2f} seconds.")

    # 2. Sequence verification and alignment check
    print("\n[2/6] Verifying predictions alignment...")
    if len(baseline_df) != len(behav_df):
        raise ValueError(f"Lengths do not match: Baseline={len(baseline_df)}, Behavioral={len(behav_df)}")
        
    if not np.array_equal(baseline_df['TransactionID'].values, behav_df['TransactionID'].values):
        raise ValueError("TransactionIDs do not match or are not in the same order!")
        
    if not np.array_equal(baseline_df['isFraud'].values, behav_df['isFraud'].values):
        raise ValueError("isFraud labels do not match between predictions files!")
    print("✅ Validation TransactionIDs and labels align perfectly.")

    # 3. Two-Stage Chronological Split of Validation set (118,108 rows)
    print("\n[3/6] Splitting validation chronologically into Meta-Development and Meta-Test...")
    total_val_rows = len(baseline_df)
    meta_dev_idx = int(total_val_rows * 0.5) # 59,054 rows
    
    # Split targets
    y_val = baseline_df['isFraud'].values
    y_dev = y_val[:meta_dev_idx]
    y_test = y_val[meta_dev_idx:]
    
    # Split probabilities
    p_tx_val = baseline_df['fraud_probability'].values
    p_behav_val = behav_df['fraud_probability'].values
    
    p_tx_dev = p_tx_val[:meta_dev_idx]
    p_behav_dev = p_behav_val[:meta_dev_idx]
    
    p_tx_test = p_tx_val[meta_dev_idx:]
    p_behav_test = p_behav_val[meta_dev_idx:]
    
    print(f"  * Meta-Development Set (Earliest 50%): {len(y_dev):,} rows")
    print(f"  * Meta-Test Set (Latest 50%):        {len(y_test):,} rows")

    # 4. Grid Search Linear Weights on Meta-Development
    print("\n[4/6] Grid searching optimal weighted fusion combination (on Meta-Dev)...")
    best_w = 1.0
    best_dev_pr_auc = 0.0
    
    # Search from 0.0 to 1.0 in steps of 0.01
    weights = np.linspace(0.0, 1.0, 101)
    for w in weights:
        fused_dev = w * p_tx_dev + (1.0 - w) * p_behav_dev
        dev_pr_auc = average_precision_score(y_dev, fused_dev)
        if dev_pr_auc > best_dev_pr_auc:
            best_dev_pr_auc = dev_pr_auc
            best_w = w
            
    print(f"  * Best weight (w_opt): {best_w:.2f} (Transaction-weight)")
    print(f"  * Best PR-AUC on Meta-Development: {best_dev_pr_auc:.5f}")

    # Apply w_opt to Meta-Test
    p_fused_test = best_w * p_tx_test + (1.0 - best_w) * p_behav_test
    # Save full val predictions
    p_fused_val = best_w * p_tx_val + (1.0 - best_w) * p_behav_val

    # 5. Train Stacked Logistic Regression on Meta-Development
    print("\n[5/6] Training stacked Logistic Regression meta-model...")
    X_dev = pd.DataFrame({'p_tx': p_tx_dev, 'p_behav': p_behav_dev})
    X_test = pd.DataFrame({'p_tx': p_tx_test, 'p_behav': p_behav_test})
    X_val = pd.DataFrame({'p_tx': p_tx_val, 'p_behav': p_behav_val})
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_dev, y_dev)
    
    lr_probs_test = lr_model.predict_proba(X_test)[:, 1]
    lr_probs_val = lr_model.predict_proba(X_val)[:, 1]
    
    print(f"  * Meta-model Coefs:  p_tx={lr_model.coef_[0][0]:.4f}, p_behav={lr_model.coef_[0][1]:.4f}")
    print(f"  * Meta-model Intercept: {lr_model.intercept_[0]:.4f}")

    # 6. Evaluate all on Meta-Test Subset (Apples-to-Apples)
    print("\n[6/6] Slicing and evaluating all models on Meta-Test Set...")
    metrics_tx = evaluate_predictions(y_test, p_tx_test)
    metrics_behav = evaluate_predictions(y_test, p_behav_test)
    metrics_weighted = evaluate_predictions(y_test, p_fused_test)
    metrics_lr = evaluate_predictions(y_test, lr_probs_test)

    print("\nCOMPARATIVE RESULTS ON META-TEST SUBSET:")
    print(f"  * Model A (Tx-Only) PR-AUC:      {metrics_tx['pr_auc']:.5f}")
    print(f"  * Model B (Behav-Only) PR-AUC:   {metrics_behav['pr_auc']:.5f}")
    print(f"  * Fused Model (Weighted) PR-AUC: {metrics_weighted['pr_auc']:.5f} (Delta: {metrics_weighted['pr_auc'] - metrics_tx['pr_auc']:+.5f})")
    print(f"  * Fused Model (Logistic) PR-AUC: {metrics_lr['pr_auc']:.5f} (Delta: {metrics_lr['pr_auc'] - metrics_tx['pr_auc']:+.5f})")

    # Serialize fused predictions
    fused_predictions_df = pd.DataFrame({
        'TransactionID': baseline_df['TransactionID'],
        'TransactionDT': baseline_df['TransactionDT'],
        'isFraud': y_val,
        'fraud_probability_tx': p_tx_val,
        'fraud_probability_behav': p_behav_val,
        'fraud_probability_weighted': p_fused_val,
        'fraud_probability_logistic': lr_probs_val
    })
    fused_predictions_df.to_parquet(fused_preds_path, engine='pyarrow', index=False)
    print(f"\nSaved fused probabilities Parquet to: {fused_preds_path}")

    # Generate Comparative PR Curves Plot
    plt.figure(figsize=(10, 8))
    plt.plot(metrics_tx['recalls_curve'], metrics_tx['precisions_curve'], color='gray', linestyle='--', lw=2, label=f"Model A (Tx-Only) (PR-AUC = {metrics_tx['pr_auc']:.5f})")
    plt.plot(metrics_behav['recalls_curve'], metrics_behav['precisions_curve'], color='blue', linestyle=':', lw=2, label=f"Model B (Behav-Only) (PR-AUC = {metrics_behav['pr_auc']:.5f})")
    plt.plot(metrics_weighted['recalls_curve'], metrics_weighted['precisions_curve'], color='darkorange', linestyle='-', lw=2.5, label=f"Weighted Fusion (w={best_w:.2f}) (PR-AUC = {metrics_weighted['pr_auc']:.5f})")
    plt.plot(metrics_lr['recalls_curve'], metrics_lr['precisions_curve'], color='red', linestyle='-.', lw=2, label=f"Logistic Stack (PR-AUC = {metrics_lr['pr_auc']:.5f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparative Precision-Recall Curves on Meta-Test')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative PR curves graph to: {curves_path}")

    # Write Markdown Performance Report
    report_content = f"""# Phase 5B: Risk Score Fusion Evaluation Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report evaluates stacked combinations of our transaction-risk and behavioral-risk models. By separating transaction-level logic (Model A LightGBM) and card history context (Model B XGBoost) and fusing their output probabilities, we measure whether stacked fusion improves predictive power compared to a transaction-only classifier.

---

## 1. Experimental Protocol & Splits

To ensure strict validation and prevent data leakage:
* **Validation Subset Size**: {total_val_rows:,} rows.
* **Meta-Development Set (Earliest 50%)**: {len(y_dev):,} rows. Used for linear weight grid search and logistic stacking training.
* **Meta-Test Set (Latest 50%)**: {len(y_test):,} rows. Used for final comparative evaluations.

All model metrics below are calculated on the **Meta-Test Set** for a clean, apples-to-apples baseline comparison.

---

## 2. Comparative Performance Matrix (Evaluated on Meta-Test)

| Metric | Model A (Transaction-Only) | Behavioral Model (XGBoost) | Weighted Fusion ($w_{{opt}}$ = {best_w:.2f}) | Logistic Stack Meta-Classifier |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Primary)** | `{metrics_tx['pr_auc']:.5f}` | `{metrics_behav['pr_auc']:.5f}` | `{metrics_weighted['pr_auc']:.5f}` | `{metrics_lr['pr_auc']:.5f}` |
| **ROC-AUC** | `{metrics_tx['roc_auc']:.5f}` | `{metrics_behav['roc_auc']:.5f}` | `{metrics_weighted['roc_auc']:.5f}` | `{metrics_lr['roc_auc']:.5f}` |
| **F1 @ Optimal** | `{metrics_tx['best_f1']:.5f}` | `{metrics_behav['best_f1']:.5f}` | `{metrics_weighted['best_f1']:.5f}` | `{metrics_lr['best_f1']:.5f}` |
| **Optimal Threshold** | `{metrics_tx['best_threshold']:.4f}` | `{metrics_behav['best_threshold']:.4f}` | `{metrics_weighted['best_threshold']:.4f}` | `{metrics_lr['best_threshold']:.4f}` |
| **FPR @ Optimal** | `{metrics_tx['fpr']:.5f}` | `{metrics_behav['fpr']:.5f}` | `{metrics_weighted['fpr']:.5f}` | `{metrics_lr['fpr']:.5f}` |

---

## 3. Key Observations & Findings

1. **Optimal Linear Combination Weight**: The grid-search selected **`w_opt = {best_w:.2f}`** as the weight for the transaction baseline, meaning the fused score is computed as:
   `fused_prob = {best_w:.2f} * p_tx + {1.0 - best_w:.2f} * p_behav`
   This shows that the transaction model remains the dominant feature, but is supplemented by a `{1.0 - best_w:.2f}` weight of the behavioral classifier.
2. **Logistic Stack Calibration**: The meta-model coefficients are `p_tx = {lr_model.coef_[0][0]:.4f}` and `p_behav = {lr_model.coef_[0][1]:.4f}` with an intercept of `{lr_model.intercept_[0]:.4f}`.
3. **PR-AUC Improvement**: Check the Delta (Fusion vs. Model A) in the matrix above. If Fused Model PR-AUC exceeds Model A's `{metrics_tx['pr_auc']:.5f}`, it demonstrates that historical behavioral risk is complementary to transaction-level logic.

---

## 4. Serialization Outputs

The following files have been saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Fused Probabilities Parquet**: [`fused_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/fused_predictions.parquet) (contains columns for transaction probabilities, behavioral probabilities, weighted fusion, and logistic stack fusion)
2. **Evaluation Metrics**: [`risk_fusion_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/risk_fusion_report.md)
3. **Comparative PR Curves Graph**: [`fused_pr_curves.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/fused_pr_curves.png)
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Saved risk fusion evaluation report to: {report_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("RISK FUSION TRAINING AND COMPARATIVE EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    main()
