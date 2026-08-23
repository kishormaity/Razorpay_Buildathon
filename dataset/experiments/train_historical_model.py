import os
import sys
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix
)

def main():
    print("=" * 70)
    print("         IEEE-CIS LIGHTGBM TRANSACTION + HISTORICAL MODEL (D)")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    features_path = os.path.join(processed_dir, 'features/historical_features.parquet')
    baseline_preds_path = os.path.join(processed_dir, 'predictions/baseline_predictions.parquet')
    
    model_save_path = os.path.join(processed_dir, 'models/historical_lgb_model.txt')
    preds_parquet_path = os.path.join(processed_dir, 'predictions/historical_predictions.parquet')
    report_path = os.path.join(processed_dir, 'reports/historical_evaluation_report.md')
    pr_curve_path = os.path.join(processed_dir, 'plots/historical_pr_curves.png')

    # 1. Load Datasets
    print("\n[1/7] Loading historical features...")
    if not os.path.exists(features_path):
        print(f"[ERROR] Historical features parquet file not found at: {features_path}")
        print("Please run historical_features.py first.")
        sys.exit(1)
        
    if not os.path.exists(baseline_preds_path):
        print(f"[ERROR] Baseline predictions file not found at: {baseline_preds_path}")
        print("Please run train_transaction_baseline.py first.")
        sys.exit(1)
        
    start_time = time.time()
    df = pd.read_parquet(features_path)
    baseline_df = pd.read_parquet(baseline_preds_path)
    print(f"Loaded datasets in {time.time() - start_time:.2f} seconds.")
    print(f"  * Enriched dataset: {df.shape}")

    # 2. Chronological Sorting & Train/Val Splitting
    print("\n[2/7] Preparing chronological splits...")
    t0 = time.time()
    
    # Sort chronologically by TransactionDT
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    total_rows = len(df)
    split_idx = int(total_rows * 0.8)
    split_dt = df['TransactionDT'].iloc[split_idx]
    
    # Verify ID Alignment
    print("Checking alignment with baseline validation set...")
    val_ids = df.iloc[split_idx:]['TransactionID'].values
    baseline_ids = baseline_df['TransactionID'].values
    
    if not np.array_equal(val_ids, baseline_ids):
        raise ValueError("Validation set TransactionIDs do not align with baseline predictions!")
    print("✅ Validation TransactionIDs align perfectly with baseline validation set.")
    
    # Group features matrix and labels
    X = df.drop(columns=['TransactionID', 'isFraud'])
    y = df['isFraud']
    
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    
    X_val = X.iloc[split_idx:].copy()
    y_val = y.iloc[split_idx:].copy()
    
    print(f"Split completed in {time.time() - t0:.2f} seconds.")
    print(f"  * Features Count:           {len(X_train.columns)} columns (393 original + 10 historical)")
    print(f"  * Training set (Earliest):  {len(X_train):,} rows")
    print(f"  * Validation set (Latest):  {len(X_val):,} rows")

    # 4. LightGBM Dataset Construction
    print("\n[3/7] Setting up LightGBM datasets...")
    # Identify categorical columns (already pandas category type)
    categorical_cols = list(X_train.select_dtypes(include=['category']).columns)
    
    # Create LightGBM datasets
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset, categorical_feature=categorical_cols)

    # 5. Model Training (Exact Same Parameters as Baseline)
    print("\n[4/7] Training LightGBM model with transaction + historical features...")
    
    params = {
        'objective': 'binary',
        'metric': 'average_precision',  # PR-AUC
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'scale_pos_weight': 1.0,         # Unweighted baseline setup
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    t0 = time.time()
    # Train the model with early stopping
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[train_dataset, val_dataset],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    print(f"Model training completed in {time.time() - t0:.2f} seconds.")
    print(f"Best iteration: {model.best_iteration}")

    # 6. Evaluate Predictions & Compare with Baseline
    print("\n[5/7] Evaluating predictions and comparing with baseline...")
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Compute current model metrics
    pr_auc = average_precision_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_preds)
    
    # Reference Threshold Evaluation (0.50)
    preds_05 = (val_preds >= 0.50).astype(int)
    p_05, r_05, f_05, _ = precision_recall_fscore_support(y_val, preds_05, average='binary')
    tn_05, fp_05, fn_05, tp_05 = confusion_matrix(y_val, preds_05).ravel()
    fpr_05 = fp_05 / (fp_05 + tn_05) if (fp_05 + tn_05) > 0 else 0.0

    # Optimal F1 Threshold Selection
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_preds)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    preds_opt = (val_preds >= best_threshold).astype(int)
    p_opt, r_opt, f_opt, _ = precision_recall_fscore_support(y_val, preds_opt, average='binary')
    tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_val, preds_opt).ravel()
    fpr_opt = fp_opt / (fp_opt + tn_opt) if (fp_opt + tn_opt) > 0 else 0.0

    # Calculate baseline metrics dynamically from the predictions file
    baseline_probs = baseline_df['fraud_probability'].values
    b_pr_auc = average_precision_score(y_val, baseline_probs)
    b_roc_auc = roc_auc_score(y_val, baseline_probs)
    
    b_preds_05 = (baseline_probs >= 0.50).astype(int)
    bp_05, br_05, bf_05, _ = precision_recall_fscore_support(y_val, b_preds_05, average='binary')
    btn_05, bfp_05, bfn_05, btp_05 = confusion_matrix(y_val, b_preds_05).ravel()
    bfpr_05 = bfp_05 / (bfp_05 + btn_05) if (bfp_05 + btn_05) > 0 else 0.0
    
    b_precisions, b_recalls, b_thresholds = precision_recall_curve(y_val, baseline_probs)
    b_f1_scores = np.divide(
        2 * b_precisions * b_recalls,
        b_precisions + b_recalls,
        out=np.zeros_like(b_precisions),
        where=(b_precisions + b_recalls) > 0
    )
    b_best_idx = np.argmax(b_f1_scores)
    b_best_threshold = b_thresholds[b_best_idx] if b_best_idx < len(b_thresholds) else 0.5
    b_best_f1 = b_f1_scores[b_best_idx]
    
    b_preds_opt = (baseline_probs >= b_best_threshold).astype(int)
    bp_opt, br_opt, bf_opt, _ = precision_recall_fscore_support(y_val, b_preds_opt, average='binary')
    btn_opt, bfp_opt, bfn_opt, btp_opt = confusion_matrix(y_val, b_preds_opt).ravel()
    bfpr_opt = bfp_opt / (bfp_opt + btn_opt) if (bfp_opt + btn_opt) > 0 else 0.0

    print("\nCROSS-MODEL METRICS COMPARISON (Baseline A vs Historical D):")
    print(f"  * PR-AUC (AP):    Baseline={b_pr_auc:.5f} -> Historical={pr_auc:.5f} (Delta={pr_auc - b_pr_auc:+.5f})")
    print(f"  * ROC-AUC:        Baseline={b_roc_auc:.5f} -> Historical={roc_auc:.5f} (Delta={roc_auc - b_roc_auc:+.5f})")
    print(f"  * Best F1-Score:  Baseline={b_best_f1:.5f} -> Historical={best_f1:.5f} (Delta={best_f1 - b_best_f1:+.5f})")

    # 7. Generate Comparative PR Curve Plot
    print("\n[6/7] Generating comparative precision-recall curve graph...")
    plt.figure(figsize=(10, 8))
    plt.plot(b_recalls, b_precisions, color='gray', linestyle='--', lw=2, label=f'Model A (Tx-Only) (PR-AUC = {b_pr_auc:.5f})')
    plt.plot(recalls, precisions, color='darkorange', linestyle='-', lw=2.5, label=f'Model D (Tx + Historical) (PR-AUC = {pr_auc:.5f})')
    plt.axvline(x=r_opt, color='blue', linestyle=':', alpha=0.8, label=f'Model D Opt F1 Recall = {r_opt:.3f}')
    plt.axhline(y=p_opt, color='blue', linestyle=':', alpha=0.8, label=f'Model D Opt F1 Precision = {p_opt:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparative Precision-Recall Curves (Baseline vs. Historical Features)')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative PR curve plot to: {pr_curve_path}")

    # Serialize Model & Predictions
    model.save_model(model_save_path)
    print(f"Saved model booster file to: {model_save_path}")

    predictions_df = pd.DataFrame({
        'TransactionID': df.iloc[split_idx:]['TransactionID'],
        'TransactionDT': df.iloc[split_idx:]['TransactionDT'],
        'isFraud': y_val,
        'fraud_probability': val_preds
    })
    predictions_df.to_parquet(preds_parquet_path, engine='pyarrow', index=False)
    print(f"Saved validation predictions to: {preds_parquet_path}")

    # 8. Write Markdown Performance Report
    print("\n[7/7] Writing Model D evaluation report...")
    
    # Feature Importance (top 20 by gain)
    importance_gain = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    feat_imp = pd.DataFrame({'feature': feature_names, 'gain_importance': importance_gain})
    feat_imp = feat_imp.sort_values('gain_importance', ascending=False).reset_index(drop=True)
    top_20_imp = feat_imp.head(20)

    report_content = f"""# Phase 5C: LightGBM Transaction + Historical Features Model Evaluation Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report compares **Model D (Transaction + Chronological expanding Historical Features)** directly against **Model A (Transaction-Only Baseline)** to measure the predictive impact of leakage-free frequencies and target encodings for key entities.

---

## 1. Metrics Comparison Table

| Metric | Model A (Transaction-Only) | Model D (Tx + Historical Features) | Absolute Delta (Δ) |
| :--- | :---: | :---: | :---: |
| **PR-AUC (Average Precision)** | `{b_pr_auc:.5f}` | `{pr_auc:.5f}` | `{pr_auc - b_pr_auc:+.5f}` |
| **ROC-AUC** | `{b_roc_auc:.5f}` | `{roc_auc:.5f}` | `{roc_auc - b_roc_auc:+.5f}` |
| **F1 @ 0.50 (Reference)** | `{bf_05:.5f}` | `{f_05:.5f}` | `{f_05 - bf_05:+.5f}` |
| **Precision @ 0.50** | `{bp_05:.5f}` | `{p_05:.5f}` | `{p_05 - bp_05:+.5f}` |
| **Recall @ 0.50** | `{br_05:.5f}` | `{r_05:.5f}` | `{r_05 - br_05:+.5f}` |
| **Optimal F1-Score** | `{b_best_f1:.5f}` | `{best_f1:.5f}` | `{best_f1 - b_best_f1:+.5f}` |
| **Optimal Threshold** | `{b_best_threshold:.4f}` | `{best_threshold:.4f}` | `{best_threshold - b_best_threshold:+.4f}` |
| **FPR @ Optimal Threshold** | `{bfpr_opt:.5f}` | `{fpr_opt:.5f}` | `{fpr_opt - bfpr_opt:+.5f}` |

---

## 2. Confusion Matrices Comparison

### Reference Threshold (0.50)
* **Model A (TN / FP / FN / TP)**: {btn_05:,} / {bfp_05:,} / {bfn_05:,} / {btp_05:,}
* **Model D (TN / FP / FN / TP)**: {tn_05:,} / {fp_05:,} / {fn_05:,} / {tp_05:,}

### Optimal Threshold
* **Model A (TN / FP / FN / TP @ {b_best_threshold:.4f})**: {btn_opt:,} / {bfp_opt:,} / {bfn_opt:,} / {btp_opt:,}
* **Model D (TN / FP / FN / TP @ {best_threshold:.4f})**: {tn_opt:,} / {fp_opt:,} / {fn_opt:,} / {tp_opt:,}

---

## 3. Top 20 Feature Importance (By Gain)

This inventory shows which attributes contributed the most gain to Model D. Note if any of our newly engineered 10 historical features rank in the top 20:

| Rank | Feature Name | Gain Importance | Description |
| :--- | :--- | :--- | :--- |
{chr(10).join([f"| {i+1} | `{row['feature']}` | {row['gain_importance']:.2f} | Model feature |" for i, row in top_20_imp.iterrows()])}

---

## 4. Artifact Directory Inventory

The following outputs have been serialized and saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Model Booster**: [`historical_lgb_model.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/historical_lgb_model.txt)
2. **Evaluation Metrics**: [`historical_evaluation_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/historical_evaluation_report.md)
3. **Comparative PR Curve**: [`historical_pr_curves.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/historical_pr_curves.png)
4. **Validation Predictions**: [`historical_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/historical_predictions.parquet)
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Saved historical evaluation report to: {report_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("HISTORICAL MODEL TRAINING AND EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    main()
