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
    print("         IEEE-CIS LIGHTGBM TRANSACTION-ONLY BASELINE MODEL")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    features_path = os.path.join(processed_dir, 'features/transaction_features.parquet')
    
    model_save_path = os.path.join(processed_dir, 'models/baseline_lgb_model.txt')
    preds_parquet_path = os.path.join(processed_dir, 'predictions/baseline_predictions.parquet')
    report_path = os.path.join(processed_dir, 'reports/baseline_evaluation_report.md')
    pr_curve_path = os.path.join(processed_dir, 'plots/baseline_pr_curve.png')

    # 1. Load Dataset
    print("\n[1/7] Loading transaction features...")
    if not os.path.exists(features_path):
        print(f"[ERROR] Features parquet file not found at: {features_path}")
        print("Please run transaction_features.py first.")
        sys.exit(1)
        
    start_time = time.time()
    df = pd.read_parquet(features_path)
    print(f"Loaded dataset: {df.shape} in {time.time() - start_time:.2f} seconds.")

    # 2. Chronological Sorting & Train/Val Splitting
    print("\n[2/7] Preparing chronological splits...")
    t0 = time.time()
    
    # Sort by TransactionDT (raw elapsed seconds) to enforce temporal causality
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    # Calculate 80/20 chronological split threshold
    total_rows = len(df)
    split_idx = int(total_rows * 0.8)
    split_dt = df['TransactionDT'].iloc[split_idx]
    
    # Split features matrix and labels
    X = df.drop(columns=['TransactionID', 'isFraud'])
    y = df['isFraud']
    
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    
    X_val = X.iloc[split_idx:].copy()
    y_val = y.iloc[split_idx:].copy()
    
    print(f"Split completed in {time.time() - t0:.2f} seconds.")
    print(f"  * Total Dataset size:       {total_rows:,} rows")
    print(f"  * Split boundary:           TransactionDT = {split_dt:,}")
    print(f"  * Training set (Earliest):  {len(X_train):,} rows ({len(X_train)/total_rows*100:.1f}%)")
    print(f"  * Validation set (Latest):  {len(X_val):,} rows ({len(X_val)/total_rows*100:.1f}%)")
    print(f"  * Training fraud cases:     {y_train.sum():,} ({y_train.mean()*100:.3f}% fraud rate)")
    print(f"  * Validation fraud cases:   {y_val.sum():,} ({y_val.mean()*100:.3f}% fraud rate)")

    # 3. LightGBM Dataset Construction
    print("\n[3/7] Setting up LightGBM datasets...")
    # Identify categorical columns (already pandas category type)
    categorical_cols = list(X_train.select_dtypes(include=['category']).columns)
    print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    # Create LightGBM dataset structures
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset, categorical_feature=categorical_cols)

    # 4. Model Training
    print("\n[4/7] Training LightGBM model (scale_pos_weight = 1.0)...")
    
    params = {
        'objective': 'binary',
        'metric': 'average_precision',  # Focus on PR-AUC (Average Precision)
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'scale_pos_weight': 1.0,         # Establish unweighted baseline
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

    # 5. Evaluate Validation Predictions
    print("\n[5/7] Evaluating predictions on the validation set...")
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Compute baseline metrics
    pr_auc = average_precision_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_preds)
    
    print(f"  * Primary Metric (PR-AUC):   {pr_auc:.5f}")
    print(f"  * Secondary Metric (ROC-AUC): {roc_auc:.5f}")

    # Reference Threshold Evaluation (0.50)
    preds_05 = (val_preds >= 0.50).astype(int)
    p_05, r_05, f_05, _ = precision_recall_fscore_support(y_val, preds_05, average='binary')
    tn_05, fp_05, fn_05, tp_05 = confusion_matrix(y_val, preds_05).ravel()
    fpr_05 = fp_05 / (fp_05 + tn_05) if (fp_05 + tn_05) > 0 else 0.0

    # Optimal F1 Threshold Selection
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_preds)
    # Avoid division by zero
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
    
    print(f"  * Reference Threshold (0.50): F1={f_05:.5f}, Precision={p_05:.5f}, Recall={r_05:.5f}, FPR={fpr_05:.5f}")
    print(f"  * Optimal Threshold ({best_threshold:.4f}): F1={best_f1:.5f}, Precision={p_opt:.5f}, Recall={r_opt:.5f}, FPR={fpr_opt:.5f}")

    # 6. Generate Artifacts & Plots
    print("\n[6/7] Generating metrics plots and serialization outputs...")
    
    # Save the LightGBM model booster text
    model.save_model(model_save_path)
    print(f"Saved model booster file to: {model_save_path}")

    # Save Predictions Parquet file
    predictions_df = pd.DataFrame({
        'TransactionID': df.iloc[split_idx:]['TransactionID'],
        'TransactionDT': df.iloc[split_idx:]['TransactionDT'],
        'isFraud': y_val,
        'fraud_probability': val_preds
    })
    predictions_df.to_parquet(preds_parquet_path, engine='pyarrow', index=False)
    print(f"Saved validation predictions to: {preds_parquet_path}")

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='darkorange', lw=2, label=f'LGBM Baseline (PR-AUC = {pr_auc:.5f})')
    plt.axvline(x=r_opt, color='blue', linestyle='--', label=f'Optimal F1 Recall = {r_opt:.3f}')
    plt.axhline(y=p_opt, color='blue', linestyle='--', label=f'Optimal F1 Precision = {p_opt:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Transaction-Only Baseline')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Precision-Recall Curve plot to: {pr_curve_path}")

    # 7. Write Markdown Evaluation Report
    print("\n[7/7] Writing baseline evaluation report...")
    
    # Feature Importance (top 20 by gain)
    importance_gain = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    feat_imp = pd.DataFrame({'feature': feature_names, 'gain_importance': importance_gain})
    feat_imp = feat_imp.sort_values('gain_importance', ascending=False).reset_index(drop=True)
    top_20_imp = feat_imp.head(20)

    report_content = f"""# Phase 2: LightGBM Transaction-Only Baseline Evaluation Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report documents the training parameters and evaluation performance of the **Transaction-Only Baseline Model** (Phase 2). This baseline establishes the detection capacity using only features derived from the transaction transaction-details, serving as a benchmark for subsequent behavioral and graph feature iterations.

---

## 1. Split & Configuration Parameters

* **Temporal Split Strategy**: Time-aware chronological partition on sorted `TransactionDT`.
* **Dataset Boundary**:
  * **Train Set (Earliest 80%)**: {len(X_train):,} rows (TransactionDT < {split_dt:,})
  * **Validation Set (Latest 20%)**: {len(X_val):,} rows (TransactionDT >= {split_dt:,})
* **Hyperparameters**:
  * `scale_pos_weight = 1.0` (Unweighted baseline setup)
  * `learning_rate = 0.05`
  * `num_leaves = 31`
  * `boosting_type = gdbt`
  * Categorical variables: Handled natively by LightGBM using fisher-splits.

---

## 2. Model Performance Summary

| Metric | Score | Description |
| :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | **{pr_auc:.5f}** | Primary performance score (imbalance-robust) |
| **ROC-AUC** | **{roc_auc:.5f}** | Overall classification performance |

### Classification Decision Trade-offs:

| Threshold | F1-Score | Precision | Recall | False Positive Rate (FPR) |
| :--- | :--- | :--- | :--- | :--- |
| **Reference (0.50)** | {f_05:.5f} | {p_05:.5f} | {r_05:.5f} | {fpr_05:.5f} |
| **Optimal F1 ({best_threshold:.4f})** | {best_f1:.5f} | {p_opt:.5f} | {r_opt:.5f} | {fpr_opt:.5f} |

---

## 3. Confusion Matrices

### Reference Threshold (0.50)
* **True Negatives (TN)**: {tn_05:,}
* **False Positives (FP)**: {fp_05:,}
* **False Negatives (FN)**: {fn_05:,}
* **True Positives (TP)**: {tp_05:,}

### Optimal F1 Threshold ({best_threshold:.4f})
* **True Negatives (TN)**: {tn_opt:,}
* **False Positives (FP)**: {fp_opt:,}
* **False Negatives (FN)**: {fn_opt:,}
* **True Positives (TP)**: {tp_opt:,}

---

## 4. Top 20 Feature Importance (By Gain)

| Rank | Feature Name | Gain Importance | Description |
| :--- | :--- | :--- | :--- |
{chr(10).join([f"| {i+1} | `{row['feature']}` | {row['gain_importance']:.2f} | Transaction-level metric |" for i, row in top_20_imp.iterrows()])}

---

## 5. Artifact Directory Inventory

The following outputs have been serialized and saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Model Booster**: [`baseline_lgb_model.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/baseline_lgb_model.txt)
2. **Evaluation Metrics**: [`baseline_evaluation_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/baseline_evaluation_report.md)
3. **PR Curve Plot**: [`baseline_pr_curve.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/baseline_pr_curve.png)
4. **Validation Predictions**: [`baseline_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/baseline_predictions.parquet)
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Saved baseline evaluation report to: {report_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("BASELINE MODEL TRAINING AND EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    main()
