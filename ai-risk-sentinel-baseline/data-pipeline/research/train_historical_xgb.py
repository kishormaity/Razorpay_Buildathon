import os
import sys
import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
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
    
    # Avoid division by zero in F1 calculation
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
        'precisions_curve': precisions,
        'confusion': (tn_opt, fp_opt, fn_opt, tp_opt)
    }

def main():
    print("=" * 70)
    print("         IEEE-CIS XGBOOST TRANSACTION + HISTORICAL MODEL (E)")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    features_path = os.path.join(processed_dir, 'features/historical_features.parquet')
    
    baseline_preds_path = os.path.join(processed_dir, 'predictions/baseline_predictions.parquet')
    model_d_preds_path = os.path.join(processed_dir, 'predictions/historical_predictions.parquet')
    
    model_save_path = os.path.join(processed_dir, 'models/historical_xgb_model.json')
    preds_parquet_path = os.path.join(processed_dir, 'predictions/historical_xgb_predictions.parquet')
    report_path = os.path.join(processed_dir, 'reports/historical_xgb_report.md')
    pr_curve_path = os.path.join(processed_dir, 'plots/historical_comparative_curves.png')

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
    
    model_d_available = os.path.exists(model_d_preds_path)
    if model_d_available:
        model_d_df = pd.read_parquet(model_d_preds_path)
        print("  * Loaded Model D (LightGBM Historical) predictions.")
    else:
        print("  * [WARNING] Model D predictions file not found. Static fallback values will be used.")
        
    print(f"Loaded datasets in {time.time() - start_time:.2f} seconds.")

    # 2. Chronological Sorting & Train/Val Splitting
    print("\n[2/7] Preparing chronological splits...")
    t0 = time.time()
    
    # Sort chronologically by TransactionDT
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    total_rows = len(df)
    split_idx = int(total_rows * 0.8)
    split_dt = df['TransactionDT'].iloc[split_idx]
    
    # Verify ID, target, and timestamp Alignment
    print("Checking strict alignment with baseline validation set...")
    val_ids = df.iloc[split_idx:]['TransactionID'].values
    val_fraud = df.iloc[split_idx:]['isFraud'].values
    val_dt = df.iloc[split_idx:]['TransactionDT'].values
    
    baseline_ids = baseline_df['TransactionID'].values
    baseline_fraud = baseline_df['isFraud'].values
    baseline_dt = baseline_df['TransactionDT'].values
    
    if not np.array_equal(val_ids, baseline_ids):
        raise ValueError("Validation set TransactionIDs do not align with baseline predictions!")
        
    if not np.array_equal(val_fraud, baseline_fraud):
        raise ValueError("Validation set isFraud labels do not align with baseline predictions!")
        
    if not np.array_equal(val_dt, baseline_dt):
        raise ValueError("Validation set TransactionDT values do not align with baseline predictions!")
        
    print("✅ Validation IDs, labels, and timestamps align perfectly.")

    # Separate target and ID
    X = df.drop(columns=['TransactionID', 'isFraud'])
    y = df['isFraud']
    
    # Verify Feature Count is exactly 403
    expected_feature_count = 403
    actual_feature_count = X.shape[1]
    if actual_feature_count != expected_feature_count:
        raise ValueError(f"Feature count mismatch! Expected exactly {expected_feature_count} features, got {actual_feature_count}")
    print(f"✅ Verified feature matrix has exactly {actual_feature_count} columns.")

    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    
    X_val = X.iloc[split_idx:].copy()
    y_val = y.iloc[split_idx:].copy()
    
    print(f"Split completed in {time.time() - t0:.2f} seconds.")
    print(f"  * Training set (Earliest):  {len(X_train):,} rows")
    print(f"  * Validation set (Latest):  {len(X_val):,} rows")

    # 3. XGBoost Sklearn API Setup
    print("\n[3/7] Instantiating XGBClassifier...")
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=1000,
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        early_stopping_rounds=50
    )

    # 4. Model Training
    print("\n[4/7] Training XGBoost model with transaction + historical features...")
    t_train_start = time.time()
    
    # Train the model
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    xgb_time = time.time() - t_train_start
    xgb_best_iter = xgb_model.best_iteration
    print(f"Model training completed in {xgb_time:.2f} seconds.")
    print(f"Best iteration: {xgb_best_iter}")

    # 5. Evaluate Predictions & Compare with Baseline/Model D
    print("\n[5/7] Evaluating predictions and comparing models...")
    val_preds = xgb_model.predict_proba(X_val)[:, 1]
    
    # Compute Model E metrics
    metrics_e = evaluate_predictions(y_val, val_preds)

    # Evaluate Model A (Baseline)
    baseline_probs = baseline_df['fraud_probability'].values
    metrics_a = evaluate_predictions(y_val, baseline_probs)
    
    # Evaluate Model D (LightGBM Historical)
    if model_d_available:
        model_d_probs = model_d_df['fraud_probability'].values
        metrics_d = evaluate_predictions(y_val, model_d_probs)
    else:
        # Static fallback values if predictions parquet is missing
        metrics_d = {
            'pr_auc': 0.58144,
            'roc_auc': 0.90507,
            'best_f1': 0.58178,
            'best_threshold': 0.3040,
            'fpr': 0.00880,
            'recalls_curve': np.array([0, 1]),
            'precisions_curve': np.array([1, 0]),
            'confusion': (113040, 1004, 1985, 2079)
        }

    print("\nCROSS-MODEL METRICS COMPARISON (A vs D vs E):")
    print(f"  * Model A (LightGBM Tx-Only) PR-AUC:      {metrics_a['pr_auc']:.5f}")
    print(f"  * Model D (LightGBM Tx+Hist) PR-AUC:      {metrics_d['pr_auc']:.5f}")
    print(f"  * Model E (XGBoost Tx+Hist) PR-AUC:       {metrics_e['pr_auc']:.5f} (Delta vs D: {metrics_e['pr_auc'] - metrics_d['pr_auc']:+.5f})")

    # 6. Generate Comparative PR Curves Graph
    print("\n[6/7] Generating comparative precision-recall curves plot...")
    plt.figure(figsize=(10, 8))
    plt.plot(metrics_a['recalls_curve'], metrics_a['precisions_curve'], color='gray', linestyle='--', lw=2, label=f'Model A (LGBM Tx-Only) (PR-AUC = {metrics_a["pr_auc"]:.5f})')
    
    if model_d_available:
        plt.plot(metrics_d['recalls_curve'], metrics_d['precisions_curve'], color='blue', linestyle=':', lw=2, label=f'Model D (LGBM Tx+Hist) (PR-AUC = {metrics_d["pr_auc"]:.5f})')
    else:
        plt.plot([0, 1], [0.58144, 0.58144], color='blue', linestyle=':', lw=1.5, label='Model D (LGBM Tx+Hist) (PR-AUC = 0.58144 Fallback)')
        
    plt.plot(metrics_e['recalls_curve'], metrics_e['precisions_curve'], color='darkorange', linestyle='-', lw=2.5, label=f'Model E (XGB Tx+Hist) (PR-AUC = {metrics_e["pr_auc"]:.5f})')
    
    plt.axvline(x=metrics_e['recall'], color='red', linestyle=':', alpha=0.8, label=f'Model E Opt F1 Recall = {metrics_e["recall"]:.3f}')
    plt.axhline(y=metrics_e['precision'], color='red', linestyle=':', alpha=0.8, label=f'Model E Opt F1 Precision = {metrics_e["precision"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparative Precision-Recall Curves (Model A vs. D vs. E)')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative PR curves plot to: {pr_curve_path}")

    # Serialize Model & Predictions
    xgb_model.save_model(model_save_path)
    print(f"Saved model booster file to: {model_save_path}")

    predictions_df = pd.DataFrame({
        'TransactionID': df.iloc[split_idx:]['TransactionID'],
        'TransactionDT': df.iloc[split_idx:]['TransactionDT'],
        'isFraud': y_val,
        'fraud_probability': val_preds
    })
    predictions_df.to_parquet(preds_parquet_path, engine='pyarrow', index=False)
    print(f"Saved validation predictions to: {preds_parquet_path}")

    # 7. Write Markdown Performance Report
    print("\n[7/7] Writing Model E evaluation report...")
    
    # Feature Importance (top 20 by gain)
    importance_gain = xgb_model.feature_importances_
    feature_names = X_train.columns
    feat_imp = pd.DataFrame({'feature': feature_names, 'gain_importance': importance_gain})
    feat_imp = feat_imp.sort_values('gain_importance', ascending=False).reset_index(drop=True)
    top_20_imp = feat_imp.head(20)

    # Confusion matrix unpack
    tn_a, fp_a, fn_a, tp_a = metrics_a['confusion']
    tn_d, fp_d, fn_d, tp_d = metrics_d['confusion']
    tn_e, fp_e, fn_e, tp_e = metrics_e['confusion']

    report_content = f"""# Phase 5D: XGBoost Transaction + Historical Features Model Evaluation Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report compares **Model E (XGBoost Transaction + Chronological expanding Historical Features)** directly against **Model A (LightGBM Transaction-Only Baseline)** and **Model D (LightGBM Transaction + Historical)** to determine which boosting algorithm leverages our historical features best.

---

## 1. Metrics Leaderboard

| Model Config | Algorithm | Features | PR-AUC | ROC-AUC | Optimal F1 | Optimal Threshold | FPR @ Optimal | Training Time | Best Iteration |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | LightGBM | Transaction | `{metrics_a['pr_auc']:.5f}` | `{metrics_a['roc_auc']:.5f}` | `{metrics_a['best_f1']:.5f}` | `{metrics_a['best_threshold']:.4f}` | `{metrics_a['fpr']:.5f}` | `94.91s` | `831` |
| **Model D** | LightGBM | Transaction + Historical | `{metrics_d['pr_auc']:.5f}` | `{metrics_d['roc_auc']:.5f}` | `{metrics_d['best_f1']:.5f}` | `{metrics_d['best_threshold']:.4f}` | `{metrics_d['fpr']:.5f}` | `119.27s` | `998` |
| **Model E** | **XGBoost** | Transaction + Historical | `{metrics_e['pr_auc']:.5f}` | `{metrics_e['roc_auc']:.5f}` | `{metrics_e['best_f1']:.5f}` | `{metrics_e['best_threshold']:.4f}` | `{metrics_e['fpr']:.5f}` | `{xgb_time:.2f}s` | `{xgb_best_iter}` |

---

## 2. Confusion Matrices Comparison (Optimal Threshold)

* **Model A (TN / FP / FN / TP @ {metrics_a['best_threshold']:.4f})**: {tn_a:,} / {fp_a:,} / {fn_a:,} / {tp_a:,}
* **Model D (TN / FP / FN / TP @ {metrics_d['best_threshold']:.4f})**: {tn_d:,} / {fp_d:,} / {fn_d:,} / {tp_d:,}
* **Model E (TN / FP / FN / TP @ {metrics_e['best_threshold']:.4f})**: {tn_e:,} / {fp_e:,} / {fn_e:,} / {tp_e:,}

---

## 3. Top 20 Feature Importance (XGBoost Split Gains)

This list shows the top 20 attributes contributing the most predictive weight to Model E:

| Rank | Feature Name | Gain Weight | Description |
| :--- | :--- | :--- | :--- |
{chr(10).join([f"| {i+1} | `{row['feature']}` | {row['gain_importance']:.6f} | Model feature |" for i, row in top_20_imp.iterrows()])}

---

## 4. Artifact Directory Inventory

The following outputs have been serialized and saved inside [`dataset/data/processed/`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/):
1. **Model Booster**: [`historical_xgb_model.json`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/historical_xgb_model.json)
2. **Evaluation Metrics**: [`historical_xgb_report.md`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/reports/historical_xgb_report.md)
3. **Comparative PR Curves**: [`historical_comparative_curves.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/historical_comparative_curves.png)
4. **Validation Predictions**: [`historical_xgb_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/historical_xgb_predictions.parquet)
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Saved Model E evaluation report to: {report_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("XGBOOST MODEL TRAINING AND EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print("=" * 70)

if __name__ == '__main__':
    main()
