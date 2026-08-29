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

def evaluate_predictions(y_true, y_prob):
    """
    Computes standard evaluation metrics (PR-AUC, ROC-AUC, optimal F1, Precision, Recall, FPR)
    """
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
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
    print("         IEEE-CIS LIGHTGBM BEHAVIORAL DEVIATION MODEL (G)")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/deviation_features.parquet')
    baseline_preds_path = os.path.join(processed_dir, 'predictions/baseline_predictions.parquet')
    model_d_preds_path = os.path.join(processed_dir, 'predictions/historical_predictions.parquet')
    
    model_save_path = os.path.join(processed_dir, 'models/deviation_lgb_model.txt')
    preds_parquet_path = os.path.join(processed_dir, 'predictions/deviation_predictions.parquet')
    report_path = os.path.join(processed_dir, 'reports/deviation_evaluation_report.md')
    pr_curve_path = os.path.join(processed_dir, 'plots/deviation_pr_curves.png')

    # 1. Load Datasets
    print("\n[1/7] Loading deviation features and baseline predictions...")
    if not os.path.exists(features_path):
        print(f"[ERROR] Deviation features parquet file not found at: {features_path}")
        sys.exit(1)
    if not os.path.exists(baseline_preds_path):
        print(f"[ERROR] Baseline predictions file not found at: {baseline_preds_path}")
        sys.exit(1)
        
    start_time = time.time()
    df = pd.read_parquet(features_path)
    baseline_df = pd.read_parquet(baseline_preds_path)
    
    model_d_available = os.path.exists(model_d_preds_path)
    if model_d_available:
        model_d_df = pd.read_parquet(model_d_preds_path)
        print("  * Loaded Model D (LightGBM Historical) predictions for comparison.")
    else:
        print("  * [WARNING] Model D predictions file not found. Static champion score (0.58144) will be referenced.")

    print(f"Loaded datasets in {time.time() - start_time:.2f} seconds.")
    print(f"  * Full dataset shape: {df.shape}")

    # 2. Chronological Sorting & Train/Val Splitting
    print("\n[2/7] Preparing chronological splits...")
    t0 = time.time()
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    total_rows = len(df)
    split_idx = int(total_rows * 0.8)
    
    # Verify ID Alignment
    print("Checking alignment with baseline validation set...")
    val_ids = df.iloc[split_idx:]['TransactionID'].values
    baseline_ids = baseline_df['TransactionID'].values
    if not np.array_equal(val_ids, baseline_ids):
        raise ValueError("Validation set TransactionIDs do not align with baseline predictions!")
    print("✅ Validation TransactionIDs align perfectly with baseline validation set.")

    # Group features matrix and labels
    # Model G: 393 Transaction + 10 Historical + 11 Deviation = 414 features (Excludes diversity metrics)
    cols_to_drop = ['TransactionID', 'isFraud', 'card_device_diversity', 'card_location_diversity']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['isFraud']
    
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    
    X_val = X.iloc[split_idx:].copy()
    y_val = y.iloc[split_idx:].copy()
    
    print(f"Split completed in {time.time() - t0:.2f} seconds.")
    print(f"  * Features Count:           {len(X_train.columns)} columns (Expected 414)")
    print(f"  * Training set:             {len(X_train):,} rows")
    print(f"  * Validation set:           {len(X_val):,} rows")

    # 3. LightGBM Dataset Construction
    print("\n[3/7] Setting up LightGBM datasets...")
    categorical_cols = list(X_train.select_dtypes(include=['category']).columns)
    
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset, categorical_feature=categorical_cols)

    # 4. Model Training (Exact Same Parameters as Model D)
    print("\n[4/7] Training LightGBM Model G with 414 features...")
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
        'scale_pos_weight': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    t0 = time.time()
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
    print(f"Model G training completed in {time.time() - t0:.2f} seconds.")
    print(f"Best iteration: {model.best_iteration}")

    # 5. Evaluate Validation Predictions
    print("\n[5/7] Evaluating predictions on the validation set...")
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    metrics_g = evaluate_predictions(y_val, val_preds)
    
    # Calculate baseline metrics dynamically from the predictions file
    baseline_probs = baseline_df['fraud_probability'].values
    metrics_a = evaluate_predictions(y_val, baseline_probs)
    
    # Calculate Model D metrics dynamically if predictions exist, otherwise use static fallback
    if model_d_available:
        model_d_probs = model_d_df['fraud_probability'].values
        metrics_d = evaluate_predictions(y_val, model_d_probs)
    else:
        metrics_d = {
            'pr_auc': 0.58144,
            'roc_auc': 0.90507,
            'best_f1': 0.58178,
            'best_threshold': 0.2858, # historical threshold
            'precision': 0.5786,
            'recall': 0.5850,
            'fpr': 0.00880,
            'recalls_curve': np.array([0, 1]),
            'precisions_curve': np.array([1, 0])
        }

    print("\nCOMPARATIVE LEADERBOARD SUMMARY:")
    print(f"  * Model A (Tx Only):       PR-AUC = {metrics_a['pr_auc']:.5f} | Best F1 = {metrics_a['best_f1']:.5f}")
    print(f"  * Model D (Tx + Hist):     PR-AUC = {metrics_d['pr_auc']:.5f} | Best F1 = {metrics_d['best_f1']:.5f}")
    print(f"  * Model G (Tx + Hist + D): PR-AUC = {metrics_g['pr_auc']:.5f} | Best F1 = {metrics_g['best_f1']:.5f}")
    
    delta_d = metrics_g['pr_auc'] - metrics_d['pr_auc']
    print(f"  * Delta relative to Champion (Model D): {delta_d:+.5f} PR-AUC")
    if delta_d > 0:
        print("🏆 SUCCESS! Model G exceeds Model D performance and becomes the NEW pipeline champion!")
    else:
        print("⚠️ Model G did not exceed Model D. Model D remains the pipeline champion.")

    # 6. Generate Comparative PR Curve Plot
    print("\n[6/7] Generating comparative precision-recall curve graph...")
    plt.figure(figsize=(10, 8))
    plt.plot(metrics_a['recalls_curve'], metrics_a['precisions_curve'], color='gray', linestyle='--', lw=2, label=f"Model A (Tx-Only) (PR-AUC = {metrics_a['pr_auc']:.5f})")
    plt.plot(metrics_d['recalls_curve'], metrics_d['precisions_curve'], color='blue', linestyle=':', lw=2, label=f"Model D (Tx + Historical) (PR-AUC = {metrics_d['pr_auc']:.5f})")
    plt.plot(metrics_g['recalls_curve'], metrics_g['precisions_curve'], color='darkorange', linestyle='-', lw=2.5, label=f"Model G (Tx + Hist + Dev) (PR-AUC = {metrics_g['pr_auc']:.5f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparative Precision-Recall Curves (Model G Experiment)')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(os.path.dirname(pr_curve_path), exist_ok=True)
    plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparative PR curve plot to: {pr_curve_path}")

    # Serialize Predictions & Model
    os.makedirs(os.path.dirname(preds_parquet_path), exist_ok=True)
    predictions_df = pd.DataFrame({
        'TransactionID': df.iloc[split_idx:]['TransactionID'],
        'TransactionDT': df.iloc[split_idx:]['TransactionDT'],
        'isFraud': y_val,
        'fraud_probability': val_preds
    })
    predictions_df.to_parquet(preds_parquet_path, engine='pyarrow', index=False)
    print(f"Saved validation predictions to: {preds_parquet_path}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path)
    print(f"Saved Model G booster to: {model_save_path}")

    # 7. Write Evaluation Report
    print("\n[7/7] Writing Model G evaluation report...")
    
    # Feature Importance (top 20 by gain)
    importance_gain = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    feat_imp = pd.DataFrame({'feature': feature_names, 'gain_importance': importance_gain})
    feat_imp = feat_imp.sort_values('gain_importance', ascending=False).reset_index(drop=True)
    top_20_imp = feat_imp.head(20)

    report_content = f"""# Phase 6: Model G (Behavioral Deviation Features) Evaluation Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report evaluates **Model G (LightGBM Transaction + Historical + Core 11 Deviation features)** against the baseline Model A and the current champion Model D.

---

## 1. Metrics Comparison

| Metric | Model A (Tx-Only) | Model D (Tx + Hist) | Model G (Tx + Hist + Dev) | Delta (G vs D) |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Primary)** | `{metrics_a['pr_auc']:.5f}` | `{metrics_d['pr_auc']:.5f}` | `{metrics_g['pr_auc']:.5f}` | `{metrics_g['pr_auc'] - metrics_d['pr_auc']:+.5f}` |
| **ROC-AUC** | `{metrics_a['roc_auc']:.5f}` | `{metrics_d['roc_auc']:.5f}` | `{metrics_g['roc_auc']:.5f}` | `{metrics_g['roc_auc'] - metrics_d['roc_auc']:+.5f}` |
| **Optimal F1-Score** | `{metrics_a['best_f1']:.5f}` | `{metrics_d['best_f1']:.5f}` | `{metrics_g['best_f1']:.5f}` | `{metrics_g['best_f1'] - metrics_d['best_f1']:+.5f}` |
| **Optimal Threshold** | `{metrics_a['best_threshold']:.4f}` | `{metrics_d['best_threshold']:.4f}` | `{metrics_g['best_threshold']:.4f}` | `{metrics_g['best_threshold'] - metrics_d['best_threshold']:+.4f}` |
| **FPR @ Optimal** | `{metrics_a['fpr']:.5f}` | `{metrics_d['fpr']:.5f}` | `{metrics_g['fpr']:.5f}` | `{metrics_g['fpr'] - metrics_d['fpr']:+.5f}` |

---

## 2. Champion Verdict

* **Success Status**: {"🏆 SUCCESS - Model G is the new pipeline champion!" if delta_d > 0 else "⚠️ Model D remains the pipeline champion."}
* **Scientific Insights**: Behavioral deviations relative to each entity's own history {"do indeed contribute complementary predictive signal to chronological risk encodings." if delta_d > 0 else "did not exceed the performance of the Bayes-smoothed chronological fraud-rate baseline in this configuration."}

---

## 3. Top 20 Feature Importance (By Gain)

The top predictors in Model G by gain importance:

| Rank | Feature Name | Gain Importance | Description |
| :--- | :--- | :--- | :--- |
{chr(10).join([f"| {i+1} | `{row['feature']}` | {row['gain_importance']:.2f} | Model Feature |" for i, row in top_20_imp.iterrows()])}

---

## 4. File Assets

The following artifacts have been created:
1. **Model G Booster**: [`deviation_lgb_model.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/models/deviation_lgb_model.txt)
2. **Comparative PR Curve**: [`deviation_pr_curves.png`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/plots/deviation_pr_curves.png)
3. **Validation Predictions**: [`deviation_predictions.parquet`](file:///c:/Users/BIT/Downloads/Razorpay_Build/dataset/data/processed/predictions/deviation_predictions.parquet)
"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Saved Model G evaluation report to: {report_path}")

    print("\n" + "=" * 70)
    print("MODEL G TRAINING AND EVALUATION COMPLETED")
    print("=" * 70)

if __name__ == '__main__':
    main()
