import os
import sys
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    confusion_matrix
)

def evaluate_predictions(y_true, y_prob):
    """
    Computes standard evaluation metrics (PR-AUC, ROC-AUC, optimal F1, FPR)
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
        'fpr': fpr_opt
    }

def train_eval_config(config_name, X_train, y_train, X_val, y_val, categorical_cols):
    """
    Trains a LightGBM booster with standard parameters and evaluates it.
    """
    print(f"\nTraining config {config_name} with {X_train.shape[1]} features...")
    
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset, categorical_feature=categorical_cols)
    
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
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0) # suppress outputs
        ]
    )
    elapsed = time.time() - t0
    
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    metrics = evaluate_predictions(y_val, val_preds)
    
    metrics['time'] = elapsed
    metrics['best_iter'] = model.best_iteration
    metrics['num_features'] = X_train.shape[1]
    
    print(f"  * Finished in {elapsed:.2f}s | Best Iteration = {model.best_iteration}")
    print(f"  * PR-AUC = {metrics['pr_auc']:.5f} | Best F1 = {metrics['best_f1']:.5f}")
    
    return metrics

def main():
    print("=" * 70)
    print("         IEEE-CIS BEHAVIORAL DEVIATION ABLATION STUDY SUITE")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/deviation_features.parquet')
    report_path = os.path.join(processed_dir, 'reports/ablation_study_report.md')

    if not os.path.exists(features_path):
        print(f"[ERROR] Deviation features parquet file not found at: {features_path}")
        sys.exit(1)
        
    start_time = time.time()
    df = pd.read_parquet(features_path)
    print(f"Loaded dataset: {df.shape} in {time.time() - start_time:.2f} seconds.")

    # Sort chronologically by TransactionDT
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)
    split_idx = int(total_rows * 0.8)

    # Base features: 393 Transaction + 10 Historical = 403 features
    # Ensure intermediate/diversity/deviation columns are excluded from base
    core_deviations = [
        'amount_vs_card_mean', 'amount_vs_card_median', 'amount_zscore',
        'tx_frequency_deviation_24h', 'tx_frequency_deviation_1h',
        'time_gap_deviation_median', 'time_gap_deviation_mean',
        'spend_velocity_deviation_1h', 'spend_velocity_deviation_24h',
        'card_device_frequency', 'card_location_frequency'
    ]
    diversity_metrics = ['card_device_diversity', 'card_location_diversity']
    metadata_cols = ['TransactionID', 'isFraud']
    
    # Identify base features
    all_features = list(df.drop(columns=metadata_cols).columns)
    base_features = [f for f in all_features if f not in core_deviations and f not in diversity_metrics]
    
    print(f"Identified {len(base_features)} baseline features (393 Tx + 10 Hist).")

    # Define ablation setups
    # (Config Name, Added Feature Lists)
    setups = [
        ("Model D (Baseline)", []),
        ("Model G1 (Amount Anomaly)", ['amount_vs_card_mean', 'amount_vs_card_median', 'amount_zscore']),
        ("Model G2 (Frequency Anomaly)", ['tx_frequency_deviation_24h', 'tx_frequency_deviation_1h']),
        ("Model G3 (Temporal Anomaly)", ['time_gap_deviation_median', 'time_gap_deviation_mean']),
        ("Model G4 (Spending Velocity Anomaly)", ['spend_velocity_deviation_1h', 'spend_velocity_deviation_24h']),
        ("Model G5 (Entity Association Anomaly)", ['card_device_frequency', 'card_location_frequency']),
        ("Model G6 (All 11 Deviations)", core_deviations),
        ("Model G7 (All 11 + Diversity)", core_deviations + diversity_metrics)
    ]

    y = df['isFraud']
    y_train = y.iloc[:split_idx].copy()
    y_val = y.iloc[split_idx:].copy()

    results = []

    for name, added_cols in setups:
        # Construct feature subset
        cols_to_use = base_features + added_cols
        X_train = df.iloc[:split_idx][cols_to_use].copy()
        X_val = df.iloc[split_idx:][cols_to_use].copy()
        
        # Identify categorical columns in the current subset
        categorical_cols = list(X_train.select_dtypes(include=['category']).columns)
        
        metrics = train_eval_config(name, X_train, y_train, X_val, y_val, categorical_cols)
        metrics['name'] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    
    # Sort leaderboard by PR-AUC descending
    results_df = results_df.sort_values('pr_auc', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("FINAL COMPARATIVE leaderboard:")
    print("=" * 70)
    print(results_df[['name', 'num_features', 'pr_auc', 'roc_auc', 'best_f1', 'best_threshold', 'fpr', 'best_iter']].to_string(index=False))
    print("=" * 70)

    # Save Markdown Report
    print(f"\nSaving ablation report to: {report_path}")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    leaderboard_rows = []
    for idx, row in results_df.iterrows():
        # Highlight if it beats Model D
        is_succ = "🏆" if (row['pr_auc'] > 0.58144 and row['name'] != "Model D (Baseline)") else ""
        leaderboard_rows.append(
            f"| {idx+1} | {is_succ} **{row['name']}** | `{row['num_features']}` | `{row['pr_auc']:.5f}` | `{row['roc_auc']:.5f}` | `{row['best_f1']:.5f}` | `{row['best_threshold']:.4f}` | `{row['fpr']:.5f}` | `{row['best_iter']}` | `{row['time']:.1f}s` |"
        )

    report_content = f"""# Phase 6: Behavioral Deviation Feature Ablation Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This study systematically isolates and measures the performance impact of each anomaly category when added individually and collectively to the **Model D (Transaction + Historical)** baseline features.

---

## 1. Comparative Leaderboard

| Rank | Model Configuration | Features | PR-AUC (Primary) | ROC-AUC | Optimal F1 | Optimal Threshold | FPR @ Optimal | Best Iter | Train Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(leaderboard_rows)}

---

## 2. Key Scientific Findings & Analysis

1. **Ablation Performance Contribution**:
   * Inspect the ranks above. The category that provides the largest PR-AUC improvement is the primary driver of behavioral contextualization.
   * If **Model G6 (All 11 Deviations)** outperforms the individual models, the anomaly categories act in a complementary manner.
   
2. **Contextualization vs. Standalone Frequencies**:
   * Earlier experiments combining raw rolling behavioral features failed due to noise. 
   * If Model G6 beats Model D (`0.58144`), it proves that **entity-relative deviation metrics (contextualized anomaly scores)** bypass high-frequency rolling noise and successfully enrich chronological Bayes-smoothed historical models.

3. **Diversity Metric Evaluation (Model G7)**:
   * Model G7 measures the marginal benefit of adding non-deviation contextual context (`card_device_diversity`, `card_location_diversity`). Check if G7 outperforms G6 to decide if diversity should be retained in the champion pipeline.

---

## 3. Serialization Log
All models trained inside this ablation study run with identical seed parameters (`random_state=42`) and an 80/20 chronological train/validation split.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Ablation study completed!")

if __name__ == '__main__':
    main()
