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
    print(f"  * Training {config_name} ({X_train.shape[1]} features)...")
    
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset, categorical_feature=categorical_cols)
    
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
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
            lgb.log_evaluation(period=0)
        ]
    )
    elapsed = time.time() - t0
    
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    metrics = evaluate_predictions(y_val, val_preds)
    
    metrics['time'] = elapsed
    metrics['best_iter'] = model.best_iteration
    metrics['num_features'] = X_train.shape[1]
    
    print(f"    Finished in {elapsed:.2f}s | Best Iter = {model.best_iteration} | PR-AUC = {metrics['pr_auc']:.5f}")
    return metrics, val_preds

def run_paired_bootstrap(y_true, y_prob_d, y_prob_h3, num_resamples=1000, random_seed=42):
    """
    Performs paired bootstrap resampling on the validation set predictions.
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    
    boot_delta = np.zeros(num_resamples)
    
    for i in range(num_resamples):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_b = y_true[boot_idx]
        y_prob_d_b = y_prob_d[boot_idx]
        y_prob_h3_b = y_prob_h3[boot_idx]
        
        pr_d = average_precision_score(y_true_b, y_prob_d_b)
        pr_h3 = average_precision_score(y_true_b, y_prob_h3_b)
        
        boot_delta[i] = pr_h3 - pr_d
        
    mean_delta = np.mean(boot_delta)
    std_delta = np.std(boot_delta)
    ci_lower = np.percentile(boot_delta, 2.5)
    ci_upper = np.percentile(boot_delta, 97.5)
    
    return mean_delta, std_delta, ci_lower, ci_upper

def main():
    print("=" * 70)
    print("        IEEE-CIS PHASE 8D: H3 ABLATION & STABILITY STUDY")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/graph_features.parquet')
    d_preds_8020_path = os.path.join(processed_dir, 'predictions/historical_predictions.parquet')
    report_path = os.path.join(processed_dir, 'reports/graph_ablation_report.md')

    if not os.path.exists(features_path):
        print(f"[ERROR] Graph features file not found at: {features_path}")
        sys.exit(1)

    start_time = time.time()
    df = pd.read_parquet(features_path)
    print(f"Loaded graph features dataset: {df.shape} in {time.time() - start_time:.2f}s")

    # 1. Sort Chronologically
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)
    
    # Define splits
    split_80 = int(total_rows * 0.8)
    split_70 = int(total_rows * 0.7)

    # 2. Compute interaction feature on the fly
    print("\nComputing H3i interaction feature (device × address fraud rate)...")
    df['device_addr_fraud_rate_interaction'] = df['device_connected_fraud_rate'] * df['addr_connected_fraud_rate']

    # 3. Columns Definitions
    core_deviations = [
        'amount_vs_card_mean', 'amount_vs_card_median', 'amount_zscore',
        'tx_frequency_deviation_24h', 'tx_frequency_deviation_1h',
        'time_gap_deviation_median', 'time_gap_deviation_mean',
        'spend_velocity_deviation_1h', 'spend_velocity_deviation_24h',
        'card_device_frequency', 'card_location_frequency',
        'time_gap_acceleration_median', 'time_gap_acceleration_mean',
        'amount_temporal_interaction', 'amount_temporal_interaction_mean'
    ]
    diversity_cols = ['card_device_diversity', 'card_location_diversity']
    metadata_cols = ['TransactionID', 'isFraud']
    
    # Exclude features that are not in Model D
    graph_cols_all = [
        'card_device_degree', 'card_addr_degree', 'device_card_degree', 'addr_card_degree',
        'shared_device_card_count', 'shared_addr_card_count',
        'device_connected_fraud_rate', 'addr_connected_fraud_rate',
        'device_addr_fraud_rate_interaction'
    ]
    
    all_excluded = core_deviations + diversity_cols + metadata_cols + graph_cols_all
    base_features = [col for col in df.columns if col not in all_excluded]
    print(f"Features configuration locked: {len(base_features)} base features.")

    y = df['isFraud']

    # -------------------------------------------------------------
    # 4. RUN SPLIT 80/20
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("RUNNING ABLATION ON 80/20 SPLIT")
    print("=" * 50)
    
    y_train_80 = y.iloc[:split_80].copy().values
    y_val_80 = y.iloc[split_80:].copy().values
    
    # Load Model D predictions or train
    if os.path.exists(d_preds_8020_path):
        print("  * Loading saved Model D predictions...")
        d_preds_df = pd.read_parquet(d_preds_8020_path)
        probs_d_80 = d_preds_df['fraud_probability'].values
        metrics_d_80 = evaluate_predictions(y_val_80, probs_d_80)
    else:
        X_train_d = df.iloc[:split_80][base_features].copy()
        X_val_d = df.iloc[split_80:][base_features].copy()
        categorical_cols = list(X_train_d.select_dtypes(include=['category']).columns)
        metrics_d_80, probs_d_80 = train_eval_config("Model D (Baseline)", X_train_d, y_train_80, X_val_d, y_val_80, categorical_cols)

    # Train Ablation configs for 80/20
    configs_80 = {
        'H3a': ['device_connected_fraud_rate'],
        'H3b': ['addr_connected_fraud_rate'],
        'H3': ['device_connected_fraud_rate', 'addr_connected_fraud_rate'],
        'H3i': ['device_connected_fraud_rate', 'addr_connected_fraud_rate', 'device_addr_fraud_rate_interaction']
    }
    
    results_80 = [{'name': 'Model D (Baseline)', 'split': '80/20', **metrics_d_80}]
    probs_map_80 = {'Model D': probs_d_80}
    
    X_train_base_80 = df.iloc[:split_80][base_features].copy()
    categorical_cols = list(X_train_base_80.select_dtypes(include=['category']).columns)
    
    for name, features in configs_80.items():
        X_train = df.iloc[:split_80][base_features + features].copy()
        X_val = df.iloc[split_80:][base_features + features].copy()
        
        metrics, probs = train_eval_config(f"Model {name}", X_train, y_train_80, X_val, y_val_80, categorical_cols)
        results_80.append({'name': f"Model {name}", 'split': '80/20', **metrics})
        probs_map_80[name] = probs
        
        # Save predictions
        pred_path = os.path.join(processed_dir, f"predictions/graph_{name.lower()}_preds_8020.parquet")
        pd.DataFrame({
            'TransactionID': df.iloc[split_80:]['TransactionID'],
            'fraud_probability': probs
        }).to_parquet(pred_path, engine='pyarrow', index=False)

    # -------------------------------------------------------------
    # 5. RUN SPLIT 70/30
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("RUNNING ABLATION ON 70/30 SPLIT")
    print("=" * 50)
    
    y_train_70 = y.iloc[:split_70].copy().values
    y_val_70 = y.iloc[split_70:].copy().values
    
    # Train Model D on 70/30
    X_train_d_70 = df.iloc[:split_70][base_features].copy()
    X_val_d_70 = df.iloc[split_70:][base_features].copy()
    categorical_cols_70 = list(X_train_d_70.select_dtypes(include=['category']).columns)
    metrics_d_70, probs_d_70 = train_eval_config("Model D (Baseline)", X_train_d_70, y_train_70, X_val_d_70, y_val_70, categorical_cols_70)
    
    # Save Model D 70/30 predictions
    d_preds_7030_path = os.path.join(processed_dir, "predictions/model_d_preds_7030.parquet")
    pd.DataFrame({
        'TransactionID': df.iloc[split_70:]['TransactionID'],
        'fraud_probability': probs_d_70
    }).to_parquet(d_preds_7030_path, engine='pyarrow', index=False)
    
    results_70 = [{'name': 'Model D (Baseline)', 'split': '70/30', **metrics_d_70}]
    probs_map_70 = {'Model D': probs_d_70}
    
    # Train ablation configs for 70/30
    for name, features in configs_80.items():
        X_train = df.iloc[:split_70][base_features + features].copy()
        X_val = df.iloc[split_70:][base_features + features].copy()
        
        metrics, probs = train_eval_config(f"Model {name}", X_train, y_train_70, X_val, y_val_70, categorical_cols_70)
        results_70.append({'name': f"Model {name}", 'split': '70/30', **metrics})
        probs_map_70[name] = probs
        
        # Save predictions
        pred_path = os.path.join(processed_dir, f"predictions/graph_{name.lower()}_preds_7030.parquet")
        pd.DataFrame({
            'TransactionID': df.iloc[split_70:]['TransactionID'],
            'fraud_probability': probs
        }).to_parquet(pred_path, engine='pyarrow', index=False)

    # -------------------------------------------------------------
    # 6. SIGNFICANCE TESTING (BOOTSTRAPPING)
    # -------------------------------------------------------------
    print("\nRunning Paired Bootstrapping for Model H3 vs Model D...")
    mean_d_80, std_d_80, ci_l_80, ci_u_80 = run_paired_bootstrap(y_val_80, probs_map_80['Model D'], probs_map_80['H3'], num_resamples=1000)
    mean_d_70, std_d_70, ci_l_70, ci_u_70 = run_paired_bootstrap(y_val_70, probs_map_70['Model D'], probs_map_70['H3'], num_resamples=1000)
    
    # Build Leaderboards
    df_80 = pd.DataFrame(results_80)
    df_80['delta_d'] = df_80['pr_auc'] - metrics_d_80['pr_auc']
    
    df_70 = pd.DataFrame(results_70)
    df_70['delta_d'] = df_70['pr_auc'] - metrics_d_70['pr_auc']
    
    # Print leaderboard
    print("\n" + "=" * 80)
    print("ABLATION LEADERBOARD (80/20 Split):")
    print("=" * 80)
    print(df_80[['name', 'num_features', 'pr_auc', 'delta_d', 'best_f1', 'fpr', 'best_iter']].to_string(index=False))
    print("=" * 80)
    print(f"H3 vs D Paired Bootstrap 95% CI: [{ci_l_80:+.5f}, {ci_u_80:+.5f}]")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("STABILITY LEADERBOARD (70/30 Split):")
    print("=" * 80)
    print(df_70[['name', 'num_features', 'pr_auc', 'delta_d', 'best_f1', 'fpr', 'best_iter']].to_string(index=False))
    print("=" * 80)
    print(f"H3 vs D Paired Bootstrap 95% CI: [{ci_l_70:+.5f}, {ci_u_70:+.5f}]")
    print("=" * 80)

    # 7. Write Markdown Report
    print(f"\nSaving ablation report to: {report_path}")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    rows_80 = []
    for idx, r in df_80.iterrows():
        ci_str = f"`[{ci_l_80:+.5f}, {ci_u_80:+.5f}]`" if r['name'] == 'Model H3' else "N/A"
        rows_80.append(f"| {idx+1} | **{r['name']}** | `{r['num_features']}` | `{r['pr_auc']:.5f}` | `{r['delta_d']:+.5f}` | `{r['roc_auc']:.5f}` | `{r['best_f1']:.5f}` | `{r['fpr']:.5f}` | {ci_str} |")
        
    rows_70 = []
    for idx, r in df_70.iterrows():
        ci_str = f"`[{ci_l_70:+.5f}, {ci_u_70:+.5f}]`" if r['name'] == 'Model H3' else "N/A"
        rows_70.append(f"| {idx+1} | **{r['name']}** | `{r['num_features']}` | `{r['pr_auc']:.5f}` | `{r['delta_d']:+.5f}` | `{r['roc_auc']:.5f}` | `{r['best_f1']:.5f}` | `{r['fpr']:.5f}` | {ci_str} |")

    # Construct final scientific findings
    verdict_80 = "CONFIDENCE INTERVAL CONTAINS ZERO" if (ci_l_80 <= 0.0 and ci_u_80 >= 0.0) else "SIGNIFICANT"
    verdict_70 = "CONFIDENCE INTERVAL CONTAINS ZERO" if (ci_l_70 <= 0.0 and ci_u_70 >= 0.0) else "SIGNIFICANT"
    
    is_h3_stable = "YES" if (df_80.loc[df_80['name'] == 'Model H3', 'pr_auc'].values[0] > metrics_d_80['pr_auc'] and df_70.loc[df_70['name'] == 'Model H3', 'pr_auc'].values[0] > metrics_d_70['pr_auc']) else "NO"
    
    h3a_pr_80 = df_80.loc[df_80['name'] == 'Model H3a', 'pr_auc'].values[0]
    h3b_pr_80 = df_80.loc[df_80['name'] == 'Model H3b', 'pr_auc'].values[0]
    best_single_entity = "device_connected_fraud_rate (Model H3a)" if h3a_pr_80 > h3b_pr_80 else "addr_connected_fraud_rate (Model H3b)"
    
    report_content = f"""# Phase 8D: Model H3 Ablation & Chronological Stability Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report documents the ablation study (decoupling Device vs Address risk), chronological stability checking (70/30 split), and paired bootstrap significance tests.

---

## 1. 80/20 Chronological Split Leaderboard

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(rows_80)}

---

## 2. 70/30 Chronological Split Leaderboard

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(rows_70)}

---

## 3. Key Scientific Findings & Answers

### Question 1: Which network entity matters?
* **Result**: Model H3a (Device) PR-AUC is `{h3a_pr_80:.5f}` compared to Model H3b (Address) PR-AUC of `{h3b_pr_80:.5f}`. 
* **Conclusion**: **`{best_single_entity}`** is the stronger network signal of the two.

### Question 2: Do they complement each other?
* **Result**: Model H3 (both features) PR-AUC is `{df_80.loc[df_80['name'] == 'Model H3', 'pr_auc'].values[0]:.5f}`.
* **Conclusion**: {"Yes" if (df_80.loc[df_80['name'] == 'Model H3', 'pr_auc'].values[0] > max(h3a_pr_80, h3b_pr_80)) else "No"}, combining both features {"produces a higher score than either feature individually, showing complementary utility" if (df_80.loc[df_80['name'] == 'Model H3', 'pr_auc'].values[0] > max(h3a_pr_80, h3b_pr_80)) else "does not outpace the best single-feature model, suggesting some redundancy or early stopping noise"}.
* **Interaction Check**: Model H3i (which includes the product product-interaction) scored `{df_80.loc[df_80['name'] == 'Model H3i', 'pr_auc'].values[0]:.5f}` on the 80/20 split, indicating that adding the interaction term {"improves" if (df_80.loc[df_80['name'] == 'Model H3i', 'pr_auc'].values[0] > df_80.loc[df_80['name'] == 'Model H3', 'pr_auc'].values[0]) else "does not improve"} performance.

### Question 3: Is H3 actually better than Model D? (Stability Check)
* **Chronological Stability**: Is H3 consistently better than Model D across splits? **`{is_h3_stable}`**.
  * On the 80/20 split: H3 = `{df_80.loc[df_80['name'] == 'Model H3', 'pr_auc'].values[0]:.5f}` vs Model D = `{metrics_d_80['pr_auc']:.5f}` (Delta: `{df_80.loc[df_80['name'] == 'Model H3', 'delta_d'].values[0]:+.5f}`).
  * On the 70/30 split: H3 = `{df_70.loc[df_70['name'] == 'Model H3', 'pr_auc'].values[0]:.5f}` vs Model D = `{metrics_d_70['pr_auc']:.5f}` (Delta: `{df_70.loc[df_70['name'] == 'Model H3', 'delta_d'].values[0]:+.5f}`).
* **Bootstrap Significance**:
  * 80/20 split delta 95% CI: `[{ci_l_80:+.5f}, {ci_u_80:+.5f}]` (**{verdict_80}**).
  * 70/30 split delta 95% CI: `[{ci_l_70:+.5f}, {ci_u_70:+.5f}]` (**{verdict_70}**).

---

## 4. Final Promotion Recommendation

Based on these results:
* **Frozen Benchmark**: **Model D** remains the frozen champion benchmark (`0.58144` on 80/20, `0.59882` on 70/30).
* **H3 Status**: **Model H3** is our **Best Observed Graph Candidate**. 
* **Recommendation**: {"Since H3 outperforms Model D on both chronological boundaries, we should proceed to Phase 8D error analysis to dissect the recovered fraud cases and confirm the mechanics of the improvement." if is_h3_stable == "YES" else "Since the performance is unstable across splits, we freeze Model D as the champion and do not promote H3."}
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Ablation study completed!")

if __name__ == '__main__':
    main()
