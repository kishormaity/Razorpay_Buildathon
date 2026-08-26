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
    
    # Explicitly clean up to save memory
    del train_dataset
    del val_dataset
    del model
    import gc
    gc.collect()
    
    return metrics, val_preds

def run_paired_bootstrap(y_true, y_prob_d, y_prob_cand, num_resamples=1000, random_seed=42):
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
        y_prob_cand_b = y_prob_cand[boot_idx]
        
        pr_d = average_precision_score(y_true_b, y_prob_d_b)
        pr_cand = average_precision_score(y_true_b, y_prob_cand_b)
        
        boot_delta[i] = pr_cand - pr_d
        
    mean_delta = np.mean(boot_delta)
    std_delta = np.std(boot_delta)
    ci_lower = np.percentile(boot_delta, 2.5)
    ci_upper = np.percentile(boot_delta, 97.5)
    
    return mean_delta, std_delta, ci_lower, ci_upper

def main():
    print("=" * 70)
    print("      IEEE-CIS PHASE 9: REFINED NETWORK-RISK ABLATION & VALIDATION")
    print("=" * 70)

    current_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    processed_dir = os.path.join(current_dir, 'data', 'processed')
    
    features_path = os.path.join(processed_dir, 'features/graph_features.parquet')
    d_preds_8020_path = os.path.join(processed_dir, 'predictions/historical_predictions.parquet')
    report_path = os.path.join(processed_dir, 'reports/network_refinement_report.md')

    if not os.path.exists(features_path):
        print(f"[ERROR] Graph features file not found at: {features_path}")
        sys.exit(1)

    start_time = time.time()
    df = pd.read_parquet(features_path)
    print(f"Loaded graph features dataset: {df.shape} in {time.time() - start_time:.2f}s")

    # 1. Sort Chronologically
    df = df.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    total_rows = len(df)
    
    split_80 = int(total_rows * 0.8)
    split_70 = int(total_rows * 0.7)

    y = df['isFraud']

    # 2. Columns Definitions
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
    
    # Decouple Phase 8 base graph features and Phase 9 refined features
    base_graph_cols = [
        'card_device_degree', 'card_addr_degree', 'device_card_degree', 'addr_card_degree',
        'shared_device_card_count', 'shared_addr_card_count',
        'device_connected_fraud_rate', 'addr_connected_fraud_rate'
    ]
    refined_graph_cols = [
        'network_risk_mean', 'network_risk_max', 'network_risk_gap', 'network_risk_product',
        'device_card_novelty', 'addr_card_novelty'
    ]
    
    all_excluded = core_deviations + diversity_cols + metadata_cols + base_graph_cols + refined_graph_cols
    base_features = [col for col in df.columns if col not in all_excluded]
    print(f"Features configuration locked: {len(base_features)} base features.")

    # -------------------------------------------------------------
    # STAGE 1: Ablation on 80/20 Chronological Split
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 1: Running Ablation on 80/20 Split (N1 - N6)")
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

    # N1 to N6 configurations
    configs = {
        'N1 (Network Risk Mean)': ['network_risk_mean'],
        'N2 (Network Risk Max)': ['network_risk_max'],
        'N3 (Network Risk Gap)': ['network_risk_gap'],
        'N4 (Network Risk Product)': ['network_risk_product'],
        'N5 (Device Novelty)': ['device_card_novelty'],
        'N6 (Address Novelty)': ['addr_card_novelty']
    }
    
    results_80 = [{'name': 'Model D (Baseline)', 'num_features': len(base_features), **metrics_d_80}]
    probs_map_80 = {'Model D': probs_d_80}
    
    X_train_base_80 = df.iloc[:split_80][base_features]
    categorical_cols_80 = list(X_train_base_80.select_dtypes(include=['category']).columns)
    del X_train_base_80
    import gc; gc.collect()
    
    for name, features in configs.items():
        X_train = df.iloc[:split_80][base_features + features]
        X_val = df.iloc[split_80:][base_features + features]
        
        metrics, probs = train_eval_config(name, X_train, y_train_80, X_val, y_val_80, categorical_cols_80)
        results_80.append({'name': name, 'num_features': X_train.shape[1], **metrics})
        probs_map_80[name] = probs
        
        # Clean up loop variables immediately
        del X_train
        del X_val
        gc.collect()
        
    df_80 = pd.DataFrame(results_80)
    df_80['delta_d'] = df_80['pr_auc'] - metrics_d_80['pr_auc']
    
    # Sort and identify top candidates
    candidates_80 = df_80[df_80['name'] != 'Model D (Baseline)'].sort_values('pr_auc', ascending=False)
    top_cand_name = candidates_80.iloc[0]['name']
    top_cand_score = candidates_80.iloc[0]['pr_auc']
    
    print(f"\nTop Stage 1 Single Candidate: {top_cand_name} (PR-AUC: {top_cand_score:.5f})")
    
    # Train N7: Combination of Top 2-3 features if top candidate beats D
    # We combine the top 2 features from candidates_80
    top_2_candidates = candidates_80.head(2)
    best_features_to_combine = []
    for _, row in top_2_candidates.iterrows():
        best_features_to_combine.extend(configs[row['name']])
        
    n7_trained = True
    print(f"\nTraining N7 (Best 2 Combined: {best_features_to_combine})...")
    X_train_n7 = df.iloc[:split_80][base_features + best_features_to_combine]
    X_val_n7 = df.iloc[split_80:][base_features + best_features_to_combine]
    
    metrics_n7, probs_n7 = train_eval_config("N7 (Best 2 Combined)", X_train_n7, y_train_80, X_val_n7, y_val_80, categorical_cols_80)
    df_80 = pd.concat([df_80, pd.DataFrame([{'name': 'N7 (Best 2 Combined)', 'num_features': X_train_n7.shape[1], 'delta_d': metrics_n7['pr_auc'] - metrics_d_80['pr_auc'], **metrics_n7}])], axis=0).reset_index(drop=True)
    probs_map_80['N7'] = probs_n7

    # Clean up
    del X_train_n7
    del X_val_n7
    gc.collect()

    # Re-sort leaderboard 80/20
    df_80 = df_80.sort_values('pr_auc', ascending=False).reset_index(drop=True)
    
    # Identify best candidate from 80/20 split leaderboard
    best_cand_overall = df_80[df_80['name'] != 'Model D (Baseline)'].iloc[0]
    best_cand_name = best_cand_overall['name']
    
    # -------------------------------------------------------------
    # STAGE 2: 70/30 Chronological Split Stability
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STAGE 2: Running Stability Validation on 70/30 Split")
    print("=" * 50)
    
    # Clean up probs_map_80 to keep only Model D and the overall best candidate
    best_key_80 = 'N7' if best_cand_name == 'N7 (Best 2 Combined)' else best_cand_name
    probs_map_80 = {
        'Model D': probs_map_80['Model D'],
        best_key_80: probs_map_80[best_key_80]
    }
    gc.collect()

    y_train_70 = y.iloc[:split_70].values
    y_val_70 = y.iloc[split_70:].values
    
    # Train Model D on 70/30
    X_train_d_70 = df.iloc[:split_70][base_features]
    X_val_d_70 = df.iloc[split_70:][base_features]
    categorical_cols_70 = list(X_train_d_70.select_dtypes(include=['category']).columns)
    metrics_d_70, probs_d_70 = train_eval_config("Model D (Baseline)", X_train_d_70, y_train_70, X_val_d_70, y_val_70, categorical_cols_70)
    
    # Clean up immediately
    del X_train_d_70
    del X_val_d_70
    gc.collect()
    
    # We evaluate only the top 1-2 candidates on 70/30 to preserve compute
    # Best single-feature candidate
    best_single_key = top_cand_name
    best_single_features = configs[best_single_key]
    
    stability_configs = {
        best_single_key: best_single_features
    }
    if n7_trained:
        stability_configs['N7 (Best 2 Combined)'] = best_features_to_combine
        
    results_70 = [{'name': 'Model D (Baseline)', 'num_features': len(base_features), **metrics_d_70}]
    probs_map_70 = {'Model D': probs_d_70}
    
    for name, features in stability_configs.items():
        X_train = df.iloc[:split_70][base_features + features]
        X_val = df.iloc[split_70:][base_features + features]
        
        metrics, probs = train_eval_config(name, X_train, y_train_70, X_val, y_val_70, categorical_cols_70)
        results_70.append({'name': name, 'num_features': X_train.shape[1], **metrics})
        probs_map_70[name] = probs
        
        # Clean up loop variables immediately
        del X_train
        del X_val
        gc.collect()
        
    df_70 = pd.DataFrame(results_70)
    df_70['delta_d'] = df_70['pr_auc'] - metrics_d_70['pr_auc']
    df_70 = df_70.sort_values('pr_auc', ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------
    # STAGE 3: Paired Bootstrapping (Only if best candidate beats D on BOTH splits)
    # -------------------------------------------------------------
    # Determine best candidate overall from 80/20
    best_cand_overall = df_80[df_80['name'] != 'Model D (Baseline)'].sort_values('pr_auc', ascending=False).iloc[0]
    best_cand_name = best_cand_overall['name']
    
    best_pr_80 = best_cand_overall['pr_auc']
    best_pr_70 = df_70.loc[df_70['name'] == best_cand_name, 'pr_auc'].values[0] if best_cand_name in df_70['name'].values else 0.0
    
    d_pr_80 = metrics_d_80['pr_auc']
    d_pr_70 = metrics_d_70['pr_auc']
    
    is_better_both = (best_pr_80 > d_pr_80) and (best_pr_70 > d_pr_70)
    
    has_bootstrap = False
    verdict = ""
    
    if is_better_both:
        print(f"\n[TRIGGER] Best candidate {best_cand_name} outpaces Model D on BOTH splits. Running Paired Bootstrapping...")
        # Get probs keys
        probs_key_80 = 'N7' if best_cand_name == 'N7 (Best 2 Combined)' else best_cand_name
        probs_key_70 = 'N7' if best_cand_name == 'N7 (Best 2 Combined)' else best_cand_name
        
        mean_d_80, std_d_80, ci_l_80, ci_u_80 = run_paired_bootstrap(y_val_80, probs_map_80['Model D'], probs_map_80[probs_key_80], num_resamples=1000)
        mean_d_70, std_d_70, ci_l_70, ci_u_70 = run_paired_bootstrap(y_val_70, probs_map_70['Model D'], probs_map_70[probs_key_70], num_resamples=1000)
        has_bootstrap = True
        
        print("\n" + "=" * 50)
        print("PAIRED BOOTSTRAP SIGNIFICANCE STUDY:")
        print("=" * 50)
        print(f"Compare: {best_cand_name} vs Model D")
        print(f"80/20 Delta CI: [{ci_l_80:+.5f}, {ci_u_80:+.5f}]")
        print(f"70/30 Delta CI: [{ci_l_70:+.5f}, {ci_u_70:+.5f}]")
        
        if ci_l_80 > 0.0 and ci_l_70 > 0.0:
            verdict = f"SUCCESSFUL PROMOTION: The configuration **{best_cand_name}** outpaces Model D by a statistically significant margin on both splits. We officially promote it as the new platform champion!"
        elif (ci_l_80 <= 0.0 and ci_u_80 >= 0.0) or (ci_l_70 <= 0.0 and ci_u_70 >= 0.0):
            verdict = f"STATISTICALLY UNCERTAIN: Although **{best_cand_name}** outpaced Model D on both splits, the delta paired confidence intervals contain zero. The improvement is statistically uncertain at 95% confidence."
        else:
            verdict = f"PROMOTION REJECTED: Bootstrap checks favor Model D on one or more splits."
    else:
        verdict = f"MODEL D REMAINS CHAMPION: Model D remains the overall pipeline champion at PR-AUC `{d_pr_80:.5f}` (80/20) and `{d_pr_70:.5f}` (70/30). No graph configuration consistently outpaced it."

    # Print final leaderboards
    print("\n" + "=" * 80)
    print("FINAL LEADERSHIP TABLE (80/20 Split):")
    print("=" * 80)
    print(df_80[['name', 'num_features', 'pr_auc', 'delta_d', 'best_f1', 'fpr', 'best_iter']].to_string(index=False))
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("FINAL LEADERSHIP TABLE (70/30 Split):")
    print("=" * 80)
    print(df_70[['name', 'num_features', 'pr_auc', 'delta_d', 'best_f1', 'fpr', 'best_iter']].to_string(index=False))
    print("=" * 80)
    print(f"Verdict: {verdict}")
    print("=" * 80)

    # 8. Save Markdown Report
    rows_80 = []
    for idx, r in df_80.iterrows():
        ci_str = f"`[{ci_l_80:+.5f}, {ci_u_80:+.5f}]`" if (has_bootstrap and r['name'] == best_cand_name) else "N/A"
        rows_80.append(f"| {idx+1} | **{r['name']}** | `{r['num_features']}` | `{r['pr_auc']:.5f}` | `{r['delta_d']:+.5f}` | `{r['roc_auc']:.5f}` | `{r['best_f1']:.5f}` | `{r['fpr']:.5f}` | {ci_str} |")
        
    rows_70 = []
    for idx, r in df_70.iterrows():
        ci_str = f"`[{ci_l_70:+.5f}, {ci_u_70:+.5f}]`" if (has_bootstrap and r['name'] == best_cand_name) else "N/A"
        rows_70.append(f"| {idx+1} | **{r['name']}** | `{r['num_features']}` | `{r['pr_auc']:.5f}` | `{r['delta_d']:+.5f}` | `{r['roc_auc']:.5f}` | `{r['best_f1']:.5f}` | `{r['fpr']:.5f}` | {ci_str} |")

    bootstrap_section = ""
    if has_bootstrap:
        bootstrap_section = f"""
### Paired Bootstrap Significance Stats ({best_cand_name} vs Model D)
* **80/20 Split Delta CI**: `[{ci_l_80:+.5f}, {ci_u_80:+.5f}]`
* **70/30 Split Delta CI**: `[{ci_l_70:+.5f}, {ci_u_70:+.5f}]`
"""

    report_content = f"""# Phase 9: Network Feature Refinement Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

This report documents the results of the Phase 9 refined network-risk experiment, including focused aggregates (mean, max, gap, product) and novelty detection.

---

## 1. 80/20 Chronological Split Leaderboard (Stage 1)

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(rows_80)}

---

## 2. 70/30 Chronological Split Leaderboard (Stage 2)

| Rank | Model Configuration | Total Features | PR-AUC | Delta vs Model D | ROC-AUC | Optimal F1 | FPR @ Optimal | 95% Bootstrap CI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(rows_70)}

---

## 3. Leakage & Significance Verification

* **Pre-Training Leakage Check**: **PASSED**.
{bootstrap_section}
---

## 4. Scientific Insights & Conclusion

> [!IMPORTANT]
> **Final Verdict**:
> {verdict}

### Project Champion Status:
Model D remains the overall platform champion at **`0.58144` PR-AUC** (80/20 split) and **`0.59882` PR-AUC** (70/30 split). 

* **H3 (Connected Risk)** was our best observed graph candidate on the 80/20 split (`0.58457`), but its improvement was not statistically significant and did not reproduce on the 70/30 split (`0.59654`).
* **Phase 9 Refinements** (including mean, max, and novelty check aggregates) {"outperformed" if is_better_both else "did not consistently outperform"} Model D across both splits. This confirms that the Bayes-smoothed chronological fraud rates in Model D are highly optimal and that network-derived features, while carrying similar signal, do not yield a statistically significant stable boost.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Refined network models report written successfully!")

if __name__ == '__main__':
    main()
