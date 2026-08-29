# AI Risk Sentinel - Project Structure Guide (Baseline)

Welcome to the **AI Risk Sentinel** baseline workspace. This guide outlines the project's file structure, directory tree, and module classifications to help developers and judges navigate the codebase.

---

## 🏗️ Core Directory Architecture

```text
ai-risk-sentinel-baseline/
├── README.md                  # High-level platform architecture and instructions
├── PROJECT_GUIDE.md           # This file (conceptual directory mapping and classification)
├── CLAUDE.md                  # Command cheatsheet (start dev servers, run scripts)
│
└── data-pipeline/             # PYTHON DATA PIPELINE: Pipeline, modeling, and API servers
    ├── requirements.txt       # Python library dependencies
    ├── pipeline/              # Ingestion, raw dataset merge, and data auditing
    ├── features/              # Feature engineering scripts (by module/feature set)
    ├── models/                # Frozen serialized booster binaries and feature configs
    ├── app/                   # FastAPI portal application and CLI demo
    ├── research/              # Model training experiments & analytical scripts
    ├── validation/            # Statistical validations & verification scripts
    └── data/                  # Git-ignored local database store (raw & processed data)
```

---

## 📁 Detailed Directory Breakdown

### 1. Python Modeling & API Server (`data-pipeline/`)
This workspace handles downloading the raw Kaggle dataset, merging tables, running evaluations, exporting final model binaries, and hosting the backend FastAPI server.

#### `data-pipeline/pipeline/`
* **[`download.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/pipeline/download.py)**: Downloads raw ZIP files from the IEEE-CIS Fraud Detection competition via the Kaggle API.
* **[`build_merged_dataset.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/pipeline/build_merged_dataset.py)**: Performs a high-reliability chronological outer join on `transaction` and `identity` tables.
* **[`preprocess.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/pipeline/preprocess.py)**: Cleans data types and prepares splits.
* **[`audit.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/pipeline/audit.py)**: Analyzes statistical attributes and creates an audit report.
* **[`schema.sql`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/pipeline/schema.sql)**: SQL configuration for storing risk transactions.

#### `data-pipeline/features/`
* **[`abuse_ring_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/features/abuse_ring_features.py)**: Main module that builds the final 8 Sentinel features.
* **[`transaction_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/features/transaction_features.py)**: Baseline transaction-level characteristics.
* **[`historical_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/features/historical_features.py)**: Entity sharing and card velocity properties.
* **[`card_novelty_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/features/card_novelty_features.py)**: Tracks historical card issuance timelines.
* **[`deviation_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/features/deviation_features.py)**: Normal behavioral baseline deviation metrics.
* **[`error_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/features/error_features.py)**: Preconditions for transaction validation anomalies.
* **[`graph_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/features/graph_features.py)**: Computes device-address card intersection degrees.

#### `data-pipeline/models/`
* **[`model_d_final.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/models/model_d_final.txt)**: Serialized LightGBM booster containing the 403-feature Model D.
* **[`abuse_ring_sentinel_final.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/models/abuse_ring_sentinel_final.txt)**: Serialized LightGBM booster containing the 8-feature Sentinel model.
* **[`model_d_features.json`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/models/model_d_features.json)**: Feature names list for Model D alignment.
* **[`sentinel_features.json`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/models/sentinel_features.json)**: Feature names list for Sentinel model alignment.

---

### 2. Application & Live Portal (`data-pipeline/app/`)
Exposes the backend REST APIs and serves the embedded live portal web UI.
* **[`sentinel_app.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/app/sentinel_app.py)**: FastAPI app server that serves the interactive Verifiable Live Inference Portal (Demo A).
* **[`sentinel_demo.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/app/sentinel_demo.py)**: Interactive transaction scoring simulation terminal demo.

---

### 3. Model Training & Analysis (`data-pipeline/research/`)
Houses model training pipelines, ablation studies, and diagnostic experiments.
* **[`train_final_models.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_final_models.py)**: Chronological model fitter that exports frozen `.txt` LightGBM models.
* **[`train_abuse_ring_sentinel.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_abuse_ring_sentinel.py)**: Script to fit and validate the secondary-review Sentinel model.
* **[`train_transaction_baseline.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_transaction_baseline.py)**: Fits baseline LightGBM.
* **[`train_behavioral_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_behavioral_model.py)**: Fits transaction and behavioral models.
* **[`train_top_behavioral_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_top_behavioral_model.py)**: Fits models with top selected behavioral features.
* **[`train_behavior_only_models.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_behavior_only_models.py)**: Baseline GBDT benchmarks.
* **[`train_risk_fusion.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_risk_fusion.py)**: Linear stacking fusion of prediction probabilities.
* **[`train_historical_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_historical_model.py)**: Model D GBDT training.
* **[`train_historical_xgb.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_historical_xgb.py)**: Model E XGBoost training.
* **[`train_combined_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_combined_model.py)**: Model F combined GBDT training.
* **[`train_deviation_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_deviation_model.py)**: Explores behavioral deviations.
* **[`train_graph_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_graph_model.py)**: Explores graph topological models.
* **[`train_refined_network_models.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/train_refined_network_models.py)**: Fits refined network GBDT modules.
* **[`ablation_study.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/ablation_study.py)**: Measures validation AUC degradation upon feature removals.
* **[`model_d_deep_dive.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/model_d_deep_dive.py)**: Features ranking and split analysis.
* **[`residual_fn_analysis.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/residual_fn_analysis.py)**: Evaluates false negatives of baseline models.
* **[`card_interaction_analysis.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/card_interaction_analysis.py)**: Analyzes card payment network dynamics.
* **[`error_analysis.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/error_analysis.py)**: Analyzes prediction error behaviors.
* **[`error_analysis_h3.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/error_analysis_h3.py)**: Sub-group failure checks.
* **[`error_analysis_v2.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/error_analysis_v2.py)**: Expanded baseline failure profiling.
* **[`audit_ring_proxy.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/audit_ring_proxy.py)**: Builds proxy target labels for weak supervision.
* **[`audit_sentinel_predictions.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/audit_sentinel_predictions.py)**: Audits Sentinel risk scoring distributions.
* **[`analyze_behavioral.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/analyze_behavioral.py)**: Computes behavioral profiles.
* **[`evaluate_production_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/evaluate_production_model.py)**: Tabulates comparative metrics.
* **[`optimize_combinations.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/research/optimize_combinations.py)**: Finds optimal prediction fusions.

---

### 4. Validation & Verification (`data-pipeline/validation/`)
Contains script sanity checks, bootstrappers, and statistical tests.
* **[`validate_sentinel_routing.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/validation/validate_sentinel_routing.py)**: Replicates test-split routing flow statistics.
* **[`validate_statistics.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/validation/validate_statistics.py)**: Verifies feature statistics alignment.
* **[`validate_card_novelty.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/validation/validate_card_novelty.py)**: Sanity checks card timeline age trackers.
* **[`validate_deviation_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/validation/validate_deviation_features.py)**: Tests normal baseline deviations.
* **[`validate_error_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/validation/validate_error_features.py)**: Tests pipeline validation errors.
* **[`validate_h3_ablation.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/validation/validate_h3_ablation.py)**: Verifies GBDT subset scores.
* **[`validate_c5_bootstrap.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/validation/validate_c5_bootstrap.py)**: Bootstraps statistical bounds.
* **[`verify_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/ai-risk-sentinel-baseline/data-pipeline/validation/verify_features.py)**: Verifies baseline features shapes.
