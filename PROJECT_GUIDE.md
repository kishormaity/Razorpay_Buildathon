# AI Risk Sentinel - Project Structure Guide

Welcome to the **AI Risk Sentinel** workspace. This guide outlines the project's file structure, directory tree, and module classifications to help developers and judges navigate the codebase.

---

## 🏗️ Core Directory Architecture

```text
Razorpay_Build/
├── README.md                  # High-level platform architecture and instructions
├── PROJECT_GUIDE.md           # This file (conceptual directory mapping and classification)
├── CLAUDE.md                  # Command cheatsheet (start dev servers, run scripts)
├── AGENTS.md                  # Custom agent system guidelines and credentials policies
│
├── backend/                   # PYTHON BACKEND: Pipeline, modeling, and API servers
│   ├── requirements.txt       # Python library dependencies
│   ├── pipeline/              # Ingestion, raw dataset merge, and data auditing
│   ├── features/              # Feature engineering scripts (by module/feature set)
│   ├── models/                # Frozen serialized booster binaries and feature configs
│   ├── experiments/           # Active research scripts, model training, and web portal
│   └── data/                  # Git-ignored local database store (raw & processed data)
│
└── frontend/                  # FRONTEND APPLICATION: Analyst dashboard UI
    ├── package.json           # npm library dependencies
    └── src/                   # Next.js pages, UI components, and API client state
```

---

## 📁 Detailed Directory Breakdown

### 1. Python Modeling & API Server (`backend/`)
This workspace handles downloading the raw Kaggle dataset, merging tables, running evaluations, exporting final model binaries, and hosting the backend FastAPI server.

#### `backend/pipeline/`
* **[`download.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/pipeline/download.py)**: Downloads raw ZIP files from the IEEE-CIS Fraud Detection competition via the Kaggle API.
* **[`build_merged_dataset.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/pipeline/build_merged_dataset.py)**: Performs a high-reliability chronological outer join on `transaction` and `identity` tables.
* **[`preprocess.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/pipeline/preprocess.py)**: Cleans data types and prepares splits.
* **[`audit.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/pipeline/audit.py)**: Analyzes statistical attributes and creates an audit report.
* **[`schema.sql`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/pipeline/schema.sql)**: SQL configuration for storing risk transactions.

#### `backend/features/`
* **[`abuse_ring_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/features/abuse_ring_features.py)**: Main module that builds the final 8 Sentinel features.
* **[`transaction_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/features/transaction_features.py)**: Baseline transaction-level characteristics.
* **[`historical_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/features/historical_features.py)**: Entity sharing and card velocity properties.
* **[`card_novelty_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/features/card_novelty_features.py)**: Tracks historical card issuance timelines.
* **[`deviation_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/features/deviation_features.py)**: Normal behavioral baseline deviation metrics.
* **[`error_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/features/error_features.py)**: Preconditions for transaction validation anomalies.
* **[`graph_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/features/graph_features.py)**: Computes device-address card intersection degrees.

#### `backend/models/`
* **[`model_d_final.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/models/model_d_final.txt)**: Serialized LightGBM booster containing the 403-feature Model D.
* **[`abuse_ring_sentinel_final.txt`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/models/abuse_ring_sentinel_final.txt)**: Serialized LightGBM booster containing the 8-feature Sentinel model.
* **[`model_d_features.json`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/models/model_d_features.json)**: Feature names list for Model D alignment.
* **[`sentinel_features.json`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/models/sentinel_features.json)**: Feature names list for Sentinel model alignment.

---

### 2. Conceptual Classification of Experiments (`backend/experiments/`)
To preserve relative imports and launch commands, all research scripts are kept in the `experiments/` directory. They can be classified into 5 functional categories:

#### Class A: Main Application & Final Pipeline
* **[`sentinel_app.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/sentinel_app.py)**: Upgraded FastAPI app server that serves the interactive Verifiable Live Inference Portal (Demo A).
* **[`train_final_models.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/train_final_models.py)**: Chronological model fitter that exports frozen `.txt` LightGBM models.

#### Class B: Sentinel Network Experiments
* **[`train_abuse_ring_sentinel.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/train_abuse_ring_sentinel.py)**: Script to fit and validate the secondary-review Sentinel model.
* **[`audit_ring_proxy.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/audit_ring_proxy.py)**: Weak supervision label creation script.
* **[`audit_sentinel_predictions.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/audit_sentinel_predictions.py)**: Scans Sentinel score distribution behaviors.

#### Class C: Feature Importance & Ablation Studies
* **[`ablation_study.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/ablation_study.py)**: Measures validation AUC degradation upon feature removals.
* **[`model_d_deep_dive.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/model_d_deep_dive.py)**: Ranks features and plots splits.
* **[`residual_fn_analysis.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/residual_fn_analysis.py)**: Evaluates false negatives of Model D.
* **[`card_interaction_analysis.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/card_interaction_analysis.py)**: Analyzes card usage patterns.

#### Class D: Architectural Validation Scripts
* **[`validate_sentinel_routing.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/validate_sentinel_routing.py)**: Tests transaction flow routing logic.
* **[`validate_statistics.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/validate_statistics.py)**: Checks feature statistics.
* **[`validate_card_novelty.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/validate_card_novelty.py)**: Validates card timeline issuance logic.
* **[`validate_deviation_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/validate_deviation_features.py)**: Validates normal-behavior deviation metrics.
* **[`validate_error_features.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/validate_error_features.py)**: Validates raw error checking constraints.
* **[`validate_h3_ablation.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/validate_h3_ablation.py)**: Checks feature contribution layers.
* **[`validate_c5_bootstrap.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/validate_c5_bootstrap.py)**: Tests robustness bootstrapper.

#### Class E: Sub-model Explorations
* **[`train_deviation_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/train_deviation_model.py)**: Explores deviation sub-model fitting.
* **[`train_graph_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/train_graph_model.py)**: Fits graph topological connections.
* **[`train_historical_model.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/train_historical_model.py)**: Fits entities velocity profile.
* **[`train_refined_network_models.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/train_refined_network_models.py)**: Fits refined network parameters.
* **[`train_risk_fusion.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/train_risk_fusion.py)**: Explores probability fusion metrics.
* **[`train_transaction_baseline.py`](file:///c:/Users/BIT/Downloads/Razorpay_Build/backend/experiments/train_transaction_baseline.py)**: Fits baseline LightGBM.

---

### 3. Analyst Dashboard UI (`frontend/`)
Next.js React dashboard for real-time monitoring and network telemetry graph visuals.
* **`src/app/`**: Framework layouts, pages, and API routing.
* **`src/components/`**: UI visualizations for active transaction logs and risk profiles.
* **`src/services/`**: Simulated local state and mock database managers.
