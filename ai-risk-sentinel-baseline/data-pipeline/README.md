# AI Risk Manager Dataset Workbench

This directory houses the data pipeline scripts to acquire, sample, and preprocess the **IEEE-CIS Fraud Detection** dataset from Kaggle. 

This dataset is chosen because it is a widely used benchmark for payment fraud detection and provides both transaction and identity attributes. It consists of two complementary segments:
1. **Transaction Telemetry:** Amount, card details, email domain, address matches, and distance velocity metrics.
2. **Identity Attributes:** Device fingerprint details, OS type, IP subnets, browser versions, and screen parameters (crucial for linking coordinated abuse rings).

---

## Directory Structure

```text
data-pipeline/
├── README.md                 # Pipeline documentation & instructions
├── requirements.txt          # Python dependencies
├── kaggle.json               # Kaggle API authentication credentials
├── pipeline/                 # Data Pipeline Scripts
│   ├── download.py           # Downloads raw CSVs via Kaggle API
│   ├── build_merged_dataset.py # Validation-based transaction-identity merger
│   ├── preprocess.py         # Preprocesses, samples, and exports to SQLite database
│   ├── schema.sql            # Relational database schema matching frontend
│   ├── audit.py              # Dynamic chunked data audit & report generator
│   └── data_audit_report.md  # Audit diagnostics report
├── features/                 # Feature Engineering Components
│   ├── transaction_features.py # Baseline transaction telemetry processor
│   ├── behavioral_features.py # Rolling temporal behavioral feature engineering
│   └── historical_features.py # Expanding chronological target/frequency features
├── app/                      # Backend REST API and Live Portal
│   ├── sentinel_app.py       # FastAPI web server and embedded portal page
│   └── sentinel_demo.py      # Terminal command-line simulator
├── research/                 # Model training experiments & analytical scripts
│   ├── train_final_models.py # Trains and serializes production Model D & Sentinel
│   ├── train_historical_model.py # Trains GBDT Model D (LGBM Tabular + Historical)
│   └── (23 more scripts)     # Ablation studies, combinations, linear fusions, etc.
├── validation/               # Statistical validations & verification scripts
│   ├── validate_sentinel_routing.py # Replicates test-split routing flow statistics
│   └── (7 more validation scripts) # Deviances, bootstrap, novelty checks, etc.
└── data/                     # Git-ignored data repository
    ├── raw/                  # Full raw CSV datasets from Kaggle
    └── processed/            # Structured data workbench artifacts
        ├── features/         # Feature parquet tables (transaction, behavioral, historical)
        ├── predictions/      # Parquet files containing model validation predictions
        ├── models/           # Serialized LightGBM/XGBoost model files
        ├── reports/          # Markdown evaluation metrics reports
        ├── plots/            # Precision-Recall curve PNG plots
        ├── metadata/         # Model pipeline metadata tracking files
        └── risk_sentinel.db  # Relational SQLite database (dashboard input)
```

---

## Prerequisites & Setup

To execute the data scripts, you need Python 3 installed with the required libraries.

### 1. Install Dependencies
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle Credentials
To download the dataset automatically, you need a Kaggle API token:
1. Log in to [Kaggle](https://www.kaggle.com/).
2. Navigate to your **Account Settings** page (click your profile photo, then **Settings**).
3. Scroll down to the **API** section and click **Create New API Token**.
4. This downloads a file named `kaggle.json`.
5. Place this file in the appropriate directory on your operating system:
   * **Windows:** `C:\Users\<Your-Username>\.kaggle\kaggle.json`
   * **macOS/Linux:** `~/.kaggle/kaggle.json`
6. Make sure you accept the competition rules by visiting the [IEEE-CIS Competition Page](https://www.kaggle.com/c/ieee-fraud-detection) and clicking "I Understand and Accept".

---

## Execution Guide

### Step 1: Download the Data
Run the download script. It will connect to Kaggle, accept credentials, and pull down the dataset zip files into `data/raw/` before unzipping them:
```bash
python pipeline/download.py
```

### Step 2: Preprocess and Structure the Data
Because the raw dataset is very large (590,540 transactions, >2.5 GB on disk), we sample it for the lightweight web application database. The preprocessing script does the following:
1. Loads the Transaction and Identity files.
2. Performs balanced sampling (retaining all fraud cases in the sample, paired with a matching subset of non-fraud records to mitigate class imbalance for the dashboard interface).
3. Extracts entities (Users, Devices, IPs, Payments) and builds the graph relationships table (shared device logins, shared IP addresses, card binding links).
4. Creates a clean SQLite database at `data/processed/risk_sentinel.db` which is fully structured according to our frontend data models.

Run the preprocessing script:
```bash
python pipeline/preprocess.py
```

---

## 🧪 Model Development Workbench

This directory also functions as a machine learning experimentation environment for payment-fraud detection. The pipeline evaluates incremental feature sets and multiple tree-based classifiers using a strict chronological 80/20 train/validation split.

The primary evaluation metric is **PR-AUC (Average Precision)** because the dataset is highly imbalanced, with approximately 3.499% fraudulent transactions (20,663 fraud cases out of 590,540 total transactions population).

### 📁 Model Pipeline Files

* **`train_transaction_baseline.py`**  
  Trains **Model A**, the LightGBM transaction-only baseline.  
  **PR-AUC: `0.57181`**.

* **`behavioral_features.py`**  
  Generates 12 leakage-free temporal and behavioral features based primarily on card transaction history, spending velocity, location/email consistency, and device/location novelty.

* **`train_behavioral_model.py`**  
  Trains **Model B**, LightGBM using transaction features plus all 12 behavioral features. Performance decreased to **PR-AUC `0.54628`**, indicating that the raw rolling behavioral features introduced noise when directly combined with the transaction feature space.

* **`train_top_behavioral_model.py`**  
  Trains **Model C**, LightGBM using transaction features plus the six highest-ranked behavioral features. Achieved **PR-AUC `0.56722`**.

* **`train_behavior_only_models.py`**  
  Benchmarks LightGBM, XGBoost, and CatBoost using only the 12 behavioral features. XGBoost achieved the best standalone behavioral performance with **PR-AUC `0.09396`**, compared with the random/prevalence baseline of approximately `0.03499`.

* **`train_risk_fusion.py`**  
  Evaluates weighted prediction fusion and logistic stacking between the transaction-risk model and the standalone behavioral-risk model.

* **`historical_features.py`**  
  Generates 10 leakage-free chronological historical features: four past-transaction frequency features and six Bayesian-smoothed historical fraud-rate features for high-cardinality entities and combinations.

* **`train_historical_model.py`**  
  Trains **Model D**, LightGBM using transaction + historical features. This is currently the **best-performing model**, achieving **PR-AUC `0.58144`**.

* **`train_historical_xgb.py`**  
  Trains **Model E**, XGBoost using transaction + historical features. Achieved **PR-AUC `0.55965`**.

* **`train_combined_model.py`**  
  Trains **Model F**, LightGBM using transaction + historical features + the six selected behavioral features. Achieved **PR-AUC `0.57394`**. The behavioral features did not provide additional PR-AUC improvement over Model D in this configuration.

---

### 📊 Performance Leaderboard

All Models A–F are evaluated using the same chronological validation methodology where applicable. PR-AUC is the primary metric.

| Model Config | Algorithm | Features Included / Mode | PR-AUC (Primary) | ROC-AUC | Optimal F1 | FPR @ Optimal | Note |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Model A** | LightGBM | Transaction-Only Baseline | `0.57181` | `0.91223` | `0.56645` | `0.00948` | Baseline |
| **Model B** | LightGBM | Transaction + All 12 Behavioral | `0.54628` | `0.90603` | `0.55230` | `0.00800` | Overfitting noise |
| **Model C** | LightGBM | Transaction + Top-6 Behavioral | `0.56722` | `0.91233` | `0.56120` | `0.00735` | Lower FPR |
| 🥇 **Model D** | **LightGBM** | **Transaction + Historical** | **`0.58144`** | `0.90507` | **`0.58178`** | **`0.00880`** | **Pipeline Champion** |
| **Model E** | XGBoost | Transaction + Historical | `0.55965` | `0.89799` | `0.56932` | `0.00943` | XGBoost comparison |
| **Model F** | LightGBM | Transaction + Historical + Top-6 Behav | `0.57394` | `0.90275` | `0.57162` | `0.01005` | No additional PR-AUC gain |
| **LGBM Behavior** | LightGBM | 12 Behavioral Features Only | `0.08936` | `0.72475` | `0.17132` | `0.06985` | Standalone benchmark |
| **XGB Behavior** | XGBoost | 12 Behavioral Features Only | **`0.09396`** | `0.72462` | `0.17269` | `0.05659` | Standalone winner |
| **CatB Behavior** | CatBoost | 12 Behavioral Features Only | `0.09180` | `0.72463` | `0.17139` | `0.05942` | Standalone benchmark |
| **Weighted Fusion** | Linear | $w_{opt}=1.00$ (Tx-only weight) | `0.54831` | `0.90667` | `0.54939` | `0.01054` | Evaluated on Meta-Test subset |
| **Logistic Stack** | Logistic Reg | Tx Model + Behavioral Model | `0.54529` | `0.90729` | `0.54991` | `0.01052` | Evaluated on Meta-Test subset |

> **Note:** Fusion metrics were evaluated on the Meta-Test subset, so they should not be directly compared with the full validation metrics of Models A–F.

---

## 🏆 Current Best Model

**Model D — LightGBM Transaction + Historical Features**

- Algorithm: **LightGBM**
- Feature space: **403 model features**
  - 393 transaction features
  - 4 historical frequency features
  - 6 historical Bayesian-smoothed fraud-rate features
- PR-AUC: **`0.58144`**
- ROC-AUC: `0.90507`
- Optimal F1: **`0.58178`**
- FPR @ Optimal: **`0.00880`**

Model D currently provides the highest PR-AUC among all evaluated configurations. The historical features provide additional predictive information that was not captured effectively by directly adding the rolling behavioral features to the LightGBM transaction model.

---

## 🚀 Running the Champion Pipeline

### 1. Generate Historical Features
Run the feature extraction script to calculate leakage-free expanding frequency counts and Bayesian-smoothed historical fraud-rates strictly looking backward:
```bash
python features/historical_features.py
```
This saves the enriched features to `data/processed/features/historical_features.parquet`.

### 2. Train the Champion Model
Train the LightGBM transaction + historical features model under chronological 80/20 splitting:
```bash
python experiments/train_historical_model.py
```
This saves the trained model to `data/processed/models/historical_lgb_model.txt`, exports predictions, plots comparative Precision-Recall curves, and writes a detailed evaluation report.

### Current Pipeline Architecture

```text
IEEE-CIS Transactions + Identity
                │
                ▼
       Feature Engineering
                │
                ▼
       Transaction Features
          (393 features)
                │
        ┌───────┴────────┐
        │                │
        ▼                ▼
 Behavioral Features   Historical Features
    (12 features)        (10 features)
        │                │
        │                ▼
        │          Bayesian-smoothed
        │          chronological rates
        │                │
        └───────┬────────┘
                │
                ▼
       Model Experiments
                │
                ▼
     ┌──────────────────────┐
     │ Model D              │
     │ LightGBM             │
     │ Tx + Historical      │
     │ PR-AUC = 0.58144     │
     └──────────────────────┘
                │
                ▼
          Fraud Risk Score
```

> **Note:** Behavioral-only models remain useful as an independent risk signal, but directly adding the selected behavioral features to the historical LightGBM model did not improve PR-AUC in the current experiments.
