# AI Risk Sentinel

AI Risk Sentinel is an enterprise-grade platform for payment fraud detection, network-based abuse management, and machine learning model telemetry. 

The platform integrates a dynamic Next.js-based analyst dashboard with a high-performance Python data workbench designed around the widely used **IEEE-CIS Fraud Detection** benchmark dataset, which contains 590,540 transactions with complementary transaction and identity information.

---

## 🏗️ Platform Architecture

The project consists of two primary components:

1. **AI Risk Dashboard (Frontend)**: A Next.js, React, and TypeScript application for real-time risk alerts, transaction queue management, entity graph visualization, and model monitoring (policies, feature drift, and coordination rings).
2. **Dataset Workbench (Backend Pipeline)**: A Python pipeline that automates downloading the raw IEEE-CIS dataset, merging transaction and identity data, running dynamic audits, and structuring tables for database schemas.

```text
                  IEEE-CIS Kaggle Raw Data
                             │
                             ▼
                    [Dataset Workbench]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ML Modeling Dataset]             [Application Schema]
     (merged_train.parquet)            (schema.sql)
                                              │
                                              ▼
                                       [SQLite Database]
                                      (risk_sentinel.db)
                                              │
                                              ▼
                                     [Analyst Dashboard]
                                    (Next.js App Server)
```

---

## 📁 Directory Structure

```text
Razorpay_Build/
├── README.md                 # Root platform documentation
├── CLAUDE.md                 # Developer CLI command cheatsheet
├── AGENTS.md                 # Custom rules and guidelines
├── backend/                  # Python Data Engineering Pipeline & API Server
│   ├── README.md             # Backend pipeline instructions
│   ├── requirements.txt      # Python dependencies
│   ├── pipeline/             # Data ingestion and preprocessing scripts
│   ├── features/             # Feature engineering components
│   ├── experiments/          # Model experiments, training scripts, and FastAPI app
│   └── data/                 # Raw and processed datasets (Git-ignored)
└── frontend/                 # Next.js TypeScript App
    ├── package.json          # Node dependencies
    ├── src/
    │   ├── app/              # Page layouts & router endpoints
    │   ├── components/       # Core UI elements (Rings, Search, Stats)
    │   ├── services/         # API & simulated database connection state
    │   └── data/             # Frontend typescript schemas and mock arrays
```

---

## 🛠️ Getting Started

### 1. Backend & Data Pipeline Setup

Configure your Python environment and prepare the IEEE-CIS raw data:

```bash
# 1. Navigate to the backend workbench
cd backend

# 2. Set up virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate       # On Windows (cmd/powershell)
source .venv/bin/activate    # On Unix/macOS
pip install -r requirements.txt

# 3. Configure your Kaggle credentials (kaggle.json API token)
# Standard location: ~/.kaggle/kaggle.json

# 4. Download raw dataset from Kaggle
python pipeline/download.py

# 5. Run the high-reliability dataset merger (produces merged_train.parquet)
python pipeline/build_merged_dataset.py

# 6. Run the dynamic data audit (produces data_audit_report.md)
python pipeline/audit.py
```

### 2. Frontend Development Setup

Run the Next.js development server to launch the Analyst dashboard:

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install packages
npm install

# 3. Start local development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the AI Risk Sentinel console.

---

## 🧪 ML Modeling Pipeline & Leaderboard

The Python data workbench includes a high-performance modeling pipeline that evaluates incremental feature groups and boosting algorithms under a chronological 80/20 train/validation split. The primary evaluation metric is **PR-AUC (Average Precision)** because the dataset is highly imbalanced, with approximately 3.499% fraudulent transactions (20,663 fraud cases out of 590,540 total transactions).

### 📁 Model Progression
* **Model A (Transaction-Only)**: The LightGBM transaction-only baseline model. **PR-AUC: `0.57181`**.
* **Model B (Transaction + All Behavioral)**: LightGBM using transaction features plus all 12 rolling behavioral features. Performance decreased to **PR-AUC `0.54628`**, indicating that raw rolling window aggregates introduced noise when directly combined with transaction features.
* **Model C (Transaction + Top-6 Behavioral)**: LightGBM using transaction features plus the six highest-ranked behavioral features. Achieved **PR-AUC `0.56722`**.
* **Model D (Transaction + Historical)**: LightGBM using transaction + 10 leakage-free chronological historical features. This is currently the **best-performing model**, achieving **PR-AUC `0.58144`**.
* **Model E (XGBoost Transaction + Historical)**: XGBoost version of Model D. Achieved **PR-AUC `0.55965`**.
* **Model F (Transaction + Historical + Top-6 Behavioral)**: LightGBM combining transaction, historical, and the top-6 behavioral features. Achieved **PR-AUC `0.57394`**. Top behavioral features did not provide additional PR-AUC improvement when combined with the historical features in this LightGBM configuration.

---

### 📊 Comparative Leaderboard

All Models A–F are evaluated using the same chronological validation split. PR-AUC is the primary metric.

| Model Config | Algorithm | Features Included / Mode | PR-AUC (Primary) | ROC-AUC | Optimal F1 | FPR @ Optimal | Training Time | Best Iteration | Note |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model A** | LightGBM | Transaction-Only Baseline | `0.57181` | `0.91223` | `0.56645` | `0.00948` | `94.91s` | `831` | Baseline |
| **Model B** | LightGBM | Transaction + All 12 Behavioral | `0.54628` | `0.90603` | `0.55230` | `0.00800` | `141.22s` | `393` | Overfitting noise |
| **Model C** | LightGBM | Transaction + Top-6 Behavioral | `0.56722` | `0.91233` | `0.56120` | `0.00735` | `123.28s` | `378` | FPR reduction |
| 🥇 **Model D** | **LightGBM** | **Transaction + Historical** | **`0.58144`** | `0.90507` | **`0.58178`** | **`0.00880`** | `119.27s` | `998` | **Pipeline Champion** |
| **Model E** | XGBoost | Transaction + Historical | `0.55965` | `0.89799` | `0.56932` | `0.00943` | `413.15s` | `444` | Depth-first growth |
| **Model F** | LightGBM | Transaction + Hist + Top-6 Behav | `0.57394` | `0.90275` | `0.57162` | `0.01005` | `77.35s` | `562` | Behavioral redundancy |
| **LGBM Behavior** | LightGBM | 12 Behavioral Features Only | `0.08936` | `0.72475` | `0.17132` | `0.06985` | `5.98s` | `81` | Standalone benchmark |
| **XGB Behavior** | XGBoost | 12 Behavioral Features Only | **`0.09396`** | `0.72462` | `0.17269` | `0.05659` | `10.37s` | `46` | Standalone winner |
| **CatB Behavior** | CatBoost | 12 Behavioral Features Only | `0.09180` | `0.72463` | `0.17139` | `0.05942` | `32.43s` | `504` | Standalone benchmark |
| **Weighted Fusion** | Linear | $w_{opt}=1.00$ (Tx-only weight) | `0.54831` | `0.90667` | `0.54939` | `0.01054` | `1.74s` | *N/A* | Meta-Test subset |
| **Logistic Stack** | Logistic Reg | Tx Model + Behavioral Model | `0.54529` | `0.90729` | `0.54991` | `0.01052` | `1.74s` | *N/A* | Meta-Test subset |

> **Note:** Fusion metrics were evaluated on the Meta-Test subset, so they should not be directly compared with the full validation metrics of Models A–F.

---

### 🔍 Key Scientific Insights & Takeaways

1. **Tabular Champion (Model D)**:
   * **PR-AUC**: **`0.58144`** (A relative improvement of **`+1.68%`** over the baseline Model A).
   * **Optimal F1-Score**: **`0.58178`** (An absolute improvement of **`+0.01533`**).
   * **FPR**: Reduced to **`0.00880`** (A **7.2% reduction in false positives**, reducing customer checkout friction by 77 occurrences).
   * **Metrics Trade-off**: Model D significantly improves PR-AUC, F1, and FPR, but results in a slight decrease in ROC-AUC (`0.90507` vs. `0.91223`).
2. **Expanding Chronological Encodings prevent Overfitting**:
   * Rolling window aggregates (Model B/C/F) fluctuate rapidly and introduce high-frequency noise. This caused LightGBM to overfit on specific card records and trigger premature early stopping.
   * In contrast, Model D leverages 10 **leakage-free expanding chronological target encodings and count frequencies** smoothed with Bayesian m-estimates. These features are highly stable and represent the global risk profile of entities. The model trained for **998 rounds** without overfitting, achieving our peak score.
   * In Model D/F, `card_addr_combo_historical_fraud_rate` became the **#1 most important feature** by gain globally, surpassing raw card and address IDs.
3. **Booster Growth Strategy Dynamics**:
   * LightGBM's leaf-wise (best-first) tree growth is highly optimized for high-cardinality categorical tabular datasets. It outpaced XGBoost Model E by **`+0.02179`** in PR-AUC and trained **3.5x faster** (119s vs. 413s). XGBoost's level-wise (depth-first) growth strategy overfit on features early, triggering premature early stopping.
4. **Behavioral Redundancy**:
   * Combining historical features and behavioral features in Model F degraded PR-AUC to `0.57394` (down `-0.00750` from Model D). The stable chronological features render rolling behavioral windows redundant and noisy in this configuration.

---

## 🏆 Current Best Model

**Model D — LightGBM Transaction + Historical Features**

- Algorithm: **LightGBM**
- Feature space: **403 model features** (393 transaction + 4 historical frequency + 6 Bayesian-smoothed fraud-rate features)
- PR-AUC: **`0.58144`**
- ROC-AUC: `0.90507`
- Optimal F1: **`0.58178`**
- FPR @ Optimal: **`0.00880`**

Model D currently provides the highest PR-AUC and optimal F1 balance among all evaluated configurations.

---

## 🛡️ Standalone Abuse-Ring Sentinel Layer

To address coordinated network-based exploits that bypass transaction-level models, we integrated a secondary review routing overlay:
> **The Abuse-Ring Sentinel identifies transactions exhibiting coordinated multi-entity risk patterns using a chronological weak-supervision proxy.**

### 1. Two-Layer Operational Workflow
Rather than blending risk scores, which dilutes automated checkout precision, the pipeline implements a modular routing topology:

```text
                  Incoming Transaction
                           │
                           ▼
                    [ Model D GBDT ]
                     Threshold 0.30
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
          [ BLOCK ]                   [ ALLOW ]
       (score >= 0.30)            (score < 0.30)
                                         │
                                         ▼
                               [ Sentinel Check ]
                                 Threshold 0.15
                                         │
                           ┌─────────────┴─────────────┐
                           ▼                           ▼
                       [ REVIEW ]                  [ APPROVE ]
                    (score >= 0.15)             (score < 0.15)
```

---

### 2. Locked Final Test Set Metrics

The Sentinel was evaluated once on the strictly locked chronological Test split (last 15% of transactions), delivering these results:

| Evaluation Dimension | Final Test Split Result |
| :--- | :---: |
| **Sentinel Review Population Share** | **`9.90%`** (8,773 transactions) |
| **Model D Missed Fraud (FNs) Total** | **`1,647`** cases |
| **Sentinel-Captured FNs** | **`201`** cases |
| **Missed Fraud (FN) Capture Rate (%)** | **`12.20%`** |
| **Fraud Count in Reviewed Set** | **`426`** cases |
| **FN Capture Efficiency** | **`1.23x`** |

*By capturing 12.20% of Model D's missed fraud while reviewing only 9.90% of transactions, the Sentinel routes risk with **1.23x** efficiency compared to random selection.*

---

## 🚀 Final Submission Reproducibility

To run the pipeline replication and launch the interactive scenario demo, execute:

```powershell
# 1. Run final validation report replication
.venv\Scripts\python backend/experiments/validate_sentinel_routing.py

# Run the sentinel demo visualization
.venv\Scripts\python backend/experiments/sentinel_demo.py
```

---

## 🔬 Core Data Guidelines

To prevent common ML pipeline mistakes, this repository enforces the following practices:
* **No Synthetic Fallback**: The database pipeline structures raw features directly from the source dataset. No mock fallback data is written.
* **No Label Leakage**: The target label (`isFraud`) is strictly excluded from feature inputs, preserving separation between ground truth and simulated model scores.
* **Correct Temporal Representation**: `TransactionDT` is treated as a relative offset in elapsed seconds from an anonymized reference point for relative-time feature engineering. It is not mapped to hardcoded calendar days.
* **Entity Relationships**: User identities are not assumed using `card1`. The graph model explicitly maps separate nodes for payments (`CARD`), hardware configurations (`DEVICE`), and billing locations (`REGION`).

