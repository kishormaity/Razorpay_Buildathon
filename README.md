# AI Risk Sentinel

AI Risk Sentinel is an enterprise-grade platform for payment fraud detection, network-based abuse management, and machine learning model telemetry. 

The platform integrates a dynamic Next.js-based analyst dashboard with a high-performance Python data workbench designed around the industry-standard **IEEE-CIS Fraud Detection** dataset.

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
├── dataset/                  # Python Data Engineering Pipeline
│   ├── download.py           # Kagglehub raw dataset downloader
│   ├── audit.py              # Dynamic chunked data audit & report generator
│   ├── build_merged_dataset.py # Validation-based transaction-identity merger
│   ├── preprocess.py         # DB schema setup and SQLite compiler
│   ├── schema.sql            # SQLite relational database schema
│   ├── requirements.txt      # Python dependencies
│   └── data/                 # Raw and processed files (Git-ignored)
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
# 1. Navigate to the dataset workbench
cd dataset

# 2. Set up virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate       # On Windows (cmd/powershell)
source .venv/bin/activate    # On Unix/macOS
pip install -r requirements.txt

# 3. Configure your Kaggle credentials (kaggle.json API token)
# Standard location: ~/.kaggle/kaggle.json

# 4. Download raw dataset from Kaggle
python download.py

# 5. Run the high-reliability dataset merger (produces merged_train.parquet)
python build_merged_dataset.py

# 6. Run the dynamic data audit (produces data_audit_report.md)
python audit.py
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

## 🔬 Core Data Guidelines

To prevent common ML pipeline mistakes, this repository enforces the following practices:
* **No Synthetic Fallback**: The database pipeline structures raw features directly from the source dataset. No mock fallback data is written.
* **No Label Leakage**: The target label (`isFraud`) is strictly excluded from feature inputs, preserving separation between ground truth and simulated model scores.
* **Correct Temporal Representation**: `TransactionDT` is treated as a relative offset in elapsed seconds from an anonymized reference point for relative-time feature engineering. It is not mapped to hardcoded calendar days.
* **Entity Relationships**: User identities are not assumed using `card1`. The graph model explicitly maps separate nodes for payments (`CARD`), hardware configurations (`DEVICE`), and billing locations (`REGION`).
