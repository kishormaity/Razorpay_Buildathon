# AI Risk Manager Dataset Workbench

This directory houses the data pipeline scripts to acquire, sample, and preprocess the **IEEE-CIS Fraud Detection** dataset from Kaggle. 

This dataset is chosen because it is the industry standard for payment fraud and network-based abuse detection. It consists of two complementary segments:
1. **Transaction Telemetry:** Amount, card details, email domain, address matches, and distance velocity metrics.
2. **Identity Attributes:** Device fingerprint details, OS type, IP subnets, browser versions, and screen parameters (crucial for linking coordinated abuse rings).

---

## Directory Structure

```text
dataset/
├── README.md               # Pipeline documentation & instructions
├── download.py             # Script to download raw CSV files via Kaggle API
├── preprocess.py           # Preprocesses, samples, and exports to SQLite database
├── schema.sql              # Database schema matching our frontend data model
└── data/                   # Git-ignored data repository
    ├── raw/                # Houses the raw zip files and full CSVs (~2.5 GB)
    └── processed/          # Houses clean, sampled data and SQLite database (~40 MB)
```

---

## Prerequisites & Setup

To execute the data scripts, you need Python 3 installed with the required libraries.

### 1. Install Dependencies
Run the following command to install the required packages:
```bash
pip install kaggle pandas numpy
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
python download.py
```

### Step 2: Preprocess and Structure the Data
Because the raw dataset is very large (~500k transactions, >2.5 GB on disk), we need to clean and sample it. The preprocessing script does the following:
1. Loads the Transaction and Identity files.
2. Performs balanced sampling (retaining all fraud cases, paired with a matching subset of non-fraud records to mitigate class imbalance).
3. Extracts entities (Users, Devices, IPs, Payments) and builds the graph relationships table (shared device logins, shared IP addresses, card binding links).
4. Creates a clean SQLite database at `data/processed/risk_sentinel.db` which is fully structured according to our frontend data models.

Run the preprocessing script:
```bash
python preprocess.py
```
