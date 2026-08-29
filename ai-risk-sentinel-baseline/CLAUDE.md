# Workspace Commands & Guidelines

Please refer to [AGENTS.md](file:///c:/Users/BIT/Downloads/Razorpay_Build/AGENTS.md) for the full agent guidelines and styling rules.

## Security Boundary
* **DO NOT read, inspect, or output `kaggle.json`:** Under no circumstances should an AI agent attempt to read, print, or edit the `kaggle.json` file. It contains private API authentication tokens.

## Primary Commands
* **Acquire Dataset (Kagglehub):** `.venv\Scripts\python data-pipeline\pipeline\download.py`
* **Preprocess & Structure SQL Database:** `.venv\Scripts\python data-pipeline\pipeline\preprocess.py`
* **Train and Freeze Models:** `.venv\Scripts\python data-pipeline\research\train_final_models.py`
* **Launch Inference Server:** `.venv\Scripts\python data-pipeline\app\sentinel_app.py`



