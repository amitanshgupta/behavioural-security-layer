# Behavioral Security Layer
### A Privacy-Preserving, Explainable & Context-Aware Behavioral Anomaly Detection System

6th Semester Interdisciplinary Project (IDP)  
Domain: Cybersecurity | Behavioral Analytics | Explainable AI  
Academic Year: 2025–2026

---

## Project Overview

This project implements a 5-layer indigenous security framework for detecting
malicious insider activity, unauthorized access, and behavioral anomalies in
real time — without compromising user privacy.

### Four Novel Contributions

| # | Contribution | Status |
|---|---|---|
| NC1 | Context-Aware Dynamic Baselining (role-based peer groups) | ✅ |
| NC2 | SHAP-Powered Explainability Dashboard (NLG alerts) | ✅ |
| NC3 | Privacy-Preserving Federated Architecture (FedAvg + DP) | ✅ |
| NC4 | GenAI Behavioral Threat Detection Module | ✅ |

---

## Results Summary

### Network IDS (NSL-KDD + CICIDS2017)
| Metric | Target | Achieved |
|--------|--------|----------|
| F1 Score | > 0.97 | **0.9936** |
| False Positive Rate | < 8% | **0.29%** |
| ROC-AUC | maximize | **0.9998** |
| Zero-day Detection | > 80% | **99.6%** |

### Insider Threat (CERT r6.2)
| Metric | Value |
|--------|-------|
| F1 Score | 0.7496 |
| ROC-AUC | 0.9908 |
| FPR | 5.6% |

### Federated Learning
| Metric | Target | Achieved |
|--------|--------|----------|
| Fed vs Central gap | < 3% | **1.96%** |
| Privacy budget (ε) | ≤ 0.5 | **0.119** |
| Rounds to converge | — | **2** |

### Ablation Study
| Component | F1 Impact |
|-----------|-----------|
| NC1 Peer features added | +4.10% |
| NC4 GenAI features added | +1.79% |

---

## Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| NSL-KDD | 148,517 rows | Network intrusion baseline |
| CICIDS2017 | 2,830,743 rows | Modern network attacks (15 types) |
| CERT Insider Threat r6.2 | 1,394,010 user-days | Insider threat detection |

---

## Project Structure
behavioral-security-layer/
├── ingestion/          # Dataset loaders (NSL-KDD, CICIDS, CERT)
├── preprocessing/      # Cleaning, validation, CERT cleaning
├── feature_engineering/# 68 engineered features + CERT features
├── context_engine/     # Clustering, baselines, drift, context adjustment
├── models/             # IF, LightGBM, BiLSTM, GenAI detector, ensemble
├── explainability/     # SHAP, LIME, alert generator
├── federated/          # FedAvg, differential privacy, FL server/client
├── evaluation/         # Metrics, experiments, ablation study
├── outputs/            # SHAP plots, LIME plots, alerts, metrics
├── notebooks/          # BiLSTM Colab training notebook
├── data/
│   ├── raw/            # Original datasets
│   ├── interim/        # Cleaned intermediate files
│   ├── processed/      # Feature-engineered datasets
│   └── metadata/       # Saved models, scalers, baselines
├── main.py             # Pipeline entry point
└── requirements.txt

---

## Setup

```bash
# Clone and create virtual environment
git clone <repo>
cd behavioral-security-layer
python -m venv venv

# Windows
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# Run a specific phase
python main.py --phase 1   # Data pipeline
python main.py --phase 2   # Context engine
python main.py --phase 3   # Model training
python main.py --phase 4   # Explainability
python main.py --phase 5   # Federated learning
python main.py --phase 6   # Evaluation
python main.py --phase 7   # CERT pipeline
python main.py --phase 8   # CERT models

# Run from a specific phase to end
python main.py --from-phase 3

# Run full pipeline
python main.py
```

---

## Key Files

| File | Description |
|------|-------------|
| `models/network_ids_model.py` | Best model — F1=0.994 on NSL-KDD+CICIDS |
| `models/cert_model.py` | Insider threat model — AUC=0.991 |
| `explainability/alert_generator.py` | SHAP+LIME alert generation |
| `federated/server.py` | Federated learning orchestration |
| `evaluation/ablation.py` | Novel contribution validation |
| `notebooks/bilstm_cert_colab.ipynb` | BiLSTM GPU training (Colab) |

---

## Technology Stack

| Category | Tool |
|----------|------|
| Language | Python 3.11 |
| ML | LightGBM, scikit-learn, PyTorch |
| Explainability | SHAP, LIME |
| Federated Learning | Custom FedAvg + Gaussian DP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |

---

## Academic Targets vs Achieved

| Metric | Proposal Target | Achieved |
|--------|----------------|----------|
| Detection Accuracy | > 97% | 99.4% ✅ |
| False Positive Rate | < 8% | 0.29% ✅ |
| Explainability Latency | < 100ms | < 50ms ✅ |
| Federated vs Central gap | < 3% | 1.96% ✅ |
| Privacy budget ε | ≤ 0.5 | 0.119 ✅ |
| Zero-day Detection | > 80% | 99.6% ✅ |
Save both files. Then clean up:
powershellRemove-Item temp_eval.py

Final Project Checkpoint ✅
Core Pipeline     — Phases 1-6 complete
CERT Integration  — 1.39M user-days, AUC=0.991
CICIDS2017        — 2.83M rows, 15 attack types
Network IDS       — F1=0.994, FPR=0.3%, Zero-day=99.6%
GenAI Detector    — Retrained on real http.csv, AUC=0.9996
Federated         — gap=1.96%, ε=0.119
Explainability    — SHAP+LIME+NLG alerts working
requirements.txt  — complete
README.md         — complete
Colab notebook    — ready for BiLSTM GPU training