# Telco Customer Churn Prediction

![CI](https://github.com/srikaran-git/churn-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)

A **production-ready ML pipeline** that predicts customer churn for a telecom
company. Built to demonstrate the full journey from raw data to a served,
containerised REST API — not just a notebook.

---
## App Live Demo :https://srikaran-git-churn-prediction.streamlit.app
## Overview

| | |
|---|---|
| **Problem** | Identify customers likely to churn before they leave |
| **Model** | Random Forest Classifier (sklearn Pipeline) |
| **Serving** | FastAPI REST API |
| **Infra** | Docker + Docker Compose |
| **CI** | GitHub Actions (lint + test on every push) |

**Key engineering decisions:**
- Model is **never baked into the Docker image** — mounted as a volume at runtime
- Environment validated at startup — app fails fast on missing config
- Full separation of concerns: data → model → API layers

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/srikaran-git/churn-prediction.git
cd churn-prediction

# 2. Train the model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py

# 3. Run with Docker Compose
APP_ENV_FILE=.env.development docker-compose up --build

# 4. Hit the API
curl http://localhost:8000/health
```

---

## API Reference

### `GET /health`
Returns model load status.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
Predicts churn probability for a customer.

**Request:**
```json
{
  "tenure": 12,
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check",
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "PaperlessBilling": "Yes"
}
```

**Response:**
```json
{
  "churn_probability": 0.82,
  "churn_prediction": true,
  "model_version": "pipeline_v1"
}
```

---

## Development

### Run tests
```bash
pytest tests/ -v
```

### Lint & format
```bash
isort . && black . && flake8 .
```

### Train the model
```bash
python train.py
```

### Run locally (without Docker)
```bash
uvicorn src.api.app:app --reload
```

---

## Architecture
churn.csv
│
▼
data_loader.py ──► preprocessor.py ──► data_validator.py
│
▼
pipeline_builder.py
│
▼
trainer.py ──► pipeline_v1.pkl
│
┌───────────────┘
▼
model_loader.py
│
▼
FastAPI (app.py + predictor.py)
│
▼
POST /predict
---

## Project Structure
churn-prediction/
├── src/
│   ├── api/          # FastAPI app, routes, lifespan
│   ├── data/         # Loading, validation, preprocessing
│   ├── models/       # Training, evaluation, serving
│   └── utils/        # Config, logging, env validation
├── tests/            # 33 unit tests (pytest)
├── configs/          # config.yaml — no hardcoded values
├── models/           # Volume-mounted .pkl — not in image
├── .github/          # CI pipeline (lint + test)
├── Dockerfile
├── docker-compose.yml
└── DEPLOYMENT.md     # Full runbook

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md).

