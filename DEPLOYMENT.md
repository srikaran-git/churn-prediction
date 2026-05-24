# Deployment Guide

## Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.11+ | `python --version` |
| Docker Desktop | Latest | `docker --version` |
| Git | Any | `git --version` |

---

## First-time setup

### 1. Clone the repository
```bash
git clone https://github.com/srikaran-git/churn-prediction.git
cd churn-prediction
```

### 2. Create your environment file
```bash
cp .env.example .env.development
```

Open `.env.development` and set:
ENV=development
LOG_LEVEL=DEBUG
MODEL_PATH=models/pipeline_v1.pkl
API_HOST=0.0.0.0
API_PORT=8000

### 3. Train the model (first time only)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

This produces `models/pipeline_v1.pkl`.

---

## Running the API

### With Docker (recommended)
```bash
# Start
docker compose --env-file .env.development up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

API available at: `http://localhost:8000`
Docs available at: `http://localhost:8000/docs`

### Without Docker (local dev)
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Verifying the deployment

```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status": "ok", "model_loaded": true}

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 70.35, ...}'
```

---

## Running tests

```bash
pytest tests/ -v
```

Expected: `33 passed`

---

## CI/CD

Every push to `main` triggers the CI pipeline automatically:

- Import order check (isort)
- Formatting check (black)  
- Lint (flake8)
- Full test suite (pytest)

View results: `https://github.com/srikaran-git/churn-prediction/actions`

---

## Updating the model

Because the model is volume-mounted (not baked into the image):

```bash
python train.py                          # retrain → overwrites pipeline_v1.pkl
docker compose restart api               # picks up new model, no rebuild needed
docker compose logs | grep "Pipeline loaded"   # confirm
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ConfigurationError: Missing required environment variables` | `.env.development` missing or incomplete | `cp .env.example .env.development` and fill values |
| `curl: (52) Empty reply from server` | Container still starting | Wait 3s and retry |
| `503 Service Unavailable` on `/health` | Model failed to load | Check `docker compose logs` for pipeline errors |
| Port 8000 already in use | Another process on 8000 | `docker compose down` then retry |
