# Diabetes Prediction MLOps Pipeline

A production-grade MLOps pipeline for diabetes prediction using FastAPI, React, Docker, DVC, and MLflow — with automated retraining, drift detection, and monitoring.

## 🌐 Live Demo

[Try it now!](http://54.224.141.110:3000)

Deployed on **AWS EC2** with Docker Compose. The live server runs the full stack including the React frontend, FastAPI backend, and ML model.

---

<img width="960" height="503" alt="frontend" src="https://github.com/user-attachments/assets/11f0896c-1492-40c5-bba1-1b0267a14ec9" />

<img width="960" height="503" alt="grafana" src="https://github.com/user-attachments/assets/77e7051d-d439-48e5-b2b0-3c6ba1ba5faa" />

---

## Features

- **ML Pipeline** — DVC-managed stages: ingest → validate → preprocess → train → evaluate
- **4 Models** — Random Forest, Logistic Regression, Gradient Boosting, XGBoost with GridSearchCV
- **MLflow** — Experiment tracking, model registry, automatic promotion
- **Drift Detection** — Evidently AI monitors for data drift
- **CI/CD** — GitHub Actions with automated testing, Docker builds, and deployment
- **Monitoring** — Prometheus + Grafana dashboards

---

## Architecture

```
User → React Frontend (port 3000)
              │
              ▼
       FastAPI Backend (port 8000)
         ├── POST /api/predict   (ML inference)
         ├── GET  /api/health
         └── GET  /metrics       (Prometheus)
              │
        diabetes_model.pkl
              │
     ┌────── DVC Pipeline ───────┐
     │ ingest → validate →       │
     │ preprocess → train →      │
     │ evaluate → (drift)        │
     └───────────────────────────┘
              │
        MLflow Tracking (port 5000)
        Prometheus (port 9090)
        Grafana (port 3001)
```

---

## Quick Start

### Prerequisites

| Tool | Minimum version |
|------|-----------------|
| Python | 3.11 |
| Docker Desktop | 24.x |
| Node.js (frontend dev) | 18.x |

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-pipeline.txt

# Initialize DVC (first time only)
dvc init

# Run ML pipeline
dvc repro

# Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Start frontend (in separate terminal)
cd frontend && npm install && npm run dev
```

### Docker Compose (Full Stack)

```bash
docker compose up --build -d
```

### Service URLs

| Service | Local URL | Live URL |
|---------|-----------|----------|
| Frontend | http://localhost:3000 | http://54.224.141.110:3000 |
| API | http://localhost:8000 | http://54.224.141.110:8000 |
| Swagger Docs | http://localhost:8000/docs | http://54.224.141.110:8000/docs |
| MLflow UI | http://localhost:5000 | — |
| Grafana | http://localhost:3001 | — |
| Prometheus | http://localhost:9090 | — |

**Grafana credentials:** admin / admin

---

## Project Structure

```
├── app/                          FastAPI backend
│   ├── main.py                   App entry point
│   ├── api/routes.py             /predict and /health endpoints
│   ├── model/diabetes_model.pkl  Active deployed model
│   └── tests/                    API and model tests
│
├── pipeline/                     ML pipeline scripts
│   ├── ingest.py                 Data ingestion
│   ├── validate.py               Schema validation
│   ├── preprocess.py             Preprocessing + scaling
│   ├── train.py                  Training + MLflow logging
│   ├── evaluate.py               Model promotion gate
│   └── drift.py                  Evidently drift detection
│
├── frontend/                     React + Vite frontend
├── monitoring/                   Prometheus + Grafana config
├── data/                         Raw and processed data (DVC-tracked)
│
├── dvc.yaml                      Pipeline DAG definition
├── params.yaml                   All hyperparameters
├── docker-compose.yml            Full stack orchestration
└── .github/workflows/            CI/CD pipelines
```

---

## ML Pipeline

Run with DVC:

```bash
dvc repro              # Run all stages (cached)
dvc repro --force      # Force re-run all stages
dvc dag                # View pipeline DAG
dvc metrics show       # Show current metrics
```

| Stage | Script | Description |
|-------|--------|-------------|
| `ingest` | `pipeline/ingest.py` | Copies CSV to `data/raw/` |
| `validate` | `pipeline/validate.py` | Checks schema, ranges, class balance |
| `preprocess` | `pipeline/preprocess.py` | Imputes zeros, splits, fits scaler |
| `train` | `pipeline/train.py` | 4 models, CV, GridSearchCV, MLflow |
| `evaluate` | `pipeline/evaluate.py` | Gates promotion, updates model |

---

## API Reference

### Health Check

```
GET /api/health
→ { "status": "ok", "message": "Diabetes Prediction API is running!" }
```

### Predict

```
POST /api/predict
Content-Type: application/json
```

```json
{
  "Pregnancies": 3,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 28.5,
  "DiabetesPedigreeFunction": 0.45,
  "Age": 35
}
```

Response:
```json
{ "prediction": 0, "result": "Non-Diabetic" }
```

### Field Validation

| Field | Type | Range |
|-------|------|-------|
| Pregnancies | int | 0–20 |
| Glucose | int | 0–300 |
| BloodPressure | int | 0–200 |
| SkinThickness | int | 0–100 |
| Insulin | int | 0–900 |
| BMI | float | 0.0–80.0 |
| DiabetesPedigreeFunction | float | 0.0–3.0 |
| Age | int | 0–120 |

---

## CI/CD Pipeline

Defined in `.github/workflows/ci-cd.yml`. Runs on push to `main` and all PRs.

```
[1] test-and-scan      → Lint, tests, security scan
        ↓
[2] ml-pipeline        → dvc repro (main branch only)
        ↓
[3] docker-backend  ┐
[4] docker-frontend ┘  → Build + push to GHCR
        ↓
[5] deploy             → SSH deploy to EC2 (if enabled)
```

### Scheduled Retraining

Defined in `.github/workflows/retrain.yml`. Runs **every Sunday at 02:00 UTC**.

1. Drift check with Evidently AI
2. If drift detected → `dvc repro --force`
3. Commit new model → Rebuild Docker images

---

## Monitoring

### Prometheus Metrics

The API exposes `/metrics` for Prometheus scraping:

| Metric | Description |
|--------|-------------|
| `diabetes_predictions_total{result="Diabetic"}` | Diabetic prediction count |
| `diabetes_predictions_total{result="Non-Diabetic"}` | Non-Diabetic count |
| `http_requests_total` | Total HTTP requests |
| `http_request_duration_seconds` | Latency histogram |

### Grafana Dashboard

Pre-configured dashboard includes:
- Total predictions by outcome
- HTTP request rate by endpoint
- API latency percentiles (p50, p95, p99)
- Error rate (4xx / 5xx)

---

## Configuration

All hyperparameters in `params.yaml`:

```yaml
evaluation:
  primary_metric: f1_diabetic     # accuracy | f1_diabetic | f1_macro | roc_auc
  min_improvement: 0.005
  promotion_threshold: 0.55

tuning:
  enabled: true
  cv_folds: 5
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development` or `production` |
| `API_PORT` | `8000` | API bind port |
| `MODEL_PATH` | `app/model/diabetes_model.pkl` | Model path |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |

---

## Drift Detection

Evidently AI compares live predictions against training data:

```bash
python pipeline/drift.py
```

- Requires 50+ predictions in `data/predictions.db`
- Exit code `0` = No drift
- Exit code `2` = Drift detected → triggers retraining
