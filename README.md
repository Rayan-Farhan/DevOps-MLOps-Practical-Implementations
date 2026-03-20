# Diabetes Prediction MLOps Pipeline

A full end-to-end MLOps pipeline for diabetes prediction using FastAPI, React, Docker, DVC, and MLflow — with automated retraining, drift detection, and monitoring.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Quick Start — Local Development](#5-quick-start--local-development)
6. [Running the ML Pipeline](#6-running-the-ml-pipeline)
7. [Running Everything with Docker Compose](#7-running-everything-with-docker-compose)
8. [CI/CD Pipeline](#8-cicd-pipeline)
9. [Scheduled Retraining](#9-scheduled-retraining)
10. [Monitoring (Prometheus + Grafana)](#10-monitoring-prometheus--grafana)
11. [Drift Detection (Evidently AI)](#11-drift-detection-evidently-ai)
12. [MLflow Experiment Tracking](#12-mlflow-experiment-tracking)
13. [DVC Data Versioning](#13-dvc-data-versioning)
14. [Configuration Reference](#14-configuration-reference)
15. [API Reference](#15-api-reference)
16. [Tuning the Pipeline](#16-tuning-the-pipeline)

---

## 1. Project Overview

This project predicts whether a patient is diabetic or non-diabetic based on 8 clinical features (Pima Indians Diabetes dataset). It implements a production-grade MLOps pipeline covering:

- **Data ingestion** from local CSV or URL
- **Data validation** (schema, ranges, class balance)
- **Preprocessing** (zero-imputation, leakage-safe scaling, train/test split)
- **Model training** with 4 candidate models, 5-fold cross-validation, and optional GridSearchCV tuning
- **Experiment tracking** with MLflow (all runs, metrics, and artifacts logged)
- **Model selection and promotion** via a configurable metric threshold gate
- **MLflow Model Registry** (Staging → Production transitions)
- **REST API** serving predictions (FastAPI)
- **React frontend** for user interaction
- **Prometheus + Grafana** monitoring for API metrics
- **Evidently AI** drift detection on live prediction data
- **GitHub Actions** CI/CD with automatic model promotion and weekly retraining

---

## 2. Architecture

```
User → React Frontend (port 3000)
              │
              ▼
       FastAPI Backend (port 8000)
         ├── POST /api/predict   (ML inference + logs to SQLite)
         ├── GET  /api/health
         └── GET  /metrics       (Prometheus scrape endpoint)
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

**CI/CD flow:**
```
git push → GitHub Actions
  ├── test-and-scan  (lint, tests, security)
  ├── ml-pipeline    (dvc repro, commit new model if promoted)
  ├── docker-backend (build + push to GHCR)
  └── docker-frontend
```

**Scheduled retraining (every Sunday 02:00 UTC):**
```
drift-check (Evidently AI)
  │ drift detected?
  └─ yes → retrain (dvc repro --force) → commit model → rebuild images
```

---

## 3. Project Structure

```
.
├── app/                          FastAPI backend
│   ├── main.py                   App entry point, CORS, Prometheus
│   ├── config.py                 Settings from env vars
│   ├── api/routes.py             /predict and /health endpoints
│   ├── schemas/diabetes_schema.py  Pydantic input validation
│   ├── model/
│   │   ├── diabetes.csv          Original training dataset
│   │   └── diabetes_model.pkl    Active deployed model bundle
│   └── utils/
│       ├── metrics.py            Prometheus prediction counter
│       └── logger.py             Rotating file logger
│
├── pipeline/                     ML pipeline scripts (run via dvc repro)
│   ├── ingest.py                 Stage 1: data ingestion
│   ├── validate.py               Stage 2: data validation
│   ├── preprocess.py             Stage 3: preprocessing + scaling
│   ├── train.py                  Stage 4: training + MLflow + Registry
│   ├── evaluate.py               Stage 5: model promotion gate
│   └── drift.py                  Stage 6: Evidently drift detection
│
├── frontend/                     React + Vite frontend
│   ├── src/
│   │   ├── pages/Home.jsx
│   │   └── components/
│   │       ├── PredictionForm.jsx
│   │       └── ResultCard.jsx
│   ├── Dockerfile
│   └── nginx.conf
│
├── monitoring/
│   ├── prometheus.yml            Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/prometheus.yml
│       │   └── dashboards/dashboard.yml
│       └── dashboards/diabetes_api.json
│
├── mlflow/Dockerfile             Lightweight MLflow server image
├── data/
│   ├── raw/                      Ingested CSV (DVC-tracked)
│   └── processed/                Scaled arrays, scaler.pkl, metrics
│
├── dvc.yaml                      5-stage pipeline DAG
├── params.yaml                   All hyperparameters and thresholds
├── docker-compose.yml            Full stack orchestration
├── Dockerfile                    Backend container image
├── requirements.txt              API runtime dependencies
├── requirements-pipeline.txt     Pipeline-only dependencies
└── .github/workflows/
    ├── ci-cd.yml                 Main CI/CD pipeline
    └── retrain.yml               Scheduled weekly retraining
```

---

## 4. Prerequisites

| Tool | Minimum version |
|---|---|
| Python | 3.11 |
| Docker Desktop | 24.x |
| Node.js (frontend dev only) | 18.x |
| Git | any |

---

## 5. Quick Start — Local Development

### Step 1 — Install dependencies

```bash
# API dependencies
pip install -r requirements.txt

# Pipeline dependencies (training, DVC, MLflow, Evidently, XGBoost)
pip install -r requirements-pipeline.txt
```

### Step 2 — Initialise DVC (first time only)

```bash
dvc init
```

### Step 3 — Run the ML pipeline

```bash
dvc repro
```

This runs all 5 stages in order:

| Stage | Script | What it does |
|---|---|---|
| `ingest` | `pipeline/ingest.py` | Copies CSV to `data/raw/` |
| `validate` | `pipeline/validate.py` | Checks schema, ranges, class balance |
| `preprocess` | `pipeline/preprocess.py` | Imputes zeros, splits, fits scaler |
| `train` | `pipeline/train.py` | 4 models, CV, GridSearchCV, MLflow |
| `evaluate` | `pipeline/evaluate.py` | Gates promotion, updates `.pkl` |

DVC caches every stage. Re-running without changes takes less than 1 second.

### Step 4 — Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API: `http://localhost:8000`
Swagger docs: `http://localhost:8000/docs`

### Step 5 — Open the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLflow UI: `http://localhost:5000`

### Step 6 — Start the frontend (development)

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173`

---

## 6. Running the ML Pipeline

### Run all stages

```bash
dvc repro
```

### Force re-run all stages (ignore cache)

```bash
dvc repro --force
```

### Run a single stage

```bash
dvc repro --single-item train
dvc repro --single-item evaluate
```

### View pipeline DAG

```bash
dvc dag
```

### View metrics

```bash
dvc metrics show          # current run
dvc metrics diff          # compare vs last git commit
```

### Run stages individually

```bash
python pipeline/ingest.py
python pipeline/validate.py
python pipeline/preprocess.py
python pipeline/train.py
python pipeline/evaluate.py
python pipeline/drift.py
```

### Ingest data from a URL

```bash
python pipeline/ingest.py --source url --url https://example.com/diabetes.csv
```

---

## 7. Running Everything with Docker Compose

### Start all services

```bash
docker compose up --build -d
```

### Service URLs

| Service | URL | Default credentials |
|---|---|---|
| Frontend | http://localhost:3000 | — |
| API | http://localhost:8000 | — |
| Swagger docs | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |

### Useful Docker commands

```bash
# Stop all services
docker compose down

# Stop and wipe all volumes (database, MLflow runs, Grafana data)
docker compose down -v

# View logs
docker compose logs -f backend
docker compose logs -f mlflow

# Rebuild a single service
docker compose up --build -d backend
```

---

## 8. CI/CD Pipeline

Defined in `.github/workflows/ci-cd.yml`. Runs automatically on every push to `main` and on all pull requests.

### Job order

```
[Job 1] test-and-scan      ← always runs
    ↓
[Job 2] ml-pipeline        ← push to main only (runs dvc repro)
    ↓
[Job 3] docker-backend  ┐
[Job 4] docker-frontend ┘  ← parallel, push images to GHCR
    ↓
[Job 5] deploy             ← only if DEPLOY_ENABLED secret = 'true'
```

### Required GitHub secrets (for optional features)

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | DVC S3 remote pull |
| `AWS_SECRET_ACCESS_KEY` | DVC S3 remote pull |
| `DEPLOY_ENABLED` | Set to `'true'` to activate SSH deploy |
| `DEPLOY_SSH_HOST` | Target server hostname |
| `DEPLOY_SSH_USER` | SSH username |
| `DEPLOY_SSH_KEY` | SSH private key |
| `DEPLOY_WORKDIR` | Working directory on server |

None of these are required for CI to pass — only for deploy.

---

## 9. Scheduled Retraining

Defined in `.github/workflows/retrain.yml`. Runs every **Sunday at 02:00 UTC**.

### Flow

```
[Job 1] drift-check
  python pipeline/drift.py
  exit 0 → no drift → pipeline stops (no unnecessary retrain)
  exit 2 → drift detected
      ↓
[Job 2] retrain
  dvc repro --force
  git commit (if new model promoted)
  artifact: training_report.json
      ↓
[Job 3] rebuild-images
  push :latest to GHCR
```

### Trigger manually via GitHub UI

1. Go to **Actions** tab in your repository
2. Select **"scheduled-retrain"** workflow
3. Click **"Run workflow"**
4. Optionally set `force_retrain = true`

### Trigger manually via CLI

```bash
gh workflow run retrain.yml -f force_retrain=true
```

---

## 10. Monitoring (Prometheus + Grafana)

### Start the monitoring stack

```bash
docker compose up -d prometheus grafana
```

### Grafana dashboard

- URL: `http://localhost:3001`
- Login: `admin` / `admin`
- The **"Diabetes Prediction API"** dashboard loads automatically

**Dashboard panels:**
- Total predictions (Diabetic / Non-Diabetic counters)
- HTTP request rate by endpoint
- API latency p50, p95, p99
- Prediction rate over time (Diabetic vs Non-Diabetic)
- HTTP error rate (4xx / 5xx)

### Prometheus

- URL: `http://localhost:9090`
- Scrapes backend `/metrics` every 15 seconds

**Key metrics:**

| Metric | Description |
|---|---|
| `diabetes_predictions_total{result="Diabetic"}` | Diabetic prediction count |
| `diabetes_predictions_total{result="Non-Diabetic"}` | Non-Diabetic count |
| `http_requests_total` | Total HTTP requests |
| `http_request_duration_seconds` | Latency histogram |

### Set a custom Grafana password

```bash
GRAFANA_PASSWORD=mysecurepassword docker compose up -d grafana
```

---

## 11. Drift Detection (Evidently AI)

Drift detection compares distributions of **recent live predictions** against the **original training dataset**.

### How predictions are collected

Every `POST /api/predict` call saves the input features to:
```
data/predictions.db   (SQLite, created automatically)
```

### Run drift detection

```bash
python pipeline/drift.py
```

**Requires:** at least 50 predictions logged in `data/predictions.db`.

**Exit codes:**
- `0` — No drift (or skipped due to insufficient data)
- `2` — Drift detected → trigger `dvc repro --force`

### View drift report

```bash
cat data/processed/drift_report.json
```

Fields:
- `drift_detected` — overall boolean
- `feature_drift` — per-feature drift score and flag
- `reference_rows` / `current_rows` — dataset sizes

---

## 12. MLflow Experiment Tracking

### Open MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000
```

### What is logged per training run

| Item | Detail |
|---|---|
| Parameters | All hyperparameters from `params.yaml` |
| CV metrics | `cv_mean_f1_diabetic`, `cv_std_f1_diabetic` |
| Train metrics | accuracy, f1_diabetic, f1_macro, roc_auc |
| Artifact | Serialised sklearn/XGBoost model |
| Tag | `model_type` |

### MLflow Model Registry

When promoted, the model is registered as `diabetes-classifier` and transitioned to **Production**. All prior Production versions are archived automatically.

View at: `http://localhost:5000/#/models/diabetes-classifier`

---

## 13. DVC Data Versioning

### First-time setup

```bash
dvc init
```

### Track a data file with DVC

```bash
dvc add app/model/diabetes.csv
git add app/model/diabetes.csv.dvc .gitignore
git commit -m "track diabetes.csv with DVC"
```

### Configure S3 remote (recommended for teams)

```bash
dvc remote add -d s3remote s3://your-bucket-name/dvc-cache
dvc remote modify s3remote region us-east-1
git add .dvc/config
git commit -m "add DVC S3 remote"

dvc push   # upload data to S3
dvc pull   # download data (new machine / CI)
```

### Pipeline caching

```bash
dvc repro           # skips unchanged stages
dvc repro --force   # rerun everything
dvc dag             # visualise the DAG
dvc metrics show    # show current metrics
dvc metrics diff    # diff vs last commit
```

---

## 14. Configuration Reference

All tunable values are in `params.yaml`. No code changes needed.

```yaml
data:
  source_local_path: app/model/diabetes.csv
  min_rows: 500
  test_size: 0.2
  random_state: 42

models:
  random_forest:   { n_estimators: 200, max_depth: 8, random_state: 42 }
  logistic_regression: { C: 1.0, max_iter: 1000, random_state: 42 }
  gradient_boosting:   { n_estimators: 100, learning_rate: 0.1, max_depth: 5 }
  xgboost:             { n_estimators: 100, learning_rate: 0.1, max_depth: 5 }

evaluation:
  primary_metric: f1_diabetic     # accuracy | f1_diabetic | f1_macro | roc_auc
  min_improvement: 0.005
  promotion_threshold: 0.55

tuning:
  enabled: true
  cv_folds: 5
  param_grids:
    random_forest:  { n_estimators: [100, 200], max_depth: [6, 8, 10] }
    logistic_regression: { C: [0.1, 1.0, 10.0] }
    gradient_boosting:   { n_estimators: [100, 200], learning_rate: [0.05, 0.1] }
    xgboost:             { n_estimators: [100, 200], learning_rate: [0.05, 0.1] }

drift:
  min_prediction_rows: 50
  predictions_db: data/predictions.db
```

### Backend environment variables

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | `development` or `production` |
| `API_HOST` | `0.0.0.0` | Bind host |
| `API_PORT` | `8000` | Bind port |
| `API_PREFIX` | `/api` | URL prefix |
| `CORS_ALLOW_ALL` | `true` | Allow all origins (dev) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MODEL_PATH` | `app/model/diabetes_model.pkl` | Model bundle path |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |

---

## 15. API Reference

### Health check

```
GET /api/health
```
```json
{ "status": "ok", "message": "Diabetes Prediction API is running!" }
```

### Predict

```
POST /api/predict
Content-Type: application/json
```

Request:
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

**Field validation ranges:**

| Field | Type | Range |
|---|---|---|
| Pregnancies | int | 0–20 |
| Glucose | int | 0–300 |
| BloodPressure | int | 0–200 |
| SkinThickness | int | 0–100 |
| Insulin | int | 0–900 |
| BMI | float | 0.0–80.0 |
| DiabetesPedigreeFunction | float | 0.0–3.0 |
| Age | int | 0–120 |

### Prometheus metrics

```
GET /metrics
```

---

## 16. Tuning the Pipeline

### Change the scoring metric

```yaml
# params.yaml
evaluation:
  primary_metric: roc_auc   # options: accuracy | f1_diabetic | f1_macro | roc_auc
```

Then `dvc repro` — only `train` and `evaluate` re-run.

### Disable tuning for faster runs

```yaml
tuning:
  enabled: false
```

### Tighten the promotion gate

```yaml
evaluation:
  min_improvement: 0.02       # must improve by 2%
  promotion_threshold: 0.65   # minimum F1 of 0.65
```

### View results after a run

```bash
cat data/processed/evaluation_report.json
cat data/processed/training_report.json
dvc metrics show
```
