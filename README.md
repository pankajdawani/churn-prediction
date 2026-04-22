# E-Commerce Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Model](https://img.shields.io/badge/Model-Random%20Forest%20%2B%20SMOTE-blueviolet)

A production-grade ML pipeline that predicts which e-commerce customers are likely to churn — served via a REST API with real-time risk scoring.

> **Current model recall: 44% | ROC-AUC: 0.73**  
> Improved from baseline recall of 5% through iterative feature selection and pipeline tuning.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Development Journey](#development-journey)
- [Roadmap](#roadmap)

---

## Overview

This project solves a real business problem: identifying customers who are about to stop buying before they actually leave. The pipeline covers the full ML lifecycle — from raw data ingestion and feature engineering to model training, evaluation, and a production API.

**Key design decisions:**
- All settings in `configs/config.yaml` — no hardcoded values in code
- Structured logging to both console and rotating log files
- SMOTE oversampling to handle the 82/18 class imbalance
- Risk tiers (High / Medium / Low) on top of raw probabilities for business usability
- FastAPI for async-ready, auto-documented prediction endpoints

---

## Project Structure

```
churn-prediction/
├── configs/
│   └── config.yaml                  # Hyperparameters, paths, API settings
├── data/
│   ├── raw/                         # Source data — never modified (git-ignored)
│   └── processed/                   # Transformed data (git-ignored)
├── models/
│   └── artifacts/                   # Trained model + evaluation plots (git-ignored)
├── notebooks/
│   └── churn_prediction_eda.ipynb   # EDA, feature analysis, baseline experiments
├── src/
│   ├── main.py                      # FastAPI app — prediction endpoints
│   ├── features/
│   │   └── build_features.py        # Feature engineering + preprocessing pipeline
│   ├── models/
│   │   ├── train.py                 # End-to-end training script
│   │   └── predict.py               # Inference — single + batch
│   └── utils/
│       ├── config_loader.py         # YAML config reader
│       └── logger.py                # Rotating file + console logger
├── logs/                            # Auto-generated runtime logs
├── outputs/                         # Saved plots (confusion matrix, feature importance)
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- `brew install libomp` (Mac only — required for XGBoost)

### 1. Clone and set up environment
```bash
git clone https://github.com/pankajdawani/churn-prediction.git
cd churn-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add the dataset
```bash
# Place the raw data file in:
data/raw/ecommerce_churn.csv
```
> The file is git-ignored. Request access via the project owner.

### 3. Train the model
```bash
python3 -m src.models.train
```

Expected output:
```
INFO | Loading raw data from: data/raw/ecommerce_churn.csv
INFO | Loaded 5,630 rows × 19 columns
INFO | Running 5-fold stratified cross-validation...
INFO | CV ROC-AUC : 0.7043 ± 0.0053
INFO | Hold-out ROC-AUC : 0.7318
INFO | Model saved → models/artifacts/churn_model.joblib
```

### 4. Start the API
```bash
python3 -m uvicorn src.main:app --reload
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Deployed model metadata |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Batch predictions (max 500 records) |

### Single prediction — example

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 8,
    "city_tier": 2,
    "warehouse_to_home_km": 22.0,
    "hours_on_app": 2.1,
    "devices_registered": 4,
    "preferred_login_device": "Mobile Phone",
    "preferred_payment_mode": "UPI",
    "preferred_order_cat": "Mobile",
    "gender": "Male",
    "order_count_l6m": 4,
    "order_amount_hike_pct": 28.0,
    "coupon_used": 1,
    "day_since_last_order": 32,
    "cashback_amount": 95.0,
    "satisfaction_score": 2,
    "complain": 1,
    "number_of_address": 2
  }'
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7812,
  "risk_tier": "High",
  "latency_ms": 18.4
}
```

### Risk tiers

| Tier | Probability | Suggested Action |
|------|------------|-----------------|
| High | ≥ 45% | Trigger immediate retention offer |
| Medium | 25–45% | Flag for follow-up campaign |
| Low | < 25% | No action needed |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest + SMOTE |
| Hold-out ROC-AUC | 0.73 |
| Cross-val ROC-AUC | 0.70 ± 0.01 |
| Churn Recall | 0.44 |
| Churn F1 | 0.40 |
| Training samples | 4,504 |
| Test samples | 1,126 |
| Dataset churn rate | 18.3% |

**Top predictors (by feature importance):**
1. `satisfaction_score`
2. `complain`
3. `day_since_last_order`
4. `cashback_amount`
5. `tenure_months`

---

## Configuration

Everything is controlled from `configs/config.yaml`:

```yaml
data:
  raw_path: "data/raw/ecommerce_churn.csv"
  test_size: 0.2
  random_state: 42

features:
  target: "churn"
  drop_cols: ["customer_id", "coupon_used", "order_count_l6m",
              "warehouse_to_home_km", "hours_on_app"]

model:
  name: "RandomForest_SMOTE"
  artifact_path: "models/artifacts/churn_model.joblib"
  params:
    n_estimators: 500
    max_depth: 6
    min_samples_leaf: 8
    class_weight: "balanced"

api:
  title: "E-Commerce Churn Prediction API"
  version: "1.0.0"
```

---

## Development Journey

This section tracks key decisions and what was learned along the way — useful for anyone picking up this project.

### v0.1 — Baseline 
- Initial Random Forest with default parameters
- **Result:** Recall = 0.05 — model predicted "No Churn" for almost everyone
- **Problem identified:** Class imbalance (82% no-churn) causing accuracy paradox

### v0.2 — Hyperparameter tuning 
- Reduced `max_depth` from 15 → 8, increased `min_samples_leaf`
- Added `class_weight: balanced`
- **Result:** Recall improved to 0.18 — better but still poor

### v0.3 — Feature selection
- EDA revealed `coupon_used`, `order_count_l6m`, `hours_on_app`, `warehouse_to_home_km` had near-zero difference between churners and non-churners
- Dropped these 4 features from training
- **Result:** Recall jumped to 0.44, F1 improved from 0.08 → 0.40
- **Key learning:** Feature selection matters more than algorithm tuning when signals are weak

---

## Roadmap

- [x] EDA and feature analysis notebook
- [x] Modular feature engineering pipeline
- [x] Model training with SMOTE + cross-validation
- [x] FastAPI prediction service (single + batch)
- [x] Structured logging and config management
- [ ] Docker containerisation
- [ ] CI/CD with GitHub Actions
- [ ] Cloud deployment (AWS EC2 / GCP Cloud Run)
- [ ] Model monitoring and drift detection
