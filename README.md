# 🛒 E-Commerce Customer Churn Prediction

Production-grade machine learning pipeline to predict customer churn for an e-commerce platform.  
Stack: **Python · Scikit-learn · SMOTE · FastAPI · Random Forest**

---

## 📁 Project Structure

```
churn_prediction/
├── configs/
│   └── config.yaml              # All hyperparameters & paths in one place
├── data/
│   ├── raw/                     # Original data (git-ignored)
│   └── processed/               # Cleaned/transformed data (git-ignored)
├── models/
│   └── artifacts/               # Saved model .joblib files (git-ignored)
├── notebooks/
│   └── churn_prediction_eda.ipynb  # Full EDA + training walkthrough
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── features/
│   │   └── build_features.py    # Feature engineering pipeline
│   ├── models/
│   │   ├── train.py             # Training script
│   │   └── predict.py           # Inference utilities
│   └── utils/
│       ├── config_loader.py     # YAML config loader
│       └── logger.py            # Centralised logging
├── tests/
│   └── test_api.py              # API integration tests
├── .gitignore
├── Makefile
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Clone & set up environment
```bash
git clone https://github.com/<your-username>/churn-prediction.git
cd churn-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the dataset
Place `ecommerce_churn.csv` in `data/raw/`.  
*(The file is git-ignored; share it separately via Drive/S3.)*

### 3. Train the model
```bash
make train
# or: python -m src.models.train
```

### 4. Run the API
```bash
make run
# or: uvicorn src.api.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 5. Run tests
```bash
make test
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch prediction (up to 500) |

### Example request
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

### Example response
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7812,
  "risk_tier": "High",
  "latency_ms": 18.4
}
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest + SMOTE |
| Hold-out ROC-AUC | ~0.72 |
| Cross-val ROC-AUC | 0.70 ± 0.01 |
| Dataset size | 5,630 customers |
| Churn rate | ~18% |

**Top predictors:** `satisfaction_score`, `complain`, `day_since_last_order`, `tenure_months`, `cashback_amount`

---

## 🔧 Configuration

All settings live in `configs/config.yaml` — no hardcoded values anywhere in the codebase.

```yaml
model:
  params:
    n_estimators: 300
    max_depth: 12
    class_weight: balanced
```

---

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Commit your changes: `git commit -m "feat: add my feature"`
3. Push and open a PR: `git push origin feature/my-feature`
