"""
FastAPI application for the E-Commerce Churn Prediction service.

Endpoints:
    GET  /health          – liveness probe
    GET  /model/info      – model metadata
    POST /predict         – single customer prediction
    POST /predict/batch   – batch prediction (up to 500 records)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import time

from src.models.predict import predict_single, predict_batch, load_model
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)
cfg = load_config()

app = FastAPI(
    title=cfg["api"]["title"],
    version=cfg["api"]["version"],
    description=(
        "Production API for predicting customer churn in an e-commerce platform. "
        "Returns churn probability and risk tier per customer."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Pydantic schemas ─────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, description="Months since customer joined")
    city_tier: int = Field(..., ge=1, le=3, description="City tier: 1 (metro), 2 (tier-2), 3 (tier-3)")
    warehouse_to_home_km: float = Field(..., ge=0, description="Distance from warehouse to customer home")
    hours_on_app: float = Field(..., ge=0, le=24, description="Average hours on app per day")
    devices_registered: int = Field(..., ge=1, le=10, description="Number of registered devices")
    preferred_login_device: str = Field(..., description="Mobile Phone | Computer | Tablet")
    preferred_payment_mode: str = Field(..., description="Debit Card | Credit Card | E Wallet | UPI | Cash on Delivery")
    preferred_order_cat: str = Field(..., description="Laptop & Accessory | Mobile | Fashion | Grocery | Others")
    gender: str = Field(..., description="Male | Female")
    order_count_l6m: int = Field(..., ge=0, description="Number of orders in last 6 months")
    order_amount_hike_pct: float = Field(..., ge=0, description="% increase in order amount vs last year")
    coupon_used: int = Field(..., ge=0, description="Coupons used in last month")
    day_since_last_order: int = Field(..., ge=0, description="Days since last order placed")
    cashback_amount: float = Field(..., ge=0, description="Average cashback received (INR)")
    satisfaction_score: int = Field(..., ge=1, le=5, description="Customer satisfaction score (1–5)")
    complain: int = Field(..., ge=0, le=1, description="1 if complaint raised in last month")
    number_of_address: int = Field(..., ge=1, description="Number of saved delivery addresses")

    @field_validator("preferred_login_device")
    @classmethod
    def validate_login_device(cls, v):
        allowed = {"Mobile Phone", "Computer", "Tablet"}
        if v not in allowed:
            raise ValueError(f"preferred_login_device must be one of {allowed}")
        return v

    @field_validator("preferred_payment_mode")
    @classmethod
    def validate_payment(cls, v):
        allowed = {"Debit Card", "Credit Card", "E Wallet", "UPI", "Cash on Delivery"}
        if v not in allowed:
            raise ValueError(f"preferred_payment_mode must be one of {allowed}")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in {"Male", "Female"}:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "tenure_months": 12,
            "city_tier": 2,
            "warehouse_to_home_km": 18.5,
            "hours_on_app": 3.2,
            "devices_registered": 3,
            "preferred_login_device": "Mobile Phone",
            "preferred_payment_mode": "UPI",
            "preferred_order_cat": "Mobile",
            "gender": "Male",
            "order_count_l6m": 6,
            "order_amount_hike_pct": 22.0,
            "coupon_used": 2,
            "day_since_last_order": 25,
            "cashback_amount": 145.50,
            "satisfaction_score": 2,
            "complain": 1,
            "number_of_address": 3,
        }
    }}


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_tier: str
    latency_ms: float


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., max_length=500)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]
    latency_ms: float


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load the model on startup to avoid cold-start latency."""
    try:
        load_model(cfg["model"]["artifact_path"])
        logger.info("Model pre-loaded successfully on startup.")
    except FileNotFoundError as e:
        logger.warning(str(e))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Operations"])
def health_check():
    """Liveness probe for load balancers and container orchestrators."""
    return {"status": "healthy", "service": cfg["api"]["title"]}


@app.get("/model/info", tags=["Operations"])
def model_info():
    """Return metadata about the deployed model."""
    return {
        "model_name": cfg["model"]["name"],
        "artifact_path": cfg["model"]["artifact_path"],
        "api_version": cfg["api"]["version"],
        "hyperparameters": cfg["model"]["params"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Returns churn prediction (0/1), probability (0–1), and risk tier (Low/Medium/High).
    """
    t0 = time.perf_counter()
    try:
        result = predict_single(
            features=customer.model_dump(),
            artifact_path=cfg["model"]["artifact_path"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    latency = round((time.perf_counter() - t0) * 1000, 2)
    return {**result, "latency_ms": latency}


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch_endpoint(request: BatchRequest):
    """
    Predict churn for a batch of customers (max 500 per request).
    """
    t0 = time.perf_counter()
    records = [c.model_dump() for c in request.customers]
    try:
        results = predict_batch(
            records=records,
            artifact_path=cfg["model"]["artifact_path"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal batch prediction error.")

    latency = round((time.perf_counter() - t0) * 1000, 2)
    predictions = [{**r, "latency_ms": latency / len(results)} for r in results]
    return {"count": len(predictions), "predictions": predictions, "latency_ms": latency}
