"""
Inference utilities for the churn prediction model.
Loads the saved pipeline artifact and exposes a clean predict interface.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: Dict[str, Any] = {}


def load_model(artifact_path: str = "models/artifacts/churn_model.joblib"):
    """
    Load the serialized model pipeline. Results are cached in memory.

    Args:
        artifact_path: Path to the .joblib model artifact.

    Returns:
        Loaded sklearn / imblearn Pipeline.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    if artifact_path in _MODEL_CACHE:
        return _MODEL_CACHE[artifact_path]

    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{artifact_path}'. "
            "Run `python -m src.models.train` first."
        )

    model = joblib.load(path)
    _MODEL_CACHE[artifact_path] = model
    logger.info(f"Model loaded from {artifact_path}")
    return model


def predict_single(features: Dict[str, Any], artifact_path: str = "models/artifacts/churn_model.joblib") -> Dict[str, Any]:
    """
    Generate churn prediction for a single customer record.

    Args:
        features: Dictionary mapping feature names to values.
        artifact_path: Path to the model artifact.

    Returns:
        Dictionary with:
            - churn_prediction (int): 1 = churner, 0 = retained
            - churn_probability (float): probability of churn
            - risk_tier (str): "High" / "Medium" / "Low"
    """
    model = load_model(artifact_path)
    df = pd.DataFrame([features])

    # Apply same engineered features as training
    df["recency_engagement"] = df["day_since_last_order"] / (df["hours_on_app"] + 1)
    df["coupon_order_ratio"] = df["coupon_used"] / (df["order_count_l6m"] + 1)
    df["high_value_flag"] = (df["cashback_amount"] > 180).astype(int)   # approx training median
    df["multi_device_flag"] = (df["devices_registered"] >= 3).astype(int)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    risk_tier = _get_risk_tier(probability)

    logger.info(f"Prediction: {prediction} | Probability: {probability:.4f} | Tier: {risk_tier}")

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_tier": risk_tier,
    }


def predict_batch(records: List[Dict[str, Any]], artifact_path: str = "models/artifacts/churn_model.joblib") -> List[Dict[str, Any]]:
    """
    Generate churn predictions for a list of customer records.

    Args:
        records: List of feature dictionaries.
        artifact_path: Path to the model artifact.

    Returns:
        List of prediction dictionaries (same schema as predict_single).
    """
    model = load_model(artifact_path)
    df = pd.DataFrame(records)

    df["recency_engagement"] = df["day_since_last_order"] / (df["hours_on_app"] + 1)
    df["coupon_order_ratio"] = df["coupon_used"] / (df["order_count_l6m"] + 1)
    df["high_value_flag"] = (df["cashback_amount"] > 180).astype(int)
    df["multi_device_flag"] = (df["devices_registered"] >= 3).astype(int)

    predictions = model.predict(df).tolist()
    probabilities = model.predict_proba(df)[:, 1].tolist()

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "churn_prediction": int(pred),
            "churn_probability": round(float(prob), 4),
            "risk_tier": _get_risk_tier(float(prob)),
        })

    logger.info(f"Batch prediction complete: {len(results)} records")
    return results


def _get_risk_tier(probability: float) -> str:
    """Map churn probability to a human-readable risk tier."""
    if probability >= 0.45:
        return "High"
    elif probability >= 0.25:
        return "Medium"
    return "Low"
