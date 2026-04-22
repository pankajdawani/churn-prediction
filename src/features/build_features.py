"""
Feature engineering pipeline for e-commerce churn prediction.
Handles preprocessing of numeric and categorical features.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV data and perform basic type enforcement."""
    logger.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Validate that required columns exist and log missing value stats.

    Args:
        df: Input dataframe.
        required_cols: List of column names that must be present.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    null_pct = df.isnull().mean()
    cols_with_nulls = null_pct[null_pct > 0]
    if not cols_with_nulls.empty:
        logger.warning(f"Columns with missing values:\n{cols_with_nulls.to_string()}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed derived features.

    New features:
        - recency_engagement: interaction between last order day and app hours
        - coupon_order_ratio: coupons used per order placed
        - high_value_flag: customers with above-median cashback
        - multi_device_flag: registered 3+ devices
    """
    df = df.copy()

    df["recency_engagement"] = df["day_since_last_order"] / (df["hours_on_app"] + 1)
    df["coupon_order_ratio"] = df["coupon_used"] / (df["order_count_l6m"] + 1)
    df["high_value_flag"] = (df["cashback_amount"] > df["cashback_amount"].median()).astype(int)
    df["multi_device_flag"] = (df["devices_registered"] >= 3).astype(int)

    logger.info("Feature engineering complete. New features: recency_engagement, "
                "coupon_order_ratio, high_value_flag, multi_device_flag")
    return df


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that handles:
        - Numeric: median imputation + standard scaling
        - Categorical: most-frequent imputation + one-hot encoding

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ], remainder="drop")

    return preprocessor


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features from target variable.

    Args:
        df: Full dataframe including target.
        target_col: Name of the target column.
        drop_cols: Columns to drop (e.g. IDs).

    Returns:
        Tuple of (X, y).
    """
    cols_to_drop = [c for c in drop_cols + [target_col] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    logger.info(f"Features shape: {X.shape}, Target distribution:\n{y.value_counts().to_string()}")
    return X, y
