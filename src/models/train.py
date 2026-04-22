"""
Model training pipeline for e-commerce churn prediction.
Trains a Random Forest classifier with SMOTE oversampling,
evaluates on hold-out set, and persists the model artifact.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt

from src.features.build_features import (
    load_raw_data, validate_data, engineer_features,
    build_preprocessor, split_features_target,
)
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from xgboost import XGBClassifier


logger = get_logger(__name__)


def train(config_path: str = "configs/config.yaml") -> dict:
    """
    Full training pipeline:
        1. Load & validate raw data
        2. Feature engineering
        3. Train / test split
        4. Build preprocessing + SMOTE + RF pipeline
        5. Cross-validation
        6. Final evaluation on hold-out set
        7. Save model artifact

    Args:
        config_path: Path to YAML configuration.

    Returns:
        Dictionary of evaluation metrics.
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]

    # ── 1. Load & validate ──────────────────────────────────────────────────
    df = load_raw_data(data_cfg["raw_path"])
    all_required = feat_cfg["numeric"] + feat_cfg["categorical"] + [feat_cfg["target"]]
    validate_data(df, all_required)

    # ── 2. Feature engineering ───────────────────────────────────────────────
    df = engineer_features(df)

    # Add engineered features to numeric list
    engineered_numeric = [
        "recency_engagement", "coupon_order_ratio",
        "high_value_flag", "multi_device_flag",
    ]
    numeric_features = feat_cfg["numeric"] + engineered_numeric
    categorical_features = feat_cfg["categorical"]

    X, y = split_features_target(df, feat_cfg["target"], feat_cfg["drop_cols"])

    # ── 3. Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 4. Pipeline: Preprocessor → SMOTE → RandomForest ────────────────────
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    rf_params = model_cfg["params"]
    model_pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=data_cfg["random_state"])),
        #("classifier", XGBClassifier(**rf_params)),
        ("classifier", RandomForestClassifier(**rf_params)),
    ])

    # ── 5. Cross-validation ──────────────────────────────────────────────────
    logger.info("Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=data_cfg["random_state"])
    cv_results = cross_validate(
        model_pipeline, X_train, y_train, cv=cv,
        scoring=["roc_auc", "average_precision", "f1"],
        n_jobs=-1,
    )
    logger.info(
        f"CV ROC-AUC : {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}\n"
        f"CV Avg-Prec: {cv_results['test_average_precision'].mean():.4f} ± {cv_results['test_average_precision'].std():.4f}\n"
        f"CV F1      : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}"
    )

    # ── 6. Final fit & evaluation ────────────────────────────────────────────
    logger.info("Fitting final model on full training set...")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    logger.info(f"\n{'='*55}")
    logger.info(f"Hold-out ROC-AUC         : {roc_auc:.4f}")
    logger.info(f"Hold-out Avg Precision   : {avg_prec:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    # Confusion matrix plot
    _save_confusion_matrix(y_test, y_pred)

    # Feature importance plot
    _save_feature_importance(model_pipeline, numeric_features, categorical_features)

    # ── 7. Save artifact ─────────────────────────────────────────────────────
    artifact_path = Path(model_cfg["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, artifact_path)
    logger.info(f"Model saved → {artifact_path}")

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "avg_precision": round(avg_prec, 4),
        "cv_roc_auc_mean": round(cv_results["test_roc_auc"].mean(), 4),
        "cv_roc_auc_std": round(cv_results["test_roc_auc"].std(), 4),
    }
    return metrics


def _save_confusion_matrix(y_true, y_pred, out_dir: str = "models/artifacts") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix – Hold-out Set", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {out_dir}/confusion_matrix.png")


def _save_feature_importance(pipeline, numeric_features, categorical_features, out_dir: str = "models/artifacts") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rf = pipeline.named_steps["classifier"]
    ohe_cats = pipeline.named_steps["preprocessor"] \
        .named_transformers_["cat"] \
        .named_steps["encoder"] \
        .get_feature_names_out(categorical_features).tolist()

    all_features = numeric_features + ohe_cats
    importances = rf.feature_importances_

    fi_df = (
        pd.DataFrame({"feature": all_features, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#1f77b4")
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
    ax.set_title("Top 20 Feature Importances", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=150)
    plt.close()
    logger.info(f"Feature importance plot saved → {out_dir}/feature_importance.png")


if __name__ == "__main__":
    metrics = train()
    print("\nFinal Metrics:", metrics)
