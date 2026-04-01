"""XGBoost training wrapper with MLflow integration."""

import logging

import mlflow
import mlflow.xgboost  # pyright: ignore[reportPrivateImportUsage]
import polars as pl
import xgboost as xgb
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class XGBoostConfig(BaseModel):
    """Hyperparameters for XGBoost regression."""

    objective: str = "reg:squarederror"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "hist"
    early_stopping_rounds: int = 50


def train_xgboost(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_val: pl.DataFrame,
    y_val: pl.Series,
    config: XGBoostConfig | None = None,
) -> xgb.XGBRegressor:
    """Train a single XGBoost model with early stopping.

    Logs to MLflow if an active run exists. Passes Polars DataFrames
    directly (XGBoost >=2.0 supports this).

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        config: Hyperparameters. Uses defaults if None.

    Returns:
        Fitted XGBRegressor.
    """
    if config is None:
        config = XGBoostConfig()

    model = xgb.XGBRegressor(**config.model_dump())
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    logger.info(
        "XGBoost trained: best_iteration=%d, best_score=%.4f",
        model.best_iteration,
        model.best_score,
    )

    # Log to MLflow if active run
    if mlflow.active_run():
        mlflow.log_params(config.model_dump())
        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.xgboost.log_model(xgb_model=model, name="model")  # pyright: ignore[reportPrivateImportUsage]

    return model


def train_multi_horizon(
    X_train: pl.DataFrame,
    y_trains: dict[int, pl.Series],
    X_val: pl.DataFrame,
    y_vals: dict[int, pl.Series],
    config: XGBoostConfig | None = None,
) -> dict[int, xgb.XGBRegressor]:
    """Train one XGBoost model per forecast horizon with MLflow nested runs.

    Args:
        X_train: Training features (shared across horizons).
        y_trains: Dict mapping horizon -> training target series.
        X_val: Validation features (shared across horizons).
        y_vals: Dict mapping horizon -> validation target series.
        config: Hyperparameters. Uses defaults if None.

    Returns:
        Dict mapping horizon -> fitted XGBRegressor.
    """
    models: dict[int, xgb.XGBRegressor] = {}

    for h in sorted(y_trains.keys()):
        logger.info("Training horizon h=%d (%d min ahead)", h, h * 10)

        if mlflow.active_run():
            with mlflow.start_run(run_name=f"horizon-{h}", nested=True):
                mlflow.log_params({"horizon_steps": h, "horizon_minutes": h * 10})
                model = train_xgboost(X_train, y_trains[h], X_val, y_vals[h], config)
                models[h] = model
        else:
            model = train_xgboost(X_train, y_trains[h], X_val, y_vals[h], config)
            models[h] = model

    return models
