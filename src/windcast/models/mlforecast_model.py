"""MLForecast (Nixtla) training wrapper with MLflow integration.

Wraps mlforecast for multi-domain energy forecasting with automatic
lag/rolling feature generation and recursive/direct multi-step prediction.
"""

import logging

import mlflow
import polars as pl
import xgboost as xgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MLForecastConfig(BaseModel):
    """Configuration for mlforecast training."""

    # XGBoost hyperparameters (passed to XGBRegressor)
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    # mlforecast settings
    strategy: str = "sparse_direct"  # "recursive" | "direct" | "sparse_direct"
    n_cv_windows: int = 3


DOMAIN_MLFORECAST: dict[str, dict[str, object]] = {
    "wind": {
        "target": "active_power_kw",
        "group": "turbine_id",
        "freq": "10m",
        "lags": [1, 2, 3, 6, 12, 24],
        "rolling_windows": [6, 12, 24],
    },
    "demand": {
        "target": "load_mw",
        "group": "zone_id",
        "freq": "1h",
        "lags": [1, 2, 24, 168],
        "rolling_windows": [24, 168],
    },
    "solar": {
        "target": "power_kw",
        "group": "system_id",
        "freq": "15m",
        "lags": [1, 2, 4, 8, 96],
        "rolling_windows": [4, 16, 96],
    },
}


def prepare_mlforecast_df(df: pl.DataFrame, domain: str) -> pl.DataFrame:
    """Rename columns to mlforecast convention (unique_id, ds, y).

    Keeps exogenous feature columns, drops raw/non-feature columns.

    Args:
        df: DataFrame with domain-specific column names.
        domain: One of "wind", "demand", "solar".

    Returns:
        DataFrame with unique_id, ds, y + exogenous columns.
    """
    dcfg = DOMAIN_MLFORECAST[domain]
    group_col = str(dcfg["group"])
    target_col = str(dcfg["target"])

    # Columns to drop (raw data, not features)
    drop_cols = {
        "qc_flag",
        "status_code",
        "raw_active_power_kw",
        "raw_wind_speed",
        "raw_wind_direction",
        "pitch_angle_avg_deg",
        "nacelle_direction_deg",
        "rotor_speed_rpm",
        "ambient_temp_c_raw",
        "generator_speed_rpm",
        "generator_bearing_temp_c",
        "yaw_error_deg",
        "blade_pitch_std_deg",
    }

    # Rename required columns
    rename_map = {
        group_col: "unique_id",
        "timestamp_utc": "ds",
        target_col: "y",
    }

    # Select columns to keep: renamed + exogenous (anything not in drop_cols and not renamed)
    existing_drop = drop_cols & set(df.columns)
    df_clean = df.drop(list(existing_drop))

    df_renamed = df_clean.rename(rename_map)

    # Cast unique_id to string for mlforecast
    df_renamed = df_renamed.with_columns(pl.col("unique_id").cast(pl.Utf8))

    logger.info(
        "Prepared mlforecast DataFrame: %d rows, %d columns (domain=%s)",
        len(df_renamed),
        len(df_renamed.columns),
        domain,
    )
    return df_renamed


def create_mlforecast(
    domain: str,
    config: MLForecastConfig | None = None,
    horizons: list[int] | None = None,
) -> MLForecast:
    """Create an MLForecast instance configured for the given domain.

    Args:
        domain: One of "wind", "demand", "solar".
        config: Hyperparameters. Uses defaults if None.
        horizons: Forecast horizons in steps (for sparse direct strategy info only).

    Returns:
        Configured MLForecast instance (not yet fitted).
    """
    if config is None:
        config = MLForecastConfig()

    dcfg = DOMAIN_MLFORECAST[domain]
    lags: list[int] = dcfg["lags"]  # type: ignore[assignment]
    rolling_windows: list[int] = dcfg["rolling_windows"]  # type: ignore[assignment]
    freq: str = dcfg["freq"]  # type: ignore[assignment]

    # Build lag transforms: attach rolling mean/std to lag=1
    lag_transforms_list: list[RollingMean | RollingStd] = []
    for w in rolling_windows:
        lag_transforms_list.extend(
            [
                RollingMean(window_size=w),
                RollingStd(window_size=w),
            ]
        )
    lag_transforms = {1: lag_transforms_list}

    model = xgb.XGBRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        tree_method="hist",
    )

    fcst = MLForecast(
        models={"xgb": model},
        freq=freq,
        lags=lags,
        lag_transforms=lag_transforms,  # pyright: ignore[reportArgumentType]
    )

    logger.info(
        "Created MLForecast: domain=%s, freq=%s, lags=%s, rolling_windows=%s",
        domain,
        freq,
        lags,
        rolling_windows,
    )
    return fcst


def train_mlforecast(
    df: pl.DataFrame,
    domain: str,
    config: MLForecastConfig | None = None,
    horizons: list[int] | None = None,
) -> MLForecast:
    """Train an MLForecast model on the given data.

    Args:
        df: DataFrame in mlforecast format (unique_id, ds, y + exogenous).
        domain: One of "wind", "demand", "solar".
        config: Hyperparameters. Uses defaults if None.
        horizons: Forecast horizons in steps. Default: [1, 6, 12, 24, 48].

    Returns:
        Fitted MLForecast instance.
    """
    if config is None:
        config = MLForecastConfig()
    if horizons is None:
        horizons = [1, 6, 12, 24, 48]

    fcst = create_mlforecast(domain, config, horizons)

    logger.info(
        "Training mlforecast: strategy=%s, horizons=%s, n_rows=%d",
        config.strategy,
        horizons,
        len(df),
    )

    # static_features=[] tells mlforecast all extra columns are dynamic exogenous
    fit_kwargs: dict[str, object] = {
        "id_col": "unique_id",
        "time_col": "ds",
        "target_col": "y",
        "static_features": [],
    }

    if config.strategy == "sparse_direct":
        fit_kwargs["horizons"] = horizons
    elif config.strategy == "direct":
        fit_kwargs["max_horizon"] = max(horizons)
    # recursive: no extra kwargs needed

    fcst.fit(df, **fit_kwargs)  # type: ignore[arg-type]

    # Log to MLflow if active run
    if mlflow.active_run():
        mlflow.log_params(config.model_dump())
        mlflow.log_params(
            {
                "domain": domain,
                "horizons": str(horizons),
                "strategy": config.strategy,
                "n_train_rows": len(df),
            }
        )

    logger.info("MLForecast training complete")
    return fcst


def predict_mlforecast(
    fcst: MLForecast,
    h: int,
    X_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Generate predictions from a fitted MLForecast.

    Args:
        fcst: Fitted MLForecast instance.
        h: Number of steps to predict.
        X_df: Future exogenous values (if used in training).

    Returns:
        DataFrame with columns: unique_id, ds, xgb.
    """
    preds = fcst.predict(h=h, X_df=X_df)
    logger.info("MLForecast predictions: %d rows", len(preds))
    return preds


def cross_validate_mlforecast(
    df: pl.DataFrame,
    domain: str,
    config: MLForecastConfig | None = None,
    horizons: list[int] | None = None,
    n_windows: int | None = None,
) -> pl.DataFrame:
    """Run temporal cross-validation with MLForecast.

    Args:
        df: DataFrame in mlforecast format (unique_id, ds, y + exogenous).
        domain: One of "wind", "demand", "solar".
        config: Hyperparameters. Uses defaults if None.
        horizons: Forecast horizons in steps. Default: [1, 6, 12, 24, 48].
        n_windows: Number of CV windows. Default: from config.

    Returns:
        DataFrame with columns: unique_id, ds, cutoff, y, xgb.
    """
    if config is None:
        config = MLForecastConfig()
    if horizons is None:
        horizons = [1, 6, 12, 24, 48]
    if n_windows is None:
        n_windows = config.n_cv_windows

    fcst = create_mlforecast(domain, config, horizons)

    h = max(horizons)
    logger.info(
        "Cross-validating mlforecast: h=%d, n_windows=%d, strategy=%s",
        h,
        n_windows,
        config.strategy,
    )

    fit_kwargs: dict[str, object] = {
        "id_col": "unique_id",
        "time_col": "ds",
        "target_col": "y",
        "static_features": [],
    }
    if config.strategy == "sparse_direct":
        fit_kwargs["horizons"] = horizons
    elif config.strategy == "direct":
        fit_kwargs["max_horizon"] = h

    cv_results = fcst.cross_validation(
        df=df,
        h=h,
        n_windows=n_windows,
        **fit_kwargs,  # type: ignore[arg-type]
    )

    logger.info("Cross-validation complete: %d result rows", len(cv_results))
    return cv_results
