# ML Pipeline Patterns — WindCast Reference

Covers: XGBoost >=2.0, MLflow >=2.10, Polars >=1.0, scikit-learn >=1.4, Python 3.12+

---

## 1. XGBoost — Quantile Regression with scikit-learn API

### 1.1 Single-quantile XGBRegressor

`reg:quantileerror` and `quantile_alpha` were introduced in XGBoost 1.7.0. As of XGBoost >=2.0 this is the canonical way to do quantile regression.

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.5,          # 0.5 = median, 0.1 = lower bound, 0.9 = upper bound
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=10,         # higher = more regularization, important for wind data
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",          # "hist" is fast and handles large datasets
    device="cpu",
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,                 # print eval metric every 100 rounds
)
```

### 1.2 Early Stopping (two equivalent approaches)

**Approach A — `early_stopping_rounds` in `fit()` (simplest):**

```python
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
    early_stopping_rounds=50,    # stop if no improvement for 50 rounds
)
# After fit: model.best_iteration, model.best_score are populated
# predict() automatically uses best_iteration
```

**Approach B — `xgb.callback.EarlyStopping` (more control):**

```python
early_stop = xgb.callback.EarlyStopping(
    rounds=50,
    metric_name="rmse",          # must match eval_metric
    data_name="validation_0",    # name in eval_set (auto-named "validation_0", "validation_1", ...)
    save_best=True,              # restore weights from best iteration
    min_delta=0.0,               # minimum improvement required
)

model = xgb.XGBRegressor(
    objective="reg:squarederror",  # use for mean forecast (not quantile)
    n_estimators=1000,
    learning_rate=0.05,
    callbacks=[early_stop],
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
```

**Gotcha:** When using `objective="reg:quantileerror"`, the eval metric is `"quantile"` (pinball loss), not `"rmse"`. Set `eval_metric="quantile"` explicitly or let XGBoost infer it.

### 1.3 Multiple quantiles in a single model (XGBoost >=2.0)

```python
# Train ONE model for multiple quantiles simultaneously — more efficient
model = xgb.XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=[0.1, 0.5, 0.9],   # list of quantiles
    n_estimators=500,
    tree_method="hist",
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

# predict() returns shape (n_samples, n_quantiles) when quantile_alpha is a list
preds = model.predict(X_test)  # shape: (n_samples, 3) for 3 quantiles
```

**Pitfall:** Multi-quantile mode returns a 2D array. If you need single quantile predictions downstream, slice: `preds[:, 1]` for the median.

### 1.4 Separate models per forecast horizon (recommended pattern)

```python
from typing import Any
import xgboost as xgb
import numpy as np

HORIZONS = [1, 3, 6, 12, 24]  # steps ahead (10-min intervals → 10min, 30min, 1h, 2h, 4h)

def train_horizon_models(
    X_train: np.ndarray,
    y_trains: dict[int, np.ndarray],  # horizon -> target array
    X_val: np.ndarray,
    y_vals: dict[int, np.ndarray],
    base_params: dict[str, Any],
) -> dict[int, xgb.XGBRegressor]:
    models: dict[int, xgb.XGBRegressor] = {}
    for h in HORIZONS:
        model = xgb.XGBRegressor(**base_params)
        model.fit(
            X_train,
            y_trains[h],
            eval_set=[(X_val, y_vals[h])],
            verbose=False,
            early_stopping_rounds=50,
        )
        models[h] = model
    return models
```

**Note on target construction:** For horizon `h`, the target at row `i` is `power[i + h]`. Build this during feature engineering, then `dropna()` to remove rows without a future target.

### 1.5 Feature Importance Extraction

```python
# Method 1 — scikit-learn attribute (weight importance by default)
importances = model.feature_importances_   # np.ndarray, shape (n_features,)

# Method 2 — get_booster().get_score() for more importance types
booster = model.get_booster()
scores: dict[str, float] = booster.get_score(
    importance_type="gain",      # options: "weight", "gain", "cover", "total_gain", "total_cover"
)
# keys are feature names if you set feature_names; otherwise "f0", "f1", ...

# Method 3 — with feature names from training DataFrame
import polars as pl
feature_names = df.select(feature_cols).columns   # list[str] from Polars

model.fit(
    X_train,
    y_train,
    feature_names=feature_names,  # pass at fit() time OR set on booster
)
scores = model.get_booster().get_score(importance_type="gain")
# scores is now {"wind_speed": 142.3, "hour_sin": 87.1, ...}
```

---

## 2. MLflow — Experiment Tracking Patterns

### 2.1 Setup (always at module level)

```python
import mlflow

# For local tracking (no server needed)
mlflow.set_tracking_uri("file:./mlruns")

# Create or get experiment (idempotent)
mlflow.set_experiment("windcast-kelmarsh-xgboost")
```

### 2.2 Basic run with manual logging

```python
with mlflow.start_run(run_name="xgb-median-all-features"):
    # Log hyperparameters
    mlflow.log_params({
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "quantile_alpha": 0.5,
        "feature_set": "v2",
        "dataset": "kelmarsh",
        "turbine": "T01",
    })

    # Train model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Log evaluation metrics
    mlflow.log_metrics({
        "val_mae": float(mae),
        "val_rmse": float(rmse),
        "val_skill_score": float(skill),
        "best_iteration": model.best_iteration,
    })

    # Log the model artifact
    mlflow.xgboost.log_model(
        xgb_model=model,
        name="model",            # subdirectory name under artifacts/
        model_format="ubj",      # "ubj" (binary) is default and smallest
    )
```

### 2.3 Nested runs — parent per experiment, children per horizon

```python
import mlflow
import xgboost as xgb

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("windcast-kelmarsh-multihorizon")

HORIZONS = [1, 3, 6, 12, 24]

with mlflow.start_run(run_name="full-experiment-v1") as parent_run:
    # Log experiment-level params on the parent
    mlflow.log_params({
        "dataset": "kelmarsh",
        "turbine": "T01",
        "feature_set": "v2",
        "n_quantiles": 3,
        "horizons": str(HORIZONS),
    })

    horizon_results: dict[int, dict] = {}

    for h in HORIZONS:
        with mlflow.start_run(
            run_name=f"horizon-{h}",
            nested=True,          # CRITICAL: marks this as a child run
        ):
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=0.5,
                n_estimators=500,
            )
            model.fit(
                X_train,
                y_trains[h],
                eval_set=[(X_val, y_vals[h])],
                early_stopping_rounds=50,
                verbose=False,
            )

            mae = compute_mae(y_vals[h], model.predict(X_val))
            rmse = compute_rmse(y_vals[h], model.predict(X_val))
            skill = compute_skill(y_vals[h], model.predict(X_val), persistence_val[h])

            mlflow.log_params({"horizon_steps": h, "horizon_minutes": h * 10})
            mlflow.log_metrics({"val_mae": mae, "val_rmse": rmse, "val_skill": skill})
            mlflow.xgboost.log_model(xgb_model=model, name="model")

            horizon_results[h] = {"mae": mae, "rmse": rmse, "skill": skill}

    # Log aggregate metrics on the parent run
    avg_skill = sum(r["skill"] for r in horizon_results.values()) / len(HORIZONS)
    mlflow.log_metric("avg_skill_score", avg_skill)
```

**Gotcha:** `nested=True` in `start_run()` is mandatory. Without it, starting a second run inside an active run will raise an error in MLflow >=2.x (or silently end the parent run in older versions).

### 2.4 Logging custom artifacts

```python
import matplotlib.pyplot as plt
import pandas as pd
import mlflow

with mlflow.start_run():
    # Log a matplotlib figure
    fig, ax = plt.subplots()
    ax.bar(feature_names, importances)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Gain importance")
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)

    # Log a dict as JSON (feature importance scores)
    mlflow.log_dict(
        {"feature_importance": scores, "feature_names": feature_names},
        "artifacts/feature_importance.json",
    )

    # Log a CSV as artifact (evaluation results)
    eval_df = pd.DataFrame({
        "horizon": HORIZONS,
        "mae": [results[h]["mae"] for h in HORIZONS],
        "rmse": [results[h]["rmse"] for h in HORIZONS],
        "skill": [results[h]["skill"] for h in HORIZONS],
    })
    eval_path = "/tmp/eval_results.csv"
    eval_df.to_csv(eval_path, index=False)
    mlflow.log_artifact(eval_path, artifact_path="artifacts")

    # Log tags for filtering in UI
    mlflow.set_tag("stage", "validation")
    mlflow.set_tag("model_type", "xgboost")
```

### 2.5 Autologging (alternative to manual logging)

```python
# One line captures: all XGBRegressor params, per-round metrics,
# feature importance files and plots, trained model with signature
mlflow.xgboost.autolog(
    importance_types=["gain", "weight"],   # which importance types to log
    log_input_examples=False,              # set True to log sample inputs
    log_models=True,
    model_format="ubj",
)

with mlflow.start_run():
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    # Everything logged automatically — no mlflow.log_* calls needed
```

**Gotcha:** Autologging logs the model with a default `name="model"`. If you run nested child runs with autolog, each child gets its own model artifact.

---

## 3. Polars → numpy / XGBoost Interop

### 3.1 Direct pass-through (recommended, no conversion needed)

As of XGBoost >=2.0 and scikit-learn >=1.4, both libraries accept Polars DataFrames directly via the `__dataframe__` protocol. Internal conversion to numpy happens transparently.

```python
import polars as pl
import xgboost as xgb

feature_cols = ["wind_speed", "wind_direction", "hour_sin", "hour_cos", "lag_1", "lag_3"]
target_col = "power_kw"

# Pass Polars DataFrame slices directly
X = df.select(feature_cols)     # pl.DataFrame
y = df.get_column(target_col)   # pl.Series

model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X, y)                  # no .to_numpy() needed
predictions = model.predict(X)   # returns np.ndarray
```

**Gotcha:** Feature names are inferred from the Polars column names — this is good! It means `model.feature_importances_` and `get_booster().get_score()` will return named keys.

### 3.2 Explicit numpy conversion (when you need it)

```python
import numpy as np

# DataFrame → 2D float64 array
X_np: np.ndarray = df.select(feature_cols).to_numpy(allow_copy=True)

# Series → 1D array
y_np: np.ndarray = df.get_column(target_col).to_numpy()

# For DMatrix (native XGBoost API — slightly faster training)
dtrain = xgb.DMatrix(X_np, label=y_np, feature_names=feature_cols)
```

**Pitfall:** `to_numpy()` will fail or copy on non-contiguous memory. Pass `allow_copy=True` to avoid silent errors. For Float32 columns, numpy gets float32 — XGBoost handles this fine, but be explicit about dtypes.

### 3.3 Temporal train/val/test split in Polars

**Never shuffle time series data.** Use date-based splits:

```python
import polars as pl

def temporal_split(
    df: pl.DataFrame,
    train_end: str,
    val_end: str,
    timestamp_col: str = "timestamp_utc",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split by timestamp. train_end and val_end are ISO date strings."""
    train = df.filter(pl.col(timestamp_col) < pl.lit(train_end).str.to_datetime())
    val = df.filter(
        (pl.col(timestamp_col) >= pl.lit(train_end).str.to_datetime())
        & (pl.col(timestamp_col) < pl.lit(val_end).str.to_datetime())
    )
    test = df.filter(pl.col(timestamp_col) >= pl.lit(val_end).str.to_datetime())
    return train, val, test


# Usage
train_df, val_df, test_df = temporal_split(
    df,
    train_end="2022-01-01",
    val_end="2023-01-01",
)
```

**Alternative — split by fraction (simpler for exploration):**

```python
n = len(df)
train_df = df.slice(0, int(n * 0.7))
val_df   = df.slice(int(n * 0.7), int(n * 0.15))
test_df  = df.slice(int(n * 0.85), n - int(n * 0.85))
```

### 3.4 Lag features in Polars

```python
import polars as pl

def add_lag_features(
    df: pl.DataFrame,
    col: str,
    lags: list[int],
) -> pl.DataFrame:
    """Add lag columns for a given signal. Rows with nulls at the start are expected."""
    return df.with_columns([
        pl.col(col).shift(lag).alias(f"{col}_lag{lag}")
        for lag in lags
    ])


def add_rolling_features(
    df: pl.DataFrame,
    col: str,
    windows: list[int],
) -> pl.DataFrame:
    """Add rolling mean and std. Uses only past data (min_periods=1)."""
    exprs = []
    for w in windows:
        exprs.extend([
            pl.col(col).shift(1).rolling_mean(window_size=w).alias(f"{col}_roll_mean_{w}"),
            pl.col(col).shift(1).rolling_std(window_size=w).alias(f"{col}_roll_std_{w}"),
        ])
    return df.with_columns(exprs)


# Example usage
df = add_lag_features(df, "power_kw", lags=[1, 3, 6, 12, 24])
df = add_rolling_features(df, "power_kw", windows=[3, 6, 12])

# Drop rows where any lag is null (beginning of series)
df = df.drop_nulls()
```

**Gotcha:** `rolling_mean()` in Polars is a look-ahead risk if you forget `.shift(1)`. Always shift by 1 before rolling so the window only uses data available at prediction time.

**Pitfall in Polars >=1.0:** The `rolling_mean()` API requires `window_size` as a keyword argument (not positional). Use `rolling_mean(window_size=6)`.

### 3.5 Cyclic encoding for temporal and directional features

Polars has native `.sin()` and `.cos()` on `Expr` objects. Chain with `.dt` accessor for datetime features.

```python
import polars as pl
import math

def add_cyclic_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add sin/cos cyclic encoding for hour, day-of-week, month, wind direction."""
    return df.with_columns([
        # Hour of day (period = 24)
        (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24)).sin().alias("hour_sin"),
        (pl.col("timestamp_utc").dt.hour().cast(pl.Float64) * (2 * math.pi / 24)).cos().alias("hour_cos"),

        # Day of week (period = 7, Monday=0)
        (pl.col("timestamp_utc").dt.weekday().cast(pl.Float64) * (2 * math.pi / 7)).sin().alias("dow_sin"),
        (pl.col("timestamp_utc").dt.weekday().cast(pl.Float64) * (2 * math.pi / 7)).cos().alias("dow_cos"),

        # Month (period = 12)
        (pl.col("timestamp_utc").dt.month().cast(pl.Float64) * (2 * math.pi / 12)).sin().alias("month_sin"),
        (pl.col("timestamp_utc").dt.month().cast(pl.Float64) * (2 * math.pi / 12)).cos().alias("month_cos"),

        # Wind direction (period = 360 degrees)
        (pl.col("wind_direction") * (math.pi / 180)).sin().alias("wind_dir_sin"),
        (pl.col("wind_direction") * (math.pi / 180)).cos().alias("wind_dir_cos"),
    ])
```

**Why cyclic encoding matters:** A model treating hour 23 and hour 0 as far apart will fail on midnight transitions. Sin/cos encodes the circular continuity. Same for wind direction (359° and 1° are adjacent).

---

## 4. scikit-learn Metrics

### 4.1 Standard regression metrics

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,   # added in sklearn 1.4 — use this, NOT mean_squared_error(squared=False)
)
import numpy as np

y_true = np.array([...])
y_pred = np.array([...])

mae  = mean_absolute_error(y_true, y_pred)           # mean |y - ŷ|
rmse = root_mean_squared_error(y_true, y_pred)        # sqrt(mean (y - ŷ)²)
mape = mean_absolute_percentage_error(y_true, y_pred) # mean |y - ŷ| / |y|  (fraction, not %)
```

**Gotcha:** `mean_squared_error(squared=False)` was deprecated in sklearn 1.4 and removed in 1.6. Always use `root_mean_squared_error()` for >=1.4.

**Gotcha:** MAPE is undefined when `y_true` contains zeros (division by zero). For wind power, zero output is common at low wind speeds — filter out curtailment periods before computing MAPE, or use MAE as primary metric.

### 4.2 Custom skill score

The skill score measures improvement over the persistence baseline:

```python
def compute_skill_score(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_persistence: np.ndarray,
) -> float:
    """
    Skill score = 1 - RMSE_model / RMSE_persistence.
    Range: (-inf, 1]. 1 = perfect, 0 = same as persistence, <0 = worse than persistence.
    """
    from sklearn.metrics import root_mean_squared_error
    rmse_model = root_mean_squared_error(y_true, y_pred_model)
    rmse_persistence = root_mean_squared_error(y_true, y_pred_persistence)
    if rmse_persistence == 0:
        return 1.0 if rmse_model == 0 else -float("inf")
    return 1.0 - (rmse_model / rmse_persistence)
```

---

## 5. Persistence Baseline Model

### 5.1 Simple functional implementation

For initial validation and benchmarking, a function is simpler than an estimator:

```python
import numpy as np
import polars as pl


def persistence_forecast(
    df: pl.DataFrame,
    horizon: int,
    power_col: str = "power_kw",
    timestamp_col: str = "timestamp_utc",
) -> pl.DataFrame:
    """
    Naive persistence: forecast at t+horizon = observed power at t.
    Returns DataFrame with columns [timestamp_utc, y_true, y_pred_persistence].
    """
    return df.with_columns([
        pl.col(power_col).alias("y_true"),
        pl.col(power_col).shift(0).alias("y_pred_persistence"),  # current value as forecast
    ]).select([timestamp_col, "y_true", "y_pred_persistence"]).slice(0, len(df) - horizon)
```

For evaluation, you need actual vs. persistence at the same horizon:

```python
def make_persistence_targets(
    power: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_pred_persistence) aligned at the same indices.
    y_true[i] = power[i + horizon]   (future value to predict)
    y_pred_persistence[i] = power[i]  (last known value = persistence forecast)
    """
    y_true = power[horizon:]               # actual future values
    y_persistence = power[:-horizon]       # last known values
    return y_true, y_persistence
```

### 5.2 scikit-learn compatible estimator

Use this when you need to plug the persistence model into sklearn pipelines or scoring utilities:

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class PersistenceForecaster(BaseEstimator, RegressorMixin):
    """
    Naive persistence baseline: predicts the last observed value of the target.

    In a direct multi-step setup, X must contain a column of the current
    (t=0) power value. The estimator simply returns that column as the forecast.

    Parameters
    ----------
    current_power_idx : int
        Column index in X that contains the current (t=0) power value.
        Default: -1 (last column, assuming lag_1 is appended last).
    """

    def __init__(self, current_power_idx: int = -1) -> None:
        self.current_power_idx = current_power_idx

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PersistenceForecaster":
        # sklearn convention: fit() must set at least one fitted attribute
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        return X[:, self.current_power_idx].copy()


# Usage example
from sklearn.metrics import root_mean_squared_error

persistence = PersistenceForecaster(current_power_idx=feature_cols.index("power_kw_lag1"))
persistence.fit(X_train)

y_pred_persistence = persistence.predict(X_val)
rmse_baseline = root_mean_squared_error(y_val, y_pred_persistence)
print(f"Persistence RMSE: {rmse_baseline:.2f} kW")
```

---

## 6. Full Integration Pattern — Training Script Skeleton

```python
"""
scripts/train.py — multi-horizon quantile forecasting with MLflow tracking.
"""
import math

import mlflow
import mlflow.xgboost
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from windcast.features import add_cyclic_features, add_lag_features
from windcast.metrics import compute_skill_score
from windcast.models import PersistenceForecaster

HORIZONS = [1, 3, 6, 12, 24]
QUANTILES = [0.1, 0.5, 0.9]
FEATURE_COLS = [
    "wind_speed", "wind_direction",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "wind_dir_sin", "wind_dir_cos",
    "power_kw_lag1", "power_kw_lag3", "power_kw_lag6",
    "power_kw_roll_mean_6", "power_kw_roll_std_6",
]

XGB_PARAMS = {
    "objective": "reg:quantileerror",
    "quantile_alpha": 0.5,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
}

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("windcast-kelmarsh-v1")


def build_horizon_dataset(
    df: pl.DataFrame, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    target = df.get_column("power_kw").shift(-horizon).alias(f"target_h{horizon}")
    df_h = df.with_columns(target).drop_nulls()
    X = df_h.select(FEATURE_COLS).to_numpy()
    y = df_h.get_column(f"target_h{horizon}").to_numpy()
    return X, y


def main() -> None:
    df = pl.read_parquet("data/processed/kelmarsh_T01.parquet")
    df = add_lag_features(df, "power_kw", lags=[1, 3, 6, 12, 24])
    df = add_cyclic_features(df)
    df = df.drop_nulls()

    n = len(df)
    train_df = df.slice(0, int(n * 0.70))
    val_df   = df.slice(int(n * 0.70), int(n * 0.15))
    test_df  = df.slice(int(n * 0.85), n - int(n * 0.85))

    with mlflow.start_run(run_name="kelmarsh-T01-multihorizon"):
        mlflow.log_params({
            **XGB_PARAMS,
            "feature_cols": FEATURE_COLS,
            "horizons": str(HORIZONS),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        })

        for h in HORIZONS:
            X_train, y_train = build_horizon_dataset(train_df, h)
            X_val, y_val     = build_horizon_dataset(val_df, h)

            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )

            # Persistence baseline
            persistence_idx = FEATURE_COLS.index("power_kw_lag1")
            baseline = PersistenceForecaster(current_power_idx=persistence_idx)
            baseline.fit(X_val)
            y_persistence = baseline.predict(X_val)

            # Metrics
            y_pred = model.predict(X_val)
            mae   = mean_absolute_error(y_val, y_pred)
            rmse  = root_mean_squared_error(y_val, y_pred)
            skill = compute_skill_score(y_val, y_pred, y_persistence)

            with mlflow.start_run(run_name=f"h{h:02d}-{h*10}min", nested=True):
                mlflow.log_params({"horizon_steps": h, "horizon_min": h * 10})
                mlflow.log_metrics({
                    "val_mae": mae,
                    "val_rmse": rmse,
                    "val_skill": skill,
                    "best_iteration": model.best_iteration,
                })
                mlflow.xgboost.log_model(xgb_model=model, name="model")


if __name__ == "__main__":
    main()
```

---

## Quick Reference — Common Gotchas

| Issue | Cause | Fix |
|-------|-------|-----|
| `early_stopping_rounds` ignored | No `eval_set` passed to `fit()` | Always pass `eval_set=[(X_val, y_val)]` |
| Multi-quantile predict returns 2D array | `quantile_alpha` is a list | Slice: `preds[:, 1]` for median |
| `mean_squared_error(squared=False)` error | Removed in sklearn 1.6 | Use `root_mean_squared_error()` |
| Feature importance keys are `"f0"`, `"f1"` | No feature names set | Pass Polars DataFrame directly to fit() or set `feature_names=` |
| `nested=True` missing → run error | MLflow >=2.x strict nesting | Always `mlflow.start_run(nested=True)` in child runs |
| Rolling features use future data | Forgot `.shift(1)` before `.rolling_mean()` | Always: `pl.col(...).shift(1).rolling_mean(window_size=N)` |
| MAPE is inf or NaN | Zero values in y_true | Filter zero-power rows before MAPE, use MAE as primary |
| `to_numpy()` fails silently | Non-contiguous chunks | Use `to_numpy(allow_copy=True)` |
| Polars `rename` raises on missing keys | `strict=True` by default | `df.rename(map, strict=False)` |
