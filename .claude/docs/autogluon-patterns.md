# AutoGluon Patterns — WindCast Reference

Version: 1.5.0 | Researched: 2026-04-01

## Relevant Modules

- **AutoGluon-TimeSeries**: primary candidate for wind power forecasting
- **AutoGluon-Tabular**: quantile regression baseline comparison
- **AutoGluon-Cloud**: SageMaker training/deployment (optional)

---

## AutoGluon-TimeSeries

### Core API

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Polars → pandas required at boundary
train_data = TimeSeriesDataFrame.from_data_frame(
    df.to_pandas(),             # AutoGluon does NOT support Polars natively
    id_column="item_id",        # turbine ID
    timestamp_column="timestamp"
)

predictor = TimeSeriesPredictor(
    prediction_length=6,        # 6 steps × 10 min = 1h ahead
    target="power_kw",
    eval_metric="MASE",
    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
    path="./ag-timeseries-models",  # local artifact store (NOT MLflow)
)

predictor.fit(
    train_data,
    presets="medium_quality",   # or "high_quality", "best_quality"
    time_limit=600,
)

predictions = predictor.predict(train_data)
# Returns DataFrame with columns: "mean", "0.1", "0.25", "0.5", "0.75", "0.9"
# One row per (item_id, timestamp) forecast step
```

### Supported Models (full zoo)

| Category | Models |
|---|---|
| Statistical | NaiveModel, SeasonalNaive, Average, SeasonalAverage, ETS, AutoARIMA, AutoETS, AutoCES, Theta, NPTS |
| Deep Learning | DeepAR, DLinear, PatchTST, SimpleFeedForward, TemporalFusionTransformer, TiDE, WaveNet |
| Tabular/ML | DirectTabular (LightGBM), PerStepTabular, RecursiveTabular |
| Pretrained | Chronos2 (zero-shot, T5-based), Chronos, Toto |
| Sparse | ADIDA, Croston variants, IMAPA |

DirectTabular and RecursiveTabular use LightGBM internally — directly comparable to XGBoost quantile baseline.

### Multi-Horizon

Native. Set `prediction_length=N` where N is number of 10-min steps. All models handle this internally.

### Probabilistic Output

Native. `quantile_levels` at predictor init. Output columns are string keys: `"0.1"`, `"0.5"`, `"0.9"`, etc.

---

## AutoGluon-Tabular

### Quantile Regression

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label="power_kw",
    problem_type="quantile",
    quantile_levels=[0.1, 0.5, 0.9],
)
predictor.fit(train_df.to_pandas(), presets="medium_quality")
predictions = predictor.predict(test_df.to_pandas())
```

### Models Trained Automatically

LightGBM, LightGBMXT, LightGBMLarge, XGBoost, CatBoost, RandomForest (Gini/Entropy), ExtraTrees (Gini/Entropy), NeuralNetFastAI, NeuralNetTorch → combined into WeightedEnsemble_L2.

---

## AutoGluon-Cloud (SageMaker)

```python
from autogluon.cloud import TabularCloudPredictor

cloud_predictor = TabularCloudPredictor(cloud_output_path="s3://bucket/path")
cloud_predictor.fit(
    predictor_init_args={"label": "power_kw", "problem_type": "quantile"},
    predictor_fit_args={"train_data": train_df, "time_limit": 600},
    instance_type="ml.m5.2xlarge",
)
cloud_predictor.deploy()
predictions = cloud_predictor.predict_real_time(test_df)

# Exit path — convert to local predictor for portability
local_predictor = cloud_predictor.to_local_predictor()
```

Pricing: SageMaker instance rates + S3 storage only. No AutoGluon surcharge.
Setup: IAM role required via CloudFormation or manual config.

---

## MLflow Integration (Manual)

AutoGluon does NOT emit to MLflow automatically. Wrap it:

```python
import mlflow
from autogluon.timeseries import TimeSeriesPredictor

mlflow.set_tracking_uri("file:./mlruns")

with mlflow.start_run(run_name="autogluon-timeseries"):
    predictor = TimeSeriesPredictor(prediction_length=6, target="power_kw")
    predictor.fit(train_data, presets="medium_quality", time_limit=600)

    lb = predictor.leaderboard(test_data)
    best = lb.iloc[0]
    mlflow.log_metric("MASE", best["score_test"])
    mlflow.log_metric("val_MASE", best["score_val"])
    mlflow.log_param("best_model", best["model"])
    mlflow.log_param("preset", "medium_quality")

    # Log leaderboard as artifact
    lb.to_csv("/tmp/ag_leaderboard.csv", index=False)
    mlflow.log_artifact("/tmp/ag_leaderboard.csv")
```

---

## WindCast-Specific Gotchas

| Issue | Mitigation |
|---|---|
| Polars not supported | Call `.to_pandas()` at the AutoGluon boundary; keep Polars for all upstream processing |
| SCADA gaps (maintenance, curtailment) break regular frequency assumption | Pre-impute or resample to regular 10-min grid before `TimeSeriesDataFrame` construction |
| TemporalFusionTransformer requires GPU/cuDNN | Set `excluded_model_types=["TemporalFusionTransformer"]` on CPU-only machines |
| No physics constraints | Post-process: clip predictions to `[0, rated_power]`; enforce monotonicity if needed |
| AutoGluon artifact store != MLflow | Use AutoGluon `path=` for its own checkpoints; log summary metrics to MLflow manually |
| Hill of Towie end-of-period timestamps | Shift -10 min BEFORE converting to `TimeSeriesDataFrame` |
| AeroUp performance discontinuity | Add retrofit date as `static_features` covariate or split dataset at discontinuity |

### Excluding Models on CPU

```python
predictor.fit(
    train_data,
    excluded_model_types=["TemporalFusionTransformer", "DeepAR", "WaveNet"],
    time_limit=600,
)
```

### Static Covariates (turbine metadata)

```python
static_features = pd.DataFrame({
    "item_id": ["T01", "T02"],
    "hub_height_m": [100, 80],
    "rotor_diameter_m": [126, 112],
    "has_aeroup": [1, 0],
})
train_data = TimeSeriesDataFrame.from_data_frame(
    df.to_pandas(),
    id_column="item_id",
    timestamp_column="timestamp",
    static_features_df=static_features,
)
```

---

## Key Limitations

- No native Polars support — pandas boundary required
- MLflow not auto-integrated — manual logging wrapper needed
- Deep learning models (TFT, DeepAR) need GPU for production performance
- Ensembling is a black box — hard to audit individual model contributions
- No physics constraints (power curve cap, monotonicity)
- Regular time grid required — SCADA gaps need pre-imputation
- AWS lock-in for Cloud module (exit path: `to_local_predictor()`)
- Presets control tradeoff between speed and accuracy; fine-grained HPO requires custom `hyperparameters` dict

---

## Docs

- Main: https://auto.gluon.ai/stable/index.html
- TimeSeries quickstart: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html
- Model zoo: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html
- Tabular quickstart: https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html
- Cloud/SageMaker: https://auto.gluon.ai/cloud/stable/tutorials/autogluon-cloud.html
