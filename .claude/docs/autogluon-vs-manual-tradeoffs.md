# AutoGluon vs Manual ML Pipeline — WindCast Tradeoff Analysis

Researched: 2026-04-01 | AutoGluon version: 1.5.x

---

## Executive Summary

AutoGluon-TimeSeries is a legitimate accelerator for a POC but carries real costs in
debuggability, CQR control, and MLflow ergonomics. For WindCast's current architecture —
custom feature engineering + XGBoost quantile regression + CQR calibration — the manual
pipeline is a better fit. AutoGluon adds value as an ensemble benchmark, not as a replacement.

---

## Question-by-Question Findings

### 1. Can AutoGluon handle custom feature engineering?

**Partially — via `known_covariates` and `past_covariates` columns in `TimeSeriesDataFrame`.**

You can pre-compute features (wind_cubed, turbulence_intensity, stability_proxy, NWP
temperature, NWP wind) as extra columns and pass them at fit/predict time. All AutoGluon
models accept these through the `covariate_regressor` hyperparameter.

Constraint: AutoGluon also applies its own internal `TimeSeriesFeatureGenerator`
(normalization, imputation, lag creation). You cannot disable this. Your custom features
are added on top — they are not the only features. This creates a hybrid that is hard to
audit.

Known-covariates (features available at forecast time, e.g., NWP) are supported natively.
Past-covariates (only available in history, e.g., SCADA operational flags) are also
supported.

Bottom line: custom features work, but you cannot fully control the feature set. For
physics-informed models (power curve, wake, stability regime), this opacity is a real risk.

### 2. Multi-horizon quantile forecasting?

**Yes, fully native.**

- `prediction_length=N` sets how many 10-min steps ahead.
- `quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]` at predictor init.
- All models produce the full quantile set for every horizon step.
- Output columns are string keys: `"0.1"`, `"0.5"`, `"0.9"`, plus `"mean"`.

For H+1 to H+48 in WindCast (1, 6, 12, 24, 48 steps at 10-min resolution), you would
need `prediction_length=48`. All intermediate horizons are part of the output — no need
to train separate models per horizon.

**Important gap: no CQR (Conformalized Quantile Regression).**
AutoGluon produces raw quantile outputs, not conformally calibrated intervals. If
calibrated coverage guarantees are a project requirement, CQR must be applied as a
post-processing step on top of AutoGluon predictions — the same as the manual pipeline.
This diminishes the automation advantage.

### 3. MLflow integration?

**No native integration. Manual wrapper required.**

AutoGluon is aware of MLflow conflicts (`warn_if_mlflow_autologging_is_enabled()`
function in TimeSeries predictor source) but does not emit to MLflow automatically.
As of April 2026, there is no official AutoGluon flavor in MLflow (open issue: mlflow#13214).

What you must do manually:
- Wrap `predictor.fit()` in `mlflow.start_run()`
- Extract metrics from `predictor.leaderboard()` and log with `mlflow.log_metric()`
- Log the leaderboard CSV as an artifact
- Log hyperparameters separately

AutoGluon uses its own artifact store (`path=` argument). The two stores coexist but are
not linked — you get duplicate bookkeeping.

With the manual pipeline, every XGBoost and LightGBM run is a single clean MLflow run.
With AutoGluon, one AutoGluon "run" spawns 10-20 sub-models internally with no MLflow
visibility into individual model performance.

### 4. Learning curve?

**Weekend for basic use. 2-4 weeks to master edge cases.**

Basic fit/predict with covariates: one day if you know the pandas/MLflow ecosystem.

Hard parts that take longer:
- Understanding which internal transformations fire and why
- Debugging `TimeSeriesDataFrame` frequency inference errors (SCADA gaps will trigger these)
- Configuring `hyperparameters` dict for custom HPO (underdocumented)
- Disabling deep learning models on CPU without breaking the ensemble
- Integrating CQR post-processing cleanly

The Polars boundary is a one-liner (`.to_pandas()`) but timestamp handling (end-of-period
shifts, UTC normalization) must be done before conversion — AutoGluon will silently accept
wrong timestamps.

### 5. AutoGluon-Cloud + SageMaker — worth it for solo POC?

**No. Clear overkill.**

Cost reality:
- SageMaker adds ~20% premium over equivalent EC2 instance rates.
- Always-on inference endpoints: $150+/month. Training-only: $0.02-0.16/job.
- For a POC with occasional retraining, total cost is low either way — but SageMaker adds
  IAM setup, CloudFormation, S3 bucket management, and a deployment workflow that is
  irrelevant until production.

AutoGluon runs perfectly on a local MacBook or a single EC2 spot instance.
`mlflow.set_tracking_uri("file:./mlruns")` is all the infrastructure you need for POC.

Reserve SageMaker for: multi-turbine production rollout with real-time inference endpoints
and enterprise compliance requirements.

### 6. AutoGluon for wind/energy forecasting — field evidence?

No peer-reviewed AutoGluon wind paper found. The energy forecasting literature uses:
- XGBoost/Random Forest for tabular baselines (strong performers)
- Deep learning (LSTM, TFT, WaveNet) for longer horizons
- WindDragon (automated deep learning, CNN+ViT on NWP maps) for regional forecasting

AutoGluon's `DirectTabular` model internally uses LightGBM with lag features — effectively
what a well-tuned manual LightGBM pipeline does, but with less control over lag window
selection and feature importance.

The consensus from benchmark papers: ensemble methods beat single models, but physics-
informed features (wind³, stability regime) are the key differentiator for wind
specifically — not the choice of AutoML framework.

---

## Benefits of AutoGluon

| Benefit | Concrete value for WindCast |
|---|---|
| Model zoo breadth | Tests statistical (ETS, Theta), ML (LightGBM, XGBoost), and DL (DeepAR, TFT) in one call |
| Chronos-2 zero-shot | Can benchmark a pre-trained foundation model with zero training time |
| Ensemble automatically | Weighted ensemble often outperforms best single model |
| Probabilistic native | Quantile output built-in, no extra code |
| Multi-horizon native | Single `prediction_length` covers H+1 through H+48 |
| Saves HPO time | `medium_quality` preset is a reasonable starting point |

## Risks of AutoGluon

| Risk | Severity for WindCast |
|---|---|
| Black-box feature engineering | High — physics interpretation is a project goal |
| No CQR built-in | Medium — must still implement post-processing layer |
| MLflow friction | Medium — double bookkeeping, no per-submodel visibility |
| SCADA gap handling | High — irregular 10-min series triggers frequency errors |
| Deep learning cost | Low — can exclude TFT/DeepAR on CPU |
| Polars boundary | Low — one `.to_pandas()` call |
| AWS lock-in risk | Low for training, real for Cloud/SageMaker module |
| Interpretability | High — leaderboard shows MASE but not feature importance per model |

---

## Recommendation

**Keep the manual pipeline. Add AutoGluon as a benchmark lane.**

Concrete split:

| Task | Tool |
|---|---|
| Feature engineering | Polars (manual) — full control, physics-informed |
| Primary model | XGBoost quantile regression (manual) — auditable, tunable |
| Comparison baseline | AutoGluon `DirectTabular` or `RecursiveTabular` preset — one-shot benchmark |
| Probabilistic calibration | CQR on top of both (manual) — coverage guarantees |
| Foundation model zero-shot | AutoGluon Chronos-2 — useful lower bound with no training |
| Experiment tracking | MLflow (manual wrapper for AutoGluon runs) |
| HPO | Optuna on manual models; AutoGluon presets for its own models |
| Cloud | Local / EC2 spot — skip SageMaker until production |

This gives you the best of both: your physics-informed manual pipeline is the primary
artifact, AutoGluon provides a benchmark that would have taken weeks to replicate manually.

Expected time to add AutoGluon benchmark lane: 1-2 days.

---

## Implementation Notes

Custom covariate passthrough pattern:

```python
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Pre-compute WindCast features in Polars, then convert at boundary
features_pd = features_polars.to_pandas()  # must be regular 10-min grid, UTC

ts_df = TimeSeriesDataFrame.from_data_frame(
    features_pd,
    id_column="turbine_id",
    timestamp_column="timestamp",
    # known_covariates = columns available at forecast time
    # AutoGluon infers them from column presence at predict() time
)

predictor = TimeSeriesPredictor(
    prediction_length=48,     # H+48 = 8 hours at 10-min resolution
    target="power_kw",
    known_covariates_names=["nwp_wind_speed", "nwp_wind_dir", "nwp_temp"],
    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
    path="./ag-models",
    eval_metric="MASE",
)

predictor.fit(
    ts_df,
    presets="medium_quality",
    time_limit=1800,
    excluded_model_types=["TemporalFusionTransformer", "DeepAR"],  # CPU-safe
)
```

MLflow wrapper:

```python
import mlflow

with mlflow.start_run(run_name="autogluon-benchmark"):
    predictor.fit(ts_df, presets="medium_quality", time_limit=1800)
    lb = predictor.leaderboard(test_ts_df, silent=True)
    best = lb.iloc[0]
    mlflow.log_metric("MASE_best", float(best["score_test"]))
    mlflow.log_param("best_model", best["model"])
    mlflow.log_param("preset", "medium_quality")
    lb.to_csv("/tmp/ag_leaderboard.csv", index=False)
    mlflow.log_artifact("/tmp/ag_leaderboard.csv")
```

---

## Sources

- [AutoGluon TimeSeries In-Depth](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html)
- [AutoGluon Model Zoo](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html)
- [AutoGluon vs XGBoost — Medium/TDS](https://medium.com/data-science/autogluon-vs-xgboost-will-automl-replace-data-scientists-dc1220010102)
- [MLflow AutoGluon flavor request (open issue)](https://github.com/mlflow/mlflow/issues/13214)
- [Databricks: AutoGluon MLflow integration](https://community.databricks.com/t5/machine-learning/autogluon-mlflow-integration/td-p/111423)
- [AutoGluon-Cloud SageMaker tutorial](https://auto.gluon.ai/cloud/dev/tutorials/autogluon-cloud.html)
- [Cost-effective AutoML on AWS vs SageMaker](https://dev.to/aws-builders/building-a-cost-effective-automl-platform-on-aws-10-25month-vs-150-for-sagemaker-endpoints-1m99)
- [WindDragon automated deep learning paper](https://arxiv.org/html/2402.14385v1)
- [Review of ML/AutoML for time series (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9159649/)
