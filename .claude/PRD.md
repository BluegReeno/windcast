# WindCast — Product Requirements Document

**Version**: 1.0
**Last Updated**: 2026-03-31
**Target**: MVP (Phase 1 + Phase 2)

---

## 1. Executive Summary

WindCast is a standardized ML framework for wind power forecasting. It turns raw SCADA data from any wind farm into calibrated, probabilistic power forecasts using reproducible pipelines, MLflow experiment tracking, and open datasets.

The project builds on WattCast (electricity spot price forecasting for EPEX SPOT FR) but targets a fundamentally different domain: turbine-level power production from 10-minute SCADA measurements. The key methodological upgrade is **MLflow from day 1** — every experiment is logged, compared, and reproducible.

**MVP goal:** Given a new wind farm SCADA dataset, produce a ranked set of forecast models (with confidence intervals) in under 24 hours of compute and under 2 hours of human setup. Prove this on 3 datasets from 2 different OEMs.

---

## 2. Mission

**Standardize the weather-to-power calibration loop so that onboarding a new wind farm takes hours, not weeks.**

### Core Principles

1. **Dataset-agnostic** — one pipeline, many datasets. Adding a farm = writing a parser, not redesigning the ML.
2. **Reproducible by default** — every run tracked in MLflow with features, params, metrics, artifacts.
3. **Simple over clever** — scripts over frameworks, Parquet over databases, XGBoost over Transformers.
4. **Fail fast** — detailed errors, strict schemas, no silent data corruption.
5. **Beat persistence or go home** — if the model can't outperform "last known power", it's not useful.

---

## 3. Target Users

### Primary: Solo ML engineer / data scientist (i.e., the developer)

- Deep Python experience, familiar with XGBoost/scikit-learn ecosystem
- Understands wind energy basics (power curves, SCADA signals, NWP)
- Wants a reusable template to apply to new wind farm datasets
- Pain points: repetitive data cleaning, non-reproducible experiments, no standardized evaluation

### Secondary: Wind energy team lead (portfolio perspective)

- Evaluates forecast quality across multiple sites
- Needs MLflow comparison UI to rank models and track improvements
- Pain point: each data scientist has their own scripts, no cross-site benchmarking

---

## 4. MVP Scope

### In Scope (Phase 1 + 2)

**Data Foundation:**
- ✅ Canonical SCADA schema (Polars-native, typed)
- ✅ Kelmarsh v4 parser + signal mapping (primary dataset)
- ✅ Hill of Towie parser (Siemens OEM — generalization test)
- ✅ Penmanshiel v3 parser (Senvion cross-site test)
- ✅ QC pipeline: outlier detection, maintenance filtering, curtailment flagging, gap-fill
- ✅ Open-Meteo NWP client (wind speed/direction at 100m, temperature, pressure)
- ✅ Parquet storage (local `data/` directory)

**Feature Engineering:**
- ✅ Baseline features: wind speed, direction (sin/cos), power lags, rolling stats (1h/6h/24h)
- ✅ Enriched features: V³ (cube law), stability proxy, direction sectors
- ✅ NWP-augmented features: Open-Meteo forecasts aligned to prediction horizon
- ✅ Calendar features: hour, day of week, month (cyclic encoding)
- ✅ Standardized feature sets (baseline / enriched / full) per dataset

**ML Models:**
- ✅ Persistence benchmark (naive: last known power)
- ✅ XGBoost quantile regression (multi-horizon)
- ✅ LightGBM comparison
- ✅ Probabilistic output: quantile regression + Conformalized Quantile Regression (CQR)
- ✅ Multi-horizon forecasts: H+1, H+6, H+12, H+24, H+48

**Experiment Tracking:**
- ✅ MLflow setup with local file tracking (`file:./mlruns`)
- ✅ Every run logged: features, hyperparameters, all metrics, artifacts
- ✅ Standardized evaluation: MAE, RMSE, MAPE, skill score vs persistence
- ✅ Error analysis by wind regime (low/medium/high)
- ✅ Cross-site comparison in MLflow UI

**Quality:**
- ✅ ruff linting + formatting
- ✅ pyright type checking
- ✅ pytest test suite
- ✅ Pydantic config validation

### Out of Scope (future phases)

**Phase 3 — Production:**
- ❌ MLflow Model Registry (staging → production → archived)
- ❌ FastAPI prediction endpoint
- ❌ Streamlit dashboard
- ❌ Drift monitoring + automated re-calibration
- ❌ Supabase storage backend
- ❌ Docker containerization

**Not planned:**
- ❌ Deep learning models (LSTM, Transformer) — XGBoost/LightGBM are sufficient for POC
- ❌ Kedro pipeline orchestration — overkill for solo developer
- ❌ Real-time streaming — batch processing only
- ❌ Turbine-level forecasting — farm-level aggregated power for MVP
- ❌ FINO1/NWTC M2 wind-only pipeline (available for manual NWP validation)

---

## 5. User Stories

1. **As an ML engineer**, I want to ingest a new SCADA dataset by writing only a parser function, so that I can onboard a new wind farm in < 2 hours.
   - *Example:* Write `kelmarsh.py` with column mapping, run `uv run python scripts/ingest_kelmarsh.py`, get clean Parquet output.

2. **As an ML engineer**, I want standardized feature sets that work across any dataset, so that I don't reinvent feature engineering for each farm.
   - *Example:* Call `build_features(df, feature_set="enriched")` and get the same V³, sin/cos direction, rolling stats regardless of source dataset.

3. **As an ML engineer**, I want every experiment automatically logged to MLflow, so that I can compare models across runs and datasets.
   - *Example:* Run `uv run python scripts/train.py --dataset kelmarsh --model xgboost --features enriched`, see full run in `mlflow ui`.

4. **As an ML engineer**, I want to evaluate models with skill score vs persistence, so that I can prove the forecast adds value over a naive baseline.
   - *Example:* Evaluation report shows "XGBoost enriched: skill score 0.42 at H+24" — meaning 42% RMSE reduction vs persistence.

5. **As an ML engineer**, I want probabilistic forecasts (confidence intervals), so that I can quantify forecast uncertainty.
   - *Example:* Model outputs P10/P50/P90 quantiles; CQR calibration ensures 80% of observations fall within P10–P90.

6. **As an ML engineer**, I want to run the same pipeline on Hill of Towie (Siemens) after building it on Kelmarsh (Senvion), so that I can prove the framework is OEM-agnostic.
   - *Example:* Write `hill_of_towie.py` parser, run same `train.py` and `evaluate.py` scripts, compare results in MLflow.

7. **As an ML engineer**, I want error analysis broken down by wind regime, so that I can identify where the model struggles.
   - *Example:* Evaluation shows MAE by wind regime: low (<5 m/s): 120 kW, medium (5-12 m/s): 85 kW, high (>12 m/s): 210 kW.

---

## 6. Core Architecture

### Directory Structure

```
windcast/
├── pyproject.toml                  # Dependencies, project config
├── CLAUDE.md                       # AI assistant guidelines
├── README.md
│
├── src/windcast/                   # Main package
│   ├── __init__.py
│   ├── config.py                   # Pydantic Settings, constants
│   │
│   ├── data/                       # Data ingestion & QC
│   │   ├── __init__.py
│   │   ├── schema.py               # Canonical SCADA schema (Polars)
│   │   ├── kelmarsh.py             # Kelmarsh v4 parser
│   │   ├── hill_of_towie.py        # Hill of Towie parser
│   │   ├── penmanshiel.py          # Penmanshiel v3 parser
│   │   ├── qc.py                   # Quality control pipeline
│   │   └── open_meteo.py           # NWP weather data client
│   │
│   ├── features/                   # Feature engineering
│   │   ├── __init__.py
│   │   └── pipeline.py             # Standardized feature sets
│   │
│   └── models/                     # ML training & evaluation
│       ├── __init__.py
│       ├── persistence.py          # Naive persistence benchmark
│       ├── xgboost.py              # XGBoost quantile regression
│       ├── lightgbm.py             # LightGBM comparison
│       └── evaluation.py           # Metrics, skill scores, analysis
│
├── scripts/                        # CLI entry points
│   ├── ingest_kelmarsh.py          # Parse & QC Kelmarsh data
│   ├── ingest_hill_of_towie.py     # Parse & QC Hill of Towie data
│   ├── ingest_penmanshiel.py       # Parse & QC Penmanshiel data
│   ├── build_features.py           # Feature engineering
│   ├── train.py                    # Train models (logged to MLflow)
│   └── evaluate.py                 # Evaluate + compare
│
├── data/                           # Local data (GITIGNORED)
│   ├── raw/                        # Downloaded ZIPs
│   ├── processed/                  # Clean Parquet files
│   └── features/                   # Feature-engineered Parquet
│
├── mlruns/                         # MLflow tracking (GITIGNORED)
│
├── tests/                          # Mirrors src/ structure
│   ├── data/
│   │   ├── test_schema.py
│   │   ├── test_kelmarsh.py
│   │   ├── test_qc.py
│   │   └── test_open_meteo.py
│   ├── features/
│   │   └── test_pipeline.py
│   └── models/
│       ├── test_persistence.py
│       ├── test_xgboost.py
│       └── test_evaluation.py
│
└── docs/
    └── research/                   # Brainstorming, methodology, dataset catalog
```

### Key Design Patterns

1. **Parser pattern** — Each dataset has a dedicated parser module that maps native column names → canonical schema. The parser is the only dataset-specific code; everything downstream is generic.

2. **Schema-first** — Canonical SCADA schema defined as Polars-native types. All parsers must produce this exact schema. QC, features, and models only accept the canonical schema.

3. **Feature set registry** — Named feature sets (`baseline`, `enriched`, `full`) defined declaratively. Models request a feature set by name, not by constructing features manually.

4. **MLflow-everything** — Every script that produces a result logs to MLflow. No result exists outside MLflow. Evaluation metrics, feature importance, residual plots — all as artifacts.

5. **Script-as-CLI** — No framework orchestration. Each script is a standalone CLI entry point. Pipeline = running scripts in order. Simple, debuggable, no magic.

### Data Flow

```
Raw CSV/ZIP (Zenodo)
    │
    ▼
[Parser] kelmarsh.py / hill_of_towie.py / penmanshiel.py
    │   Maps native columns → canonical schema
    ▼
[QC] qc.py
    │   Filters maintenance, outliers, curtailment. Gap-fills < 30 min.
    ▼
Clean Parquet (data/processed/)
    │
    ├──► [Features] pipeline.py
    │       Builds feature sets (baseline/enriched/full)
    │       Adds NWP via open_meteo.py
    │       ▼
    │   Feature Parquet (data/features/)
    │
    ├──► [Train] xgboost.py / lightgbm.py / persistence.py
    │       Temporal train/val/test split
    │       Logs everything to MLflow
    │       ▼
    │   MLflow run (mlruns/)
    │
    └──► [Evaluate] evaluation.py
            MAE, RMSE, skill score, regime analysis
            Compares across models and datasets in MLflow
```

---

## 7. Feature Specifications

### 7.1 Canonical SCADA Schema

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `timestamp_utc` | `Datetime` | UTC | Observation time (start of period) |
| `dataset_id` | `Utf8` | — | Dataset identifier (e.g., "kelmarsh") |
| `turbine_id` | `Utf8` | — | Turbine identifier |
| `active_power_kw` | `Float64` | kW | Active power output |
| `wind_speed_ms` | `Float64` | m/s | Hub-height wind speed |
| `wind_direction_deg` | `Float64` | ° | Wind direction (0-360, north=0) |
| `pitch_angle_deg` | `Float64` | ° | Blade pitch angle |
| `rotor_rpm` | `Float64` | RPM | Rotor speed |
| `nacelle_direction_deg` | `Float64` | ° | Nacelle orientation |
| `ambient_temp_c` | `Float64` | °C | Ambient temperature |
| `nacelle_temp_c` | `Float64` | °C | Nacelle temperature |
| `status_code` | `Int32` | — | Operational status |
| `is_curtailed` | `Boolean` | — | Curtailment flag (derived) |
| `is_maintenance` | `Boolean` | — | Maintenance flag (derived) |
| `qc_flag` | `UInt8` | — | QC result (0=ok, 1=suspect, 2=bad) |

### 7.2 Feature Sets

**Baseline:**
- `wind_speed_ms`, `wind_direction_sin`, `wind_direction_cos`
- `active_power_lag_1` through `active_power_lag_6` (10-min lags)
- `wind_speed_rolling_mean_1h`, `_6h`, `_24h`
- `wind_speed_rolling_std_1h`, `_6h`, `_24h`
- `active_power_rolling_mean_1h`, `_6h`, `_24h`

**Enriched** (baseline +):
- `wind_speed_cubed` (V³ — power curve proxy)
- `stability_proxy` (nacelle_temp - ambient_temp)
- `wind_direction_sector` (8 or 16 bins)
- `turbulence_intensity` (wind_speed_std / wind_speed_mean over 1h)

**Full** (enriched +):
- `nwp_wind_speed_100m` (Open-Meteo forecast at prediction horizon)
- `nwp_wind_direction_100m`
- `nwp_temperature_2m`
- `nwp_pressure_msl`
- `hour_sin`, `hour_cos` (cyclic encoding)
- `month_sin`, `month_cos`
- `day_of_week_sin`, `day_of_week_cos`

### 7.3 QC Pipeline

| Check | Rule | Action |
|-------|------|--------|
| Maintenance | status_code indicates non-operational | Flag `is_maintenance=True`, exclude from training |
| Negative power | active_power_kw < 0 | Flag `qc_flag=2` |
| Over-rated power | active_power_kw > rated_power × 1.05 | Flag `qc_flag=1` |
| Negative wind speed | wind_speed_ms < 0 | Flag `qc_flag=2` |
| Extreme wind speed | wind_speed_ms > 40 m/s | Flag `qc_flag=1` |
| Curtailment | Power below curve at high wind + pitch > threshold | Flag `is_curtailed=True` |
| Small gaps | Missing timestamps < 30 min | Forward-fill |
| Large gaps | Missing timestamps ≥ 30 min | Leave as null, do not interpolate |
| Frozen sensor | Same value repeated > 1 hour | Flag `qc_flag=1` |

### 7.4 Evaluation Framework

**Metrics (computed per horizon):**

| Metric | Formula | Purpose |
|--------|---------|---------|
| MAE | mean(\|y - ŷ\|) | Primary error metric |
| RMSE | sqrt(mean((y - ŷ)²)) | Penalizes large errors |
| MAPE | mean(\|y - ŷ\| / \|y\|) × 100 | Relative error (caution: unstable near zero) |
| Skill Score | 1 - RMSE_model / RMSE_persistence | **Key metric** — must be > 0 to be useful |
| Bias | mean(ŷ - y) | Systematic over/under prediction |
| CRPS | Continuous Ranked Probability Score | Probabilistic forecast quality |

**Analysis dimensions:**
- By forecast horizon (H+1 through H+48)
- By wind regime (low: <5 m/s, medium: 5-12 m/s, high: >12 m/s)
- By season (DJF, MAM, JJA, SON)
- By time of day (night, morning, afternoon, evening)
- Power curve residuals (error vs wind speed scatter)

---

## 8. Technology Stack

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| **Language** | Python | 3.12+ | ML ecosystem |
| **Package manager** | uv | latest | Fast, lockfile-based |
| **Data processing** | Polars | ≥1.0 | Primary DataFrame library. No pandas. |
| **ML: gradient boosting** | XGBoost | ≥2.0 | Quantile regression objective |
| **ML: gradient boosting** | LightGBM | ≥4.0 | Comparison benchmark |
| **ML: utilities** | scikit-learn | ≥1.4 | Metrics, preprocessing |
| **Experiment tracking** | MLflow | ≥2.10 | File-based tracking, UI |
| **Hyperparameter tuning** | Optuna | ≥3.0 | Bayesian optimization (used conservatively) |
| **Weather data** | open-meteo-requests | latest | NWP forecasts + historical |
| **Config** | Pydantic | ≥2.0 | Settings, validation |
| **Linting** | ruff | latest | Lint + format |
| **Type checking** | pyright | latest | Static analysis |
| **Testing** | pytest | latest | Unit + integration |
| **Serialization** | Parquet | via Polars | Columnar, compressed |

### Not Used (and why)

| Technology | Reason |
|-----------|--------|
| pandas | Polars is faster for 10-min SCADA at scale. Single library policy. |
| Kedro | Overhead > benefit for solo POC. Scripts + MLflow cover 80% of value. |
| OpenSTEF | Requires InfluxDB + MySQL. Too heavy. |
| PyTorch/TensorFlow | XGBoost/LightGBM sufficient. Don't add GPU dependency for POC. |
| Docker | Not needed for local development POC. |

---

## 9. Configuration

### Pydantic Settings (`src/windcast/config.py`)

```python
class WindCastSettings(BaseSettings):
    # Paths
    data_dir: Path = Path("data")
    mlflow_tracking_uri: str = "file:./mlruns"

    # Dataset
    dataset_id: str = "kelmarsh"

    # Training
    train_years: int = 5
    val_years: int = 1
    test_years: int = 1
    forecast_horizons: list[int] = [1, 6, 12, 24, 48]  # in 10-min steps

    # Turbine specs (per dataset)
    rated_power_kw: float = 2050.0  # Senvion MM92 default

    # QC thresholds
    max_wind_speed_ms: float = 40.0
    max_gap_fill_minutes: int = 30
    frozen_sensor_threshold_minutes: int = 60
```

### Environment Variables

```env
# Optional overrides
WINDCAST_DATA_DIR=./data
WINDCAST_MLFLOW_TRACKING_URI=file:./mlruns
WINDCAST_DATASET_ID=kelmarsh
```

No secrets required for MVP (all datasets are open, Open-Meteo is free without API key).

---

## 10. Datasets

### Selected (5 datasets, 3 categories)

Full evaluation and catalog: [`docs/research/datasets-catalog-2026-03-31.md`](../docs/research/datasets-catalog-2026-03-31.md)

#### SCADA (wind + production)

| Dataset | Turbines | OEM | Period | Size | Source |
|---------|----------|-----|--------|------|--------|
| **Kelmarsh v4** | 6 × Senvion MM92 | Senvion | 2016–2024 | 3.8 GB | [Zenodo](https://zenodo.org/records/16807551) |
| **Hill of Towie** | 21 × Siemens SWT-2.3 | Siemens | 2016–2024 | 12.6 GB | [Zenodo](https://zenodo.org/records/14870023) |
| **Penmanshiel v3** | 13 × Senvion MM82 | Senvion | 2016–2024 | 7.5 GB | [Zenodo](https://zenodo.org/records/16807304) |

#### Wind-only (NWP validation & features)

| Dataset | Type | Period | Source |
|---------|------|--------|--------|
| **FINO1** | Offshore mast, North Sea | 2003–present | [BSH](https://login.bsh.de) |
| **NWTC M2** | Onshore mast 82m, Colorado | 1996–present | [NREL MIDC](https://midcdmz.nrel.gov/apps/sitehome.pl?site=NWTC) |

### Generalization Path

```
Phase 1:  Kelmarsh v4       → prove the pipeline (6 turbines, Senvion)
Phase 2a: Hill of Towie      → different OEM (21 turbines, Siemens)
Phase 2b: Penmanshiel v3     → same OEM, different site (13 turbines, Senvion)
Phase 2c: FINO1 + NWTC M2   → NWP validation, wind shear features
```

---

## 11. Success Criteria

### MVP is successful when:

1. ✅ Kelmarsh pipeline runs end-to-end: raw CSV → clean Parquet → features → model → evaluation
2. ✅ At least one model beats persistence at all horizons (H+1 through H+48) with skill score > 0
3. ✅ XGBoost enriched achieves skill score ≥ 0.3 at H+24 on Kelmarsh
4. ✅ Hill of Towie runs through the same pipeline with only a new parser (no model code changes)
5. ✅ All experiments are logged in MLflow with full reproducibility (features, params, metrics)
6. ✅ Probabilistic forecasts (P10/P50/P90) have calibrated coverage (80% ± 5% within P10–P90)
7. ✅ Error analysis by wind regime is available for every evaluation run
8. ✅ `ruff check`, `pyright`, and `pytest` all pass

### Quality Indicators

- **Time to onboard new dataset**: < 2 hours (write parser + run pipeline)
- **MLflow experiment comparison**: visual cross-site, cross-model comparison in one UI
- **Code coverage**: > 80% on core modules (schema, QC, features, evaluation)
- **Documentation**: README with quick start, dataset info, and results summary

---

## 12. Implementation Phases

### Phase 1 — Data Foundation (1 week)

**Goal:** Ingest Kelmarsh data, clean it, and produce feature-ready Parquet files.

**Deliverables:**
- ✅ 1.1 Project setup: `pyproject.toml`, uv, ruff, pyright, pytest, MLflow
- ✅ 1.2 Canonical SCADA schema (`src/windcast/data/schema.py`)
- ✅ 1.3 Kelmarsh parser + signal mapping (`src/windcast/data/kelmarsh.py`)
- ✅ 1.4 QC pipeline (`src/windcast/data/qc.py`)
- ✅ 1.5 Pydantic config (`src/windcast/config.py`)
- ✅ 1.6 Feature engineering pipeline (`src/windcast/features/pipeline.py`)
- ✅ 1.7 Open-Meteo NWP client (`src/windcast/data/open_meteo.py`)
- ✅ 1.8 Ingestion script (`scripts/ingest_kelmarsh.py`)
- ✅ 1.9 Feature building script (`scripts/build_features.py`)
- ✅ 1.10 Tests for schema, parser, QC, features

**Validation:** `uv run python scripts/ingest_kelmarsh.py` produces clean Parquet. `uv run python scripts/build_features.py` produces feature Parquet. All tests pass.

### Phase 2 — ML Experimentation (1–2 weeks)

**Goal:** Train, evaluate, and compare models on Kelmarsh with full MLflow tracking.

**Deliverables:**
- ✅ 2.1 Persistence benchmark (`src/windcast/models/persistence.py`)
- ✅ 2.2 XGBoost quantile regression (`src/windcast/models/xgboost.py`)
- ✅ 2.3 LightGBM comparison (`src/windcast/models/lightgbm.py`)
- ✅ 2.4 Evaluation framework with skill scores (`src/windcast/models/evaluation.py`)
- ✅ 2.5 CQR probabilistic calibration
- ✅ 2.6 Training script with MLflow logging (`scripts/train.py`)
- ✅ 2.7 Evaluation script with regime analysis (`scripts/evaluate.py`)
- ✅ 2.8 Tests for models and evaluation

**Validation:** `mlflow ui` shows ranked experiments. At least one model has skill score > 0 at all horizons. P10–P90 coverage is calibrated.

### Phase 2b — Generalization Test (1 week)

**Goal:** Prove the pipeline is dataset-agnostic by running on Hill of Towie and Penmanshiel.

**Deliverables:**
- ✅ 2b.1 Hill of Towie parser (`src/windcast/data/hill_of_towie.py`)
- ✅ 2b.2 Penmanshiel parser (`src/windcast/data/penmanshiel.py`)
- ✅ 2b.3 Ingestion scripts for both datasets
- ✅ 2b.4 Run ML pipeline unchanged on both datasets
- ✅ 2b.5 Cross-site comparison in MLflow UI

**Validation:** Same `train.py` and `evaluate.py` scripts work without modification. MLflow shows side-by-side comparison across 3 datasets.

### Phase 3 — Production (future, deferred)

**Goal:** Deployment, monitoring, and automated re-calibration.

- ❌ 3.1 MLflow Model Registry integration
- ❌ 3.2 FastAPI prediction endpoint
- ❌ 3.3 Streamlit dashboard
- ❌ 3.4 Drift monitoring + re-calibration triggers

**Verdict:** Phase 1 + 2 deliver the core value. Phase 3 is real MLOps — defer until the POC proves the concept.

---

## 13. Future Considerations

- **SunCast**: If the framework is truly standardized, adding solar PV should be straightforward (irradiance instead of wind speed, different power curve). Design interfaces with this in mind but don't build for it.
- **Foundation model benchmark**: Compare WindCast XGBoost against WindFM (zero-shot Transformer) on same datasets.
- **Supabase backend**: Replace local Parquet with Supabase PostgreSQL for multi-user access.
- **Rolling-window retraining**: Daily recalibration like WattCast's LEAR approach — consider from Phase 2 onwards.
- **Business metrics**: Imbalance cost reduction, trading value — requires market data integration.
- **FINO1/NWTC M2 integration**: Use wind-only datasets for NWP bias correction features (wind shear exponent, stability indices).

---

## 14. Risks & Mitigations

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 1 | **Power curve non-linearity** — flat at rated wind speed, model struggles in high-wind regime | Medium — poor forecasts > 12 m/s | V³ feature, wind regime segmentation, regime-specific evaluation |
| 2 | **Curtailment distortion** — curtailed periods look like low production at high wind | High — corrupts power-wind relationship | QC pipeline flags curtailment (pitch + power vs curve); exclude from training |
| 3 | **NWP temporal misalignment** — using observation-time NWP instead of forecast-time | High — data leakage, overoptimistic results | Strict alignment: NWP at forecast issuance time, not observation time (WattCast v5.1 lesson) |
| 4 | **Optuna overfitting** — tuning on validation set, degrading on test | Medium — v6-worse-than-v5 WattCast lesson | Conservative defaults first; tune only with held-out test; limited trial budget (100-200) |
| 5 | **Hill of Towie signal gaps** — wind speed may not be in standard SCADA fields | Medium — blocks generalization test | Download description CSVs first; verify signal availability before building parser |

---

## 15. Appendix

### Key Documents

| Document | Purpose |
|----------|---------|
| [`docs/research/brainstorming-2026-03-31.md`](../docs/research/brainstorming-2026-03-31.md) | Feasibility study, architecture decisions, WattCast lessons |
| [`docs/research/datasets-catalog-2026-03-31.md`](../docs/research/datasets-catalog-2026-03-31.md) | Full dataset evaluation (WRAG + web research) |
| [`docs/research/methodology-scaling-pipeline.md`](../docs/research/methodology-scaling-pipeline.md) | Broader methodology for scaling weather-to-energy pipelines |

### WattCast Lessons Applied

| Lesson | Application |
|--------|------------|
| Training data volume > features | Use ≥ 3 years for training (Kelmarsh has 9) |
| Future covariates matter | NWP at prediction horizon, not observation time |
| Optuna can overfit | Conservative defaults first, limited trial budget |
| CQR is simple and effective | Same 15-line numpy pattern |
| Feature pruning doesn't help XGBoost | Let `colsample_bytree` handle it |
| Daily recalibration >> fixed split | Consider rolling-window from Phase 2 |

### Existing Open-Source Projects (reference only, not dependencies)

| Project | Relevance |
|---------|-----------|
| [OpenSTEF](https://github.com/OpenSTEF/openstef) | Architecture inspiration (MLflow + XGBoost pattern) |
| [NREL OpenOA](https://github.com/NREL/OpenOA) | QC and PlantData schema patterns |
| [WindFM](https://github.com/shiyu-coder/WindFM) | Future benchmark (zero-shot Transformer) |
