# EnerCast — Product Requirements Document

**Version**: 2.1
**Last Updated**: 2026-04-02
**Context**: ML framework for energy engineering professionals — demonstrated via WeatherNews challenge
**Target**: Production-ready framework across 3 energy domains (Wind, Demand, Solar)

---

## 1. Executive Summary

EnerCast is an ML framework for energy engineering professionals. It provides pre-built data connectors, canonical schemas, QC pipelines, baseline models, and experiment tracking — so domain experts can focus on what actually drives forecast accuracy: feature engineering, model selection, and domain-specific evaluation.

The framework follows a single pipeline pattern — parse, validate, QC, engineer features, train, evaluate, track — that works across fundamentally different energy forecasting domains: wind generation, power demand, and solar generation. Domain-specific code is limited to a parser (~100 lines) and a feature configuration.

EnerCast is demonstrated as a **technical case study for WeatherNews (WN)**, proving that standardizing the mechanical parts of the ML pipeline frees engineer time for what matters: understanding client-specific data and context. This directly addresses WN's core tension: *standardize to liberate brain time, without blocking innovation*.

**Key value proposition:** Engineers who know their domain (spot prices, wake effects, clearsky ratios) and know ML (XGBoost, feature engineering) get a framework where data plumbing, schemas, QC, and tracking are already done. They iterate on features, models, and KPIs — not on boilerplate.

---

## 2. Mission

**Give energy engineers a framework where the plumbing is solved, so they spend time on features, models, and domain expertise — not rebuilding pipelines.**

### Core Principles

1. **Domain-agnostic core** — one pipeline pattern, many domains. Adding a domain = writing a parser + feature config.
2. **Expert-friendly** — full control over feature sets, model choice, evaluation metrics. No black boxes.
3. **Reproducible by default** — every run tracked in MLflow with features, params, metrics, artifacts.
4. **Simple over clever** — scripts over frameworks, Parquet over databases, XGBoost over Transformers.
5. **Fail fast** — detailed errors, strict schemas, no silent data corruption.
6. **Extensible, not rigid** — custom metrics, custom features, custom models plug in without touching the core.

---

## 3. Target Users

### Primary: Energy domain professionals

Engineers and data scientists who work in energy forecasting (wind, solar, demand, gas). They:
- **Know their domain** — power curves, wake effects, spot price dynamics, demand seasonality, solar degradation
- **Know ML** — comfortable with XGBoost/LightGBM, feature engineering, train/val/test splits, hyperparameter tuning
- **Need to iterate fast** — swap feature sets, compare models, test domain hypotheses in minutes
- **Don't want to rebuild plumbing** — data ingestion, QC, schemas, MLflow logging should be ready to use

What they need from EnerCast:
- Pre-built connectors to common data sources (SCADA, ENTSO-E, PVDAQ, Open-Meteo)
- Canonical schemas they can extend with domain-specific fields
- Parameterizable QC rules they can tune per client/site
- Named feature sets (baseline/enriched/full) they can modify and compare
- Pluggable models and custom evaluation metrics (e.g., "MAPE when spot price > X")
- MLflow tracking for reproducibility and cross-experiment comparison

### WeatherNews evaluation panel (demo context)

- **Yoel Chetboun** (Resp. Energie) — evaluates domain relevance, pain point coverage
- **Michel Kolasinski** (Head of Ops) — evaluates operational feasibility, incremental adoption
- **Craig West** (remote) — evaluates technical depth, architecture quality

What they need to see:
- That standardization is **possible** across their 3 domains
- That it **doesn't kill flexibility** (custom metrics, custom features, new models)
- That onboarding a new client/dataset is **fast** (hours, not days)
- That experiments are **reproducible and comparable** (MLflow)
- That the approach is **incremental** (no big bang, R can coexist)

---

## 4. MVP Scope

### In Scope (Demo)

**Multi-Domain Core:**
- ✅ Domain-agnostic schema system (wind, solar, demand each have a typed schema)
- ✅ Parser pattern (1 parser per dataset, maps to canonical schema)
- ✅ Generic QC pipeline (parameterizable rules per domain)
- ✅ Feature set registry (named sets: baseline / enriched / full, per domain)
- ✅ Generic train/evaluate loop with MLflow logging
- ✅ Cross-domain experiment comparison in MLflow UI

**Wind Domain (Kelmarsh):**
- ✅ Kelmarsh v4 parser + signal mapping (DONE)
- ✅ SCADA schema — 15 columns, typed (DONE)
- ✅ QC pipeline — 9 rules (DONE)
- ✅ Wind feature sets (baseline: lags + rolling stats, enriched: V3 + stability, full: + NWP)
- ✅ Open-Meteo NWP client (DONE)
- ✅ XGBoost + persistence benchmark, skill scores

**Demand Domain (Spain ENTSO-E via Kaggle):**
- ✅ Spain demand parser (hourly load + weather + prices, CC0)
- ✅ Demand schema (timestamp, load_mw, temperature, wind_speed, price_eur)
- ✅ Demand QC (outlier detection, holiday handling, gap-fill)
- ✅ Demand feature sets (baseline: lags H-1/D-1/W-1 + calendar, enriched: + temperature + rolling, full: + weather ensemble)
- ✅ XGBoost + persistence benchmark, MAPE, skill scores

**Solar Domain (PVDAQ — stretch goal):**
- ✅ PVDAQ System 2 parser (NREL, Golden CO, CC-BY)
- ✅ Solar schema (timestamp, power_kw, ghi, temperature, etc.)
- ✅ Solar feature sets (baseline: irradiance + lags, enriched: + clearsky ratio + temperature)
- ✅ Same train/evaluate pipeline, MLflow comparison

**Quality:**
- ✅ ruff linting + formatting
- ✅ pyright type checking
- ✅ pytest test suite
- ✅ Pydantic config validation

### Out of Scope

- ❌ Deep learning models (LSTM, Transformer)
- ❌ Real-time deployment / API
- ❌ Shiny/Streamlit dashboard
- ❌ AWS deployment (mentioned in presentation narrative, not built)
- ❌ Gas-specific forecasting (demand demo covers the pattern)
- ❌ Drift monitoring, automated retraining
- ❌ Hill of Towie / Penmanshiel parsers (deferred — Kelmarsh proves the point)
- ❌ CQR probabilistic calibration (deferred to after demo)

---

## 5. User Stories

1. **As a WN evaluator**, I want to see the same pipeline handle wind and demand data, so that I believe standardization is possible across our 3 domains.

2. **As a WN evaluator**, I want to see that adding a new dataset requires only a parser (~100 lines), so that I believe onboarding a new client takes hours, not days.

3. **As a WN evaluator**, I want to see MLflow tracking across domains, so that I believe experiment reproducibility and comparison is built-in.

4. **As a WN evaluator**, I want to see custom metrics plugged in easily, so that I believe the framework doesn't block client-specific evaluation (e.g., "accuracy when spot price is high").

5. **As an ML engineer**, I want standardized feature sets per domain, so that I start with proven features and customize from there.

6. **As an ML engineer**, I want the QC pipeline to be parameterizable, so that I can adjust thresholds per client without rewriting QC code.

7. **As an ML engineer**, I want to swap models (XGBoost → LightGBM → new model) without changing the pipeline, so that I can innovate freely.

---

## 6. Core Architecture

### Directory Structure

```
enercast/                               # (currently src/windcast/, rename later)
├── config.py                           # Pydantic Settings, dataset configs
│
├── schemas/                            # Domain-specific schemas
│   ├── wind.py                         # SCADA schema (15 cols)
│   ├── demand.py                       # Demand schema
│   └── solar.py                        # Solar schema (stretch)
│
├── parsers/                            # One parser per dataset
│   ├── kelmarsh.py                     # Wind — Kelmarsh v4
│   ├── spain_demand.py                 # Demand — Kaggle Spain ENTSO-E
│   └── pvdaq.py                        # Solar — NREL PVDAQ (stretch)
│
├── qc/                                 # Generic QC, parameterized
│   └── pipeline.py                     # Rules engine (works on any schema)
│
├── features/                           # Feature engineering
│   ├── registry.py                     # Feature set registry (baseline/enriched/full)
│   ├── wind.py                         # Wind-specific features
│   ├── demand.py                       # Demand-specific features
│   └── solar.py                        # Solar-specific features (stretch)
│
├── models/                             # ML models
│   ├── persistence.py                  # Naive benchmark (any domain)
│   ├── xgboost_model.py               # XGBoost (any domain)
│   └── evaluation.py                   # Metrics, skill scores, analysis
│
└── tracking/                           # MLflow integration
    └── mlflow.py                       # Logging, comparison utilities
```

### Key Design Patterns

1. **Schema-per-domain** — Each domain (wind, demand, solar) has a typed Polars schema. Parsers must produce this exact schema. Everything downstream is generic.

2. **Parser pattern** — Each dataset has a dedicated parser. The parser is the ONLY dataset-specific code. ~100 lines each.

3. **Feature set registry** — Named feature sets per domain, defined declaratively. Models request a set by name.

4. **Model-agnostic training** — Train loop accepts any scikit-learn-compatible model. Swap XGBoost for LightGBM or a new model with zero pipeline changes.

5. **Pluggable metrics** — Evaluation accepts custom metric functions. "Accuracy when spot price > X" = one lambda.

6. **MLflow-everything** — Every script logs to MLflow. Cross-domain comparison in one UI.

### Data Flow (Generic)

```
Raw data (CSV/ZIP/API)
    │
    ▼
[Parser] domain-specific mapping → canonical schema
    │
    ▼
[QC] parameterized rules → flagged data
    │
    ▼
Clean Parquet (data/processed/)
    │
    ▼
[Features] domain feature set → feature Parquet
    │
    ▼
[Train] temporal split → model → MLflow run
    │
    ▼
[Evaluate] metrics + skill scores + regime analysis → MLflow artifacts
```

---

## 7. Feature Specifications

### 7.1 Wind Schema (existing — SCADA)

15 columns: `timestamp_utc`, `dataset_id`, `turbine_id`, `active_power_kw`, `wind_speed_ms`, `wind_direction_deg`, `pitch_angle_deg`, `rotor_rpm`, `nacelle_direction_deg`, `ambient_temp_c`, `nacelle_temp_c`, `status_code`, `is_curtailed`, `is_maintenance`, `qc_flag`

### 7.2 Demand Schema (new)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `timestamp_utc` | `Datetime` | UTC | Observation time |
| `dataset_id` | `Utf8` | — | Dataset identifier |
| `zone_id` | `Utf8` | — | Grid zone / country |
| `load_mw` | `Float64` | MW | Actual demand |
| `temperature_c` | `Float64` | °C | Temperature (weighted avg) |
| `wind_speed_ms` | `Float64` | m/s | Wind speed |
| `humidity_pct` | `Float64` | % | Relative humidity |
| `price_eur_mwh` | `Float64` | EUR/MWh | Day-ahead spot price (if available) |
| `is_holiday` | `Boolean` | — | Public holiday flag |
| `is_dst_transition` | `Boolean` | — | DST change flag |
| `qc_flag` | `UInt8` | — | QC result (0=ok, 1=suspect, 2=bad) |

### 7.3 Solar Schema (stretch)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `timestamp_utc` | `Datetime` | UTC | Observation time |
| `dataset_id` | `Utf8` | — | Dataset identifier |
| `system_id` | `Utf8` | — | PV system identifier |
| `power_kw` | `Float64` | kW | AC power output |
| `ghi_wm2` | `Float64` | W/m² | Global Horizontal Irradiance |
| `poa_wm2` | `Float64` | W/m² | Plane of Array irradiance |
| `ambient_temp_c` | `Float64` | °C | Ambient temperature |
| `module_temp_c` | `Float64` | °C | Module temperature |
| `wind_speed_ms` | `Float64` | m/s | Wind speed |
| `qc_flag` | `UInt8` | — | QC result |

### 7.4 Feature Sets by Domain

**Wind — Baseline:** wind_speed, direction (sin/cos), power lags (1-6), rolling stats (1h/6h/24h)
**Wind — Enriched:** + V³, stability proxy, direction sectors, turbulence intensity
**Wind — Full:** + NWP (Open-Meteo), calendar (cyclic)

**Demand — Baseline:** load lags (H-1, H-2, D-1, W-1), hour/dow/month (cyclic)
**Demand — Enriched:** + temperature, heating/cooling degree days, rolling load stats
**Demand — Full:** + wind speed, humidity, price lags, holiday features

**Solar — Baseline:** GHI, power lags, hour (cyclic), clearsky GHI ratio
**Solar — Enriched:** + temperature, module temp, wind speed, rolling irradiance stats
**Solar — Full:** + NWP forecasts, cloud cover

### 7.5 Evaluation Framework

**Metrics (all domains):**

| Metric | Formula | Purpose |
|--------|---------|---------|
| MAE | mean(\|y - y_hat\|) | Primary error metric |
| RMSE | sqrt(mean((y - y_hat)²)) | Penalizes large errors |
| MAPE | mean(\|y - y_hat\| / \|y\|) × 100 | Relative error |
| Skill Score | 1 - RMSE_model / RMSE_persistence | **Key metric** — must be > 0 |
| Bias | mean(y_hat - y) | Systematic over/under |

**Custom metric support:** Evaluation accepts `Callable[[array, array], float]` — plug in "accuracy when spot > X" trivially.

**Analysis dimensions:** by forecast horizon, by regime (wind: low/med/high wind; demand: weekday/weekend, season; solar: clear/cloudy).

---

## 8. Technology Stack

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| **Language** | Python | 3.12+ | ML ecosystem |
| **Package manager** | uv | latest | Fast, lockfile-based |
| **Data processing** | Polars | >=1.0 | Primary. No pandas. |
| **ML** | XGBoost | >=2.0 | Quantile regression |
| **ML** | LightGBM | >=4.0 | Comparison |
| **ML utilities** | scikit-learn | >=1.4 | Metrics, preprocessing |
| **Experiment tracking** | MLflow | >=2.10 | File-based tracking |
| **Weather data** | Open-Meteo | latest | Free NWP, no API key |
| **Config** | Pydantic | >=2.0 | Settings, validation |
| **Linting** | ruff | latest | Lint + format |
| **Type checking** | pyright | latest | Static analysis |
| **Testing** | pytest | latest | Unit + integration |

---

## 9. Configuration

### Pydantic Settings

```python
class EnerCastSettings(BaseSettings):
    data_dir: Path = Path("data")
    mlflow_tracking_uri: str = "file:./mlruns"
    domain: str = "wind"  # wind | demand | solar
    dataset_id: str = "kelmarsh"
    forecast_horizons: list[int] = [1, 6, 12, 24, 48]
```

### Dataset Configs

Each dataset has a `DatasetConfig` with domain-specific metadata:
- Wind: `rated_power_kw`, `hub_height_m`, `rotor_diameter_m`, `n_turbines`, `lat/lon`
- Demand: `zone_id`, `population`, `lat/lon` (for weather), `timezone`
- Solar: `capacity_kw`, `tilt_deg`, `azimuth_deg`, `lat/lon`

No secrets required (all datasets are open, Open-Meteo is free).

---

## 10. Datasets

### Selected for Demo

| Domain | Dataset | Source | Resolution | Period | License |
|--------|---------|--------|-----------|--------|---------|
| **Wind** | Kelmarsh v4 | Zenodo | 10 min | 2016-2024 | CC-BY |
| **Demand** | Spain Energy+Weather | Kaggle (ENTSO-E) | 1h | 2015-2018 | CC0 |
| **Solar** | PVDAQ System 2 | NREL/AWS S3 | 15 min | 2005-present | CC-BY 4.0 |

### Why These Datasets

- **Kelmarsh**: Already integrated, 6 turbines, clean SCADA, proves wind pipeline
- **Spain**: Demand + weather + prices in one file, CC0, zero-friction download, 4 MB
- **PVDAQ**: Longest open PV dataset, programmatic access via pvlib, co-located irradiance

---

## 11. Success Criteria

### Demo is successful when:

1. ✅ Wind pipeline runs end-to-end: raw → clean → features → model → MLflow
2. ✅ Demand pipeline runs through the SAME framework with only a parser + feature config
3. ✅ At least one model beats persistence in both wind and demand (skill score > 0)
4. ✅ MLflow UI shows wind and demand experiments side by side
5. ✅ Adding demand required ZERO changes to core pipeline code (QC, train, evaluate)
6. ✅ Custom metric example works (e.g., "MAPE on peak hours only")
7. ✅ `ruff check`, `pyright`, and `pytest` all pass

### Stretch Goals

8. ✅ Solar pipeline also works (3 domains in one framework)
9. ✅ Cross-domain MLflow comparison dashboard

---

## 12. Implementation Phases

### Phase 1 — Wind End-to-End (Day 1)

**Goal:** Complete the wind pipeline from data to MLflow results.

**Deliverables:**
- ✅ 1.1 Feature engineering pipeline (`features/wind.py` + `features/registry.py`)
- ✅ 1.2 Feature building script (`scripts/build_features.py`)
- ✅ 1.3 Persistence benchmark (`models/persistence.py`)
- ✅ 1.4 XGBoost training with MLflow (`models/xgboost_model.py`)
- ✅ 1.5 Evaluation with skill scores (`models/evaluation.py`)
- ✅ 1.6 Training script (`scripts/train.py`)
- ✅ 1.7 Evaluation script (`scripts/evaluate.py`)
- ✅ 1.8 Tests for features, models, evaluation

**Validation:** `mlflow ui` shows Kelmarsh experiments with skill score > 0 at all horizons.

### Phase 2 — Demand Domain (Day 2-3)

**Goal:** Prove the framework is domain-agnostic by adding power demand forecasting.

**Deliverables:**
- ✅ 2.1 Demand schema (`schemas/demand.py`)
- ✅ 2.2 Spain demand parser (`parsers/spain_demand.py`)
- ✅ 2.3 Demand QC rules (outlier detection, holiday handling)
- ✅ 2.4 Demand feature sets (`features/demand.py`)
- ✅ 2.5 Demand dataset config in `config.py`
- ✅ 2.6 Ingestion script (`scripts/ingest_spain_demand.py`)
- ✅ 2.7 Same `train.py` and `evaluate.py` work unchanged
- ✅ 2.8 Tests for demand parser, features

**Validation:** Same training and evaluation scripts work with `--domain demand --dataset spain`. MLflow shows wind + demand side by side.

### Phase 3 — Solar Domain (Day 3-4, stretch)

**Goal:** Add solar as a third domain to complete the trifecta.

**Deliverables:**
- ✅ 3.1 Solar schema (`schemas/solar.py`)
- ✅ 3.2 PVDAQ parser (`parsers/pvdaq.py`)
- ✅ 3.3 Solar feature sets (`features/solar.py`)
- ✅ 3.4 Same pipeline, same scripts, solar results in MLflow

**Validation:** 3 domains in MLflow UI. Zero core pipeline changes for solar.

### Phase 4 — Presentation (Day 5)

**Goal:** Deliver a compelling technical presentation for WN.

**Deliverables:**
- ✅ 4.1 Slide deck (English) — "I understood / Here's proof / Here's the roadmap"
- ✅ 4.2 Live demo script (MLflow UI, code walkthrough)
- ✅ 4.3 WN-specific roadmap (3 horizons: quick wins / consolidation / innovation)
- ✅ 4.4 Dry run

---

## 13. Relevance to WeatherNews

### How EnerCast Maps to WN Workflows

| WN Workflow | EnerCast Equivalent | What It Proves |
|---|---|---|
| Wind & Solar Generation | Wind + Solar domains | Same pipeline, different parsers |
| Gas & Power Demand | Demand domain | Framework handles structurally different data |
| Specific Use Cases (DLR, Power Loss) | Extensibility pattern | Custom features + custom metrics plug in |

### WN Pain Points Addressed

| Pain Point | EnerCast Answer |
|---|---|
| **Productivity** (3/3 domains) | Automated QC, standardized features, one-command training |
| **Accuracy (solar)** | Framework enables rapid A/B testing of feature sets and models |
| **MARS legacy (demand)** | XGBoost/LightGBM as drop-in replacements, proven on same data |
| **R-only deployment** | Python-native, MLflow tracking, AWS-ready architecture |
| **Reproducibility** | MLflow logs everything — features, params, metrics, artifacts |
| **Innovation blocked** | New model = implement `fit/predict`, plug into existing pipeline |

### WN Roadmap (Proposed in Presentation)

| Horizon | What | Impact |
|---|---|---|
| **Quick wins (1-3 months)** | Standardized pipeline for Wind. MARS → XGBoost for Demand. | Productivity + accuracy |
| **Consolidation (3-6 months)** | Shared feature store (weather). MLflow for all domains. Use Cases migration R → Python. | Reproducibility + scalability |
| **Innovation (6-12 months)** | AI weather models integration. CI/CD retraining. Drift monitoring. | Innovation + robustness |

---

## 14. Risks & Mitigations

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 1 | **Demo too ambitious for timeline** | Can't show 3 domains | Priority order: Wind > Demand > Solar. Wind + Demand alone proves the point. |
| 2 | **Spain dataset too clean** — doesn't show real QC challenges | Demo looks trivial | Add realistic QC scenarios, show parameterizable rules |
| 3 | **Demand is structurally different from wind** — shared abstractions may be forced | Framework feels artificial | Keep domain-specific code in separate modules, share only the orchestration pattern |
| 4 | **Presentation audience is mixed** (technical + operational) | Miss one audience | Structure: business narrative (Michel) + architecture (Craig) + demo (Yoel) |
| 5 | **Overengineering the framework** instead of showing results | No concrete accuracy numbers | Focus on end-to-end results first, refactor second |

---

## 15. Appendix

### Key Documents

| Document | Purpose |
|----------|---------|
| `docs/WNchallenge/CR WeatherNews...` | Meeting notes — WN challenge briefing |
| `docs/WNchallenge/Analyse Presentation...` | Detailed analysis of WN slides |
| `docs/research/brainstorming-2026-03-31.md` | Original feasibility study |
| `docs/research/datasets-catalog-2026-03-31.md` | Full dataset evaluation |

### Datasets Sources

| Dataset | Access |
|---------|--------|
| Kelmarsh v4 | `data/KelmarshV4/16807551.zip` (local) |
| Spain Energy+Weather | `kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather` |
| PVDAQ System 2 | `pvlib.iotools.get_pvdaq_data(system_id=2)` or S3: `s3://oedi-data-lake/pvdaq/` |

### Original WindCast Scope (preserved for reference)

The original PRD (v1.0) focused exclusively on wind power forecasting with 3 SCADA datasets (Kelmarsh, Hill of Towie, Penmanshiel). That scope is preserved as a subset of Phase 1. The multi-domain extension was driven by the WeatherNews challenge requirement to demonstrate cross-domain standardization.
