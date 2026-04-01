# WindCast — Brainstorming & Research Report

**Date:** 2026-03-31
**Context:** Feasibility study for a standardized ML framework for wind power forecasting, building on WattCast experience (electricity spot price forecasting for EPEX SPOT FR).

---

## 1. Motivation

WattCast demonstrated that a solo developer can build a production-quality probabilistic forecasting engine using open data and XGBoost. But the approach was artisanal — tightly coupled to French EPEX SPOT prices, with features, evaluation, and data pipelines handcrafted for that single use case.

WindCast asks: **can we standardize the methodology so that onboarding a new wind farm dataset takes hours, not weeks?**

The key insight (from `methodology.md`): the bottleneck is never the ML. It's everything around it — data ingestion, format normalization, QC, experiment tracking, reproducibility.

---

## 2. Open Datasets — Best Candidates for POC

### 2.1 Selected Datasets (5 datasets, 3 categories)

**Full catalog and evaluation:** See [`datasets-catalog-2026-03-31.md`](datasets-catalog-2026-03-31.md)

#### SCADA (wind + production) — for ML training & evaluation

| Dataset | Turbines | OEM | Period | Size | License | Source |
|---------|----------|-----|--------|------|---------|--------|
| **Kelmarsh v4** (primary) | 6 × Senvion MM92 | Senvion | 2016–2024 | 3.8 GB | CC-BY-4.0 | [Zenodo](https://zenodo.org/records/16807551) |
| **Hill of Towie** (generalization) | 21 × Siemens SWT-2.3 | Siemens | 2016–2024 | 12.6 GB | CC-BY-4.0 | [Zenodo](https://zenodo.org/records/14870023) |
| **Penmanshiel v3** (cross-site) | 13 × Senvion MM82 | Senvion | 2016–2024 | 7.5 GB | CC-BY-4.0 | [Zenodo](https://zenodo.org/records/16807304) |

#### Wind-only offshore — for NWP validation

| Dataset | Type | Period | Source |
|---------|------|--------|--------|
| **FINO1** | Offshore mast, 7+ levels (33–100 m), North Sea | 2003–present | [BSH](https://login.bsh.de) (free registration) |

#### Wind-only onshore — for pipeline testing & wind features

| Dataset | Type | Period | Source |
|---------|------|--------|--------|
| **NWTC M2** | 82 m mast, 6 levels (2–80 m), 1-min resolution | 1996–present | [NREL MIDC](https://midcdmz.nrel.gov/apps/sitehome.pl?site=NWTC) (open, API) |

### 2.2 Why Start with Kelmarsh

- **Longest coverage**: 9 years (2016–2024), v4 published August 2025
- **Most cited**: widely used in recent wind energy ML literature
- **Manageable size**: 6 turbines — enough diversity, not overwhelming
- **Same OEM as Penmanshiel** (Senvion) — enables cross-site generalization test with minimal parser changes
- **Well-documented**: signal mapping spreadsheet included

### 2.3 Generalization Path (revised)

1. **Phase 1**: Kelmarsh v4 → prove the pipeline works
2. **Phase 2a**: Hill of Towie → different OEM (Siemens), 21 turbines, proves dataset-agnostic design
3. **Phase 2b**: Penmanshiel v3 → same OEM (Senvion), different site → proves cross-site calibration
4. **Phase 2c**: FINO1 + NWTC M2 → NWP validation, wind shear features

> **Note:** La Haute Borne (ENGIE) replaced by Hill of Towie — ENGIE portal offline as of March 2026, and Hill of Towie is superior on every dimension (21 vs 4 turbines, 8.5 vs 3+ years, different OEM, reliable Zenodo access).

### 2.4 Other Datasets Worth Knowing

| Dataset | Notes |
|---------|-------|
| **Brazilian UEPS+UEBB** | 52 turbines (Enercon), 1 year, NetCDF4, CC BY-SA. Good for geographic diversity. |
| **SDWPF** | 134 turbines (China), 6 months. **CC BY-NC-ND** — benchmark only. |
| **SMARTEOLE** | 7 × Senvion MM82, 3 months, 1-min. Wake steering experiments. |
| **Cabauw/CESAR** | 213 m mast, 50 years, Netherlands. Portal down — contact KNMI. |
| **WIND Toolkit** (NREL) | 126,691 US sites, synthetic. Overkill for POC. |
| **Ørsted/EDP/DTU** | Various portals offline or NDA-gated as of March 2026. |

---

## 3. Existing Open-Source Projects — Landscape

### 3.1 Directly Relevant

| Project | What it does | Stack | Strengths | Weaknesses | Relevance |
|---------|-------------|-------|-----------|------------|-----------|
| **[OpenSTEF](https://github.com/OpenSTEF/openstef)** (LF Energy / Alliander) | AutoML pipeline for short-term energy forecasting. MLflow model storage, XGBoost/LightGBM, feature engineering. | Python, XGBoost, MLflow, InfluxDB, MySQL | Industrial-grade, 2800+ commits, LF Energy backing | Heavy infra (InfluxDB + MySQL mandatory), grid-operator oriented, not wind-farm-owner oriented | **High** — architecture inspiration, but too opinionated to adopt wholesale |
| **[NREL OpenOA](https://github.com/NREL/OpenOA)** | Operational analysis of wind farms: AEP calculation, power curve fitting, data filtering, QC. | Python, Pandas, scikit-learn | Excellent `PlantData` schema, battle-tested QC, NREL authority | No forecasting ML, no experiment tracking | **Medium** — great for Phase 1 (data/QC), not for Phase 2 (ML) |
| **[WindFM](https://github.com/shiyu-coder/WindFM)** | Foundation model for zero-shot wind power forecasting. 8.1M param Transformer. | Python, PyTorch | Zero-shot transfer across geographies, probabilistic output | Requires WIND Toolkit pre-training, PyTorch complexity, research-grade | **Low for POC** — interesting for future comparison benchmark |
| **[Kedro + MLflow Energy Pipeline](https://github.com/labrijisaad/Kedro-Energy-Forecasting-Machine-Learning-Pipeline)** | Modular MLOps pipeline for energy forecasting with Kedro structure + MLflow tracking. | Python, Kedro, MLflow, Docker | Clean separation of concerns, reproducible, Dockerized | Kedro learning curve, abstraction overhead for small projects | **High** — exact pattern we want, but Kedro may be overkill |

### 3.2 Key Takeaways

1. **No one has built exactly what we want**: a lightweight, dataset-agnostic wind power forecasting framework with MLflow tracking that a solo developer can use. OpenSTEF is closest but requires InfluxDB/MySQL infrastructure.

2. **OpenOA's `PlantData` schema is worth studying**: standardized data structures for SCADA + met tower + reanalysis. We should design our canonical schema with similar thinking.

3. **MLflow is the consensus choice** for experiment tracking in this space. Both OpenSTEF and Kedro pipelines use it. No need to evaluate alternatives.

4. **Kedro is a maybe**: great for teams, overhead for solo. We can adopt its *principles* (modular pipelines, data catalog) without the framework.

---

## 4. Architecture Decision: Extend WattCast vs. New Project

### 4.1 Options Evaluated

| Option | Description | Verdict |
|--------|-------------|---------|
| **A. Extend WattCast** | Add wind power forecasting to existing repo | **No** — domains too different (market prices ≠ turbine power), risks destabilizing a delivered POC |
| **B. New project (recommended)** | Fresh repo, inspired by WattCast patterns | **Yes** — clean separation, better for portfolio, can become a reusable template |
| **C. Monorepo with shared packages** | `energy-forecast/` workspace with `core/`, `wattcast/`, `windcast/` | **Premature** — factorize later if patterns emerge |

### 4.2 Rationale for Option B

- **WattCast is done** (Phase 7 complete, POC report delivered). Don't touch a working system.
- **Fundamentally different data**: SCADA 10-min turbine measurements vs. hourly electricity prices. Different features, different horizons, different evaluation metrics.
- **Different ML objective**: power curve calibration + production forecast vs. price forecasting with market features.
- **Portfolio value**: two projects showing the same methodology applied to different domains > one monolithic project.
- **MLflow from day 1**: WattCast didn't use MLflow. WindCast can showcase proper experiment tracking as a methodological upgrade.

### 4.3 What to Reuse from WattCast (copy & adapt, not import)

| Pattern | WattCast file | Adaptation for WindCast |
|---------|--------------|------------------------|
| Pydantic config | `src/wattcast/config.py` | Same pattern, different constants (turbine specs, SCADA signals) |
| Supabase storage | `src/wattcast/data/storage.py` | Same pattern, `wcast_` → `wind_` table prefix |
| Open-Meteo client | `src/wattcast/data/open_meteo.py` | Reuse almost as-is (weather for turbine locations) |
| XGBoost quantile | `src/wattcast/models/xgboost.py` | Same model, different features and objective |
| Evaluation metrics | `src/wattcast/models/evaluation.py` | Add wind-specific metrics (skill score vs persistence) |
| FastAPI structure | `src/wattcast/api/` | Same pattern if we reach Phase 3 |
| Streamlit dashboard | `dashboard.py` | Same pattern, different visualizations |

---

## 5. Difficulty Assessment by Phase

### Phase 1 — Data Ingestion & QC (Difficulty: 2/5 — Moderate)

**What to build:**
- Canonical SCADA schema: `timestamp_utc`, `turbine_id`, `active_power_kw`, `wind_speed_ms`, `wind_direction_deg`, `pitch_angle_deg`, `rotor_rpm`, `nacelle_temp_c`, `status_code`
- Dataset parsers: one per dataset (Kelmarsh, La Haute Borne, Penmanshiel) that map native column names → canonical schema
- QC pipeline: filter maintenance periods (status codes), detect outliers (power > rated, negative wind speed), forward-fill small gaps (< 30 min), flag curtailment
- Storage: Parquet files initially, Supabase later

**Risks:**
- Signal mapping inconsistencies (same OEM, different naming across datasets)
- Status code interpretation varies by OEM and supervision platform
- Missing data patterns differ (Kelmarsh has gaps, La Haute Borne has different coverage)

**Estimated effort:** 2–3 days per dataset, 1 week total for Kelmarsh + framework.

### Phase 2 — ML Experimentation (Difficulty: 3/5 — Significant)

**What to build:**
- **Feature engineering** (standardized sets):
  - *Baseline*: wind speed, wind direction (sin/cos), active power lags, rolling stats (mean, std, min, max over 1h/6h/24h)
  - *Enriched*: V³ (cube law proxy for power curve), stability proxy (nacelle temp vs ambient), direction sectors (binned)
  - *NWP-augmented*: Open-Meteo forecasts at turbine location (wind speed/direction at 100m, temperature, pressure)
  - *Calendar*: hour of day, day of week, month (cyclic encoding), holidays
- **Multi-model comparison**: XGBoost, LightGBM, Ridge regression (baseline)
- **Multi-horizon**: H+1 to H+48 (10-min steps) or aggregated (1h, 6h, 24h, 48h)
- **MLflow tracking**: every run logged with features, params, metrics, artifacts
- **Evaluation framework**: MAE, RMSE, MAPE, skill score vs persistence, error by wind regime (low/medium/high), power curve residual analysis

**Risks:**
- Power curve non-linearity (flat at rated wind speed) makes forecasting tricky
- Curtailment events distort the power-wind relationship — need proper filtering
- NWP temporal alignment (forecast vs observation) is critical (lesson from WattCast v5.1)

**Estimated effort:** 1–2 weeks for framework + first model, then < 1 day per new dataset.

### Phase 3 — Deployment & Monitoring (Difficulty: 4/5 — High)

**What to build:**
- MLflow Model Registry (staging → production → archived)
- FastAPI prediction endpoint
- Performance monitoring (rolling metrics, drift detection)
- Automated re-calibration triggers

**Verdict: skip for POC.** Phase 1 + 2 deliver the core value (standardized, reproducible ML pipeline). Phase 3 is real MLOps — valuable but not needed to prove the concept.

---

## 6. Recommended Tech Stack

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Language** | Python 3.12+ | ML ecosystem, consistent with WattCast |
| **Package manager** | uv | Fast, lockfile-based, proven in WattCast |
| **Data processing** | Polars (primary) | Performance on 10-min SCADA, WattCast precedent |
| **QC / Power curves** | Custom + OpenOA patterns | Don't reinvent filtering; adapt PlantData ideas |
| **ML models** | XGBoost, LightGBM, scikit-learn (Ridge) | Standard, proven, fast |
| **Experiment tracking** | **MLflow** | The key addition vs WattCast. Model registry, comparison UI, artifacts |
| **Hyperparameter tuning** | Optuna | Same as WattCast (but careful — Optuna overfitting lessons apply) |
| **Weather data** | Open-Meteo | Free NWP, same client as WattCast |
| **Storage** | Parquet local → Supabase (later) | Start simple, scale when needed |
| **Linting** | ruff | Consistent with WattCast |
| **Type checking** | pyright | Consistent with WattCast |
| **Testing** | pytest | Consistent with WattCast |
| **API** (Phase 3) | FastAPI | If/when needed |
| **Dashboard** (Phase 3) | Streamlit + ECharts | If/when needed |
| **Pipeline orchestration** | **None** (scripts) | Kedro is overkill for solo POC. Simple `scripts/` directory. |

### Why NOT Kedro

Kedro provides excellent structure for teams, but for a solo POC:
- Learning curve (~2 days to be productive)
- Boilerplate overhead (catalog YAML, nodes, hooks)
- We get 80% of the benefit from MLflow alone + well-structured scripts
- Can always migrate to Kedro later if the project scales

### Why NOT OpenSTEF

- Requires InfluxDB + MySQL — too much infrastructure for a POC
- Oriented toward grid operators (load forecasting), not wind farm owners (production forecasting)
- Opinionated about data flow — would constrain our design rather than help it
- Worth studying for architecture patterns, not adopting as a dependency

---

## 7. Proposed Roadmap

### Phase 1 — Data Foundation (1 week)

| Step | Task | Output |
|------|------|--------|
| 1.1 | Project setup (uv, ruff, pyright, pytest, MLflow) | Working dev environment |
| 1.2 | Define canonical SCADA schema | `src/windcast/data/schema.py` |
| 1.3 | Kelmarsh parser + signal mapping | `src/windcast/data/kelmarsh.py` |
| 1.4 | QC pipeline (filtering, outliers, gap-fill) | `src/windcast/data/qc.py` |
| 1.5 | Feature engineering baseline | `src/windcast/features/pipeline.py` |
| 1.6 | MLflow setup + first empty experiment | MLflow tracking server running |

### Phase 2 — ML Experimentation (1–2 weeks)

| Step | Task | Output |
|------|------|--------|
| 2.1 | Persistence benchmark (naive: last known power) | Baseline metrics in MLflow |
| 2.2 | XGBoost power forecast (multi-horizon) | First model in MLflow |
| 2.3 | LightGBM comparison | Second model in MLflow |
| 2.4 | NWP features (Open-Meteo for Kelmarsh location) | Enhanced feature set |
| 2.5 | Standardized evaluation (MAE, skill score, by wind regime) | Evaluation framework |
| 2.6 | Probabilistic output (quantile regression + CQR) | Confidence intervals |

### Phase 2b — Generalization Test (1 week)

| Step | Task | Output |
|------|------|--------|
| 2b.1 | La Haute Borne parser | Second dataset ingested |
| 2b.2 | Run ML pipeline unchanged on La Haute Borne | Cross-site metrics in MLflow |
| 2b.3 | Compare results across sites in MLflow UI | Proof of dataset-agnostic design |

### Phase 3 — Production (future, if needed)

| Step | Task | Output |
|------|------|--------|
| 3.1 | MLflow Model Registry integration | Model lifecycle management |
| 3.2 | FastAPI prediction endpoint | REST API |
| 3.3 | Streamlit dashboard | Visualization |
| 3.4 | Drift monitoring + re-calibration triggers | Operational system |

---

## 8. Key Lessons from WattCast to Apply

| Lesson | How it applies to WindCast |
|--------|--------------------------|
| **Training data volume > features** (2yr >> 1yr) | Kelmarsh has 9 years — use ≥3 years for training |
| **Future covariates matter** (delivery-time alignment) | NWP wind forecasts at prediction horizon, not observation time |
| **Optuna can overfit** (v6 was worse than v5 defaults) | Use conservative defaults first, tune only with held-out validation |
| **CQR is simple and effective** (15 lines of numpy) | Apply same conformalized quantile regression pattern |
| **Feature pruning doesn't help XGBoost with subsampling** | Don't waste time on feature selection, let colsample_bytree handle it |
| **Daily recalibration >> fixed-split** (LEAR lesson) | Consider rolling-window retraining from the start |
| **Measure in business terms** (EUR uplift, not just MAE) | Define a business metric (e.g., imbalance cost reduction) |

---

## 9. Open Questions

1. **Forecasting target**: turbine-level or farm-level aggregated power? (Start farm-level, simpler)
2. **Horizon**: 10-min steps (H+1 to H+48) or hourly aggregates? (Start hourly, align with market settlement)
3. **NWP model**: Open-Meteo ECMWF IFS (same as WattCast) or GFS? (IFS, proven)
4. **SunCast later?** If the framework is truly standardized, adding solar should be straightforward (irradiance instead of wind speed, different power curve). Design with this in mind but don't build for it yet.
5. **Business metric**: What decision does the forecast serve? (trading, maintenance scheduling, grid balancing?) — determines evaluation priorities.

---

## 10. Sources

- [Kelmarsh Wind Farm Data — Zenodo](https://zenodo.org/records/16807551)
- [Penmanshiel Wind Farm Data — Zenodo](https://zenodo.org/records/5946808)
- [La Haute Borne — ENGIE Open Data](https://opendata-renewables.engie.com/)
- [OpenSTEF — GitHub](https://github.com/OpenSTEF/openstef) | [Architecture docs](https://openstef.github.io/openstef/architecture_methodology_components.html)
- [NREL OpenOA — GitHub](https://github.com/NREL/OpenOA) | [Documentation](https://openoa.readthedocs.io/)
- [WindFM — arXiv](https://arxiv.org/html/2509.06311v1) | [GitHub](https://github.com/shiyu-coder/WindFM)
- [Kedro Energy Forecasting Pipeline — GitHub](https://github.com/labrijisaad/Kedro-Energy-Forecasting-Machine-Learning-Pipeline)
- [Effenberger (2022) — Open-source wind and wind power datasets catalog](https://onlinelibrary.wiley.com/doi/10.1002/we.2766)
- [MLOps for Wind Energy Forecasting — MDPI](https://www.mdpi.com/2076-3417/14/9/3725)
- [WattCast methodology document](../windcast/methodology.md) (internal)
