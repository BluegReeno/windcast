# Scaling a Weather-to-Energy Forecasting Pipeline
## From Artisanal Calibration to Industrial Delivery

---

## The Problem

Weather-to-energy forecasting companies typically deliver two core products: **site-specific weather forecasts** (wind, irradiance, temperature, pressure) and **production forecasts** (MW output). Both require calibrating ML models against client-provided historical data — SCADA measurements, lidar/mast observations, pyranometer readings.

The challenge is that this calibration process **doesn't scale**. Each new client farm means:

- **Ingesting heterogeneous data** — every client sends data in a different format, from a different supervision platform (Bazefield, PI System, Greenbyte, proprietary exports), with different OEM signal naming conventions (Vestas, Siemens Gamesa, Enercon, Nordex, GE Vernova).

- **Running a full ML experimentation cycle** — feature engineering, train/test splitting, algorithm selection, hyperparameter tuning, evaluation, validation. Often done manually in individual notebooks, with no shared framework across the team.

- **Deploying and maintaining models in production** — pushing trained models into the operational pipeline, monitoring forecast quality, re-calibrating when performance drifts (sensor changes, turbine replacements, seasonal shifts).

When this process is artisanal — each data scientist with their own scripts, their own environment, their own conventions — the result is predictable: **setup takes days to weeks per farm**, knowledge stays in people's heads, and the team spends most of its time on repetitive integration work instead of improving forecast quality.

The bottleneck is not the ML. The bottleneck is everything around it.

---

## The Approach: Three Incremental Phases

Each phase delivers immediate value while building the foundation for the next. No big bang. No 6-month infrastructure project before seeing results.

---

### Phase 1 — Standardize Data Ingestion

**Goal:** Any new client farm reaches a clean, unified dataset in hours instead of days.

**The core problem:** Data arrives in as many formats as there are clients. OEM signal naming is inconsistent (Vestas calls it `WindSpeed_Hub`, Siemens calls it `ActPwr_kW`). Timestamps are misaligned (NWP in UTC, client in local time, SCADA in its own convention). Spatial alignment between NWP grid points and actual farm coordinates requires interpolation.

**What to build:**

- **OEM signal dictionary** — A mapping table covering the 6 major turbine manufacturers × their 5 most common models. For each standard signal (active power, hub wind speed, wind direction, pitch angle, rotor RPM, nacelle temperature, status code), map the native OEM name to an internal standard. This single artifact covers ~90% of the installed European fleet.

- **Platform connectors** — Standardized parsers for the 4-5 most common supervision platforms (Bazefield/DNV, Greenbyte/Powerfactors, PI System/AVEVA, Clir Renewables) plus a generic CSV/API fallback. Each connector handles authentication, data retrieval, and format normalization.

- **Automated QC pipeline** — Outlier detection, gap identification, maintenance period filtering, curtailment flagging, sensor change detection. Rule-based first (statistical thresholds), ML-assisted later (anomaly detection).

- **Temporal and spatial alignment** — Automated timestamp normalization to UTC, NWP grid-to-point interpolation using farm coordinates, resampling to common time steps.

**Output:** A standardized, quality-controlled dataset ready for calibration. Same format regardless of the client, the OEM, or the supervision platform.

**Metric:** Time from data reception to calibration-ready dataset (target: < 4 hours for a standard farm).

---

### Phase 2 — Automate the ML Calibration Loop

**Goal:** Calibrating a new farm goes from a week-long manual process to a single command that produces a ranked set of model candidates.

**The core problem:** Each calibration is treated as a unique data science project. Feature engineering is redone from scratch, hyperparameter tuning is manual, experiment results live in individual notebooks, and there is no systematic way to compare approaches across farms or across team members.

**What to build:**

- **Standardized feature sets by product type:**
  - *Wind correction:* lags, rolling averages, wind speed cubed (V³), direction decomposition (sin/cos), atmospheric stability indicators, NWP model spread.
  - *Irradiance correction:* clear-sky index, cloud type indicators, solar angle, satellite-derived features, temperature.
  - *Production forecast:* corrected weather variables + turbine power curve + wake loss estimates + curtailment history.
  - Each set exists in 2-3 variants (baseline, enriched, full) to enable systematic comparison.

- **Automated experimentation pipeline** — Given a calibration-ready dataset and a product type, the system explores a predefined matrix: feature set variants × algorithms (XGBoost, LightGBM, optionally LSTM) × hyperparameter ranges × train/test split strategies. Optimization is Bayesian (Optuna) rather than exhaustive grid search — converges in ~100-200 runs instead of thousands.

- **Experiment tracking (MLflow)** — Every run is automatically logged: features used, hyperparameters, all evaluation metrics (RMSE, MAE, bias, skill score vs. persistence), per forecast horizon (H+1, H+6, D+1, D+3, D+7). The full team — regardless of whether they work in Python or R — sees results in a shared UI.

- **Standardized evaluation framework** — Consistent metrics across all farms and all products. Statistical metrics (RMSE, MAE, bias) plus business metrics (skill score vs. persistence, skill score vs. raw NWP, error distribution by wind regime / cloud condition). Results are automatically benchmarked against the current production model.

**Output:** A ranked leaderboard of model candidates with full traceability. The data scientist's role shifts from "the person who tunes" to "the person who validates and improves the framework."

**Metric:** Time from calibration-ready dataset to validated model candidate (target: < 24 hours of compute, < 2 hours of human review).

---

### Phase 3 — Industrialize Deployment and Monitoring

**Goal:** Models go from validated candidate to production with one click, and performance degradation is detected automatically.

**The core problem:** Even when a great model is trained, the path to production is often manual and fragile. Models may run as cron jobs on individual machines, with no versioning, no rollback capability, and no systematic performance monitoring. When a model degrades (sensor replacement, turbine upgrade, seasonal drift), it may go unnoticed for weeks.

**What to build:**

- **Model Registry** — Every trained model is versioned and stored with full metadata: training data period, feature set, hyperparameters, evaluation metrics, training date. Each client farm has a clear model lifecycle: *staging* → *production* → *archived*. Rollback to a previous version is one command.

- **Automated deployment pipeline** — Promoting a model from staging to production triggers automated integration tests (does it run? does it produce sensible output on recent data?) and a canary period (run in parallel with the current production model for N days, compare metrics). Only then does it replace the current model.

- **Performance monitoring and drift detection** — Continuous comparison of forecast accuracy against recent actuals. Automated alerts when metrics degrade beyond a threshold (e.g., RMSE increases by >15% over a 30-day rolling window). Distinguish between data issues (sensor failure, missing data) and model issues (genuine drift requiring re-calibration).

- **Automated re-calibration triggers** — When drift is confirmed, automatically launch a Phase 2 calibration run on recent data. The resulting candidate is flagged for human review before promotion.

**Output:** A self-monitoring production system where model quality is maintained proactively, not reactively.

**Metric:** Mean time to detect performance degradation (target: < 48 hours). Mean time to deploy a re-calibrated model (target: < 1 week including validation).

---

## Summary

| Phase | Focus | Key Deliverable | Business Impact |
|-------|-------|----------------|-----------------|
| **Phase 1** | Data Ingestion | OEM dictionary + platform connectors + QC pipeline | New farm setup: days → hours |
| **Phase 2** | ML Calibration | Automated experimentation + experiment tracking + standard features | Calibration: weeks → days |
| **Phase 3** | Deployment & Monitoring | Model registry + CI/CD + drift detection | Model quality: reactive → proactive |

Each phase is independent and delivers value on its own. Phase 1 is the prerequisite — without standardized data, automation is built on sand. Phase 2 multiplies team capacity. Phase 3 protects revenue by ensuring forecast quality over time.

The end state: **same team, 10x more client farms, higher forecast quality, lower operational risk.**