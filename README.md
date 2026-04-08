# EnerCast

**The plumbing layer for energy forecasting ML.** Handles data ingestion, QC, schemas, experiment tracking, and evaluation — so domain experts spend their time on what actually improves accuracy: feature engineering, model selection, and understanding client data.

Every hour an engineer spends writing QC scripts, debugging deployment artifacts, or rebuilding data pipelines is an hour NOT spent on accuracy. EnerCast solves the plumbing once. Engineers iterate on features, models, and KPIs — not on boilerplate.

## What the Framework Handles (Plumbing)

| Layer | What's Solved | What the Engineer Controls |
|-------|--------------|---------------------------|
| **Data ingestion** | Parsers for SCADA, ENTSO-E, PVDAQ, Open-Meteo NWP | Write a new parser (~100 lines) for a new dataset |
| **Schemas** | Typed canonical schemas per domain (wind, demand, solar) | Extend with domain-specific fields |
| **QC pipeline** | Parameterizable rules (outliers, gaps, frozen sensors, holidays) | Adjust thresholds per client/site |
| **Feature sets** | Named sets (baseline / enriched / full) per domain | Modify features, create custom sets |
| **Training** | Temporal splits, MLflow logging, multi-horizon training | Choose model (XGBoost, LightGBM, any sklearn-compatible), tune hyperparameters |
| **Evaluation** | MAE, RMSE, MAPE, skill scores, regime analysis | Add custom KPIs (e.g., "accuracy when spot price > X") |
| **Tracking** | MLflow logs everything — features, params, metrics, artifacts | Compare experiments across domains in one UI |

## Quick Start

```bash
# Install dependencies
uv sync

# Wind pipeline (Kelmarsh dataset)
uv run python scripts/ingest_kelmarsh.py        # Parse → QC → Parquet
uv run python scripts/build_features.py          # Feature engineering
uv run python scripts/train.py                   # Train → MLflow
uv run python scripts/evaluate.py                # Evaluate → MLflow

# Demand pipeline (Spain ENTSO-E dataset) — same scripts, different domain
uv run python scripts/ingest_spain_demand.py
uv run python scripts/build_features.py --domain demand --dataset spain_demand
uv run python scripts/train.py --domain demand --dataset spain_demand
uv run python scripts/evaluate.py --domain demand --dataset spain_demand

# View results
mlflow ui
```

## Adding a New Client/Dataset

1. Write a parser (~100 lines) that maps raw data to the domain schema
2. Add a `DatasetConfig` in `config.py` (coordinates, capacity, timezone)
3. Run the pipeline: `ingest → build_features → train → evaluate`

Zero changes to the core pipeline. The parser is the only dataset-specific code.

## Domains

| Domain | Dataset | Source | Resolution | What It Demonstrates |
|--------|---------|--------|-----------|---------------------|
| **Wind** | Kelmarsh v4 (6 turbines) | Zenodo | 10 min | SCADA ingestion, power curve modeling, NWP integration |
| **Demand** | Spain ENTSO-E | Kaggle | 1 hour | Load forecasting, calendar features, MARS replacement |
| **Solar** | PVDAQ System 4 | NREL | 15 min | Irradiance-based forecasting, clearsky ratio |

Same pipeline pattern, different parsers and feature configs.

## Pipeline Pattern

```
Raw data (CSV/ZIP/API)
    → [Parser] domain-specific mapping → canonical schema
    → [QC] parameterized rules → flagged data → clean Parquet
    → [Features] domain feature set → feature Parquet
    → [Train] temporal split → model → MLflow run
    → [Evaluate] metrics + skill scores + regime analysis → MLflow artifacts
```

**Framework zones** (grey — solved once): Parser scaffolding, schema validation, QC engine, temporal splits, MLflow logging, evaluation metrics.

**Engineer zones** (blue — where accuracy comes from): Signal mapping, feature design, model choice, QC thresholds, custom KPIs, regime definitions.

## Project Structure

```
src/windcast/
├── config.py                # Pydantic Settings + dataset configs
├── data/                    # Data ingestion & QC
│   ├── schema.py            # Wind SCADA schema (15 cols)
│   ├── demand_schema.py     # Demand schema (11 cols)
│   ├── solar_schema.py      # Solar schema (10 cols)
│   ├── kelmarsh.py          # Wind parser (Kelmarsh v4)
│   ├── spain_demand.py      # Demand parser (ENTSO-E)
│   ├── pvdaq.py             # Solar parser (PVDAQ)
│   ├── qc.py                # Wind QC (9 rules)
│   ├── demand_qc.py         # Demand QC
│   ├── solar_qc.py          # Solar QC
│   └── open_meteo.py        # NWP weather client (Open-Meteo)
├── features/                # Feature engineering
│   ├── registry.py          # Feature set registry (18 sets across 3 domains)
│   ├── wind.py              # Wind-specific features
│   ├── demand.py            # Demand-specific features
│   └── solar.py             # Solar-specific features
├── models/                  # ML models (domain-agnostic)
│   ├── xgboost_model.py     # XGBoost trainer
│   ├── mlforecast_model.py  # mlforecast (Nixtla) trainer
│   ├── persistence.py       # Naive persistence benchmark
│   └── evaluation.py        # Metrics, skill scores, regime analysis
└── tracking/
    └── mlflow_utils.py      # MLflow logging utilities
```

## Tech Stack

Python 3.12+ · uv · Polars · XGBoost · LightGBM · mlforecast (Nixtla) · scikit-learn · MLflow · Optuna · Open-Meteo · pytest · ruff · pyright

## Quality

- 234 tests passing
- ruff lint + format clean
- pyright type checking clean
- Strict temporal train/val/test splits (no data leakage)

## Documentation

- **[PRD](.claude/PRD.md)** — Product requirements and architecture
- **[Status](.claude/STATUS.md)** — Current sprint and priorities
- **[Research](docs/research/)** — Dataset catalog, methodology, brainstorming

## License

MIT
