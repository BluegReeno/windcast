# EnerCast

**ML framework for energy engineering professionals.** Pre-built pipelines, data connectors, schemas, and baseline models — so domain experts can focus on what drives accuracy: feature engineering, model selection, and result comparison.

EnerCast handles the mechanical parts (data ingestion, QC, experiment tracking) while giving full control over the parts that require domain expertise (feature sets, model configuration, custom KPIs).

## Who Is This For?

Energy engineers and data scientists who:
- **Know their domain** — spot price dynamics, wind farm performance assessment, solar degradation patterns
- **Know ML** — comfortable with XGBoost, feature engineering, train/val/test splits
- **Don't want to rebuild plumbing** — data connectors, schemas, QC rules, MLflow logging should be ready to use
- **Need to iterate fast** — swap feature sets, compare models, test hypotheses in minutes, not days

## What's Included

| Layer | What You Get | What You Control |
|-------|-------------|-----------------|
| **Data connectors** | Parsers for SCADA, ENTSO-E, PVDAQ, Open-Meteo NWP | Add your own parser (~100 lines) |
| **Schemas** | Typed canonical schemas per domain (wind, demand, solar) | Extend with domain-specific fields |
| **QC pipeline** | Parameterizable rules (outliers, gaps, frozen sensors, holidays) | Adjust thresholds per client/site |
| **Feature sets** | Named sets (baseline / enriched / full) per domain | Modify features, create custom sets |
| **Models** | XGBoost + persistence benchmark, ready to train | Swap to LightGBM, add new models |
| **Evaluation** | MAE, RMSE, MAPE, skill scores, regime analysis | Add custom KPIs (e.g., "accuracy when spot > X") |
| **Experiment tracking** | MLflow logs everything — features, params, metrics, artifacts | Compare across domains in one UI |

## Quick Start

```bash
# Install dependencies
uv sync

# Wind pipeline (Kelmarsh dataset)
uv run python scripts/ingest_kelmarsh.py        # Parse → QC → Parquet
uv run python scripts/build_features.py          # Feature engineering
uv run python scripts/train.py                   # Train → MLflow
uv run python scripts/evaluate.py                # Evaluate → MLflow

# Demand pipeline (Spain ENTSO-E dataset)
uv run python scripts/ingest_spain_demand.py     # Parse → QC → Parquet
uv run python scripts/build_features.py --domain demand --dataset spain_demand
uv run python scripts/train.py --domain demand --dataset spain_demand
uv run python scripts/evaluate.py --domain demand --dataset spain_demand

# View results
mlflow ui
```

## Domains

| Domain | Dataset | Source | Resolution | What It Demonstrates |
|--------|---------|--------|-----------|---------------------|
| **Wind** | Kelmarsh v4 (6 turbines) | Zenodo | 10 min | SCADA ingestion, power curve modeling, NWP integration |
| **Demand** | Spain ENTSO-E | Kaggle | 1 hour | Load forecasting, calendar features, price correlation |
| **Solar** | PVDAQ System 2 | NREL | 15 min | Irradiance-based forecasting, clearsky ratio *(planned)* |

Same pipeline pattern, different parsers and feature configs. Adding a new domain = writing a parser + feature set.

## Project Structure

```
src/windcast/
├── config.py                # Pydantic Settings + dataset configs
├── data/                    # Data ingestion & QC
│   ├── schema.py            # Wind SCADA schema (15 cols)
│   ├── demand_schema.py     # Demand schema (11 cols)
│   ├── kelmarsh.py          # Wind parser (Kelmarsh v4)
│   ├── spain_demand.py      # Demand parser (ENTSO-E)
│   ├── qc.py                # Wind QC (9 rules)
│   ├── demand_qc.py         # Demand QC
│   └── open_meteo.py        # NWP weather client (Open-Meteo)
├── features/                # Feature engineering
│   ├── registry.py          # Feature set registry (baseline/enriched/full)
│   ├── wind.py              # Wind-specific features
│   └── demand.py            # Demand-specific features
├── models/                  # ML models (domain-agnostic)
│   ├── xgboost_model.py     # XGBoost trainer
│   ├── persistence.py       # Naive persistence benchmark
│   └── evaluation.py        # Metrics, skill scores, regime analysis
└── tracking/
    └── mlflow_utils.py      # MLflow logging utilities
```

## Pipeline Pattern

```
Raw data (CSV/ZIP/API)
    → [Parser] domain-specific mapping → canonical schema
    → [QC] parameterized rules → flagged data → clean Parquet
    → [Features] domain feature set → feature Parquet
    → [Train] temporal split → model → MLflow run
    → [Evaluate] metrics + skill scores + regime analysis → MLflow artifacts
```

## Tech Stack

Python 3.12+ · uv · Polars · XGBoost · LightGBM · scikit-learn · MLflow · Optuna · Open-Meteo · pytest · ruff · pyright

## Documentation

- **[PRD](.claude/PRD.md)** — Product requirements and architecture
- **[Status](.claude/STATUS.md)** — Current sprint and priorities
- **[Research](docs/research/)** — Dataset catalog, methodology, brainstorming

## License

MIT
