# WindCast

Standardized ML framework for wind power forecasting. Turns raw SCADA data into calibrated power forecasts using reproducible pipelines, MLflow experiment tracking, and open datasets.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v
```

## Project Structure

```
windcast/
├── CLAUDE.md                    # AI assistant guidelines
├── README.md                    # This file
├── pyproject.toml               # Dependencies and project config
│
├── src/windcast/                # Main package
│   ├── config.py                # Settings (Pydantic), constants
│   ├── data/                    # Data ingestion & QC
│   │   ├── schema.py            # Canonical SCADA schema
│   │   ├── kelmarsh.py          # Kelmarsh dataset parser
│   │   ├── qc.py                # Quality control pipeline
│   │   └── open_meteo.py        # NWP weather data
│   ├── features/                # Feature engineering
│   │   └── pipeline.py          # Standardized feature sets
│   ├── models/                  # ML training & evaluation
│   │   ├── xgboost.py           # XGBoost quantile regression
│   │   ├── lightgbm.py          # LightGBM benchmark
│   │   └── evaluation.py        # Metrics & skill scores
│   └── api/                     # REST API (Phase 3)
│
├── scripts/                     # CLI entry points
├── data/                        # Local data (GITIGNORED)
├── mlruns/                      # MLflow tracking (GITIGNORED)
├── tests/                       # Mirrors src/ structure
│
└── docs/
    └── research/                # Brainstorming & research documents
```

## Documentation

- **[PRD](.claude/PRD.md)** — Product Requirements Document
- **[Status](.claude/STATUS.md)** — Current sprint and priorities
- **[Brainstorming](docs/research/brainstorming-2026-03-31.md)** — Initial research & architecture decisions
- **[Methodology](docs/research/methodology-scaling-pipeline.md)** — Scaling weather-to-energy pipelines

## Datasets

### SCADA (wind + production)

| Dataset | Turbines | OEM | Period | Source |
|---------|----------|-----|--------|--------|
| **Kelmarsh v4** (primary) | 6 × Senvion MM92 | Senvion | 2016–2024 | [Zenodo](https://zenodo.org/records/16807551) |
| **Hill of Towie** | 21 × Siemens SWT-2.3 | Siemens | 2016–2024 | [Zenodo](https://zenodo.org/records/14870023) |
| **Penmanshiel v3** | 13 × Senvion MM82 | Senvion | 2016–2024 | [Zenodo](https://zenodo.org/records/16807304) |

### Wind-only (NWP validation & features)

| Dataset | Type | Period | Source |
|---------|------|--------|--------|
| **FINO1** | Offshore mast, North Sea | 2003–present | [BSH](https://login.bsh.de) |
| **NWTC M2** | Onshore mast 82m, Colorado | 1996–present | [NREL MIDC](https://midcdmz.nrel.gov/apps/sitehome.pl?site=NWTC) |

## Tech Stack

Python 3.12+ | uv | Polars | XGBoost | LightGBM | MLflow | Open-Meteo | pytest | ruff | pyright

## License

MIT
