# Data Ingestion Pipeline

## Overview

WindCast uses a **standardized ingestion pipeline** that converts raw SCADA data from any wind farm into a unified format. The goal: onboard a new dataset by writing only a parser function, while reusing schema validation, quality control, and storage logic.

```
Raw data        →  Parser (custom)  →  Canonical schema  →  QC pipeline  →  Parquet
(vendor format)    (1 file per          (15 columns,         (9 rules,       (1 file per
                    dataset)             unified types)       reusable)       turbine)
```

---

## Architecture

### Modules

| Module | Path | Role |
|--------|------|------|
| **Config** | `src/windcast/config.py` | Dataset metadata, QC thresholds, paths |
| **Schema** | `src/windcast/data/schema.py` | Canonical 15-column schema + validation |
| **Kelmarsh parser** | `src/windcast/data/kelmarsh.py` | Kelmarsh v4 ZIP → canonical DataFrame |
| **QC pipeline** | `src/windcast/data/qc.py` | 9 quality control rules |
| **Open-Meteo client** | `src/windcast/data/open_meteo.py` | Historical NWP weather data |
| **CLI script** | `scripts/ingest_kelmarsh.py` | End-to-end: parse → QC → Parquet |

### Data flow

```
data/raw/kelmarsh/16807551.zip
    │
    ▼  parse_kelmarsh()
pl.DataFrame (canonical schema, qc_flag=0)
    │
    ▼  validate_schema()
    │  (fails fast if columns/types don't match)
    │
    ▼  run_qc_pipeline()
pl.DataFrame (qc_flag updated, curtailment/maintenance flagged, small gaps filled)
    │
    ▼  write_parquet()
data/processed/
    ├── kelmarsh_kwf1.parquet
    ├── kelmarsh_kwf2.parquet
    └── ...
```

---

## Canonical SCADA Schema

Every parser must produce a DataFrame with exactly these 15 columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_utc` | `Datetime(us, UTC)` | Observation time (start of period) |
| `dataset_id` | `String` | Dataset identifier (e.g. `"kelmarsh"`) |
| `turbine_id` | `String` | Turbine identifier (e.g. `"KWF1"`) |
| `active_power_kw` | `Float64` | Active power output in kW |
| `wind_speed_ms` | `Float64` | Hub-height wind speed in m/s |
| `wind_direction_deg` | `Float64` | Wind direction in degrees (0-360) |
| `pitch_angle_deg` | `Float64` | Blade pitch angle in degrees |
| `rotor_rpm` | `Float64` | Rotor speed in RPM |
| `nacelle_direction_deg` | `Float64` | Nacelle orientation in degrees |
| `ambient_temp_c` | `Float64` | Ambient temperature in °C |
| `nacelle_temp_c` | `Float64` | Nacelle temperature in °C |
| `status_code` | `Int32` | Operational status (0 = normal) |
| `is_curtailed` | `Boolean` | Curtailment flag (set by QC) |
| `is_maintenance` | `Boolean` | Maintenance flag (set by QC) |
| `qc_flag` | `UInt8` | QC result: 0=ok, 1=suspect, 2=bad |

Parsers initialize `qc_flag=0`, `is_curtailed=False`, `is_maintenance=False`. The QC pipeline updates these flags.

---

## Quality Control Pipeline

`run_qc_pipeline()` applies 9 rules in sequence. Each rule is an independent function, testable in isolation.

| # | Rule | Condition | Action |
|---|------|-----------|--------|
| 1 | Maintenance | `status_code != 0` | `is_maintenance = True` |
| 2 | Negative power | `active_power_kw < 0` | `qc_flag = 2` (bad) |
| 3 | Over-rated power | `active_power_kw > rated × 1.05` | `qc_flag = 1` (suspect) |
| 4 | Negative wind | `wind_speed_ms < 0` | `qc_flag = 2` (bad) |
| 5 | Extreme wind | `wind_speed_ms > 40 m/s` | `qc_flag = 1` (suspect) |
| 6 | Frozen sensors | Same value > 60 min | `qc_flag = 1` (suspect) |
| 7 | Curtailment | High wind + high pitch + low power | `is_curtailed = True` |
| 8 | Small gap fill | Missing < 30 min | Forward-fill |
| 9 | Large gaps | Missing ≥ 30 min | Left as null |

The **worst flag wins**: if a row is flagged by multiple rules, it keeps the highest `qc_flag` value.

All thresholds are configurable via `QCConfig` in `config.py` or environment variables (`WINDCAST_QC__MAX_WIND_SPEED_MS`, etc.).

---

## Storage Format

Output is written as **one Parquet file per turbine**:

```
data/processed/
├── kelmarsh_kwf1.parquet
├── kelmarsh_kwf2.parquet
├── kelmarsh_kwf3.parquet
├── kelmarsh_kwf4.parquet
├── kelmarsh_kwf5.parquet
└── kelmarsh_kwf6.parquet
```

Parquet settings: `compression="zstd"`, `compression_level=3`, `statistics=True`.

To read all turbines back:

```python
import polars as pl

df = pl.scan_parquet("data/processed/kelmarsh_*.parquet").collect()
```

---

## Configuration

`WindCastSettings` (Pydantic Settings) manages all configuration with env var overrides:

```python
from windcast.config import get_settings

settings = get_settings()
settings.data_dir         # Path("data")
settings.raw_dir          # Path("data/raw")
settings.processed_dir    # Path("data/processed")
settings.dataset_id       # "kelmarsh"
settings.dataset_config   # DatasetConfig(rated_power_kw=2050, ...)
settings.qc               # QCConfig(max_wind_speed_ms=40, ...)
```

Override via environment variables:

```bash
WINDCAST_DATASET_ID=hill_of_towie
WINDCAST_DATA_DIR=/mnt/data
WINDCAST_QC__MAX_WIND_SPEED_MS=45
```

### Pre-defined datasets

| Dataset | Rated Power | Hub Height | Turbines | Location |
|---------|-------------|------------|----------|----------|
| Kelmarsh | 2050 kW | 78.5 m | 6 | 52.40°N, 0.94°W |
| Hill of Towie | 2300 kW | 80.0 m | 21 | 57.34°N, 2.65°W |
| Penmanshiel | 2050 kW | 78.5 m | 13 | 55.91°N, 2.29°W |

---

## Adding a New Dataset

To onboard a new wind farm, you need **one custom file** (the parser) and **one config entry**. Everything else is reused.

### Step 1: Add dataset config

In `src/windcast/config.py`, add a `DatasetConfig` and register it in `DATASETS`:

```python
NEW_FARM = DatasetConfig(
    dataset_id="new_farm",
    rated_power_kw=3000.0,
    hub_height_m=90.0,
    rotor_diameter_m=110.0,
    n_turbines=10,
    latitude=55.0,
    longitude=-3.0,
)

DATASETS["new_farm"] = NEW_FARM
```

### Step 2: Write the parser

Create `src/windcast/data/new_farm.py`. The parser must:

1. **Read the raw files** (ZIP, CSV, whatever format the vendor provides)
2. **Map vendor column names → canonical names** via a signal mapping dict
3. **Handle unit conversions** (kWh → kW, local time → UTC, etc.)
4. **Return a DataFrame matching `SCADA_SCHEMA`** with `qc_flag=0`, `is_curtailed=False`, `is_maintenance=False`

Minimal template:

```python
"""New Farm dataset parser."""

import polars as pl
from pathlib import Path
from windcast.data.schema import SCADA_SCHEMA

SIGNAL_MAP: dict[str, str] = {
    "VendorTimestamp": "timestamp_raw",
    "VendorWindSpeed": "wind_speed_ms",
    "VendorPower": "active_power_kw",
    # ... map all signals
}

def parse_new_farm(raw_path: Path) -> pl.DataFrame:
    """Parse New Farm data into canonical SCADA DataFrame."""
    df = pl.read_csv(raw_path, ...)

    # Rename columns
    df = df.rename(SIGNAL_MAP)

    # Parse timestamps, convert units, etc.
    # ...

    # Add identifiers and default flags
    df = df.with_columns(
        pl.lit("new_farm").alias("dataset_id"),
        pl.lit("T01").alias("turbine_id"),
        pl.lit(0).cast(pl.Int32).alias("status_code"),
        pl.lit(False).alias("is_curtailed"),
        pl.lit(False).alias("is_maintenance"),
        pl.lit(0).cast(pl.UInt8).alias("qc_flag"),
    )

    # Ensure schema compliance
    df = df.select(
        [pl.col(col).cast(dtype) for col, dtype in SCADA_SCHEMA.items()]
    )
    return df
```

### Step 3: Write the CLI script

Create `scripts/ingest_new_farm.py` following the same pattern as `ingest_kelmarsh.py`:

```python
df = parse_new_farm(raw_path)
errors = validate_schema(df)
df = run_qc_pipeline(df, rated_power_kw=NEW_FARM.rated_power_kw)
df.write_parquet(output_path, compression="zstd")
```

### Step 4: Validate

```bash
uv run python scripts/ingest_new_farm.py --raw-path data/raw/new_farm/
uv run pytest tests/data/test_new_farm.py -v
```

### What you reuse vs. what you write

| Component | Reused | Custom |
|-----------|--------|--------|
| Canonical schema | ✓ | |
| Schema validation | ✓ | |
| QC pipeline (9 rules) | ✓ | |
| Config system | ✓ | Add 1 `DatasetConfig` entry |
| Open-Meteo client | ✓ | |
| Parquet write logic | ✓ | |
| **Signal mapping** | | ✓ One dict per dataset |
| **File reading logic** | | ✓ ZIP/CSV structure varies |
| **Unit conversions** | | ✓ kWh→kW, timezone, pitch avg |

---

## Running the Pipeline

### Kelmarsh ingestion

```bash
# With explicit path
uv run python scripts/ingest_kelmarsh.py --raw-path data/raw/kelmarsh/16807551.zip

# Auto-detect (looks in data/raw/kelmarsh/)
uv run python scripts/ingest_kelmarsh.py

# Custom output directory
uv run python scripts/ingest_kelmarsh.py --output-dir /tmp/processed
```

### Open-Meteo weather data (for future feature engineering)

```python
from windcast.data.open_meteo import fetch_historical_weather

df = fetch_historical_weather(
    latitude=52.4016,
    longitude=-0.9436,
    start_date="2016-01-01",
    end_date="2021-12-31",
)
# Returns hourly DataFrame: wind_speed_100m, wind_direction_100m,
# temperature_2m, pressure_msl, etc.
```

Results are cached indefinitely (historical data doesn't change) — subsequent calls are instant.

---

## Key Design Decisions

1. **Parser returns pre-QC data**: parsers set `qc_flag=0` everywhere. QC is always a separate explicit step. This keeps parsers simple and QC testable independently.

2. **One Parquet per turbine**: enables per-turbine processing, reduces memory pressure, and allows Polars glob-read.

3. **Status code parsing deferred**: Kelmarsh status codes are poorly documented. Initial implementation sets `status_code=0`. Proper status mapping will be refined when real status CSVs are examined.

4. **Power unit auto-detection**: the parser checks `max(power)` against rated power. If suspiciously low, it assumes kWh per 10-min and converts to kW with a logged warning.

5. **Curtailment detection is approximate**: without a proper power curve model (coming in Phase 2), a simple heuristic is used: wind > 8 m/s AND pitch > 3° AND power < 50% rated.
