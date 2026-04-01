# Feature: Phase 1.2 — Data Ingestion & QC Pipeline

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types and models. Import from the right files.

## Feature Description

Implement the complete data ingestion pipeline for WindCast: canonical SCADA schema definition, Kelmarsh v4 dataset parser with signal mapping, quality control pipeline (outlier detection, maintenance filtering, curtailment flagging, gap-filling), Pydantic configuration, Open-Meteo NWP client, and CLI ingestion script. This is the data foundation upon which all ML work depends.

## User Story

As an ML engineer
I want to ingest a new SCADA dataset by writing only a parser function
So that I can onboard a new wind farm in < 2 hours with clean, schema-compliant Parquet output

## Problem Statement

Raw Kelmarsh SCADA data lives in nested ZIPs with 330+ columns per CSV, Greenbyte signal naming conventions, power stored as cumulative kWh (not instantaneous kW), inconsistent NaN encoding, and no QC flags. This data cannot be used directly for ML training. We need a standardized ingestion pipeline that produces clean, typed, schema-compliant Parquet files.

## Solution Statement

Build a 5-module pipeline:
1. **schema.py** — Canonical SCADA schema as Polars dtype dict + validation function
2. **config.py** — Pydantic Settings with dataset configs, QC thresholds, paths
3. **kelmarsh.py** — Parser that reads nested ZIPs, maps Greenbyte signals → canonical names, converts power kWh → kW
4. **qc.py** — Quality control pipeline applying PRD rules (maintenance, outliers, curtailment, gaps)
5. **open_meteo.py** — Historical weather client for NWP features (cached, retry-enabled)
6. **ingest_kelmarsh.py** — CLI script orchestrating parse → QC → Parquet

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: High
**Primary Systems Affected**: `src/windcast/data/`, `src/windcast/config.py`, `scripts/`
**Dependencies**: polars, pydantic-settings, openmeteo-requests, requests-cache, retry-requests

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

- `.claude/PRD.md` (lines 246-266) — Canonical SCADA schema specification (15 fields, exact types)
- `.claude/PRD.md` (lines 268-290) — Feature sets (baseline/enriched/full) — informs what schema must support
- `.claude/PRD.md` (lines 292-303) — QC pipeline rules table (9 rules with thresholds)
- `.claude/PRD.md` (lines 358-395) — Configuration spec (WindCastSettings class)
- `.claude/PRD.md` (lines 136-198) — Directory structure (data/raw, data/processed, data/features)
- `.claude/PRD.md` (lines 200-241) — Data flow diagram and key design patterns
- `CLAUDE.md` (lines 82-100) — Code style, naming conventions
- `src/windcast/__init__.py` — Root package (docstring only)
- `src/windcast/data/__init__.py` — Empty, will need exports
- `tests/test_setup.py` — Existing smoke tests pattern
- `pyproject.toml` — Dependencies already installed, tool configs

### Relevant Documentation — READ BEFORE IMPLEMENTING

**Library patterns documented in `.claude/docs/`:**

- `.claude/docs/polars-patterns.md` — Schema definition, `read_csv`/`scan_csv`, timestamp parsing, null handling, rename, validation, Parquet write. **MUST READ** before writing schema.py and kelmarsh.py.
- `.claude/docs/pydantic-settings-patterns.md` — `BaseSettings` with `WINDCAST_` prefix, nested models, singleton pattern. **MUST READ** before writing config.py.
- `.claude/docs/openmeteo-patterns.md` — Client setup, historical archive endpoint, response extraction to Polars, rate limits, gotchas. **MUST READ** before writing open_meteo.py.

**Fallback URLs:**
- [Polars API reference](https://docs.pola.rs/api/python/stable/reference/) — DataFrame operations
- [Open-Meteo Historical Archive API](https://open-meteo.com/en/docs/historical-weather-api) — Endpoint params
- [Pydantic Settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — BaseSettings config

### New Files to Create

| File | Purpose |
|------|---------|
| `src/windcast/config.py` | Pydantic Settings: paths, dataset config, QC thresholds |
| `src/windcast/data/schema.py` | Canonical SCADA schema + validation function |
| `src/windcast/data/kelmarsh.py` | Kelmarsh v4 parser (ZIP → canonical DataFrame) |
| `src/windcast/data/qc.py` | Quality control pipeline (flags, gap-fill, filtering) |
| `src/windcast/data/open_meteo.py` | Open-Meteo historical weather client |
| `scripts/ingest_kelmarsh.py` | CLI script: parse → QC → save Parquet |
| `tests/data/__init__.py` | Test sub-package |
| `tests/data/test_schema.py` | Schema validation tests |
| `tests/data/test_kelmarsh.py` | Parser tests with synthetic data |
| `tests/data/test_qc.py` | QC pipeline tests |
| `tests/data/test_open_meteo.py` | Open-Meteo client tests (mocked) |
| `tests/test_config.py` | Config loading tests |

### Patterns to Follow

**Naming Conventions:**
- Files: `snake_case.py`
- Classes: `PascalCase` (e.g., `WindCastSettings`, `DatasetConfig`)
- Functions: `snake_case` (e.g., `parse_kelmarsh`, `run_qc_pipeline`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `SCADA_SCHEMA`, `KELMARSH_SIGNAL_MAP`)

**Import Groups (ruff isort, already configured):**
```python
# stdlib
from pathlib import Path

# third-party
import polars as pl
from pydantic_settings import BaseSettings

# local
from windcast.config import get_settings
from windcast.data.schema import SCADA_SCHEMA
```

**Error Handling (from CLAUDE.md):**
- Fail fast with detailed errors
- Use `ValueError` for bad data, `FileNotFoundError` for missing files
- No silent failures, no graceful degradation for corrupt data

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation (config + schema)

Create the configuration system and canonical schema that all other modules depend on.

**Tasks:**
- `config.py` — Pydantic Settings with paths, dataset metadata, QC thresholds
- `schema.py` — SCADA schema dict, validation function, constants

### Phase 2: Core Parser (kelmarsh.py)

Implement the Kelmarsh parser: ZIP extraction, signal mapping, power conversion, timestamp handling.

**Tasks:**
- Signal mapping constant (Greenbyte → canonical)
- ZIP navigation (nested ZIPs, turbine data CSVs)
- CSV parsing with Polars (schema_overrides, null handling)
- Power kWh → kW conversion
- Pitch angle averaging (3 blades → 1 value)
- Output as canonical schema DataFrame

### Phase 3: Quality Control (qc.py)

Implement the 9 QC rules from the PRD, producing flagged DataFrames.

**Tasks:**
- Maintenance detection from status codes
- Outlier detection (negative power, over-rated, extreme wind)
- Curtailment detection (power below curve + high pitch)
- Gap detection and forward-fill for small gaps
- Frozen sensor detection
- QC flag assignment (0=ok, 1=suspect, 2=bad)

### Phase 4: NWP Client (open_meteo.py)

Implement cached Open-Meteo client for historical weather data.

**Tasks:**
- Client setup with cache + retry
- Historical archive fetch function
- Polars DataFrame output with UTC timestamps
- Multi-location support (per-turbine)

### Phase 5: CLI Script (ingest_kelmarsh.py)

Wire everything together in a runnable script.

**Tasks:**
- Argparse CLI with dataset path argument
- Parse → QC → Save pipeline
- Logging with summary statistics
- Parquet output to `data/processed/`

### Phase 6: Testing

Comprehensive tests for all modules.

**Tasks:**
- Schema validation tests
- Parser tests with synthetic Kelmarsh-like CSV
- QC tests for each rule independently
- Open-Meteo tests with mocked responses
- Config loading tests

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/windcast/config.py`

Pydantic Settings configuration module.

```python
"""WindCast configuration — Pydantic Settings with env override support."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    """Per-dataset metadata."""

    dataset_id: str
    rated_power_kw: float
    hub_height_m: float
    rotor_diameter_m: float
    n_turbines: int
    latitude: float  # farm centroid for NWP
    longitude: float


# Pre-defined dataset configs
KELMARSH = DatasetConfig(
    dataset_id="kelmarsh",
    rated_power_kw=2050.0,
    hub_height_m=78.5,  # most turbines; KWF3/KWF6 are 68.5m
    rotor_diameter_m=92.0,
    n_turbines=6,
    latitude=52.4016,
    longitude=-0.9436,
)

HILL_OF_TOWIE = DatasetConfig(
    dataset_id="hill_of_towie",
    rated_power_kw=2300.0,
    hub_height_m=80.0,
    rotor_diameter_m=82.0,
    n_turbines=21,
    latitude=57.34,
    longitude=-2.65,
)

PENMANSHIEL = DatasetConfig(
    dataset_id="penmanshiel",
    rated_power_kw=2050.0,
    hub_height_m=78.5,
    rotor_diameter_m=82.0,
    n_turbines=13,
    latitude=55.905,
    longitude=-2.29,
)

DATASETS: dict[str, DatasetConfig] = {
    "kelmarsh": KELMARSH,
    "hill_of_towie": HILL_OF_TOWIE,
    "penmanshiel": PENMANSHIEL,
}


class QCConfig(BaseModel):
    """Quality control thresholds."""

    max_wind_speed_ms: float = 40.0
    max_gap_fill_minutes: int = 30
    frozen_sensor_threshold_minutes: int = 60
    rated_power_tolerance: float = 1.05  # flag if > rated × 1.05
    min_pitch_curtailment_deg: float = 3.0  # pitch above this at high wind = curtailment


class WindCastSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WINDCAST_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Path("data")

    # Active dataset
    dataset_id: str = "kelmarsh"

    # Training split
    train_years: int = 5
    val_years: int = 1
    test_years: int = 1
    forecast_horizons: list[int] = [1, 6, 12, 24, 48]

    # QC
    qc: QCConfig = Field(default_factory=QCConfig)

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def dataset_config(self) -> DatasetConfig:
        return DATASETS[self.dataset_id]


@lru_cache(maxsize=1)
def get_settings() -> WindCastSettings:
    return WindCastSettings()
```

- **PATTERN**: Follow `.claude/docs/pydantic-settings-patterns.md` — singleton via `@lru_cache`, nested BaseModel for QC config
- **GOTCHA**: `env_nested_delimiter="__"` means `WINDCAST_QC__MAX_WIND_SPEED_MS` overrides `qc.max_wind_speed_ms`
- **GOTCHA**: Use `@property` for derived paths, not `field_validator` with `mkdir` — don't auto-create dirs at import time
- **VALIDATE**: `uv run python -c "from windcast.config import get_settings; s = get_settings(); print(s.dataset_id, s.raw_dir)"`

---

### Task 2: CREATE `src/windcast/data/schema.py`

Canonical SCADA schema definition and validation.

```python
"""Canonical SCADA schema for wind farm time-series data."""

import polars as pl

# Canonical schema — every parser must produce a DataFrame matching this exactly
SCADA_SCHEMA: dict[str, pl.DataType] = {
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "dataset_id": pl.String,
    "turbine_id": pl.String,
    "active_power_kw": pl.Float64,
    "wind_speed_ms": pl.Float64,
    "wind_direction_deg": pl.Float64,
    "pitch_angle_deg": pl.Float64,
    "rotor_rpm": pl.Float64,
    "nacelle_direction_deg": pl.Float64,
    "ambient_temp_c": pl.Float64,
    "nacelle_temp_c": pl.Float64,
    "status_code": pl.Int32,
    "is_curtailed": pl.Boolean,
    "is_maintenance": pl.Boolean,
    "qc_flag": pl.UInt8,
}

# Column names in order
SCADA_COLUMNS: list[str] = list(SCADA_SCHEMA.keys())

# Signal columns (numeric, excludes identifiers and flags)
SIGNAL_COLUMNS: list[str] = [
    "active_power_kw",
    "wind_speed_ms",
    "wind_direction_deg",
    "pitch_angle_deg",
    "rotor_rpm",
    "nacelle_direction_deg",
    "ambient_temp_c",
    "nacelle_temp_c",
]

# QC flag values
QC_OK: int = 0
QC_SUSPECT: int = 1
QC_BAD: int = 2


def validate_schema(
    df: pl.DataFrame,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate DataFrame conforms to canonical SCADA schema.

    Args:
        df: DataFrame to validate.
        strict: If True, reject extra columns not in schema.

    Returns:
        List of error messages. Empty list = valid.
    """
    errors: list[str] = []
    actual = dict(zip(df.columns, df.dtypes))

    for col, expected_dtype in SCADA_SCHEMA.items():
        if col not in actual:
            errors.append(f"Missing column: {col!r}")
        elif actual[col] != expected_dtype:
            errors.append(f"Column {col!r}: expected {expected_dtype}, got {actual[col]}")

    if strict:
        extra = set(actual) - set(SCADA_SCHEMA)
        for col in sorted(extra):
            errors.append(f"Unexpected column: {col!r}")

    return errors


def empty_scada_frame() -> pl.DataFrame:
    """Create an empty DataFrame with the canonical SCADA schema. Useful for tests."""
    return pl.DataFrame(schema=SCADA_SCHEMA)
```

- **PATTERN**: Follow `.claude/docs/polars-patterns.md` — schema as dict, manual validation function
- **GOTCHA**: Polars Datetime dtype constructor is `pl.Datetime("us", "UTC")` — microsecond precision, UTC timezone
- **GOTCHA**: `pl.String` not `pl.Utf8` — Polars ≥1.0 uses `String` (Utf8 is deprecated alias)
- **IMPORTANT**: Verify the exact `pl.Datetime` constructor syntax against installed Polars 1.39 API
- **VALIDATE**: `uv run python -c "from windcast.data.schema import SCADA_SCHEMA, validate_schema, empty_scada_frame; df = empty_scada_frame(); print(df.schema); assert validate_schema(df) == []"`

---

### Task 3: CREATE `src/windcast/data/kelmarsh.py`

Kelmarsh v4 parser: nested ZIP → canonical DataFrame.

**Key implementation details:**

1. **File structure**: Kelmarsh v4 on Zenodo is a large ZIP containing annual ZIP archives. Each annual archive contains per-turbine CSVs named `Turbine_Data_Kelmarsh_N_YYYY-MM-DD_-_YYYY-MM-DD_###.csv` plus status files `Status_Kelmarsh_N_*.csv`.

2. **Signal mapping** (Greenbyte → canonical):
   ```python
   # Map from Greenbyte column headers in CSV to canonical names
   # NOTE: Actual column headers must be verified against real CSV files
   # Headers use format "Signal Name (Unit)" e.g. "Wind speed (m/s)"
   KELMARSH_SIGNAL_MAP: dict[str, str] = {
       "# Date and time": "timestamp_raw",
       "Wind speed (m/s)": "wind_speed_ms",
       "Wind direction (°)": "wind_direction_deg",
       "Power (kW)": "active_power_kw",  # May be kWh — verify!
       "Nacelle position (°)": "nacelle_direction_deg",
       "Rotor speed (rpm)": "rotor_rpm",
       "Blade angle (pitch position) A (°)": "pitch_a_deg",
       "Blade angle (pitch position) B (°)": "pitch_b_deg",
       "Blade angle (pitch position) C (°)": "pitch_c_deg",
       "Nacelle ambient temperature (°C)": "ambient_temp_c",
       "Nacelle temperature (°C)": "nacelle_temp_c",
   }
   ```

3. **Power conversion**: The PRD brainstorming mentions power may be in kWh per 10-min interval. If so: `power_kw = power_kwh * 6` (since 10 min = 1/6 hour). **Must verify with real data** — if the column is already instantaneous kW, no conversion needed. The signal mapping CSV says unit is "kWh" for signal ID 5.

4. **Pitch averaging**: Average of blade A/B/C angles → single `pitch_angle_deg`.

5. **Turbine ID extraction**: Parse from filename (e.g., `Kelmarsh_1` → `KWF1`).

6. **Status codes**: Parse from separate `Status_*.csv` files. Map to `status_code` (Int32). Initially use a simple mapping: 0 = normal operation, negative = fault/maintenance.

```python
"""Kelmarsh v4 dataset parser — ZIP archives to canonical SCADA schema."""

import logging
import zipfile
from pathlib import Path

import polars as pl

from windcast.data.schema import SCADA_SCHEMA

logger = logging.getLogger(__name__)

# Greenbyte CSV headers → canonical column names
# IMPORTANT: verify these against actual CSV headers on first run
KELMARSH_SIGNAL_MAP: dict[str, str] = {
    # Timestamp
    "# Date and time": "timestamp_raw",
    # Core signals
    "Wind speed (m/s)": "wind_speed_ms",
    "Wind direction (°)": "wind_direction_deg",
    "Power (kW)": "active_power_kw",
    "Nacelle position (°)": "nacelle_direction_deg",
    "Rotor speed (rpm)": "rotor_rpm",
    # Pitch (3 blades → average later)
    "Blade angle (pitch position) A (°)": "pitch_a_deg",
    "Blade angle (pitch position) B (°)": "pitch_b_deg",
    "Blade angle (pitch position) C (°)": "pitch_c_deg",
    # Temperatures
    "Nacelle ambient temperature (°C)": "ambient_temp_c",
    "Nacelle temperature (°C)": "nacelle_temp_c",
}

# Columns to select from CSV (keys of KELMARSH_SIGNAL_MAP)
KELMARSH_COLUMNS: list[str] = list(KELMARSH_SIGNAL_MAP.keys())

DATASET_ID = "kelmarsh"
TURBINE_IDS = [f"KWF{i}" for i in range(1, 7)]


def parse_kelmarsh(raw_path: Path) -> pl.DataFrame:
    """Parse Kelmarsh v4 ZIP archive into canonical SCADA DataFrame.

    Args:
        raw_path: Path to the Kelmarsh ZIP archive (or directory of extracted CSVs).

    Returns:
        DataFrame conforming to SCADA_SCHEMA (before QC — qc_flag/is_curtailed/is_maintenance
        are initialized to defaults).
    """
    # Implementation:
    # 1. Open outer ZIP
    # 2. Find inner annual ZIPs (Kelmarsh_SCADA_YYYY_*.zip)
    # 3. For each inner ZIP, find turbine CSVs (Turbine_Data_Kelmarsh_N_*.csv)
    # 4. Read each CSV with schema_overrides, selecting only mapped columns
    # 5. Rename columns via KELMARSH_SIGNAL_MAP
    # 6. Average pitch angles A/B/C
    # 7. Parse timestamp → UTC datetime
    # 8. Add dataset_id, turbine_id columns
    # 9. Initialize qc_flag=0, is_curtailed=False, is_maintenance=False
    # 10. Cast to SCADA_SCHEMA dtypes
    # 11. Concatenate all turbines and years
    ...


def _extract_turbine_id(filename: str) -> str:
    """Extract turbine ID from Kelmarsh filename.

    Example: 'Turbine_Data_Kelmarsh_1_2020-01-01_-_2020-12-31_1234.csv' → 'KWF1'
    """
    # Parse the number after 'Kelmarsh_'
    ...


def _read_turbine_csv(
    csv_bytes: bytes,
    turbine_id: str,
) -> pl.DataFrame:
    """Read a single Kelmarsh turbine CSV and map to canonical columns.

    Handles:
    - Column selection (only mapped signals)
    - Column renaming (Greenbyte → canonical)
    - Pitch angle averaging (A/B/C → single value)
    - Timestamp parsing (naive → UTC)
    - Power unit verification
    - Default flags (qc_flag=0, is_curtailed=False, is_maintenance=False)
    """
    ...
```

- **IMPORTS**: `zipfile`, `io.BytesIO`, `polars`, `logging`
- **PATTERN**: Parser returns pre-QC DataFrame. QC is a separate step in qc.py.
- **GOTCHA**: Kelmarsh CSVs have 330+ columns. Use `columns=` parameter in `pl.read_csv` to select only needed signals, or read all and select after rename.
- **GOTCHA**: The signal map column headers MUST be verified against real CSV files. The exact header format (parentheses, spaces, special chars like `°`) must match exactly. Run the script once with `print(df.columns[:20])` to see actual headers.
- **GOTCHA**: Nested ZIP handling — use `zipfile.ZipFile` with `BytesIO` for inner archives.
- **GOTCHA**: Power unit — if kWh per 10-min period, multiply by 6 to get kW. Verify by checking: max power should be ~2050 kW (rated). If max is ~340 (= 2050/6), it's kWh.
- **GOTCHA**: Kelmarsh timestamps are UTC (per research). No timezone conversion needed — just parse and tag as UTC.
- **GOTCHA**: Status files are separate CSVs — parse them separately and join on timestamp + turbine. For Phase 1.2, a simplified approach is acceptable: set `status_code=0` (normal) as default, refine status parsing in a follow-up if needed.
- **VALIDATE**: `uv run python -c "from windcast.data.kelmarsh import KELMARSH_SIGNAL_MAP; print(len(KELMARSH_SIGNAL_MAP), 'signals mapped')"`

---

### Task 4: CREATE `src/windcast/data/qc.py`

Quality control pipeline implementing PRD Section 7.3 rules.

```python
"""Quality control pipeline for SCADA data."""

import logging

import polars as pl

from windcast.config import QCConfig
from windcast.data.schema import QC_BAD, QC_OK, QC_SUSPECT

logger = logging.getLogger(__name__)


def run_qc_pipeline(
    df: pl.DataFrame,
    rated_power_kw: float,
    qc_config: QCConfig | None = None,
) -> pl.DataFrame:
    """Apply QC rules to canonical SCADA DataFrame.

    Modifies qc_flag, is_curtailed, is_maintenance columns.
    Returns the same DataFrame with updated flags.

    Rules applied in order:
    1. Flag maintenance periods (status_code-based)
    2. Flag negative power → qc_flag=2
    3. Flag over-rated power → qc_flag=1
    4. Flag negative wind speed → qc_flag=2
    5. Flag extreme wind speed → qc_flag=1
    6. Flag frozen sensors → qc_flag=1
    7. Detect curtailment → is_curtailed=True
    8. Fill small gaps (< 30 min) with forward-fill
    9. Leave large gaps as null

    Args:
        df: Canonical SCADA DataFrame.
        rated_power_kw: Rated power of turbine in kW.
        qc_config: QC thresholds. Uses defaults if None.

    Returns:
        DataFrame with updated QC columns.
    """
    ...


def _flag_maintenance(df: pl.DataFrame) -> pl.DataFrame:
    """Flag rows where status_code indicates non-operational state."""
    ...


def _flag_power_outliers(df: pl.DataFrame, rated_power_kw: float, tolerance: float) -> pl.DataFrame:
    """Flag negative power (qc=2) and over-rated power (qc=1)."""
    ...


def _flag_wind_outliers(df: pl.DataFrame, max_wind_ms: float) -> pl.DataFrame:
    """Flag negative wind speed (qc=2) and extreme wind (qc=1)."""
    ...


def _flag_frozen_sensors(df: pl.DataFrame, threshold_minutes: int) -> pl.DataFrame:
    """Flag periods where a signal is constant for > threshold_minutes."""
    ...


def _detect_curtailment(
    df: pl.DataFrame, rated_power_kw: float, min_pitch_deg: float
) -> pl.DataFrame:
    """Detect curtailment: high wind + high pitch + low power."""
    ...


def _fill_small_gaps(df: pl.DataFrame, max_gap_minutes: int) -> pl.DataFrame:
    """Forward-fill gaps shorter than max_gap_minutes."""
    ...


def qc_summary(df: pl.DataFrame) -> dict[str, int | float]:
    """Generate QC summary statistics for logging."""
    ...
```

- **PATTERN**: Main function `run_qc_pipeline` calls private helpers in order. Each helper is independently testable.
- **GOTCHA**: QC flag assignment uses max() logic — if a row is flagged by multiple rules, keep the worst flag (2 > 1 > 0).
- **GOTCHA**: Frozen sensor detection: use `pl.col(...).rle()` (run-length encoding) to find consecutive identical values. Check wind_speed, active_power, pitch_angle.
- **GOTCHA**: Curtailment detection is approximate in Phase 1.2 — simple heuristic: wind > 8 m/s AND pitch > 3° AND power < 0.5 × rated. Refine with proper power curve in Phase 2.
- **GOTCHA**: Gap-filling must be done per-turbine (group by turbine_id). Use `pl.DataFrame.upsample` to create a regular 10-min grid, then forward-fill.
- **VALIDATE**: `uv run python -c "from windcast.data.qc import run_qc_pipeline; print('qc importable')"`

---

### Task 5: CREATE `src/windcast/data/open_meteo.py`

Open-Meteo historical weather client.

```python
"""Open-Meteo historical weather data client."""

import logging

import openmeteo_requests
import polars as pl
import requests_cache
from retry_requests import retry

logger = logging.getLogger(__name__)

# Variables for wind power forecasting
WIND_VARIABLES: list[str] = [
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_10m",
    "wind_direction_10m",
    "temperature_2m",
    "pressure_msl",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def build_client(
    cache_dir: str = ".cache",
    expire_after: int = -1,
    retries: int = 5,
    backoff_factor: float = 0.2,
) -> openmeteo_requests.Client:
    """Build a cached, auto-retrying Open-Meteo client."""
    cache_session = requests_cache.CachedSession(cache_dir, expire_after=expire_after)
    retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=retry_session)


def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
    client: openmeteo_requests.Client | None = None,
) -> pl.DataFrame:
    """Fetch hourly historical weather from Open-Meteo archive.

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        start_date: Start date ISO format "YYYY-MM-DD".
        end_date: End date ISO format "YYYY-MM-DD".
        variables: Weather variables to fetch. Defaults to WIND_VARIABLES.
        client: Pre-built client. Creates new one if None.

    Returns:
        Polars DataFrame with timestamp_utc + weather variable columns.
    """
    ...
```

- **PATTERN**: Follow `.claude/docs/openmeteo-patterns.md` exactly — client setup, fetch function, Polars extraction
- **GOTCHA**: Always set `"wind_speed_unit": "ms"` — default is km/h
- **GOTCHA**: Always set `"timezone": "UTC"`
- **GOTCHA**: `Variables(i)` index follows exact order of `params["hourly"]` list
- **GOTCHA**: Use `pl.Float32` for weather variables (saves memory, ERA5 precision doesn't warrant Float64)
- **GOTCHA**: Cache with `expire_after=-1` — historical data never changes
- **NOTE**: This module is not used by the ingestion script yet (NWP features come in Phase 1.3). But we build it now since it's part of the data layer.
- **VALIDATE**: `uv run python -c "from windcast.data.open_meteo import build_client, WIND_VARIABLES; print(len(WIND_VARIABLES), 'variables')"`

---

### Task 6: UPDATE `src/windcast/data/__init__.py`

Add exports for the public API.

```python
"""Data ingestion and quality control modules."""

from windcast.data.schema import SCADA_SCHEMA, validate_schema

__all__ = ["SCADA_SCHEMA", "validate_schema"]
```

- **PATTERN**: Only export schema constants — parsers and QC are imported directly by name
- **VALIDATE**: `uv run python -c "from windcast.data import SCADA_SCHEMA; print(len(SCADA_SCHEMA), 'columns')"`

---

### Task 7: CREATE `scripts/ingest_kelmarsh.py`

CLI entry point for Kelmarsh data ingestion.

```python
"""Ingest Kelmarsh SCADA data: parse ZIP → QC → save Parquet.

Usage:
    uv run python scripts/ingest_kelmarsh.py [--raw-path PATH] [--output-dir PATH]

If --raw-path is not specified, looks in data/raw/kelmarsh/ for ZIP files.
"""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from windcast.config import KELMARSH, get_settings
from windcast.data.kelmarsh import parse_kelmarsh
from windcast.data.qc import qc_summary, run_qc_pipeline
from windcast.data.schema import validate_schema

logger = logging.getLogger(__name__)


def main() -> None:
    """Run Kelmarsh ingestion pipeline."""
    # 1. Parse CLI args
    # 2. Load settings
    # 3. Parse Kelmarsh ZIP → canonical DataFrame
    # 4. Validate schema
    # 5. Run QC pipeline
    # 6. Log QC summary
    # 7. Create output directory
    # 8. Write per-turbine Parquet files to data/processed/
    # 9. Print final summary (row counts, file sizes)
    ...


if __name__ == "__main__":
    main()
```

- **PATTERN**: Script-as-CLI — no framework, just argparse + logging + main()
- **GOTCHA**: Create `data/processed/` directory if it doesn't exist
- **GOTCHA**: Write one Parquet per turbine: `kelmarsh_kwf1.parquet`, `kelmarsh_kwf2.parquet`, etc.
- **GOTCHA**: Use `compression="zstd"`, `compression_level=3`, `statistics=True` for Parquet
- **VALIDATE**: `uv run python scripts/ingest_kelmarsh.py --help` (should show usage without error)

---

### Task 8: CREATE `tests/data/__init__.py`

Empty test sub-package.

- **VALIDATE**: `uv run python -c "import tests.data"`

---

### Task 9: CREATE `tests/test_config.py`

Test configuration module.

```python
"""Tests for windcast.config module."""

import os

from windcast.config import DATASETS, WindCastSettings, get_settings


def test_default_settings():
    """Default settings load without errors."""
    get_settings.cache_clear()
    settings = WindCastSettings()
    assert settings.dataset_id == "kelmarsh"
    assert settings.data_dir.name == "data"
    assert settings.qc.max_wind_speed_ms == 40.0


def test_datasets_registry():
    """All expected datasets are registered."""
    assert "kelmarsh" in DATASETS
    assert "hill_of_towie" in DATASETS
    assert "penmanshiel" in DATASETS
    assert DATASETS["kelmarsh"].rated_power_kw == 2050.0


def test_env_override(monkeypatch):
    """Environment variables override defaults."""
    get_settings.cache_clear()
    monkeypatch.setenv("WINDCAST_DATASET_ID", "hill_of_towie")
    settings = WindCastSettings()
    assert settings.dataset_id == "hill_of_towie"


def test_derived_paths():
    """Derived path properties work correctly."""
    settings = WindCastSettings()
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.processed_dir == settings.data_dir / "processed"
    assert settings.features_dir == settings.data_dir / "features"
```

- **GOTCHA**: Call `get_settings.cache_clear()` before tests that set env vars
- **VALIDATE**: `uv run pytest tests/test_config.py -v`

---

### Task 10: CREATE `tests/data/test_schema.py`

Test schema definition and validation.

```python
"""Tests for windcast.data.schema module."""

import polars as pl

from windcast.data.schema import (
    QC_BAD,
    QC_OK,
    QC_SUSPECT,
    SCADA_COLUMNS,
    SCADA_SCHEMA,
    empty_scada_frame,
    validate_schema,
)


def test_schema_has_expected_columns():
    """Schema defines all 15 canonical columns."""
    assert len(SCADA_SCHEMA) == 15
    assert "timestamp_utc" in SCADA_SCHEMA
    assert "active_power_kw" in SCADA_SCHEMA
    assert "qc_flag" in SCADA_SCHEMA


def test_empty_frame_matches_schema():
    """empty_scada_frame produces a valid schema-compliant DataFrame."""
    df = empty_scada_frame()
    errors = validate_schema(df)
    assert errors == []
    assert df.shape == (0, 15)


def test_validate_schema_detects_missing_column():
    """Validation catches missing columns."""
    df = pl.DataFrame({"timestamp_utc": [], "dataset_id": []})
    errors = validate_schema(df)
    assert len(errors) > 0
    assert any("Missing column" in e for e in errors)


def test_validate_schema_detects_wrong_type():
    """Validation catches wrong column types."""
    df = empty_scada_frame().cast({"active_power_kw": pl.Int32})
    errors = validate_schema(df)
    assert any("active_power_kw" in e for e in errors)


def test_qc_flag_constants():
    """QC flag constants are correctly defined."""
    assert QC_OK == 0
    assert QC_SUSPECT == 1
    assert QC_BAD == 2
```

- **VALIDATE**: `uv run pytest tests/data/test_schema.py -v`

---

### Task 11: CREATE `tests/data/test_kelmarsh.py`

Test Kelmarsh parser with synthetic data.

```python
"""Tests for windcast.data.kelmarsh parser."""

# Test strategy:
# - Create synthetic CSV data matching Kelmarsh format (with correct headers)
# - Package in a temp ZIP mimicking Kelmarsh nested structure
# - Run parser and verify output schema, row counts, column values
# - Test edge cases: NaN values, missing columns, power conversion

# Key tests:
# - test_parse_kelmarsh_produces_canonical_schema
# - test_signal_mapping_correct
# - test_pitch_angle_averaged
# - test_power_conversion (if kWh → kW needed)
# - test_timestamp_is_utc
# - test_turbine_id_extracted
# - test_handles_nan_values
```

- **PATTERN**: Use `tmp_path` fixture for temporary ZIP creation
- **GOTCHA**: Must create realistic ZIP structure: outer ZIP → inner annual ZIP → turbine CSVs
- **GOTCHA**: Test headers must exactly match `KELMARSH_SIGNAL_MAP` keys
- **VALIDATE**: `uv run pytest tests/data/test_kelmarsh.py -v`

---

### Task 12: CREATE `tests/data/test_qc.py`

Test QC pipeline rules independently.

```python
"""Tests for windcast.data.qc module."""

# Test strategy:
# - Create small canonical DataFrames with known issues
# - Run each QC rule independently
# - Verify flags are set correctly

# Key tests:
# - test_negative_power_flagged_bad
# - test_over_rated_power_flagged_suspect
# - test_negative_wind_flagged_bad
# - test_extreme_wind_flagged_suspect
# - test_frozen_sensor_detected
# - test_small_gaps_filled
# - test_large_gaps_preserved
# - test_curtailment_detected
# - test_qc_summary_counts
```

- **PATTERN**: Each test creates a small DataFrame (5-20 rows) with one specific issue
- **GOTCHA**: For frozen sensor test, need 7+ rows with identical values (> 60 min at 10-min resolution)
- **GOTCHA**: For gap-fill test, create DataFrame with missing timestamps, run QC, verify filled vs unfilled
- **VALIDATE**: `uv run pytest tests/data/test_qc.py -v`

---

### Task 13: CREATE `tests/data/test_open_meteo.py`

Test Open-Meteo client with mocked responses.

```python
"""Tests for windcast.data.open_meteo module."""

# Test strategy:
# - Mock the openmeteo_requests.Client to avoid real API calls
# - Verify client builds correctly
# - Verify output DataFrame schema

# Key tests:
# - test_build_client_returns_client
# - test_wind_variables_defined
# - test_fetch_returns_polars_dataframe (mocked)
# - test_fetch_timestamps_are_utc (mocked)
```

- **PATTERN**: Use `unittest.mock.patch` or `monkeypatch` to mock API calls
- **GOTCHA**: Don't make real API calls in tests — mock the `client.weather_api()` method
- **VALIDATE**: `uv run pytest tests/data/test_open_meteo.py -v`

---

### Task 14: RUN full validation suite

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
uv run pytest tests/ -v
```

- All 4 commands must pass with exit code 0
- Expected test count: ~20-25 tests (3 existing + ~17-22 new)

---

## TESTING STRATEGY

### Unit Tests

| Module | Test File | Key Tests |
|--------|-----------|-----------|
| `config.py` | `tests/test_config.py` | Default settings, env override, derived paths, dataset registry |
| `schema.py` | `tests/data/test_schema.py` | Schema completeness, validation, empty frame, QC constants |
| `kelmarsh.py` | `tests/data/test_kelmarsh.py` | Signal mapping, pitch averaging, power conversion, ZIP parsing |
| `qc.py` | `tests/data/test_qc.py` | Each QC rule independently, gap-fill, summary stats |
| `open_meteo.py` | `tests/data/test_open_meteo.py` | Client setup, variable list, mocked fetch |

### Integration Tests

- `scripts/ingest_kelmarsh.py` with synthetic data (optional — can be manual test with real data)

### Edge Cases

- CSV with all-NaN columns (sensor failure for entire period)
- CSV with zero rows (empty file in archive)
- Power values that suggest kWh vs kW (test both paths)
- Timestamps with DST transitions (UK local time)
- Status file missing (no status mapping available)
- ZIP containing unexpected file types (KMZ, XLSX — should be skipped)

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
```

### Level 2: Type Checking

```bash
uv run pyright src/
```

### Level 3: Tests

```bash
uv run pytest tests/ -v
```

### Level 4: Import Verification

```bash
uv run python -c "from windcast.config import get_settings; print(get_settings().dataset_id)"
uv run python -c "from windcast.data.schema import SCADA_SCHEMA; print(len(SCADA_SCHEMA), 'columns')"
uv run python -c "from windcast.data.kelmarsh import KELMARSH_SIGNAL_MAP; print(len(KELMARSH_SIGNAL_MAP), 'signals')"
uv run python -c "from windcast.data.qc import run_qc_pipeline; print('qc OK')"
uv run python -c "from windcast.data.open_meteo import WIND_VARIABLES; print(len(WIND_VARIABLES), 'NWP vars')"
```

### Level 5: Real Data Test (manual, after Kelmarsh download completes)

```bash
uv run python scripts/ingest_kelmarsh.py --raw-path data/raw/kelmarsh/16807551.zip
ls -la data/processed/kelmarsh_*.parquet
```

---

## ACCEPTANCE CRITERIA

- [ ] `config.py` loads defaults and respects `WINDCAST_*` env overrides
- [ ] `schema.py` defines 15-column SCADA schema matching PRD Section 7.1
- [ ] `validate_schema()` catches missing columns, wrong types
- [ ] `kelmarsh.py` reads nested ZIPs, maps Greenbyte signals → canonical names
- [ ] Power values are in kW (not kWh) — max should be ~2050
- [ ] Pitch angle is averaged from 3 blades
- [ ] Timestamps are UTC Datetime in output
- [ ] `qc.py` implements all 9 QC rules from PRD
- [ ] QC flags: 0=ok, 1=suspect, 2=bad — worst flag wins
- [ ] Small gaps (< 30 min) are forward-filled
- [ ] Large gaps are left as null
- [ ] `open_meteo.py` builds cached client and fetches to Polars DataFrame
- [ ] `ingest_kelmarsh.py` runs end-to-end: ZIP → QC → Parquet
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Linting passes: `uv run ruff check src/ tests/ scripts/`
- [ ] Type checking passes: `uv run pyright src/`
- [ ] Output Parquet files are schema-compliant (validate_schema returns [])

---

## COMPLETION CHECKLIST

- [ ] Task 1: config.py created and importable
- [ ] Task 2: schema.py created, SCADA_SCHEMA defined
- [ ] Task 3: kelmarsh.py created with parser and signal map
- [ ] Task 4: qc.py created with QC pipeline
- [ ] Task 5: open_meteo.py created with weather client
- [ ] Task 6: data/__init__.py updated with exports
- [ ] Task 7: ingest_kelmarsh.py CLI script created
- [ ] Task 8: tests/data/__init__.py created
- [ ] Task 9: test_config.py passing
- [ ] Task 10: test_schema.py passing
- [ ] Task 11: test_kelmarsh.py passing
- [ ] Task 12: test_qc.py passing
- [ ] Task 13: test_open_meteo.py passing
- [ ] Task 14: Full validation suite passing (ruff, pyright, pytest)
- [ ] All acceptance criteria met

---

## NOTES

### Design Decisions

1. **Parser returns pre-QC data**: `kelmarsh.py` outputs with `qc_flag=0, is_curtailed=False, is_maintenance=False`. QC is always a separate explicit step via `qc.py`. This keeps parsers simple and QC testable independently.

2. **One Parquet per turbine**: Output as `kelmarsh_kwf1.parquet`, not one giant file. Enables per-turbine processing and reduces memory pressure. Polars can glob-read them back: `pl.scan_parquet("data/processed/kelmarsh_*.parquet")`.

3. **Status code parsing deferred**: Kelmarsh status codes are poorly documented. Initial implementation sets `status_code=0` for all rows. Proper status mapping will be refined once we examine real status CSVs. The QC pipeline's `_flag_maintenance` will initially be a no-op until status codes are understood.

4. **Power unit verification at runtime**: The signal mapping says power is kWh, but this must be verified with real data. The parser should check: if max(power) ≈ rated_power_kw, it's already kW; if max(power) ≈ rated_power_kw/6, it's kWh. Log a warning either way.

5. **Curtailment detection is approximate**: Without a proper power curve model (Phase 2), we use a simple heuristic: wind > 8 m/s AND pitch > 3° AND power < 50% rated. This catches obvious curtailment. False positives are acceptable in Phase 1.2.

6. **Open-Meteo not wired into ingestion script yet**: The NWP client is built in Phase 1.2 but only used in Phase 1.3 (feature engineering). Building it now ensures the data layer is complete.

7. **No argparse for dataset selection yet**: `ingest_kelmarsh.py` is Kelmarsh-specific. A generic `ingest.py --dataset kelmarsh` pattern can be added when Hill of Towie parser exists (Phase 2b).

### Key Risks

1. **Signal map mismatch** — The Greenbyte CSV headers may not exactly match our mapping. Mitigation: first run should print actual headers and fail fast with clear error if mapping keys don't match.

2. **Power unit ambiguity** — kWh vs kW. Mitigation: runtime check against rated_power_kw, with clear logging.

3. **Nested ZIP complexity** — File structure may vary between Kelmarsh versions. Mitigation: defensive ZIP traversal with logging of all found files.

### Confidence Score: 7/10

High confidence on schema, config, QC logic, and Open-Meteo client. Medium confidence on Kelmarsh parser — the exact CSV header format and ZIP structure must be verified against real data. The signal map may need adjustment on first run.
