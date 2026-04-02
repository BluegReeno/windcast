# Feature: Phase 2 — Demand Domain

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Add power demand forecasting as a second domain to the EnerCast framework, proving that the same pipeline pattern (parse → QC → features → train → evaluate → MLflow) works across structurally different energy domains. Uses the Spain ENTSO-E dataset (Kaggle) with national grid load as target, 5-city weather as covariates.

## User Story

As a WeatherNews evaluator,
I want to see the same pipeline handle wind and demand data,
So that I believe standardization is possible across our 3 domains.

## Problem Statement

The current codebase is wind-only: schema, QC, features, and scripts are tightly coupled to SCADA concepts (turbine_id, wind_speed, pitch_angle, etc.). Adding a second domain requires domain-agnostic abstractions while keeping domain-specific code isolated.

## Solution Statement

Create a demand-specific data layer (schema, parser, QC) and feature layer, then refactor the CLI scripts to accept a `--domain` argument that dispatches to the right modules. The core training/evaluation/MLflow logic remains 100% unchanged — only the data loading and feature selection changes.

## Feature Metadata

**Feature Type**: New Capability (second domain)
**Estimated Complexity**: High (touches all layers, but patterns are established)
**Primary Systems Affected**: data/, features/, config.py, scripts/
**Dependencies**: None new — Polars, XGBoost, MLflow already in pyproject.toml

---

## CONTEXT REFERENCES

### Relevant Codebase Files — YOU MUST READ THESE BEFORE IMPLEMENTING!

**Data Layer (template patterns):**
- `src/windcast/data/schema.py` (all) — Schema pattern: dict of col→dtype, validate_schema(), QC constants. **Mirror this for demand.**
- `src/windcast/data/kelmarsh.py` (all) — Parser pattern: public `parse_X()` entry point, signal map, `_read_*_csv()` core transform, ensure all schema cols present, cast & reorder. **Mirror structure for Spain parser.**
- `src/windcast/data/qc.py` (all) — QC pattern: `run_qc_pipeline()` orchestrator, private `_flag_*()` functions, `qc_summary()`. Uses `pl.max_horizontal()` to never downgrade flags. **Adapt for demand-specific rules.**
- `src/windcast/data/__init__.py` (all) — Exports only schema + validation. **Update to export demand schema too.**
- `src/windcast/config.py` (all) — DatasetConfig, DATASETS registry, QCConfig, WindCastSettings. **Add DemandDatasetConfig, SPAIN_DEMAND, DemandQCConfig.**

**Feature Layer (template patterns):**
- `src/windcast/features/registry.py` (all) — FeatureSet dataclass, declarative WIND_* sets, FEATURE_REGISTRY dict, get/list functions. **Add DEMAND_* sets to registry.**
- `src/windcast/features/wind.py` (all) — `build_wind_features()` pattern: filter QC, sort, apply transforms, return expanded DataFrame. **Mirror for `build_demand_features()`.**
- `src/windcast/features/__init__.py` (all) — Exports registry + wind builder. **Add demand builder.**

**Scripts (must be refactored):**
- `scripts/ingest_kelmarsh.py` (all) — Ingestion CLI pattern: argparse, settings, parse → QC → write parquet. **Create parallel `ingest_spain_demand.py`.**
- `scripts/build_features.py` (all) — Feature CLI pattern: load parquet, build features, drop nulls, write. **Add `--domain` flag to dispatch.**
- `scripts/train.py` (lines 24-45: temporal split, lines 48-64: horizon target, lines 118-121: feature filtering, lines 134-197: MLflow loop) — Training CLI. **Add `--domain` flag, parameterize target_col and group_col.**
- `scripts/evaluate.py` (lines 42-63: model loading, lines 157-218: eval loop, lines 198-206: regime analysis) — Eval CLI. **Add `--domain` flag, parameterize regime analysis.**

**Tests (template patterns):**
- `tests/data/test_schema.py` (all) — Schema test pattern: check columns, validate errors, empty frame factory
- `tests/data/test_kelmarsh.py` (all) — Parser test pattern: `_make_csv_bytes()` builder, `_make_zip()`, test classes per function
- `tests/data/test_qc.py` (all) — QC test pattern: `_make_scada_df()` factory with **overrides, test class per rule
- `tests/features/test_registry.py` (all) — Registry test pattern: hierarchy (enriched extends baseline), error case, frozen dataclass
- `tests/features/test_wind.py` (all) — Feature test pattern: `_make_wind_df()` factory, lag validation, rolling no-look-ahead, cyclic range
- `tests/test_config.py` (all) — Config test: defaults, registry, env override, derived paths

### New Files to Create

**Data layer:**
- `src/windcast/data/demand_schema.py` — Demand canonical schema (11 cols)
- `src/windcast/data/spain_demand.py` — Spain ENTSO-E parser (energy CSV + weather CSV → canonical demand)
- `src/windcast/data/demand_qc.py` — Demand QC pipeline (outlier, gap, holiday, DST)

**Feature layer:**
- `src/windcast/features/demand.py` — Demand feature builders

**Scripts:**
- `scripts/ingest_spain_demand.py` — Demand ingestion CLI

**Tests:**
- `tests/data/test_demand_schema.py` — Demand schema tests
- `tests/data/test_spain_demand.py` — Spain parser tests
- `tests/data/test_demand_qc.py` — Demand QC tests
- `tests/features/test_demand.py` — Demand feature tests

### Relevant Documentation

**Spain Dataset Details** (from research):

The dataset is at `kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather`.
Download: `kaggle datasets download -d nicholasjhana/energy-consumption-generation-prices-and-weather --unzip`

**File 1: `energy_dataset.csv`** — 35,064 rows x 29 cols, hourly, 2015-01-01 to 2018-12-31
- Timestamp column: `time` (format: `2015-01-01 00:00:00+01:00`, Europe/Madrid TZ)
- Key columns for demand:
  - `total load actual` (MW) — **PRIMARY TARGET** (range: 18,041–41,015 MW)
  - `total load forecast` (MW) — TSO benchmark
  - `price day ahead` (EUR/MWh)
  - `price actual` (EUR/MWh)
  - `generation wind onshore` (MW) — wind generation (secondary)
  - `generation solar` (MW) — solar generation (secondary)
- ~18 NaN per generation column, 36 NaN in `total load actual`
- Drop 8 columns: 2 entirely null, 6 always zero

**File 2: `weather_features.csv`** — 178,396 rows x 17 cols, hourly, 5 cities (long format)
- Timestamp: `dt_iso` (same TZ format as energy)
- Cities: Barcelona, Bilbao, Madrid, Seville, Valencia
- **CRITICAL**: Barcelona has leading space in `city_name` — must `.str.strip()`
- **CRITICAL**: `temp` is in **Kelvin** — subtract 273.15
- **CRITICAL**: Barcelona `pressure` has 45 outliers (up to 1,008,371 hPa)
- **CRITICAL**: Valencia `wind_speed` has 3 outliers >50 m/s (max=133)
- Key weather columns: `temp` (K), `humidity` (%), `wind_speed` (m/s), `pressure` (hPa), `clouds_all` (%), `rain_1h` (mm)
- Must pivot from long (5 cities) to wide (aggregated mean) for joining with energy

### Patterns to Follow

**Parser Pattern** (from kelmarsh.py):
```python
# Public entry point
def parse_spain_demand(energy_path: Path, weather_path: Path) -> pl.DataFrame:
    ...

# Private helpers
def _read_energy_csv(path: Path) -> pl.DataFrame:
    ...

def _read_weather_csv(path: Path) -> pl.DataFrame:
    ...

def _aggregate_weather(df: pl.DataFrame) -> pl.DataFrame:
    # Pivot 5 cities → mean per timestamp
    ...
```

**QC Pattern** (from qc.py):
```python
def run_demand_qc_pipeline(
    df: pl.DataFrame,
    qc_config: DemandQCConfig | None = None,
) -> pl.DataFrame:
    df = _flag_load_outliers(df, ...)
    df = _flag_weather_outliers(df, ...)
    df = _detect_holidays(df)
    df = _detect_dst_transitions(df)
    df = _fill_small_gaps(df, ...)
    return df
```

**Feature Builder Pattern** (from wind.py):
```python
def build_demand_features(
    df: pl.DataFrame,
    feature_set: str = "demand_baseline",
) -> pl.DataFrame:
    fs = get_feature_set(feature_set)
    df = df.filter(pl.col("qc_flag") == QC_OK)
    df = df.sort("zone_id", "timestamp_utc")
    # Apply feature transforms...
    return df
```

**Naming Conventions:**
- Files: `snake_case.py`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_prefix`
- Schemas: `DEMAND_SCHEMA` dict
- Feature sets: `DEMAND_BASELINE`, `DEMAND_ENRICHED`, `DEMAND_FULL`

**Error Handling:**
- `raise FileNotFoundError(...)` for missing files
- `raise ValueError(...)` for invalid data
- `logger.warning(...)` for non-critical issues (missing columns, skipped data)
- Never swallow exceptions silently

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — Demand Schema + Config

Create the demand canonical schema and dataset configuration. This is the foundation everything else depends on.

**Tasks:**
- Define 11-column demand schema with Polars types
- Add DemandDatasetConfig and DemandQCConfig to config.py
- Add SPAIN_DEMAND to DATASETS registry
- Add domain concept to WindCastSettings (rename to EnerCastSettings or add domain field)

### Phase 2: Parser — Spain Demand

Parse the 2 Spain CSVs (energy + weather) into the canonical demand schema. This is the most complex new module.

**Tasks:**
- Read energy CSV: parse timestamps to UTC, select relevant columns, handle NaN
- Read weather CSV: strip city names, convert Kelvin→Celsius, filter outliers, pivot to wide, aggregate
- Join energy + weather on timestamp
- Map to canonical demand schema
- Add default QC columns

### Phase 3: Demand QC

Implement demand-specific quality control rules.

**Tasks:**
- Flag load outliers (negative, extreme high)
- Flag weather outliers (extreme temp, pressure, wind)
- Detect holidays (Spain public holidays 2015-2018)
- Detect DST transitions
- Forward-fill small gaps (< 3 hours for hourly data)

### Phase 4: Demand Features

Implement demand-specific feature engineering.

**Tasks:**
- Add demand feature sets to registry (baseline/enriched/full)
- Build demand feature builder with load lags, calendar encoding, temperature features

### Phase 5: Script Integration

Create ingestion script and refactor existing scripts for multi-domain support.

**Tasks:**
- Create ingest_spain_demand.py
- Add --domain flag to build_features.py, train.py, evaluate.py
- Parameterize target column and group column per domain

### Phase 6: Tests

Full test coverage for all new modules.

**Tasks:**
- Schema tests, parser tests, QC tests, feature tests
- Integration test: full pipeline demand data → features

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

---

### Task 1: CREATE `src/windcast/data/demand_schema.py`

**IMPLEMENT**: Demand canonical schema following the exact pattern from `schema.py`.

```python
DEMAND_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "dataset_id": pl.Utf8,
    "zone_id": pl.Utf8,
    "load_mw": pl.Float64,
    "temperature_c": pl.Float64,
    "wind_speed_ms": pl.Float64,
    "humidity_pct": pl.Float64,
    "price_eur_mwh": pl.Float64,
    "is_holiday": pl.Boolean,
    "is_dst_transition": pl.Boolean,
    "qc_flag": pl.UInt8,
}
```

Functions to implement:
- `DEMAND_COLUMNS: list[str]` — ordered column list
- `DEMAND_SIGNAL_COLUMNS: list[str]` — subset of numeric measurement columns (load_mw, temperature_c, wind_speed_ms, humidity_pct, price_eur_mwh)
- `validate_demand_schema(df, *, strict=False) -> list[str]` — mirror `validate_schema()` logic exactly
- `empty_demand_frame() -> pl.DataFrame` — test factory

**PATTERN**: Mirror `src/windcast/data/schema.py` structure exactly.
**IMPORTS**: `import polars as pl` + reuse `QC_OK`, `QC_SUSPECT`, `QC_BAD` from `schema.py`
**GOTCHA**: Reuse QC flag constants from schema.py (don't redefine). Import them.
**VALIDATE**: `uv run pyright src/windcast/data/demand_schema.py`

---

### Task 2: CREATE `tests/data/test_demand_schema.py`

**IMPLEMENT**: Tests for demand schema, mirroring `tests/data/test_schema.py` pattern.

Tests to write:
- `test_demand_schema_has_expected_columns()` — check 11 columns
- `test_empty_demand_frame_matches_schema()` — validate empty frame
- `test_validate_demand_schema_detects_missing_column()` — drop a column, validate
- `test_validate_demand_schema_detects_wrong_type()` — cast wrong type, validate
- `test_validate_demand_schema_strict_mode()` — extra column detection
- `test_demand_columns_ordered()` — schema key order == DEMAND_COLUMNS order
- `test_demand_signal_columns()` — check 5 signal columns

**PATTERN**: Mirror `tests/data/test_schema.py` — simple test functions, no classes.
**VALIDATE**: `uv run pytest tests/data/test_demand_schema.py -v`

---

### Task 3: UPDATE `src/windcast/config.py`

**IMPLEMENT**: Add demand-specific config classes and dataset entry.

Add to config.py:
```python
class DemandDatasetConfig(BaseModel):
    """Configuration for a demand forecasting dataset."""
    dataset_id: str
    zone_id: str
    population: int | None = None
    latitude: float
    longitude: float
    timezone: str

class DemandQCConfig(BaseModel):
    """QC thresholds for demand data."""
    max_load_mw: float = 50_000.0       # Spain peak ~41k MW
    min_load_mw: float = 10_000.0       # Spain minimum ~18k MW
    max_temperature_c: float = 50.0
    min_temperature_c: float = -20.0
    max_wind_speed_ms: float = 50.0
    max_gap_fill_hours: int = 3
```

Add SPAIN_DEMAND dataset:
```python
SPAIN_DEMAND = DemandDatasetConfig(
    dataset_id="spain_demand",
    zone_id="ES",
    population=47_000_000,
    latitude=40.4168,   # Madrid
    longitude=-3.7038,
    timezone="Europe/Madrid",
)
```

Add to DATASETS registry: `"spain_demand": SPAIN_DEMAND` — but DATASETS currently maps to DatasetConfig. Two options:
- Option A: Make DATASETS a `dict[str, DatasetConfig | DemandDatasetConfig]` (simple, union type)
- Option B: Separate registries per domain

**DECISION**: Option A — single registry with union type. Keep it simple. The scripts know which config type to expect based on `--domain`.

Also add `domain: str = "wind"` field to `WindCastSettings` and `demand_qc: DemandQCConfig = Field(default_factory=DemandQCConfig)`.

**PATTERN**: Follow existing Pydantic BaseModel pattern from `DatasetConfig` and `QCConfig`.
**GOTCHA**: Don't break existing wind functionality. Keep all existing fields as-is. Add new ones.
**VALIDATE**: `uv run pytest tests/test_config.py -v && uv run pyright src/windcast/config.py`

---

### Task 4: UPDATE `tests/test_config.py`

**IMPLEMENT**: Add tests for new demand config.

New tests:
- `test_spain_demand_in_registry()` — verify `"spain_demand" in DATASETS`
- `test_spain_demand_config_values()` — check zone_id, latitude, etc.
- `test_demand_qc_defaults()` — check DemandQCConfig defaults
- `test_domain_default()` — settings.domain == "wind"
- `test_domain_env_override(monkeypatch)` — WINDCAST_DOMAIN=demand

**PATTERN**: Mirror existing `test_config.py` — simple functions, `get_settings.cache_clear()` before env tests.
**VALIDATE**: `uv run pytest tests/test_config.py -v`

---

### Task 5: CREATE `src/windcast/data/spain_demand.py`

**IMPLEMENT**: Spain demand parser — the largest new module (~200-250 lines).

Public API:
```python
def parse_spain_demand(
    energy_path: Path,
    weather_path: Path,
) -> pl.DataFrame:
    """Parse Spain ENTSO-E energy + weather CSVs into canonical demand schema."""
```

Implementation steps:

1. `_read_energy_csv(path: Path) -> pl.DataFrame`:
   - Read CSV with `pl.read_csv()`, infer_schema_length=10_000
   - Parse `time` column: it has timezone info (`+01:00`/`+02:00`) — parse as string, use `pl.col("time").str.to_datetime("%Y-%m-%d %H:%M:%S%:z")` then `.dt.convert_time_zone("UTC")`
   - Select relevant columns: `total load actual`, `total load forecast`, `price day ahead`, `price actual`, `generation wind onshore`, `generation solar`
   - Rename to canonical: `total load actual` → `load_mw`, `price day ahead` → `price_eur_mwh`
   - Drop 8 useless columns (6 always-zero generation, 2 entirely null)

2. `_read_weather_csv(path: Path) -> pl.DataFrame`:
   - Read CSV
   - Strip whitespace from `city_name`: `.str.strip_chars()`
   - Parse `dt_iso` timestamp same way as energy `time`
   - Convert temperature: `temp - 273.15` → `temperature_c`
   - Filter Barcelona pressure outliers: keep 900-1100 hPa
   - Filter Valencia wind outliers: keep <= 50 m/s
   - Select: timestamp, city_name, temperature_c, humidity, wind_speed, pressure, clouds_all, rain_1h

3. `_aggregate_weather(df: pl.DataFrame) -> pl.DataFrame`:
   - Group by timestamp, compute mean across 5 cities for each weather variable
   - Result: one row per timestamp with averaged weather

4. Main `parse_spain_demand()`:
   - Call _read_energy_csv() + _read_weather_csv()
   - Aggregate weather
   - Join on timestamp (inner join)
   - Add `dataset_id = "spain_demand"`, `zone_id = "ES"`
   - Add default flags: `is_holiday = False`, `is_dst_transition = False`, `qc_flag = 0`
   - Ensure all DEMAND_SCHEMA columns present, cast to schema dtypes, reorder
   - Sort by timestamp_utc

**PATTERN**: Mirror `kelmarsh.py` structure — public entry, private helpers, signal map, schema compliance at end.
**IMPORTS**: `import logging`, `from pathlib import Path`, `import polars as pl`, `from windcast.data.demand_schema import DEMAND_SCHEMA, DEMAND_COLUMNS`
**GOTCHA**:
- Timestamp format has timezone offset (`+01:00` or `+02:00` for DST) — must parse with TZ then convert to UTC
- Barcelona `city_name` has leading space — `.str.strip_chars()` on load
- Temperature is Kelvin — subtract 273.15
- Weather file has unequal rows per city — group-by handles this naturally
- ~36 NaN in `total load actual` — keep as null, QC will handle
**VALIDATE**: `uv run pyright src/windcast/data/spain_demand.py`

---

### Task 6: CREATE `tests/data/test_spain_demand.py`

**IMPLEMENT**: Parser tests using synthetic CSV data (like test_kelmarsh.py pattern).

Helpers to create:
- `_make_energy_csv_bytes(n_rows=48)` — 2 days of hourly data with realistic column names
- `_make_weather_csv_bytes(n_rows=48, n_cities=5)` — matching weather data in long format

Test classes:
- `TestReadEnergyCsv`:
  - `test_parses_timestamps_to_utc()` — verify UTC conversion
  - `test_selects_relevant_columns()` — check column renaming
  - `test_handles_nan_values()` — NaN in load column
- `TestReadWeatherCsv`:
  - `test_strips_city_names()` — Barcelona space issue
  - `test_converts_kelvin_to_celsius()` — verify conversion
  - `test_filters_pressure_outliers()` — extreme values removed
- `TestAggregateWeather`:
  - `test_aggregates_to_one_row_per_timestamp()` — 5 cities → 1 row
- `TestParseSpainDemand`:
  - `test_returns_schema_compliant_frame()` — validate_demand_schema() passes
  - `test_sorts_by_timestamp()` — chronological order
  - `test_file_not_found()` — missing files raise FileNotFoundError

**PATTERN**: Mirror `tests/data/test_kelmarsh.py` — `_make_csv_bytes()` helpers, test classes, `tmp_path` fixture.
**GOTCHA**: Use realistic column names with spaces (e.g., `"total load actual"`) in test CSV builders.
**VALIDATE**: `uv run pytest tests/data/test_spain_demand.py -v`

---

### Task 7: CREATE `src/windcast/data/demand_qc.py`

**IMPLEMENT**: Demand-specific QC pipeline, following `qc.py` pattern.

Public API:
```python
def run_demand_qc_pipeline(
    df: pl.DataFrame,
    qc_config: DemandQCConfig | None = None,
) -> pl.DataFrame:
```

QC rules (in order):
1. `_flag_load_outliers(df, min_load, max_load)` — negative or extreme load → qc_flag
2. `_flag_temperature_outliers(df, min_temp, max_temp)` — extreme temperature → qc_flag
3. `_flag_wind_outliers(df, max_wind)` — extreme wind speed → qc_flag
4. `_detect_holidays(df)` — set `is_holiday = True` for Spain public holidays 2015-2018
5. `_detect_dst_transitions(df)` — set `is_dst_transition = True` for March/October DST change hours
6. `_fill_small_gaps(df, max_gap_hours)` — forward-fill signal columns, limit to max_gap_hours

Holiday detection: Define a list of Spain public holiday dates (fixed holidays + movable ones for 2015-2018). Check `timestamp_utc.dt.date()` against this list.

DST detection: Spain uses EU rules — last Sunday of March (spring forward) and last Sunday of October (fall back). Flag rows on those dates between 01:00-03:00 UTC.

Utility:
```python
def demand_qc_summary(df: pl.DataFrame) -> dict[str, int | float]:
    # Same pattern as qc_summary() from qc.py
```

**PATTERN**: Mirror `src/windcast/data/qc.py` — orchestrator + private functions + summary.
**IMPORTS**: `polars as pl`, `logging`, `from windcast.config import DemandQCConfig`, `from windcast.data.schema import QC_BAD, QC_SUSPECT`
**GOTCHA**:
- Use `pl.max_horizontal()` pattern to never downgrade flags (same as wind QC)
- Forward-fill limit is in rows (hourly = 1 row per hour), not minutes
- Holiday list must cover 2015-2018 (dataset period)
- Don't over-engineer holidays — a simple date list is fine for the demo
**VALIDATE**: `uv run pyright src/windcast/data/demand_qc.py`

---

### Task 8: CREATE `tests/data/test_demand_qc.py`

**IMPLEMENT**: QC tests mirroring `tests/data/test_qc.py` pattern.

Helper:
```python
def _make_demand_df(n_rows=24, **overrides) -> pl.DataFrame:
    """Factory for demand DataFrames with sensible defaults."""
```

Test classes:
- `TestFlagLoadOutliers`: negative load, extreme load, normal load
- `TestFlagTemperatureOutliers`: extreme cold, extreme hot, normal
- `TestFlagWindOutliers`: extreme wind speed
- `TestDetectHolidays`: known holiday date, non-holiday date
- `TestDetectDstTransitions`: March DST date, October DST date, normal date
- `TestFillSmallGaps`: gap <= threshold filled, gap > threshold not filled
- `TestDemandQcSummary`: verify summary dict keys

**PATTERN**: Mirror `tests/data/test_qc.py` — `_make_demand_df()` factory with **overrides, test class per rule.
**VALIDATE**: `uv run pytest tests/data/test_demand_qc.py -v`

---

### Task 9: UPDATE `src/windcast/data/__init__.py`

**IMPLEMENT**: Add demand schema exports.

```python
from windcast.data.demand_schema import DEMAND_SCHEMA, validate_demand_schema
from windcast.data.schema import SCADA_SCHEMA, validate_schema

__all__ = [
    "DEMAND_SCHEMA",
    "SCADA_SCHEMA",
    "validate_demand_schema",
    "validate_schema",
]
```

**VALIDATE**: `uv run pyright src/windcast/data/__init__.py`

---

### Task 10: ADD demand feature sets to `src/windcast/features/registry.py`

**IMPLEMENT**: Add 3 demand feature sets to the registry.

```python
DEMAND_BASELINE = FeatureSet(
    name="demand_baseline",
    columns=[
        "load_mw_lag1",      # H-1
        "load_mw_lag2",      # H-2
        "load_mw_lag24",     # D-1
        "load_mw_lag168",    # W-1
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ],
    description="Demand baseline: load lags (H-1, H-2, D-1, W-1) + calendar cyclic",
)

DEMAND_ENRICHED = FeatureSet(
    name="demand_enriched",
    columns=[
        # All baseline columns +
        "load_mw_lag1", "load_mw_lag2", "load_mw_lag24", "load_mw_lag168",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Temperature
        "temperature_c",
        "heating_degree_days",   # max(0, 18 - temp)
        "cooling_degree_days",   # max(0, temp - 24)
        # Rolling load stats
        "load_mw_roll_mean_24",  # 24h rolling mean
        "load_mw_roll_std_24",   # 24h rolling std
        "load_mw_roll_mean_168", # 7-day rolling mean
    ],
    description="Demand enriched: + temperature, HDD/CDD, rolling load stats",
)

DEMAND_FULL = FeatureSet(
    name="demand_full",
    columns=[
        # All enriched columns +
        "load_mw_lag1", "load_mw_lag2", "load_mw_lag24", "load_mw_lag168",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "temperature_c", "heating_degree_days", "cooling_degree_days",
        "load_mw_roll_mean_24", "load_mw_roll_std_24", "load_mw_roll_mean_168",
        # Weather
        "wind_speed_ms",
        "humidity_pct",
        # Price
        "price_eur_mwh",
        "price_lag1",
        "price_lag24",
        # Holiday
        "is_holiday",
    ],
    description="Demand full: + wind, humidity, price lags, holiday flag",
)
```

Add all three to `FEATURE_REGISTRY`.

**PATTERN**: Mirror WIND_BASELINE/ENRICHED/FULL pattern. Enriched extends baseline, full extends enriched.
**GOTCHA**: Lag naming must match what `build_demand_features()` will produce. Demand lags are in hours (lag1=1h, lag24=1day, lag168=1week), not 10-min steps like wind.
**VALIDATE**: `uv run pytest tests/features/test_registry.py -v`

---

### Task 11: CREATE `src/windcast/features/demand.py`

**IMPLEMENT**: Demand feature builder, mirroring `wind.py` pattern.

Public API:
```python
def build_demand_features(
    df: pl.DataFrame,
    feature_set: str = "demand_baseline",
) -> pl.DataFrame:
```

Feature transforms:
1. Filter by `qc_flag == QC_OK`
2. Sort by `zone_id`, `timestamp_utc`
3. **Always (baseline)**:
   - Load lags: `_add_lag_features(df, "load_mw", [1, 2, 24, 168])` — use `.shift(lag).over("zone_id")`
   - Calendar cyclic: hour (period=24), day-of-week (period=7), month (period=12)
4. **Enriched adds**:
   - Temperature as-is (already in df)
   - HDD: `pl.max_horizontal(pl.lit(0.0), pl.lit(18.0) - pl.col("temperature_c"))`
   - CDD: `pl.max_horizontal(pl.lit(0.0), pl.col("temperature_c") - pl.lit(24.0))`
   - Rolling load stats: 24h and 168h windows (shift(1) before rolling to prevent look-ahead)
5. **Full adds**:
   - wind_speed_ms, humidity_pct (already in df)
   - price_eur_mwh (already in df)
   - Price lags: shift(1), shift(24) over zone_id
   - is_holiday (already in df, cast to Int8 for model)

Private helpers (mirror wind.py pattern):
- `_add_lag_features(df, col, lags)` — same pattern as wind but over "zone_id"
- `_add_rolling_features(df, col, windows)` — same pattern, shift(1) first
- `_add_cyclic_calendar(df)` — hour, dow, month (reuse pattern from wind.py)
- `_add_temperature_features(df)` — HDD, CDD
- `_add_price_features(df)` — price lags

**PATTERN**: Mirror `src/windcast/features/wind.py` — filter QC, sort, progressively add features per set level.
**IMPORTS**: `polars as pl`, `math`, `logging`, `from windcast.data.schema import QC_OK`, `from windcast.features.registry import get_feature_set`
**GOTCHA**:
- Demand uses `zone_id` for `.over()` instead of `turbine_id`
- Lag 168 = 1 week for hourly data (168 hours)
- Rolling windows: 24 = 1 day, 168 = 1 week (in rows, since hourly)
- `is_holiday` is Boolean — cast to numeric for XGBoost
**VALIDATE**: `uv run pyright src/windcast/features/demand.py`

---

### Task 12: UPDATE `src/windcast/features/__init__.py`

**IMPLEMENT**: Export demand feature builder.

Add:
```python
from windcast.features.demand import build_demand_features
```

Update `__all__` to include `build_demand_features`.

**VALIDATE**: `uv run pyright src/windcast/features/__init__.py`

---

### Task 13: CREATE `tests/features/test_demand.py`

**IMPLEMENT**: Demand feature tests, mirroring `tests/features/test_wind.py` pattern.

Helper:
```python
def _make_demand_df(n_rows=200, n_zones=1) -> pl.DataFrame:
    """Factory for demand DataFrames with realistic hourly data."""
    # Use sine patterns for load (daily pattern), temperature (seasonal)
    # Include all DEMAND_SCHEMA columns
```

Test classes:
- `TestLagFeatures`:
  - `test_lag_values_are_shifted()` — verify lag1 == previous row value
  - `test_lag_per_zone()` — multi-zone: no cross-zone leakage
  - `test_weekly_lag()` — lag168 correct
- `TestRollingFeatures`:
  - `test_no_look_ahead()` — rolling uses shift(1)
  - `test_rolling_window_size()` — 24h window produces correct mean
- `TestCalendarFeatures`:
  - `test_hour_cyclic_range()` — sin/cos in [-1, 1]
  - `test_dow_cyclic_range()` — sin/cos in [-1, 1]
- `TestTemperatureFeatures`:
  - `test_hdd_positive_when_cold()` — temp < 18 → HDD > 0
  - `test_cdd_positive_when_hot()` — temp > 24 → CDD > 0
  - `test_hdd_zero_when_warm()` — temp >= 18 → HDD == 0
- `TestBuildDemandFeatures`:
  - `test_qc_filter_removes_bad_rows()` — QC_BAD filtered
  - `test_baseline_produces_expected_columns()` — check feature set columns
  - `test_enriched_extends_baseline()` — enriched has more columns

**PATTERN**: Mirror `tests/features/test_wind.py` — `_make_demand_df()` factory, sorted data, test per feature group.
**VALIDATE**: `uv run pytest tests/features/test_demand.py -v`

---

### Task 14: UPDATE `tests/features/test_registry.py`

**IMPLEMENT**: Add tests for demand feature sets.

New tests:
- `test_demand_baseline_exists()` — get_feature_set("demand_baseline") works
- `test_demand_enriched_extends_baseline()` — all baseline cols in enriched
- `test_demand_full_extends_enriched()` — all enriched cols in full
- `test_list_includes_demand()` — list_feature_sets() includes demand sets

**VALIDATE**: `uv run pytest tests/features/test_registry.py -v`

---

### Task 15: CREATE `scripts/ingest_spain_demand.py`

**IMPLEMENT**: Demand ingestion CLI, mirroring `ingest_kelmarsh.py` pattern.

```python
def main():
    parser = argparse.ArgumentParser(description="Ingest Spain demand data")
    parser.add_argument("--energy-path", type=Path, default=None)
    parser.add_argument("--weather-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    energy_path = args.energy_path or settings.raw_dir / "spain" / "energy_dataset.csv"
    weather_path = args.weather_path or settings.raw_dir / "spain" / "weather_features.csv"
    output_dir = args.output_dir or settings.processed_dir

    # Parse
    df = parse_spain_demand(energy_path, weather_path)

    # Validate
    errors = validate_demand_schema(df)
    if errors:
        logger.error("Schema validation failed: %s", errors)
        sys.exit(1)

    # QC
    df = run_demand_qc_pipeline(df, settings.demand_qc)
    summary = demand_qc_summary(df)
    logger.info("QC summary: %s", summary)

    # Write single file (demand has one zone, not per-turbine)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "spain_demand.parquet"
    df.write_parquet(out_path, compression="zstd", compression_level=3)
    logger.info("Wrote %d rows to %s (%.1f MB)", len(df), out_path, out_path.stat().st_size / 1e6)
```

**PATTERN**: Mirror `scripts/ingest_kelmarsh.py` — argparse, settings fallback, parse → validate → QC → write.
**GOTCHA**: Demand writes ONE parquet file (not per-turbine), since there's only one zone.
**VALIDATE**: `uv run pyright scripts/ingest_spain_demand.py`

---

### Task 16: UPDATE `scripts/build_features.py` — Add `--domain` flag

**IMPLEMENT**: Add `--domain` argument to dispatch between wind and demand feature builders.

Changes:
1. Add argument: `parser.add_argument("--domain", choices=["wind", "demand"], default="wind")`
2. Update `choices` for `--feature-set`: make dynamic based on domain (filter registry by prefix)
3. Dispatch feature builder: `build_wind_features()` vs `build_demand_features()` based on domain
4. Update file glob pattern: wind = `kelmarsh_*.parquet`, demand = `spain_demand.parquet`
5. Update default feature set: wind → `wind_baseline`, demand → `demand_baseline`

**PATTERN**: Keep backward-compatible — `--domain wind` is the default, existing behavior unchanged.
**GOTCHA**: Feature set choices depend on domain. Use conditional filtering of list_feature_sets().
**VALIDATE**: `uv run pyright scripts/build_features.py`

---

### Task 17: UPDATE `scripts/train.py` — Add `--domain` flag

**IMPLEMENT**: Add domain-awareness to training script.

Changes:
1. Add `--domain` argument (choices=["wind", "demand"], default="wind")
2. **Parameterize target column**: wind → `"active_power_kw"`, demand → `"load_mw"`
3. **Parameterize group column**: wind → `"turbine_id"`, demand → `"zone_id"`
4. **Parameterize lag column**: wind → `"active_power_kw_lag1"`, demand → `"load_mw_lag1"`
5. Update file glob: demand uses `spain_demand_features.parquet` (or similar)
6. Update experiment name default: include domain
7. Update feature set default: domain-specific

Create a small helper dict or function:
```python
DOMAIN_CONFIG = {
    "wind": {"target": "active_power_kw", "group": "turbine_id", "lag1": "active_power_kw_lag1"},
    "demand": {"target": "load_mw", "group": "zone_id", "lag1": "load_mw_lag1"},
}
```

**PATTERN**: Minimal changes — replace hardcoded strings with lookups. Keep same MLflow structure.
**GOTCHA**:
- Temporal split works the same way (year-based) for both domains
- Horizon semantics differ: wind = 10-min steps, demand = hourly steps. The shift(-h) logic is identical, but the time interpretation changes. For now, keep horizon as "steps" and document the difference.
- Demand has only 1 zone, so the "per-turbine" loop becomes a single iteration
**VALIDATE**: `uv run pyright scripts/train.py`

---

### Task 18: UPDATE `scripts/evaluate.py` — Add `--domain` flag

**IMPLEMENT**: Add domain-awareness to evaluation script.

Changes:
1. Add `--domain` argument
2. Parameterize target column and lag column (same DOMAIN_CONFIG pattern)
3. Update regime analysis: wind uses wind_speed_ms thresholds, demand uses load-based or time-based regimes
4. Update file glob for demand data

For demand regime analysis, use **time-of-day regimes** instead of wind speed:
- "off_peak": hours 0-7
- "shoulder": hours 8-17
- "peak": hours 18-23

Add a domain-aware regime analysis dispatch:
```python
if domain == "wind" and "wind_speed_ms" in test_h.columns:
    regimes = regime_analysis(...)
elif domain == "demand":
    regimes = demand_regime_analysis(...)  # New function or inline
```

**PATTERN**: Minimal changes. Regime analysis is the main domain-specific part.
**GOTCHA**: Model loading from MLflow is domain-agnostic (just loads XGBoost models by horizon). No changes needed there.
**VALIDATE**: `uv run pyright scripts/evaluate.py`

---

### Task 19: Run full validation suite

**VALIDATE**: Run all validation in sequence:
```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
uv run pytest tests/ -v
```

All must pass with zero errors.

---

## TESTING STRATEGY

### Unit Tests

- **Schema**: Column presence, type validation, empty frame factory
- **Parser**: CSV parsing, column renaming, type conversion, timestamp handling, join correctness
- **QC**: Each rule in isolation with synthetic data, flag never downgraded, gap fill limits
- **Features**: Lag correctness, rolling no-look-ahead, cyclic ranges, HDD/CDD logic

### Integration Tests

- **Full pipeline test**: Synthetic CSV → parse → QC → features → verify output schema
- **Multi-domain registry**: Both wind and demand feature sets coexist, no name collisions

### Edge Cases

- Empty weather file (no cities match)
- All-null load column
- DST transition hours (duplicate/missing hours)
- Holiday on weekend (still flagged)
- Single-row DataFrame (no lags possible)
- Temperature exactly at HDD/CDD threshold (18°C, 24°C)

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

### Level 3: Unit Tests
```bash
uv run pytest tests/ -v
```

### Level 4: Manual Pipeline Test (after dataset download)
```bash
# Download dataset
kaggle datasets download -d nicholasjhana/energy-consumption-generation-prices-and-weather --unzip -p data/raw/spain/

# Run demand pipeline
uv run python scripts/ingest_spain_demand.py
uv run python scripts/build_features.py --domain demand
uv run python scripts/train.py --domain demand --dataset spain_demand
uv run python scripts/evaluate.py --domain demand

# Verify MLflow
mlflow ui  # Should show both wind and demand experiments
```

---

## ACCEPTANCE CRITERIA

- [ ] Demand schema defined with 11 typed columns
- [ ] Spain parser handles both CSVs (energy + weather), produces schema-compliant output
- [ ] Weather aggregation: 5 cities → 1 row per timestamp (mean)
- [ ] Known data quality issues handled (Kelvin→Celsius, Barcelona space, outliers)
- [ ] Demand QC: load outliers, weather outliers, holidays, DST transitions, gap fill
- [ ] 3 demand feature sets registered (baseline/enriched/full)
- [ ] Demand features: load lags, calendar cyclic, HDD/CDD, rolling stats, price lags
- [ ] Ingestion script works: `ingest_spain_demand.py` → Parquet
- [ ] `build_features.py --domain demand` works
- [ ] `train.py --domain demand` works with same MLflow tracking
- [ ] `evaluate.py --domain demand` works with demand-appropriate regimes
- [ ] All existing wind tests still pass (no regressions)
- [ ] ruff, pyright, pytest all pass with zero errors
- [ ] Adding demand required ZERO changes to core model/evaluation/tracking code

---

## COMPLETION CHECKLIST

- [ ] All 19 tasks completed in order
- [ ] Each task validation passed
- [ ] All validation commands pass:
  - [ ] Level 1: ruff check + format
  - [ ] Level 2: pyright
  - [ ] Level 3: pytest (all tests including new demand tests)
  - [ ] Level 4: Manual pipeline test (if dataset available)
- [ ] No regressions in existing wind functionality
- [ ] All acceptance criteria met

---

## NOTES

### Key Architectural Decisions

1. **Separate schema per domain** (demand_schema.py) rather than one universal schema — keeps domain semantics clear, avoids 30-column monster schema with mostly-null columns.

2. **Separate QC per domain** (demand_qc.py) rather than parameterizing the wind QC — demand QC rules are fundamentally different (no curtailment, no frozen sensors, but holidays and DST). Shared pattern, different rules.

3. **Single DATASETS registry** with union type rather than separate registries — simpler, the scripts know what type to expect.

4. **DOMAIN_CONFIG dict in scripts** for target/group/lag column mapping — lightweight, avoids over-abstraction. Can be promoted to config.py later if more domains are added.

5. **Horizon = steps** (not time) — wind step = 10 min, demand step = 1 hour. The training loop is identical; only the interpretation changes. This matches the PRD goal of "same train/evaluate scripts."

### Risk: Hourly vs 10-min resolution

Wind data is 10-min, demand is hourly. The lag/rolling feature code uses row-based operations (shift, rolling), so it works regardless of resolution. But forecast horizons have different time meanings: horizon=6 means 1 hour for wind but 6 hours for demand. Document this clearly in experiment names.

### Estimated Confidence: 8/10

High confidence because:
- All patterns are established and well-understood
- No new dependencies needed
- Dataset format is well-documented
- Tests are comprehensive

Remaining risks:
- Timestamp parsing with timezone offsets may need iteration
- Weather aggregation edge cases (missing cities for some hours)
- Holiday list completeness
