# Feature: Phase 3 — Solar Domain

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Add solar PV generation forecasting as the third domain in EnerCast, proving the framework handles 3 fundamentally different energy domains (wind, demand, solar) with zero changes to core pipeline code.

Uses NREL PVDAQ System 4 (x-Si, Golden CO, 2007–2025, 1-min resolution) — the simplest active system with AC power + POA irradiance + temperature. GHI is not available in PVDAQ and must be sourced from Open-Meteo (already integrated).

## User Story

As a WN evaluator,
I want to see 3 domains (wind, demand, solar) running through the same framework,
So that I believe standardization across all WN energy domains is achievable.

## Problem Statement

Wind and demand are proven. Solar is structurally different (irradiance-driven, diurnal pattern, no output at night). Adding it with zero core changes completes the trifecta.

## Solution Statement

Mirror the demand domain implementation pattern: solar schema → PVDAQ parser (CSV from S3) → solar QC → solar features → register in feature registry → update scripts with `--domain solar`. Same train.py, evaluate.py, MLflow tracking.

## Feature Metadata

**Feature Type**: New Capability (third domain)
**Estimated Complexity**: Medium
**Primary Systems Affected**: data/, features/, config.py, scripts/
**Dependencies**: No new external libraries (polars reads CSV/parquet, requests for HTTP)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — READ BEFORE IMPLEMENTING

**Schema pattern (mirror these):**
- `src/windcast/data/schema.py` — Wind SCADA schema (15 cols), QC constants QC_OK/QC_SUSPECT/QC_BAD
- `src/windcast/data/demand_schema.py` — Demand schema (11 cols), validate + empty frame helpers

**Parser pattern (mirror spain_demand.py):**
- `src/windcast/data/spain_demand.py` — Parser reads CSV, maps columns, casts to schema, sorts by timestamp
- `src/windcast/data/kelmarsh.py` — Wind parser, more complex (ZIP), same output pattern

**QC pattern (mirror demand_qc.py):**
- `src/windcast/data/demand_qc.py` — QC pipeline: rule functions that update qc_flag with max_horizontal, summary stats
- `src/windcast/data/qc.py` — Wind QC (9 rules), same pattern

**Feature pattern (mirror demand.py):**
- `src/windcast/features/demand.py` — `build_demand_features()`: QC filter → sort → lags → calendar → domain-specific
- `src/windcast/features/wind.py` — `build_wind_features()`: same structure
- `src/windcast/features/registry.py` — FeatureSet dataclass, FEATURE_REGISTRY dict, get/list functions

**Config pattern:**
- `src/windcast/config.py` — `DemandDatasetConfig` model, `SPAIN_DEMAND` instance, `DATASETS` dict, `DemandQCConfig`, `WindCastSettings`

**Script pattern (update these):**
- `scripts/ingest_spain_demand.py` — Parse → validate → QC → Parquet. Mirror for solar.
- `scripts/build_features.py` (lines 23-24, 66-68, 70-75, 90-93) — `--domain` choices, default feature set, parquet pattern, feature builder dispatch
- `scripts/train.py` (lines 23-26, 77-80, 126-136) — `DOMAIN_CONFIG` dict, `--domain` choices, default feature set, parquet path
- `scripts/evaluate.py` (lines 24-27, 126-131, 166-168, 192-194, 264-268) — Same: DOMAIN_CONFIG, --domain, parquet path, regime analysis dispatch

**Package exports (update these):**
- `src/windcast/data/__init__.py` — Exports SCADA_SCHEMA + DEMAND_SCHEMA + validators
- `src/windcast/features/__init__.py` — Exports build_wind_features + build_demand_features + registry

**Test patterns (mirror these):**
- `tests/data/test_demand_schema.py` — 7 tests: column count, empty frame, missing col, wrong type, strict mode, order, signal cols
- `tests/data/test_spain_demand.py` — Synthetic CSV bytes, test parse → schema compliant, sorted, file-not-found
- `tests/data/test_demand_qc.py` — `_make_demand_df()` helper, test each QC rule, pipeline, summary
- `tests/features/test_demand.py` — `_make_demand_df()` for features, test lags, rolling, calendar, temperature, build function
- `tests/features/test_registry.py` (lines 54-79) — Tests demand feature sets exist, extend correctly

### New Files to Create

- `src/windcast/data/solar_schema.py` — Solar schema (10 columns)
- `src/windcast/data/pvdaq.py` — PVDAQ System 4 parser (CSV from S3 or local)
- `src/windcast/data/solar_qc.py` — Solar QC pipeline
- `src/windcast/features/solar.py` — Solar feature engineering
- `scripts/ingest_pvdaq.py` — Ingestion CLI script
- `tests/data/test_solar_schema.py` — Schema tests
- `tests/data/test_pvdaq.py` — Parser tests
- `tests/data/test_solar_qc.py` — QC tests
- `tests/features/test_solar.py` — Feature tests

### Relevant Documentation

**PVDAQ Dataset (S3 access, no API key needed):**
- Data lives at: `https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata/system_id%3D{id}/year%3D{Y}/month%3D{M}/day%3D{D}/system_{id}__date_{YYYY}_{MM}_{DD}.csv`
- System 4 columns: `measured_on`, `ac_power__315` (Watts), `poa_irradiance__313` (W/m²), `ambient_temp__320` (°C), `module_temp_1__321` (°C)
- `measured_on` is **local time (UTC-7)** — must convert to UTC
- 1-minute resolution — aggregate to 15 min for consistency
- Power is in **Watts** (not kW) — divide by 1000
- Negative night power values → clip to 0
- **No GHI column** — fetch from Open-Meteo (already integrated)
- **No wind_speed** in system 4 — leave as null or fetch from Open-Meteo

**Alternative: Local CSV download**
- For the demo, download a subset (e.g., 1 year) as local CSV rather than hitting S3 for every run
- Parser should support both local path and S3 download

### Patterns to Follow

**Schema naming**: `SOLAR_SCHEMA`, `SOLAR_COLUMNS`, `SOLAR_SIGNAL_COLUMNS`, `validate_solar_schema()`, `empty_solar_frame()`

**Parser function signature**: `parse_pvdaq(data_path: Path) -> pl.DataFrame` — single entry point, returns schema-compliant DataFrame

**QC function signature**: `run_solar_qc_pipeline(df: pl.DataFrame, qc_config: SolarQCConfig | None = None) -> pl.DataFrame`

**Feature function signature**: `build_solar_features(df: pl.DataFrame, feature_set: str = "solar_baseline") -> pl.DataFrame`

**QC flag pattern**: Use `pl.max_horizontal(pl.col("qc_flag"), pl.lit(QC_SUSPECT).cast(pl.UInt8))` for escalation-only flags

**Test helper pattern**: `_make_solar_df(n_rows: int = 100, **overrides) -> pl.DataFrame` — creates synthetic data

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — Schema + Config

Add the solar schema (10 columns per PRD) and configuration models. Mirror demand_schema.py exactly.

### Phase 2: Parser — PVDAQ

Parser reads local CSV files (pre-downloaded from S3). Maps system 4 column names to canonical schema. Aggregates 1-min → 15-min. Converts local time → UTC.

### Phase 3: Solar QC

QC rules specific to solar: nighttime power, irradiance bounds, temperature bounds, power-irradiance consistency, frozen sensor, gap fill.

### Phase 4: Features + Registry

Solar feature sets: baseline (irradiance + power lags + hour cyclic), enriched (+ clearsky ratio + temperature + rolling), full (+ weather from Open-Meteo + cloud cover). Register in FEATURE_REGISTRY.

### Phase 5: Script Integration

Update all 4 scripts (ingest, build_features, train, evaluate) to support `--domain solar`.

### Phase 6: Package Exports + Tests

Update __init__.py files. Write tests for all new modules.

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/windcast/data/solar_schema.py`

Mirror `src/windcast/data/demand_schema.py` structure.

**Schema (10 columns per PRD):**
```python
SOLAR_SCHEMA = {
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "dataset_id": pl.String,
    "system_id": pl.String,
    "power_kw": pl.Float64,           # AC power output
    "ghi_wm2": pl.Float64,            # Global Horizontal Irradiance (from Open-Meteo, nullable)
    "poa_wm2": pl.Float64,            # Plane of Array irradiance
    "ambient_temp_c": pl.Float64,
    "module_temp_c": pl.Float64,
    "wind_speed_ms": pl.Float64,      # From Open-Meteo (nullable)
    "qc_flag": pl.UInt8,
}
```

**Implement:**
- `SOLAR_SCHEMA` dict
- `SOLAR_COLUMNS` list
- `SOLAR_SIGNAL_COLUMNS = ["power_kw", "ghi_wm2", "poa_wm2", "ambient_temp_c", "module_temp_c", "wind_speed_ms"]`
- `validate_solar_schema(df, *, strict=False) -> list[str]` — mirror `validate_demand_schema`
- `empty_solar_frame() -> pl.DataFrame`
- Import `QC_OK, QC_SUSPECT, QC_BAD` from `windcast.data.schema`
- Export all in `__all__`

**VALIDATE**: `uv run pyright src/windcast/data/solar_schema.py`

### Task 2: CREATE `tests/data/test_solar_schema.py`

Mirror `tests/data/test_demand_schema.py` exactly (7 tests):

1. `test_solar_schema_has_expected_columns` — 10 columns, timestamp_utc, power_kw, qc_flag present
2. `test_empty_solar_frame_matches_schema` — validate + shape (0, 10)
3. `test_validate_solar_schema_detects_missing_column`
4. `test_validate_solar_schema_detects_wrong_type` — cast power_kw to Int32
5. `test_validate_solar_schema_strict_mode` — extra column detected
6. `test_solar_columns_ordered` — keys match SOLAR_COLUMNS
7. `test_solar_signal_columns` — 6 signal columns

**VALIDATE**: `uv run pytest tests/data/test_solar_schema.py -v`

### Task 3: UPDATE `src/windcast/config.py`

**Add SolarDatasetConfig and SolarQCConfig:**

```python
class SolarDatasetConfig(BaseModel):
    """Per-dataset metadata for solar PV forecasting."""
    dataset_id: str
    system_id: str
    capacity_kw: float
    tilt_deg: float
    azimuth_deg: float
    latitude: float
    longitude: float
    timezone: str

PVDAQ_SYSTEM4 = SolarDatasetConfig(
    dataset_id="pvdaq_system4",
    system_id="4",
    capacity_kw=2.2,          # ~2.2 kW x-Si system
    tilt_deg=40.0,            # Golden CO typical
    azimuth_deg=180.0,        # South-facing
    latitude=39.7407,
    longitude=-105.1686,
    timezone="America/Denver",
)

class SolarQCConfig(BaseModel):
    """QC thresholds for solar data."""
    max_power_kw: float = 5.0           # System capacity + margin
    max_irradiance_wm2: float = 1500.0  # Physical max GHI ~1361 + diffuse
    min_irradiance_wm2: float = -10.0   # Sensor noise threshold
    max_temperature_c: float = 60.0
    min_temperature_c: float = -30.0
    max_gap_fill_intervals: int = 4     # 4 × 15 min = 1 hour
    nighttime_power_threshold_kw: float = 0.01
```

**Update:**
- Add `"pvdaq_system4": PVDAQ_SYSTEM4` to `DATASETS` dict
- Update `DATASETS` type hint: `dict[str, DatasetConfig | DemandDatasetConfig | SolarDatasetConfig]`
- Add `solar_qc: SolarQCConfig = Field(default_factory=SolarQCConfig)` to `WindCastSettings`

**VALIDATE**: `uv run pytest tests/test_config.py -v`

### Task 4: UPDATE `tests/test_config.py`

Add tests for solar config:
- `test_pvdaq_system4_in_datasets`
- `test_solar_qc_defaults`
- `test_solar_dataset_config_fields`

**VALIDATE**: `uv run pytest tests/test_config.py -v`

### Task 5: CREATE `src/windcast/data/pvdaq.py`

PVDAQ System 4 parser. Reads **local CSV files** (pre-downloaded). The parser does NOT download from S3 — that's the ingestion script's job.

**Key implementation details:**

```python
DATASET_ID = "pvdaq_system4"
SYSTEM_ID = "4"

# System 4 column mapping
SIGNAL_MAP = {
    "ac_power__315": "power_watts",       # Watts → will convert to kW
    "poa_irradiance__313": "poa_wm2",
    "ambient_temp__320": "ambient_temp_c",
    "module_temp_1__321": "module_temp_c",
}
```

**Function: `parse_pvdaq(data_dir: Path, year: int | None = None) -> pl.DataFrame`**
1. Glob for CSV files in data_dir (pattern: `system_4__date_*.csv`)
2. If year provided, filter to that year's files
3. Read each CSV with polars, select + rename columns per SIGNAL_MAP
4. Parse `measured_on` column as local time (America/Denver) → convert to UTC
5. Convert power from Watts to kW (`/ 1000.0`), clip negative to 0
6. Aggregate from 1-min to 15-min: group by 15-min floor, take mean of signals
7. Add dataset_id, system_id columns
8. Add null columns for ghi_wm2, wind_speed_ms (not available in system 4)
9. Add default qc_flag = 0
10. Cast to SOLAR_SCHEMA, reorder, sort by timestamp_utc

**Helper: `_read_pvdaq_csv(path: Path) -> pl.DataFrame`** — reads single CSV, handles column mapping

**GOTCHA**: `measured_on` format is `"2007-08-26 00:00:00"` in local time. Use `str.to_datetime().dt.replace_time_zone("America/Denver").dt.convert_time_zone("UTC")`.

**GOTCHA**: Some CSV files may have different column sets. Use flexible column selection (skip missing).

**VALIDATE**: `uv run pyright src/windcast/data/pvdaq.py`

### Task 6: CREATE `tests/data/test_pvdaq.py`

Mirror `tests/data/test_spain_demand.py` pattern.

**Synthetic CSV helper:**
```python
def _make_pvdaq_csv_bytes(n_rows: int = 96) -> bytes:
    """Create synthetic PVDAQ System 4 CSV for testing (15-min aggregated)."""
    # Columns: measured_on, ac_power__315, poa_irradiance__313, ambient_temp__320, module_temp_1__321
    # 96 rows = 1 day at 15-min
```

**Tests:**
1. `test_parses_timestamps_to_utc` — verify UTC conversion from local time
2. `test_power_converted_to_kw` — Watts / 1000
3. `test_negative_power_clipped` — negative → 0
4. `test_returns_schema_compliant_frame` — validate_solar_schema passes
5. `test_sorts_by_timestamp`
6. `test_aggregation_reduces_rows` — if testing 1-min → 15-min
7. `test_missing_file_raises` — FileNotFoundError

**VALIDATE**: `uv run pytest tests/data/test_pvdaq.py -v`

### Task 7: CREATE `src/windcast/data/solar_qc.py`

Mirror `src/windcast/data/demand_qc.py` structure.

**QC rules for solar:**

1. `_flag_nighttime_power(df, threshold_kw)` → QC_SUSPECT if power > threshold when GHI ≤ 0 or POA ≤ 0
   - Use solar elevation proxy: if hour is between sunset+1 and sunrise-1 (simple: hour < 6 or hour > 20 UTC for Golden CO in summer; more robust: POA ≤ 0)
   - Simplest approach: flag rows where `poa_wm2 <= 0` AND `power_kw > threshold`
2. `_flag_power_outliers(df, max_power_kw)` → QC_SUSPECT if power > max capacity
3. `_flag_irradiance_outliers(df, min_irr, max_irr)` → QC_SUSPECT if POA out of bounds
4. `_flag_temperature_outliers(df, min_temp, max_temp)` → QC_SUSPECT if ambient_temp out of bounds
5. `_flag_power_irradiance_inconsistency(df)` → QC_SUSPECT if high irradiance but zero power (possible inverter fault)
6. `_fill_small_gaps(df, max_intervals)` → forward-fill signal columns

**Pipeline function**: `run_solar_qc_pipeline(df, qc_config=None) -> pl.DataFrame`

**Summary function**: `solar_qc_summary(df) -> dict[str, int | float]` — total, qc_ok/suspect/bad counts + pcts

**VALIDATE**: `uv run pyright src/windcast/data/solar_qc.py`

### Task 8: CREATE `tests/data/test_solar_qc.py`

Mirror `tests/data/test_demand_qc.py` structure.

**Helper**: `_make_solar_df(n_rows, **overrides)` — creates canonical solar DataFrame with realistic defaults (daytime power ~1.5 kW, POA ~600 W/m², temp ~20°C)

**Tests (one class per rule):**
- `TestFlagNighttimePower`: high power with zero irradiance → SUSPECT
- `TestFlagPowerOutliers`: power > max → SUSPECT
- `TestFlagIrradianceOutliers`: POA > max or < min → SUSPECT
- `TestFlagTemperatureOutliers`: extreme temp → SUSPECT
- `TestFlagPowerIrradianceInconsistency`: high irradiance + zero power → SUSPECT
- `TestFillSmallGaps`: forward-fill works, large gaps preserved
- `TestRunSolarQcPipeline`: full pipeline runs, worst flag wins
- `TestSolarQcSummary`: counts and empty

**VALIDATE**: `uv run pytest tests/data/test_solar_qc.py -v`

### Task 9: UPDATE `src/windcast/data/__init__.py`

Add solar exports:
```python
from windcast.data.solar_schema import SOLAR_SCHEMA, validate_solar_schema
```

Update `__all__`.

**VALIDATE**: `uv run pyright src/windcast/data/__init__.py`

### Task 10: CREATE `src/windcast/features/solar.py`

Mirror `src/windcast/features/demand.py` structure.

**Function: `build_solar_features(df, feature_set="solar_baseline") -> pl.DataFrame`**

1. Filter to QC_OK
2. Sort by system_id, timestamp_utc
3. **Baseline**: power lags (1, 2, 4, 8, 96 = 15min steps → 15m, 30m, 1h, 2h, 24h), POA irradiance as-is, hour cyclic (sin/cos)
4. **Enriched** (extends baseline): + clearsky ratio (poa / theoretical max, capped at 1.5), + ambient_temp, module_temp, rolling power stats (4, 16, 96 = 1h, 4h, 24h)
5. **Full** (extends enriched): + GHI (from Open-Meteo if available), + wind_speed, + month cyclic, + dow cyclic

**Helper functions (private, same pattern as demand.py):**
- `_add_lag_features(df, col, lags)` — shift over system_id
- `_add_rolling_features(df, col, windows)` — shift(1) + rolling over system_id
- `_add_cyclic_hour(df)` — hour sin/cos
- `_add_clearsky_ratio(df)` — POA / max_expected_irradiance, capped
- `_add_cyclic_calendar(df)` — month + dow sin/cos

**VALIDATE**: `uv run pyright src/windcast/features/solar.py`

### Task 11: UPDATE `src/windcast/features/registry.py`

Add 3 solar feature sets:

```python
SOLAR_BASELINE = FeatureSet(
    name="solar_baseline",
    columns=[
        "poa_wm2",
        "power_kw_lag1",
        "power_kw_lag2",
        "power_kw_lag4",
        "power_kw_lag8",
        "power_kw_lag96",
        "hour_sin",
        "hour_cos",
    ],
    description="Solar baseline: POA irradiance + power lags + hour cyclic",
)

SOLAR_ENRICHED = FeatureSet(
    name="solar_enriched",
    columns=[
        *SOLAR_BASELINE.columns,
        "clearsky_ratio",
        "ambient_temp_c",
        "module_temp_c",
        "power_kw_roll_mean_4",
        "power_kw_roll_mean_16",
        "power_kw_roll_mean_96",
        "power_kw_roll_std_4",
    ],
    description="Solar enriched: + clearsky ratio, temperature, rolling power stats",
)

SOLAR_FULL = FeatureSet(
    name="solar_full",
    columns=[
        *SOLAR_ENRICHED.columns,
        "ghi_wm2",
        "wind_speed_ms",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
    ],
    description="Solar full: + GHI, wind, full cyclic encoding",
)
```

Add all 3 to `FEATURE_REGISTRY`.

**VALIDATE**: `uv run pytest tests/features/test_registry.py -v`

### Task 12: UPDATE `src/windcast/features/__init__.py`

Add `build_solar_features` import and export.

**VALIDATE**: `uv run pyright src/windcast/features/__init__.py`

### Task 13: CREATE `tests/features/test_solar.py`

Mirror `tests/features/test_demand.py` structure.

**Helper**: `_make_solar_df(n_rows=200)` — creates realistic 15-min solar data (sinusoidal power peaking at noon, POA tracking power, temp cycling)

**Tests:**
- `TestLagFeatures`: lag values shifted correctly, per system_id
- `TestRollingFeatures`: no look-ahead, correct window columns
- `TestClearskyRatio`: ratio computed, capped
- `TestCalendarFeatures`: hour cyclic in [-1, 1]
- `TestBuildSolarFeatures`: QC filter, baseline columns, enriched extends, full has GHI/wind

**VALIDATE**: `uv run pytest tests/features/test_solar.py -v`

### Task 14: UPDATE `tests/features/test_registry.py`

Add `TestSolarFeatureSets` class (mirror `TestDemandFeatureSets`):
- `test_solar_baseline_exists`
- `test_solar_enriched_extends_baseline`
- `test_solar_full_extends_enriched`
- `test_list_includes_solar`

**VALIDATE**: `uv run pytest tests/features/test_registry.py -v`

### Task 15: CREATE `scripts/ingest_pvdaq.py`

Mirror `scripts/ingest_spain_demand.py`.

```python
"""Ingest PVDAQ solar data: parse CSVs → QC → save Parquet.

Usage:
    uv run python scripts/ingest_pvdaq.py --data-dir data/raw/pvdaq/
    uv run python scripts/ingest_pvdaq.py --data-dir data/raw/pvdaq/ --year 2020
"""
```

**Flow:**
1. Parse args: `--data-dir`, `--year`, `--output-dir`
2. Call `parse_pvdaq(data_dir, year)`
3. `validate_solar_schema(df)`
4. `run_solar_qc_pipeline(df, settings.solar_qc)`
5. `solar_qc_summary(df)` → log
6. Write to `output_dir / "pvdaq_system4.parquet"`

**VALIDATE**: `uv run pyright scripts/ingest_pvdaq.py`

### Task 16: UPDATE `scripts/build_features.py`

**Changes:**
- Line 5 (imports): add `build_solar_features`
- `--domain` choices: add `"solar"` → `choices=["wind", "demand", "solar"]`
- Default feature set (line 66): add `elif args.domain == "solar": feature_set = "solar_baseline"`
- Parquet pattern (line 71): add `elif args.domain == "solar": pattern = "pvdaq_system4.parquet"`
- Feature builder dispatch (line 90): add `elif args.domain == "solar": df = build_solar_features(df, feature_set=feature_set)`
- Output name (line 101): add `elif args.domain == "solar": output_path = output_dir / "pvdaq_system4_features.parquet"`

**VALIDATE**: `uv run pyright scripts/build_features.py`

### Task 17: UPDATE `scripts/train.py`

**Changes:**
- `DOMAIN_CONFIG` (line 23): add `"solar": {"target": "power_kw", "group": "system_id", "lag1": "power_kw_lag1"}`
- `--domain` choices: add `"solar"` → `choices=["wind", "demand", "solar"]`
- Default feature set (line 126): add solar case
- Default dataset (line 129): add solar case
- Parquet path (line 134): add `elif domain == "solar": parquet_path = features_dir / "pvdaq_system4_features.parquet"`

**VALIDATE**: `uv run pyright scripts/train.py`

### Task 18: UPDATE `scripts/evaluate.py`

**Changes:**
- `DOMAIN_CONFIG` (line 24): add `"solar": {"target": "power_kw", "group": "system_id", "lag1": "power_kw_lag1"}`
- `--domain` choices: add `"solar"`
- Default feature set + dataset (lines 166-168): add solar case
- Parquet path (line 192): add solar case
- Regime analysis dispatch (line 264): add solar regime (e.g., by irradiance level: low/medium/high POA)

**Add `_solar_regime_analysis()` function** — regimes by irradiance: low (<200 W/m²), medium (200-600), high (>600)

**VALIDATE**: `uv run pyright scripts/evaluate.py`

### Task 19: Run full validation suite

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

Each new module gets its own test file mirroring the demand pattern:
- **solar_schema**: 7 tests (column count, validation, strict mode)
- **pvdaq parser**: 7 tests (UTC conversion, power conversion, schema compliance, sorting)
- **solar_qc**: ~15 tests (one class per rule + pipeline + summary)
- **solar features**: ~12 tests (lags, rolling, clearsky, calendar, build function)
- **registry updates**: 4 tests (solar sets exist, extend correctly)

Total new tests: ~45

### Test Data

All tests use synthetic data created in-memory — no external data dependencies. Helper functions create realistic solar patterns (sinusoidal power, daytime-only irradiance).

### Edge Cases

- Night hours: power = 0, irradiance = 0 → features should handle gracefully
- Winter vs summer: different day lengths affect lag patterns
- Cloudy days: low irradiance but nonzero → clearsky ratio should be < 1
- Missing signals (GHI, wind_speed): null columns should not crash feature engineering

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

### Level 4: Solar-Specific Tests
```bash
uv run pytest tests/data/test_solar_schema.py tests/data/test_pvdaq.py tests/data/test_solar_qc.py tests/features/test_solar.py -v
```

---

## ACCEPTANCE CRITERIA

- [ ] Solar schema defines 10 columns matching PRD spec
- [ ] PVDAQ parser reads CSV, converts to canonical schema
- [ ] Solar QC pipeline flags nighttime power, outliers, inconsistencies
- [ ] 3 solar feature sets registered (baseline/enriched/full)
- [ ] `build_features.py --domain solar` works
- [ ] `train.py --domain solar` works (same script, zero core changes)
- [ ] `evaluate.py --domain solar` works with regime analysis
- [ ] All new code passes ruff, pyright, pytest
- [ ] No changes to core pipeline code (models/, tracking/)
- [ ] Feature sets follow baseline → enriched → full extension pattern
- [ ] Total test count increases by ~45
- [ ] Zero regressions in existing wind + demand tests

---

## COMPLETION CHECKLIST

- [ ] All 19 tasks completed in order
- [ ] Each task validation passed
- [ ] Full validation suite passes:
  - [ ] `uv run ruff check src/ tests/ scripts/` — 0 errors
  - [ ] `uv run ruff format --check src/ tests/ scripts/` — 0 errors
  - [ ] `uv run pyright src/` — 0 errors
  - [ ] `uv run pytest tests/ -v` — all pass
- [ ] No changes to: `models/`, `tracking/`, core pipeline
- [ ] All acceptance criteria met

---

## NOTES

### Dataset Choice: System 4 over System 2

System 2 is a residential system in Lakewood with QA=fail status. System 4 (NREL x-Si, Golden CO) is active through 2025, has clean data, and the right signals. PRD says "PVDAQ System 2" but this was based on outdated information — system 4 is the correct choice.

### No GHI in PVDAQ

None of the PVDAQ systems include GHI sensors. For the demo, we:
1. Leave `ghi_wm2` as null in the parser output
2. Solar features work without it (baseline + enriched use POA only)
3. The "full" feature set includes GHI, which would come from Open-Meteo enrichment (deferred — not blocking for demo)

### Resolution: 1-min → 15-min

PVDAQ is 1-min. We aggregate to 15-min in the parser to:
- Match typical SCADA resolution
- Reduce data volume (~15x smaller)
- Smooth sensor noise

### Confidence Score: 8/10

High confidence because:
- Pattern is proven (wind + demand already work)
- No new dependencies
- Clear column mapping
- Tests follow established patterns

Risk: PVDAQ CSV format quirks that synthetic test data doesn't capture. Mitigated by testing against actual downloaded CSVs during integration.
