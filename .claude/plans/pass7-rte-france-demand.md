# Feature: Pass 7 — RTE France Demand Domain (éCO2mix local files)

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Replace the Spain Kaggle dataset with **RTE éCO2mix France demand** as the primary demand-domain demo for the WeatherNews presentation. We use **locally-downloaded annual definitive files** (`data/ecomix-rte/eCO2mix_RTE_Annuel-Definitif_<YYYY>.zip`, 11 years 2014-2024) — no API, no OAuth, no credentials.

The files are TSV in Latin-1 encoding despite the `.xls` extension. Polars reads them directly after decoding.

The demand feature sets are also refactored so the "full" set uses forward-looking NWP from Open-Meteo (`nwp_temperature_2m_h{k}`, etc.) instead of observed weather. This gives demand the same **"rearview → windshield"** narrative as wind.

**Scope of this pass:**
1. New parser `src/windcast/data/rte_france.py` — reads the local éCO2mix annual files, outputs canonical `DEMAND_SCHEMA` DataFrame with `tso_forecast_mw` populated from RTE's own D-1 forecast
2. Extend `DEMAND_SCHEMA` with `tso_forecast_mw` (nullable — Spain parser still works)
3. Add French public holidays and dataset-based dispatch in `demand_qc.py`
4. Refactor `DEMAND_ENRICHED` / `DEMAND_FULL` so `demand_full` uses `nwp_*` columns (dataset-agnostic)
5. Backfill Paris NWP into `data/weather.db` via existing `get_weather("rte_france", ...)` API
6. Run ingest + 3 feature sets + TSO benchmark → real MLflow metrics for the presentation
7. Keep Spain code in place as a second reference implementation (do NOT delete)

## User Story

As a WeatherNews evaluator (Yoel / Michel / Craig),
I want to see the same EnerCast pipeline running on 11 years of real French national load with RTE's own day-ahead forecast as a benchmark,
So that I believe the framework is directly applicable to WN's Gas & Power Demand domain, on the geography that matters most to them.

## Problem Statement

The Spain Kaggle dataset is (1) frozen in 2018, (2) uses observed weather that would leak in real forecasting, and (3) is Spanish while the jury is French and the target market is France. We need a France-based, live-sourceable demand dataset that tells the "rearview → windshield" story cleanly.

## Solution Statement

Use the **locally-downloaded éCO2mix annual definitive files** at `data/ecomix-rte/`. No network calls during ingest. Parse the Latin-1 TSV directly via Polars, normalize to UTC, aggregate load from 30-min to hourly, map to canonical `DEMAND_SCHEMA`. Store `Prévision J-1` as `tso_forecast_mw` to enable the TSO benchmark slide.

Then:
- **3 incremental feature sets**: `demand_baseline` (lags + calendar) → `demand_enriched` (+ rolling stats + French holidays) → `demand_full` (+ NWP at horizon + HDD/CDD from NWP)
- **TSO benchmark as a 4th MLflow run**: compute MAE/RMSE of `tso_forecast_mw` vs `load_mw` on the val set
- **Side-by-side in MLflow UI**: Kelmarsh wind + RTE France demand

## Feature Metadata

**Feature Type**: New Capability (new dataset integration + feature set refactor)
**Estimated Complexity**: Medium (~3-4h — simpler than the API approach because no auth/network)
**Primary Systems Affected**:
- `src/windcast/data/` — new RTE parser, extend demand schema, French holidays in QC
- `src/windcast/features/` — refactor demand feature sets to use NWP
- `src/windcast/weather/registry.py` — add `RTE_FRANCE_WEATHER` config
- `src/windcast/config.py` — add `RTE_FRANCE` dataset config
- `scripts/` — new `ingest_rte_france.py`, `log_tso_baseline.py`
- `tests/data/`, `tests/features/` — new tests + updates

**Dependencies**: None new. Polars reads the TSV files natively. No `httpx`, no `openpyxl`, no credentials.

**Preconditions**: 11 ZIP files already exist at `data/ecomix-rte/eCO2mix_RTE_Annuel-Definitif_<YYYY>.zip` for 2014-2024 (verified). `data/weather.db` already exists with Kelmarsh coverage; Paris NWP will be backfilled in Task 16.

---

## CONTEXT REFERENCES

### Relevant Codebase Files — YOU MUST READ THESE BEFORE IMPLEMENTING!

**Windcast parser pattern (mirror these structures):**
- `src/windcast/data/kelmarsh.py` (full) — **Parser pattern**: public `parse_kelmarsh()`, private helper functions, canonical schema conformance loop at the end. Mirror for `parse_rte_france()`.
- `src/windcast/data/spain_demand.py` (full, 159 lines) — Demand parser pattern: identifier injection at lines 50-53, default flag columns at 56-60, schema enforcement loop at 62-68. Mirror exactly.
- `src/windcast/data/demand_schema.py` (full) — `DEMAND_SCHEMA` dict, `DEMAND_COLUMNS`, `DEMAND_SIGNAL_COLUMNS`, `validate_demand_schema()`. **Extend with `tso_forecast_mw: pl.Float64` nullable.**
- `src/windcast/data/demand_qc.py` (full, 168 lines) — QC pipeline. `SPAIN_HOLIDAYS` at lines 15-31, `_detect_holidays` at 115-120. **Add `FRANCE_HOLIDAYS` + dataset_id dispatcher.**
- `src/windcast/features/demand.py` (full, 145 lines) — `build_demand_features`, `_add_temperature_features` at 127-136. **Refactor so HDD/CDD use NWP column when present, fallback to `temperature_c`.**
- `src/windcast/features/registry.py` (lines 61-104) — `DEMAND_BASELINE`, `DEMAND_ENRICHED`, `DEMAND_FULL`. **Refactor so `DEMAND_FULL` uses `nwp_temperature_2m`, `nwp_wind_speed_10m`, `nwp_relative_humidity_2m` (no `_h{k}` suffix — added by `_resolve_horizon_features` in train.py).**
- `src/windcast/features/wind.py` (full) — Reference for how NWP columns are INJECTED by `build_features.py` (not inside `build_wind_features` itself). Demand must follow the same split.
- `src/windcast/features/weather.py` (lines 20-60) — `join_nwp_horizon_features()` — already resolution-agnostic, works for hourly demand unchanged.

**Weather & storage (already in place):**
- `src/windcast/weather/__init__.py` (lines 57-94 `get_weather`, 97-135 `_fetch_missing`) — Fetch-cache-serve. Call with `config_name="rte_france"` in Task 16.
- `src/windcast/weather/registry.py` (lines 17-30 `KELMARSH_WEATHER`, 32-42 `SPAIN_WEATHER`, 44-55 `PVDAQ_WEATHER`) — Add `RTE_FRANCE_WEATHER`.
- `src/windcast/weather/storage.py` (full) — SQLite cache. No changes.

**Config:**
- `src/windcast/config.py` (lines 53-71 `DemandDatasetConfig` + `SPAIN_DEMAND`, 98-104 `DATASETS`, 140-145 `DOMAIN_RESOLUTION`, 148-167 `WindCastSettings`) — Add `RTE_FRANCE` dataset config.

**Scripts (already multi-domain — just need `--dataset` wiring):**
- `scripts/ingest_kelmarsh.py` (full) — CLI pattern
- `scripts/ingest_spain_demand.py` (full, 86 lines) — Demand CLI pattern, mirror for RTE
- `scripts/build_features.py` (lines 87-100, 142-180) — Has `--domain demand` but hardcodes `spain_demand.parquet`. **Must accept `--dataset` to drive the filename.**
- `scripts/train.py` (lines 170-196 domain dispatch, 181-194 dataset default resolution, 191 hardcoded `spain_demand_features.parquet`) — **Must use `dataset` in the filename.**
- `scripts/compare_runs.py` (full) — No changes.

**Test patterns:**
- `tests/data/test_spain_demand.py` (full) — Mirror structure for `test_rte_france.py`. Use small fixture TSV strings as inputs (no zip extraction in tests — feed `parse_rte_france_bytes()` directly with Latin-1 bytes).
- `tests/data/test_demand_qc.py` (full) — Extend with French holidays test cases.
- `tests/features/test_demand.py` (full) — Update for NWP-based feature set logic.
- `tests/data/test_demand_schema.py` (full) — Update column count from 11 to 12.

**Project rules:**
- `CLAUDE.md` (lines 22-80) — Tech stack, code style, core principles.
- `.claude/PRD.md` (lines 219-266) — Demand domain spec.

### New Files to Create

**Source:**
- `src/windcast/data/rte_france.py` (~150 lines) — ZIP reader + TSV parser + canonical schema mapper
- `scripts/ingest_rte_france.py` (~120 lines) — CLI: read ZIPs → parse → QC → parquet
- `scripts/log_tso_baseline.py` (~60 lines) — One-off: log `tso_forecast_mw` vs `load_mw` as an MLflow benchmark run

**Tests:**
- `tests/data/test_rte_france.py` — Parser tests with in-memory fixture bytes
- No new QC or feature test files — extend the existing ones in-place

### Relevant Documentation

**PDF spec (authoritative):**
- `data/ecomix-rte/Eco2mix - Spécifications des fichiers en puissance.pdf` (12 pages) — Official RTE file format specification. Already read; key points below.

**Format reality (what the file ACTUALLY contains, verified by inspection):**
- **Extension is `.xls` but content is plain TSV in ISO-8859-1 (Latin-1) encoding** — NOT a binary Excel file
- Separator: `\t` (tab)
- **40 columns** for France-perimeter (vs 40 advertised in PDF ≈ matches)
- **Header row**: `Périmètre`, `Nature`, `Date`, `Heures`, `Consommation`, `Prévision J-1`, `Prévision J`, `Fioul`, `Charbon`, `Gaz`, `Nucléaire`, `Eolien`, `Solaire`, `Hydraulique`, `Pompage`, `Bioénergies`, `Ech. physiques`, `Taux de Co2`, `Ech. comm. Angleterre`, `Ech. comm. Espagne`, `Ech. comm. Italie`, `Ech. comm. Suisse`, `Ech. comm. Allemagne-Belgique`, `Fioul - TAC`, `Fioul - Cogén.`, `Fioul - Autres`, `Gaz - TAC`, `Gaz - Cogén.`, `Gaz - CCG`, `Gaz - Autres`, `Hydraulique - Fil de l?eau + éclusée`, `Hydraulique - Lacs`, `Hydraulique - STEP turbinage`, `Bioénergies- Déchets`, `Bioénergies- Biomasse`, `Bioénergies- Biogaz`, `Eolien terrestre`, `Eolien offshore`, ... (exact count may vary by year; use `truncate_ragged_lines=True`)
- **Date format**: ISO `2023-01-01` (NOT `jj/mm/aaaa` as the PDF claimed — the PDF is wrong on this point)
- **Heures format**: `00:00`, `00:15`, `00:30`, `00:45`, ..., `23:45` — 96 rows per day
- **Native resolution**:
  - **Consommation**: 30-min (rows at `00:15`, `00:45`, etc. have `null` Consommation)
  - **Prévision J-1 / Prévision J**: 15-min (all 96 rows populated)
- **Null markers**: `ND` (non disponible), `-` (inexistante, for regional files only), `DC` (confidentielle), empty string
- **Trailer lines**: The last 2 rows of each file contain the warning text "RTE ne pourra être tenu responsable..." and an empty row. Both must be dropped.
- **Row count per year (definitive)**: 365 days × 96 slots + 2 trailer = **35,042** rows (2023 verified)
- **Périmètre column**: constant value `France` for all rows — no filtering needed
- **Nature column**: `Données définitives` for definitive files (or `null` on the trailer rows)
- **TZ**: Europe/Paris local time (not mentioned in PDF, but industry convention and confirmed by hourly patterns). **Must call `dt.replace_time_zone("Europe/Paris", ambiguous="earliest", non_existent="null")` then `dt.convert_time_zone("UTC")`.**
- **DST transitions**: spring forward = 1 missing slot, fall back = 1 duplicated slot. Polars handles both via `ambiguous="earliest"`.
- **Range check (2023 verified)**: Consommation 29,798 – 83,757 MW; Prévision J-1 27,500 – 83,000 MW

**Polars read recipe (verified working on 2023 file):**
```python
import polars as pl
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

def read_ecomix_file(zip_path: Path) -> pl.DataFrame:
    """Read one eCO2mix annual ZIP, return raw wide DataFrame."""
    with ZipFile(zip_path) as zf:
        xls_name = [n for n in zf.namelist() if n.endswith(".xls")][0]
        raw = zf.read(xls_name)
    decoded = raw.decode("latin-1").encode("utf-8")
    return pl.read_csv(
        BytesIO(decoded),
        separator="\t",
        infer_schema_length=10_000,
        null_values=["ND", "-", "DC", ""],
        truncate_ragged_lines=True,
        ignore_errors=True,  # trailer rows have fewer fields
    )
```

**Wattcast reference (for inspiration only, not used here):**
- `../wattcast/src/wattcast/config.py` (lines 65-74) — `TEMP_POINTS` 8 weighted French cities for demand temperature. **Ignored for Pass 7**: single Paris point is simpler, wattcast's multi-city approach is a future polish.

### Patterns to Follow

**Parser Pattern** (from `spain_demand.py:16-71`):
```python
def parse_rte_france(data_dir: Path) -> pl.DataFrame:
    """Read all eCO2mix annual ZIPs in data_dir, return canonical demand DataFrame."""
    zip_files = sorted(data_dir.glob("eCO2mix_RTE_Annuel-Definitif_*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No eCO2mix annual files found in {data_dir}")

    # Read each year, concat
    yearly_dfs = [_parse_one_year(p) for p in zip_files]
    df = pl.concat(yearly_dfs, how="vertical")

    # Add identifiers + defaults
    df = df.with_columns(
        pl.lit(DATASET_ID).alias("dataset_id"),
        pl.lit(ZONE_ID).alias("zone_id"),
        pl.lit(False).alias("is_holiday"),
        pl.lit(False).alias("is_dst_transition"),
        pl.lit(0).cast(pl.UInt8).alias("qc_flag"),
    )

    # Ensure all schema cols
    for col, dtype in DEMAND_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df = df.select([pl.col(col).cast(dtype) for col, dtype in DEMAND_SCHEMA.items()])
    return df.sort("timestamp_utc").unique(subset=["timestamp_utc"], keep="first")
```

**Naming Conventions:**
- Dataset ID: `rte_france` (snake_case)
- Zone ID: `FR`
- Parquet: `data/processed/rte_france.parquet`, `data/features/rte_france_features.parquet`
- MLflow experiment: `enercast-rte_france`
- Parent run names: `rte_france-demand_baseline`, `rte_france-demand_enriched`, `rte_france-demand_full`, `rte_france-tso_baseline`

**Error Handling:** as in existing parsers (`FileNotFoundError`, `ValueError`, `logger.warning`).

**Logging Pattern:** `logger = logging.getLogger(__name__)` + `logging.basicConfig` in CLI scripts only.

---

## IMPLEMENTATION PLAN

### Phase 1: Schema + Config Foundation

Extend `DEMAND_SCHEMA` additively with `tso_forecast_mw`. Add `RTE_FRANCE` dataset config. Add `RTE_FRANCE_WEATHER` in weather registry. Update schema/parser tests to tolerate the new column.

### Phase 2: RTE éCO2mix Parser

Implement `src/windcast/data/rte_france.py` with ZIP reading, Latin-1 decoding, Europe/Paris → UTC conversion, 30-min → hourly aggregation, and canonical schema mapping.

### Phase 3: QC + Feature Set Refactor

Add `FRANCE_HOLIDAYS` + dataset dispatch. Refactor `DEMAND_ENRICHED` / `DEMAND_FULL` for NWP-based weather. Update `build_demand_features` to read NWP temperature column if present.

### Phase 4: CLI + Backfill

Create `ingest_rte_france.py`. Add `--dataset` flag to `build_features.py` and `train.py`. Backfill Paris NWP via `get_weather()`.

### Phase 5: Run Pipeline + Benchmarks

Run ingest → 3 feature-set training runs → TSO baseline logging → comparison charts → STATUS update → regression check.

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

---

### Task 1: UPDATE `src/windcast/data/demand_schema.py` — add `tso_forecast_mw`

**IMPLEMENT**:
- Add `"tso_forecast_mw": pl.Float64,` in `DEMAND_SCHEMA` right after `"price_eur_mwh"`
- Update `DEMAND_COLUMNS` to include the new column in the same position
- Add `"tso_forecast_mw"` to `DEMAND_SIGNAL_COLUMNS`

**PATTERN**: Additive schema extension.
**GOTCHA**: Spain parser auto-handles missing columns via its `for col, dtype in DEMAND_SCHEMA.items(): if col not in df.columns: df = df.with_columns(pl.lit(None).cast(dtype)...)` loop — no Spain parser changes needed. Spain tests in Task 3 may need a column-count bump.
**VALIDATE**:
```bash
uv run python -c "
from windcast.data.demand_schema import DEMAND_SCHEMA, DEMAND_COLUMNS, DEMAND_SIGNAL_COLUMNS
assert 'tso_forecast_mw' in DEMAND_SCHEMA
assert 'tso_forecast_mw' in DEMAND_COLUMNS
assert 'tso_forecast_mw' in DEMAND_SIGNAL_COLUMNS
assert len(DEMAND_COLUMNS) == 12
print('OK: 12 columns, tso_forecast_mw added')
"
```

---

### Task 2: UPDATE `tests/data/test_demand_schema.py`

**IMPLEMENT**: Bump column-count expectations from `11` to `12`. Add an assertion that `tso_forecast_mw` is in the expected columns list. If there's a test like `test_empty_demand_frame_matches_schema`, the empty-frame factory already picks up the new column automatically — just verify the count.

**PATTERN**: Surgical test update.
**VALIDATE**: `uv run pytest tests/data/test_demand_schema.py -v`

---

### Task 3: UPDATE `tests/data/test_spain_demand.py`

**IMPLEMENT**: Adjust any hardcoded column count (`== 11`) to `12`. Spain parser itself does not need changes — its schema loop fills `tso_forecast_mw` as null.

**OPTIONAL POLISH**: In `spain_demand.py:_read_energy_csv()`, also map `total load forecast` → `tso_forecast_mw` (Spain CSV has this column). Update the corresponding test to assert non-null. Keep in one commit even if skipped — document the omission in NOTES if so.

**PATTERN**: Surgical test update.
**VALIDATE**: `uv run pytest tests/data/test_spain_demand.py -v`

---

### Task 4: UPDATE `src/windcast/config.py` — add `RTE_FRANCE` dataset config

**IMPLEMENT**: Add after `SPAIN_DEMAND` (line 71):
```python
RTE_FRANCE = DemandDatasetConfig(
    dataset_id="rte_france",
    zone_id="FR",
    population=68_000_000,
    latitude=48.8566,      # Paris
    longitude=2.3522,
    timezone="Europe/Paris",
)
```

Register in `DATASETS` dict (line 98-104):
```python
"rte_france": RTE_FRANCE,
```

**PATTERN**: Mirror `SPAIN_DEMAND`.
**GOTCHA**: No new `WindCastSettings` fields needed — we read files from `settings.data_dir / "ecomix-rte"` at ingest time.
**VALIDATE**:
```bash
uv run python -c "
from windcast.config import DATASETS
assert 'rte_france' in DATASETS
assert DATASETS['rte_france'].zone_id == 'FR'
print('OK')
"
```

---

### Task 5: UPDATE `src/windcast/weather/registry.py` — add `RTE_FRANCE_WEATHER`

**IMPLEMENT**: Add after `SPAIN_WEATHER` (line 42):
```python
RTE_FRANCE_WEATHER = WeatherConfig(
    name="rte_france",
    latitude=48.8566,
    longitude=2.3522,
    variables=[
        "temperature_2m",
        "wind_speed_10m",
        "relative_humidity_2m",
        "shortwave_radiation",  # optional for solar-demand interactions
    ],
    description="Paris — NWP for French national demand forecasting",
)
```
Register in `WEATHER_REGISTRY` (line 57):
```python
"rte_france": RTE_FRANCE_WEATHER,
```

**PATTERN**: Mirror `SPAIN_WEATHER`.
**NOTE**: Single-point Paris is a simplification. Wattcast uses 8 weighted cities (`TEMP_POINTS`). Keep single-point for Pass 7; multi-city = post-presentation polish.
**VALIDATE**:
```bash
uv run python -c "
from windcast.weather.registry import get_weather_config, WEATHER_REGISTRY
assert 'rte_france' in WEATHER_REGISTRY
c = get_weather_config('rte_france')
assert c.latitude == 48.8566
print('OK')
"
```

---

### Task 6: CREATE `src/windcast/data/rte_france.py`

**IMPLEMENT**: Parser that reads the local `data/ecomix-rte/*.zip` files and returns a canonical `DEMAND_SCHEMA` DataFrame. Target ~150 lines.

Structure:
```python
"""RTE éCO2mix annual definitive files parser — local ZIPs to canonical demand schema."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import polars as pl

from windcast.data.demand_schema import DEMAND_SCHEMA

logger = logging.getLogger(__name__)

DATASET_ID = "rte_france"
ZONE_ID = "FR"
ANNUAL_PATTERN = "eCO2mix_RTE_Annuel-Definitif_*.zip"
SOURCE_TZ = "Europe/Paris"


def parse_rte_france(data_dir: Path) -> pl.DataFrame:
    """Read all eCO2mix annual ZIPs in data_dir, return canonical demand DataFrame.

    The éCO2mix definitive files are TSV in Latin-1 encoding inside a ZIP.
    Load (Consommation) is at 30-min native resolution; we aggregate to hourly.
    TSO D-1 forecast (Prévision J-1) is at 15-min native resolution; we
    aggregate to hourly mean as well.

    Args:
        data_dir: Directory containing eCO2mix_RTE_Annuel-Definitif_<YYYY>.zip files.

    Returns:
        DataFrame conforming to DEMAND_SCHEMA, sorted by timestamp_utc, hourly.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"RTE data directory not found: {data_dir}")

    zip_files = sorted(data_dir.glob(ANNUAL_PATTERN))
    if not zip_files:
        raise FileNotFoundError(
            f"No annual definitive files found in {data_dir} (pattern: {ANNUAL_PATTERN})"
        )

    logger.info("Found %d annual files: %s → %s", len(zip_files), zip_files[0].name, zip_files[-1].name)

    yearly_dfs = []
    for zp in zip_files:
        try:
            yearly_dfs.append(_parse_one_year(zp))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", zp.name, e)
            continue

    if not yearly_dfs:
        raise ValueError("No yearly files could be parsed")

    df = pl.concat(yearly_dfs, how="vertical_relaxed")
    logger.info("Concatenated %d yearly frames: %d rows before hourly resample", len(yearly_dfs), len(df))

    # Aggregate to hourly (load is 30-min native, forecast is 15-min native)
    df = _resample_hourly(df)
    logger.info("After hourly resample: %d rows", len(df))

    # Identifier + default columns
    df = df.with_columns(
        pl.lit(DATASET_ID).alias("dataset_id"),
        pl.lit(ZONE_ID).alias("zone_id"),
        pl.lit(False).alias("is_holiday"),
        pl.lit(False).alias("is_dst_transition"),
        pl.lit(0).cast(pl.UInt8).alias("qc_flag"),
    )

    # Ensure all schema columns exist, cast, reorder
    for col, dtype in DEMAND_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
    df = df.select([pl.col(col).cast(dtype) for col, dtype in DEMAND_SCHEMA.items()])
    df = df.sort("timestamp_utc").unique(subset=["timestamp_utc"], keep="first")

    logger.info("Final RTE demand DataFrame: %d rows, %d columns", len(df), len(df.columns))
    return df


def _parse_one_year(zip_path: Path) -> pl.DataFrame:
    """Extract and parse one eCO2mix annual ZIP → (timestamp_utc, load_mw, tso_forecast_mw)."""
    with ZipFile(zip_path) as zf:
        xls_names = [n for n in zf.namelist() if n.endswith(".xls")]
        if not xls_names:
            raise ValueError(f"No .xls in {zip_path.name}")
        raw = zf.read(xls_names[0])

    # File is ISO-8859-1 TSV despite the .xls extension
    decoded = raw.decode("latin-1").encode("utf-8")

    raw_df = pl.read_csv(
        BytesIO(decoded),
        separator="\t",
        infer_schema_length=10_000,
        null_values=["ND", "-", "DC", ""],
        truncate_ragged_lines=True,
        ignore_errors=True,
    )

    # Drop trailer rows (warning + empty). They have null Date.
    raw_df = raw_df.filter(pl.col("Date").is_not_null() & pl.col("Heures").is_not_null())

    # Parse Date + Heures → naive datetime → Europe/Paris → UTC
    # Date format: "2023-01-01" (ISO, NOT jj/mm/aaaa as the PDF claims)
    # Heures format: "HH:MM"
    df = raw_df.with_columns(
        (pl.col("Date") + pl.lit(" ") + pl.col("Heures"))
        .str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M", strict=False)
        .dt.replace_time_zone(SOURCE_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .alias("timestamp_utc")
    )

    # Drop rows where timestamp parsing failed (spring DST non-existent hour)
    df = df.filter(pl.col("timestamp_utc").is_not_null())

    # Extract the columns we care about, rename
    df = df.select(
        pl.col("timestamp_utc"),
        pl.col("Consommation").cast(pl.Float64).alias("load_mw"),
        pl.col("Prévision J-1").cast(pl.Float64).alias("tso_forecast_mw"),
    )

    return df


def _resample_hourly(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 30-min load + 15-min forecast to hourly mean.

    Polars `group_by_dynamic` produces a row for each hour bucket; null
    values (e.g., load_mw at :15 / :45) are skipped by the mean aggregator.
    """
    return (
        df.sort("timestamp_utc")
        .group_by_dynamic("timestamp_utc", every="1h", closed="left")
        .agg(
            pl.col("load_mw").mean(),
            pl.col("tso_forecast_mw").mean(),
        )
    )
```

**PATTERN**: Mirror `spain_demand.py` for structure; use the verified Polars TSV read recipe from the context section.
**IMPORTS**: Only stdlib (`io`, `pathlib`, `zipfile`) + `polars` + `DEMAND_SCHEMA`. No `httpx`, no `openpyxl`.
**GOTCHA**:
- The `.xls` extension is misleading — do NOT try to use `openpyxl`, `xlrd`, or `calamine`. It's plain TSV Latin-1.
- Date is in **ISO format** (`2023-01-01`), contradicting the PDF spec which says `jj/mm/aaaa`. Trust the file, not the PDF. If an older year (2014-2016) uses `dd/mm/yyyy`, the `strptime` with `strict=False` will fall back to null — check parsing by inspecting one old year first with the validation block in Task 7.
- DST transitions: `dt.replace_time_zone("Europe/Paris", ambiguous="earliest", non_existent="null")` handles both forward (non_existent→null, dropped) and backward (ambiguous→earliest slot kept) transitions safely.
- The Consommation column has NULLs at `:15` and `:45` slots — this is correct, not a bug. Hourly mean skips them.
- `ignore_errors=True` is needed because the trailer row and some mid-file rows may have field-count mismatches.
- `vertical_relaxed` concat is needed if different years have slightly different column sets (early years lack `Stockage batteries`, `Déstockage batteries`, `Eolien terrestre`, `Eolien offshore`).
**VALIDATE**: `uv run pyright src/windcast/data/rte_france.py`

---

### Task 7: SMOKE-TEST the parser against real files

**IMPLEMENT**: Run a one-off Python command to parse a single year and verify shape.

```bash
uv run python -c "
from pathlib import Path
from windcast.data.rte_france import _parse_one_year

# Test one recent year first
df = _parse_one_year(Path('data/ecomix-rte/eCO2mix_RTE_Annuel-Definitif_2023.zip'))
print('2023 raw shape:', df.shape)
print('Date range:', df['timestamp_utc'].min(), '→', df['timestamp_utc'].max())
print('load_mw non-null:', df['load_mw'].drop_nulls().len())
print('tso_forecast non-null:', df['tso_forecast_mw'].drop_nulls().len())
print('load range:', df['load_mw'].min(), '→', df['load_mw'].max())

# Sanity check: 2023 should have ~17,520 load values (365 * 48)
assert 17_000 < df['load_mw'].drop_nulls().len() < 18_000, 'unexpected load row count'

# Test the oldest year (2014) — might use different date format
df14 = _parse_one_year(Path('data/ecomix-rte/eCO2mix_RTE_Annuel-Definitif_2014.zip'))
print('2014 raw shape:', df14.shape)
print('2014 date range:', df14['timestamp_utc'].min(), '→', df14['timestamp_utc'].max())
"
```

**PATTERN**: Exploratory smoke test.
**GOTCHA**:
- If 2014 parsing fails or returns few rows, the date format may be `jj/mm/aaaa` for older files. Fall back: add a try-except in `_parse_one_year` that tries `%Y-%m-%d %H:%M` first, then `%d/%m/%Y %H:%M`.
- If the 2014 file has drastically different columns (e.g., no `Prévision J-1`), add it to the "missing columns" handling in the final concat.
**VALIDATE**: Both years parse successfully, 2023 ≈ 17,520 load values, 2014 parses at all (even if smaller).

---

### Task 8: CREATE `tests/data/test_rte_france.py`

**IMPLEMENT**: Unit tests with in-memory Latin-1 TSV fixtures (no real ZIP extraction in tests).

Tests to write:

```python
"""Tests for RTE éCO2mix parser."""
from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest

from windcast.data.demand_schema import DEMAND_SCHEMA, validate_demand_schema
from windcast.data.rte_france import _parse_one_year, _resample_hourly, parse_rte_france


MINIMAL_TSV_HEADER = (
    "Périmètre\tNature\tDate\tHeures\tConsommation\tPrévision J-1\tPrévision J\n"
)

def _make_minimal_xls_bytes(rows: list[tuple[str, str, str, str, str, str, str]]) -> bytes:
    """Build a fake eCO2mix XLS (TSV Latin-1) with the minimal columns."""
    text = MINIMAL_TSV_HEADER
    for row in rows:
        text += "\t".join(row) + "\n"
    # Trailer warning line
    text += "RTE ne pourra être tenu responsable...\n\n"
    return text.encode("latin-1")


def _make_zip(xls_bytes: bytes, inner_name: str = "test.xls") -> bytes:
    """Wrap XLS bytes in an in-memory ZIP."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(inner_name, xls_bytes)
    return buf.getvalue()


def test_parse_one_year_basic(tmp_path: Path):
    rows = [
        ("France", "Données définitives", "2023-06-01", "00:00", "50000", "49500", "50500"),
        ("France", "Données définitives", "2023-06-01", "00:15", "", "49600", "50600"),  # null load
        ("France", "Données définitives", "2023-06-01", "00:30", "50100", "49700", "50700"),
        ("France", "Données définitives", "2023-06-01", "00:45", "", "49800", "50800"),
        ("France", "Données définitives", "2023-06-01", "01:00", "50200", "49900", "50900"),
    ]
    zip_path = tmp_path / "eCO2mix_RTE_Annuel-Definitif_2023.zip"
    zip_path.write_bytes(_make_zip(_make_minimal_xls_bytes(rows)))

    df = _parse_one_year(zip_path)

    assert "timestamp_utc" in df.columns
    assert "load_mw" in df.columns
    assert "tso_forecast_mw" in df.columns
    # Trailer rows dropped
    assert len(df) == 5
    # Null handling
    assert df["load_mw"].drop_nulls().len() == 3  # only :00, :30, :00 are present


def test_parse_one_year_drops_trailer():
    """Trailer rows with null Date/Heures must be filtered."""
    # Covered in test_parse_one_year_basic


def test_resample_hourly_aggregates_to_hour():
    # Build a synthetic 15-min DataFrame with 4 points covering one hour
    df = pl.DataFrame({
        "timestamp_utc": [
            pl.datetime(2023, 6, 1, 0, 0, time_zone="UTC"),
            pl.datetime(2023, 6, 1, 0, 15, time_zone="UTC"),
            pl.datetime(2023, 6, 1, 0, 30, time_zone="UTC"),
            pl.datetime(2023, 6, 1, 0, 45, time_zone="UTC"),
        ],
        "load_mw": [50000.0, None, 50100.0, None],
        "tso_forecast_mw": [49500.0, 49600.0, 49700.0, 49800.0],
    })
    # Note: the datetime() helper returns an expression, so use direct literals:
    # actually use pl.Datetime with a list of Python datetimes
    # (simplified here for brevity — adjust syntax)
    result = _resample_hourly(df)
    assert len(result) == 1
    assert result["load_mw"][0] == pytest.approx(50050.0)  # mean of 50000 and 50100
    assert result["tso_forecast_mw"][0] == pytest.approx(49650.0)  # mean of 4 forecasts


def test_parse_rte_france_produces_canonical_schema(tmp_path: Path):
    rows = [
        ("France", "Données définitives", "2023-06-01", "00:00", "50000", "49500", "50500"),
        ("France", "Données définitives", "2023-06-01", "00:30", "50100", "49700", "50700"),
    ]
    (tmp_path / "eCO2mix_RTE_Annuel-Definitif_2023.zip").write_bytes(
        _make_zip(_make_minimal_xls_bytes(rows))
    )

    df = parse_rte_france(tmp_path)

    errors = validate_demand_schema(df)
    assert not errors, f"Schema validation failed: {errors}"
    assert df["dataset_id"][0] == "rte_france"
    assert df["zone_id"][0] == "FR"
    assert df["tso_forecast_mw"].drop_nulls().len() >= 1


def test_parse_rte_france_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_rte_france(tmp_path / "nonexistent")


def test_parse_rte_france_empty_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No annual definitive"):
        parse_rte_france(tmp_path)


def test_parse_handles_nd_as_null(tmp_path: Path):
    """ND values should be parsed as null."""
    rows = [
        ("France", "Données définitives", "2023-06-01", "00:00", "ND", "49500", "50500"),
        ("France", "Données définitives", "2023-06-01", "00:30", "50100", "ND", "50700"),
    ]
    (tmp_path / "eCO2mix_RTE_Annuel-Definitif_2023.zip").write_bytes(
        _make_zip(_make_minimal_xls_bytes(rows))
    )
    df = parse_rte_france(tmp_path)
    # load at 00:00 should be null, tso_forecast at 00:30 should be null
    # After hourly resample (both rows in same hour 00:00-01:00):
    # load_mw mean of (null, 50100) = 50100
    # tso_forecast_mw mean of (49500, null) = 49500
    assert df["load_mw"][0] == pytest.approx(50100.0)
    assert df["tso_forecast_mw"][0] == pytest.approx(49500.0)
```

Target: ~8 tests, all in-memory, no network, no real ZIP files.

**PATTERN**: Mirror `tests/data/test_spain_demand.py` (TSV-based fixtures) and `tests/data/test_kelmarsh.py` (in-memory ZIP pattern — see `_make_zip` helper).
**GOTCHA**:
- Polars `datetime()` construction syntax — double-check with a local Polars session. The fixture DataFrame builder may need `[datetime(...)]` Python objects instead of `pl.datetime(...)` literals.
- The `Period` column in tests is latin-1 by construction (since we encode to latin-1). Accented characters in `Périmètre`, `Prévision`, `Consommation` column names must survive the round-trip.
**VALIDATE**: `uv run pytest tests/data/test_rte_france.py -v`

---

### Task 9: UPDATE `src/windcast/data/demand_qc.py` — French holidays + dispatcher

**IMPLEMENT**:

a) Add `FRANCE_HOLIDAYS` after `SPAIN_HOLIDAYS` (around line 31):
```python
# France public holidays 2014-2024 (fixed-date)
FRANCE_HOLIDAYS: set[date] = set()
for y in range(2014, 2025):
    FRANCE_HOLIDAYS.update([
        date(y, 1, 1),    # Jour de l'an
        date(y, 5, 1),    # Fête du travail
        date(y, 5, 8),    # Victoire 1945
        date(y, 7, 14),   # Fête nationale
        date(y, 8, 15),   # Assomption
        date(y, 11, 1),   # Toussaint
        date(y, 11, 11),  # Armistice
        date(y, 12, 25),  # Noël
    ])
# Easter-derived holidays (Lundi de Pâques, Jeudi Ascension, Lundi Pentecôte) 2014-2024
FRANCE_HOLIDAYS.update([
    date(2014, 4, 21), date(2014, 5, 29), date(2014, 6, 9),
    date(2015, 4, 6),  date(2015, 5, 14), date(2015, 5, 25),
    date(2016, 3, 28), date(2016, 5, 5),  date(2016, 5, 16),
    date(2017, 4, 17), date(2017, 5, 25), date(2017, 6, 5),
    date(2018, 4, 2),  date(2018, 5, 10), date(2018, 5, 21),
    date(2019, 4, 22), date(2019, 5, 30), date(2019, 6, 10),
    date(2020, 4, 13), date(2020, 5, 21), date(2020, 6, 1),
    date(2021, 4, 5),  date(2021, 5, 13), date(2021, 5, 24),
    date(2022, 4, 18), date(2022, 5, 26), date(2022, 6, 6),
    date(2023, 4, 10), date(2023, 5, 18), date(2023, 5, 29),
    date(2024, 4, 1),  date(2024, 5, 9),  date(2024, 5, 20),
])
```

b) Add dispatcher:
```python
HOLIDAYS_BY_DATASET: dict[str, set[date]] = {
    "spain_demand": SPAIN_HOLIDAYS,
    "rte_france": FRANCE_HOLIDAYS,
}
```

c) Update `_detect_holidays()` (line 115-120) to use dataset_id:
```python
def _detect_holidays(df: pl.DataFrame) -> pl.DataFrame:
    """Set is_holiday=True for public holidays of the dataset's country."""
    if len(df) == 0:
        return df.with_columns(pl.lit(False).alias("is_holiday"))
    dataset_id = df.get_column("dataset_id").head(1).to_list()[0]
    holiday_set = HOLIDAYS_BY_DATASET.get(dataset_id, set())
    if not holiday_set:
        logger.warning("No holidays configured for dataset %s", dataset_id)
        return df.with_columns(pl.lit(False).alias("is_holiday"))
    holiday_dates = pl.Series("_holiday_dates", list(holiday_set)).implode()
    return df.with_columns(
        pl.col("timestamp_utc").dt.date().is_in(holiday_dates).alias("is_holiday")
    )
```

**PATTERN**: Dataset-id dispatcher keeps the QC pipeline signature unchanged.
**GOTCHA**: DST dates (`DST_TRANSITION_DATES` line 34-43) are the same for EU countries — rename the comment from "Spain" to "EU" but don't duplicate.
**VALIDATE**:
```bash
uv run python -c "
from windcast.data.demand_qc import FRANCE_HOLIDAYS, SPAIN_HOLIDAYS, HOLIDAYS_BY_DATASET
from datetime import date
assert date(2023, 7, 14) in FRANCE_HOLIDAYS  # Fête nationale
assert date(2023, 7, 14) not in SPAIN_HOLIDAYS
assert date(2023, 4, 10) in FRANCE_HOLIDAYS  # Lundi de Pâques
assert 'rte_france' in HOLIDAYS_BY_DATASET
print('OK')
"
```

---

### Task 10: ADD French holidays test cases to `tests/data/test_demand_qc.py`

**IMPLEMENT**: Add tests verifying:
- A DataFrame with `dataset_id='rte_france'` and a timestamp on 2023-07-14 → `is_holiday=True`
- A DataFrame with `dataset_id='spain_demand'` and the same timestamp → `is_holiday=False`
- Empty DataFrame → no crash

**PATTERN**: Mirror the existing Spain holiday tests.
**VALIDATE**: `uv run pytest tests/data/test_demand_qc.py -v`

---

### Task 11: UPDATE `src/windcast/features/registry.py` — refactor demand feature sets

**IMPLEMENT**: Replace `DEMAND_ENRICHED` (lines 78-90) and `DEMAND_FULL` (lines 92-104):

```python
DEMAND_ENRICHED = FeatureSet(
    name="demand_enriched",
    columns=[
        *DEMAND_BASELINE.columns,
        "load_mw_roll_mean_24",
        "load_mw_roll_std_24",
        "load_mw_roll_mean_168",
        "is_holiday",
    ],
    description="Demand enriched: baseline + rolling load stats + holiday flag (no weather)",
)

DEMAND_FULL = FeatureSet(
    name="demand_full",
    columns=[
        *DEMAND_ENRICHED.columns,
        "nwp_temperature_2m",
        "nwp_wind_speed_10m",
        "nwp_relative_humidity_2m",
        "heating_degree_days",
        "cooling_degree_days",
    ],
    description="Demand full: enriched + NWP at horizon + HDD/CDD from NWP (operational)",
)
```

**PATTERN**: Mirror `WIND_FULL` (lines 46-59) — NWP columns use the short name; `_h{k}` suffix is resolved per horizon in `train.py:_resolve_horizon_features` (line 59).
**GOTCHA**:
- `price_eur_mwh` / `price_lag1` / `price_lag24` are removed from `demand_full` — they were Spain-specific (not in RTE). Removing them also simplifies the feature set.
- HDD/CDD are computed from `nwp_temperature_2m_h1` as a static proxy in `build_demand_features` (see Task 12).
**VALIDATE**:
```bash
uv run python -c "
from windcast.features.registry import DEMAND_FULL, DEMAND_ENRICHED
assert 'nwp_temperature_2m' in DEMAND_FULL.columns
assert 'temperature_c' not in DEMAND_FULL.columns
assert 'temperature_c' not in DEMAND_ENRICHED.columns
assert 'is_holiday' in DEMAND_ENRICHED.columns
print('OK')
"
```

---

### Task 12: UPDATE `src/windcast/features/demand.py` — NWP-aware HDD/CDD

**IMPLEMENT**:

a) Generalize `_add_temperature_features` (lines 127-136):
```python
def _add_temperature_features(df: pl.DataFrame, source_col: str = "temperature_c") -> pl.DataFrame:
    """Add heating/cooling degree days from the specified temperature column."""
    if source_col not in df.columns:
        logger.warning("HDD/CDD source column %s missing, skipping", source_col)
        return df
    return df.with_columns(
        pl.max_horizontal(pl.lit(0.0), pl.lit(18.0) - pl.col(source_col)).alias("heating_degree_days"),
        pl.max_horizontal(pl.lit(0.0), pl.col(source_col) - pl.lit(24.0)).alias("cooling_degree_days"),
    )
```

b) Update `build_demand_features` (lines 17-60):
```python
def build_demand_features(
    df: pl.DataFrame,
    feature_set: str = "demand_baseline",
) -> pl.DataFrame:
    fs = get_feature_set(feature_set)
    logger.info("Building feature set %r (%d features)", fs.name, len(fs.columns))

    n_before = len(df)
    df = df.filter(pl.col("qc_flag") == QC_OK)
    logger.info("QC filter: %d -> %d rows (dropped %d)", n_before, len(df), n_before - len(df))
    df = df.sort("zone_id", "timestamp_utc")

    # Baseline: lags + calendar
    df = _add_lag_features(df, "load_mw", DEFAULT_LOAD_LAGS)
    df = _add_cyclic_calendar(df)

    if feature_set in ("demand_enriched", "demand_full"):
        df = _add_rolling_features(df, "load_mw", DEFAULT_ROLLING_WINDOWS)
        df = df.with_columns(pl.col("is_holiday").cast(pl.Int8).alias("is_holiday"))

    if feature_set == "demand_full":
        # Prefer NWP temperature at shortest horizon; fall back to observed (Spain compat)
        if "nwp_temperature_2m_h1" in df.columns:
            df = _add_temperature_features(df, source_col="nwp_temperature_2m_h1")
        elif "temperature_c" in df.columns and df["temperature_c"].drop_nulls().len() > 0:
            df = _add_temperature_features(df, source_col="temperature_c")
        else:
            logger.warning("No temperature source for HDD/CDD — skipping")

    logger.info("Feature engineering complete: %d columns", len(df.columns))
    return df
```

c) `_add_price_features` (line 139-144) is no longer called but keep the function defined (legacy for Spain; no callers after this change).

**PATTERN**: Mirror `build_wind_features` which delegates NWP joining to `build_features.py` before calling the feature builder.
**GOTCHA**:
- NWP columns `nwp_temperature_2m_h1`, `nwp_wind_speed_10m_h1`, etc. are added BY `build_features.py` when `--weather-db` is passed. This function just consumes them.
- The "h1" horizon is 1h ahead for demand (since `DOMAIN_RESOLUTION[demand]=60`). Using h1 as a proxy for HDD/CDD is a static approximation — sufficient for the demo, polish post-presentation.
**VALIDATE**: `uv run pytest tests/features/test_demand.py -v` (may need updates in Task 13)

---

### Task 13: UPDATE `tests/features/test_demand.py` — remove observed-weather assumptions

**IMPLEMENT**:
- Remove any assertions that `temperature_c` must be in `demand_enriched` output
- Remove any assertions that `price_eur_mwh` features must be in `demand_full`
- Add a new test that feeds a DataFrame with a `nwp_temperature_2m_h1` column and asserts `heating_degree_days` / `cooling_degree_days` are computed
- Add a fallback test: feed a DataFrame WITHOUT NWP but WITH `temperature_c`, verify HDD/CDD still computed
- Keep the cyclic calendar and lag tests unchanged

**PATTERN**: Surgical test update.
**VALIDATE**: `uv run pytest tests/features/test_demand.py -v`

---

### Task 14: UPDATE `scripts/build_features.py` — accept `--dataset` flag

**IMPLEMENT**:

a) Add CLI arg after line 63:
```python
parser.add_argument("--dataset", default=None, help="Dataset ID for file lookup. Default: domain-specific")
```

b) Resolve default:
```python
domain_dataset_defaults = {"wind": "kelmarsh", "demand": "spain_demand", "solar": "pvdaq_system4"}
dataset = args.dataset or domain_dataset_defaults[args.domain]
```

c) Replace filename logic (lines 86-94):
```python
if args.domain == "demand":
    pattern = f"{dataset}.parquet"
elif args.domain == "solar":
    pattern = "pvdaq_system4.parquet"
elif args.turbine_id:
    pattern = f"kelmarsh_{args.turbine_id}.parquet"
else:
    pattern = "kelmarsh_*.parquet"
```

d) Replace output filename (around line 164):
```python
if args.domain == "demand":
    output_path = output_dir / f"{dataset}_features.parquet"
elif args.domain == "solar":
    output_path = output_dir / "pvdaq_system4_features.parquet"
else:
    output_path = output_dir / pq_file.name
```

e) Wire weather config name for demand (around line 113):
```python
domain_weather_map = {
    "wind": "kelmarsh",
    "demand": dataset if args.domain == "demand" else "spain_demand",
    "solar": "pvdaq_system4",
}
```

**PATTERN**: Additive CLI extension.
**GOTCHA**: Keep backward compat — `spain_demand` is still the default when no `--dataset` is passed with `--domain demand`.
**VALIDATE**:
```bash
uv run python scripts/build_features.py --help | grep -A1 dataset
```

---

### Task 15: UPDATE `scripts/train.py` — use `--dataset` for demand parquet path

**IMPLEMENT**: The `--dataset` CLI arg already exists (line 142-145), and `dataset` resolution already exists (line 186). Just fix the hardcoded filename at line 191:

```python
if domain == "demand":
    parquet_path = features_dir / f"{dataset}_features.parquet"
```

**PATTERN**: Surgical string-interpolation fix.
**VALIDATE**: `uv run python scripts/train.py --help`

---

### Task 16: BACKFILL Paris NWP into `data/weather.db`

**IMPLEMENT**:
```bash
uv run python -c "
from windcast.weather import get_weather
df = get_weather('rte_france', '2014-01-01', '2024-12-31')
print('NWP rows:', len(df))
print('Variables:', [c for c in df.columns if c != 'timestamp_utc'])
print('Date range:', df['timestamp_utc'].min(), '→', df['timestamp_utc'].max())
"
```

**PATTERN**: Uses existing `get_weather()` fetch-cache-serve API.
**GOTCHA**:
- First run hits Open-Meteo Archive API, ~30-90 seconds for 11 years at one location
- ERA5 lag: end date auto-clamped to `today - 5 days`
- Subsequent runs hit the cache instantly
**VALIDATE**:
```bash
uv run python -c "
from pathlib import Path
from windcast.weather.storage import WeatherStorage
s = WeatherStorage(Path('data/weather.db'))
c = s.get_coverage('48.8566_2.3522')
assert c is not None, 'Paris NWP not cached'
print('Paris coverage:', c)
s.close()
"
```

---

### Task 17: CREATE `scripts/ingest_rte_france.py`

**IMPLEMENT**:
```python
"""Ingest RTE éCO2mix France demand: read ZIPs → QC → save Parquet.

Usage:
    uv run python scripts/ingest_rte_france.py
    uv run python scripts/ingest_rte_france.py --data-dir data/ecomix-rte
"""

import argparse
import logging
import sys
from pathlib import Path

from windcast.config import get_settings
from windcast.data.demand_qc import demand_qc_summary, run_demand_qc_pipeline
from windcast.data.demand_schema import validate_demand_schema
from windcast.data.rte_france import parse_rte_france

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RTE éCO2mix France demand")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing eCO2mix annual ZIPs. Default: data/ecomix-rte",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    settings = get_settings()
    data_dir = args.data_dir or (settings.data_dir / "ecomix-rte")
    output_dir = args.output_dir or settings.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing RTE éCO2mix files from %s", data_dir)
    df = parse_rte_france(data_dir)
    logger.info("Parsed %d rows", len(df))

    errors = validate_demand_schema(df)
    if errors:
        logger.error("Schema validation failed:\n%s", "\n".join(errors))
        sys.exit(1)
    logger.info("Schema validation passed")

    df = run_demand_qc_pipeline(df, settings.demand_qc)
    summary = demand_qc_summary(df)
    for k, v in summary.items():
        logger.info("QC: %s = %s", k, v)

    out_path = output_dir / "rte_france.parquet"
    df.write_parquet(out_path, compression="zstd", compression_level=3)
    logger.info("Wrote %d rows to %s (%.1f MB)", len(df), out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
```

**PATTERN**: Mirror `scripts/ingest_spain_demand.py` structure (76 lines).
**GOTCHA**:
- `settings.demand_qc.min_load_mw = 10_000` may be too high for the range — France nightly minimum is ~30 GW so 10 GW is safe
- `settings.demand_qc.max_load_mw = 50_000` — **TOO LOW** for France (peak 83 GW). Either:
  - Override via env: `WINDCAST_DEMAND_QC__MAX_LOAD_MW=100000`
  - OR update the default in `config.py:DemandQCConfig` to `max_load_mw=100_000.0`
  - RECOMMENDED: update the default (cleaner)
**VALIDATE**: `uv run python scripts/ingest_rte_france.py --help`

---

### Task 18: UPDATE `DemandQCConfig` upper limit for France

**IMPLEMENT**: In `src/windcast/config.py` at line 117-125, change:
```python
class DemandQCConfig(BaseModel):
    max_load_mw: float = 100_000.0  # was 50_000, too low for France peak 83 GW
    min_load_mw: float = 10_000.0
    # ... rest unchanged
```

**PATTERN**: Config default update.
**GOTCHA**: Spain peak is ~41 GW so 100 GW cap still works for Spain — no Spain regression.
**VALIDATE**: `uv run pytest tests/ -q | tail -5`

---

### Task 19: RUN ingest on real RTE data

**IMPLEMENT**:
```bash
uv run python scripts/ingest_rte_france.py
```

**PATTERN**: Mirror the Kelmarsh ingestion pattern from Pass 1.
**GOTCHA**:
- Expect ~96,000 hourly rows (11 years × 8766 hours/year ≈ 96,400)
- If an older year (2014-2016) has date-format issues, the smoke test in Task 7 should have caught it — fix the parser first
- `is_dst_transition` may flag ~44 transitions (2 per year × 11 years × ~2 hours each)
**VALIDATE**:
```bash
ls -la data/processed/rte_france.parquet
uv run python -c "
import polars as pl
df = pl.read_parquet('data/processed/rte_france.parquet')
print('Rows:', len(df))
print('Columns:', df.columns)
print('Date range:', df['timestamp_utc'].min(), '→', df['timestamp_utc'].max())
print('Load MW range:', df['load_mw'].min(), '→', df['load_mw'].max())
print('TSO forecast nulls:', df['tso_forecast_mw'].null_count(), '/', len(df))
print('Holidays count:', df['is_holiday'].sum())
print('QC distribution:', df['qc_flag'].value_counts().sort('qc_flag').to_dicts())
assert len(df) > 80_000, 'Too few rows'
assert df['load_mw'].max() > 80_000, 'Peak too low'
assert df['load_mw'].min() > 25_000, 'Minimum too low'
assert df['is_holiday'].sum() > 100, 'Not enough French holidays detected'
print('OK')
"
```
**Expected**: ~90,000-96,000 rows, load 29-85 GW, tso_forecast mostly non-null, ~120+ holidays detected.

---

### Task 20: RUN the 3 training feature sets

**IMPLEMENT**:
```bash
# Clear env to ensure fresh lru_cache
unset WINDCAST_TRAIN_YEARS WINDCAST_VAL_YEARS
export WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2
# → train 2014-2021 (8y), val 2022-2023 (2y), test 2024 (1y)

# Baseline — no weather
uv run python scripts/build_features.py --domain demand --dataset rte_france --feature-set demand_baseline
uv run python scripts/train.py --domain demand --dataset rte_france --feature-set demand_baseline

# Enriched — + rolling + holidays
uv run python scripts/build_features.py --domain demand --dataset rte_france --feature-set demand_enriched
uv run python scripts/train.py --domain demand --dataset rte_france --feature-set demand_enriched

# Full — + NWP at horizon
uv run python scripts/build_features.py --domain demand --dataset rte_france --feature-set demand_full --weather-db data/weather.db
uv run python scripts/train.py --domain demand --dataset rte_france --feature-set demand_full
```

**PATTERN**: Mirror Pass 3+5 wind runs.
**GOTCHA**:
- Env vars MUST be set before ANY `uv run` call (`get_settings()` is lru_cache'd)
- `--weather-db` only for `demand_full` (others don't use NWP)
- Each feature-set run overwrites `rte_france_features.parquet` — do NOT parallelize
- Expect each XGBoost training run to take 30-60 seconds (hourly × ~70,000 train rows × 5 horizons)
**VALIDATE**:
```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('enercast-rte_france')
assert exp, 'experiment missing'
runs = client.search_runs([exp.experiment_id], filter_string=\"tags.enercast.run_type = 'parent'\", order_by=['attributes.start_time ASC'])
assert len(runs) == 3, f'Expected 3 parent runs, got {len(runs)}'
print(f'{\"feature_set\":<18} {\"h1_mae\":>10} {\"h6_mae\":>10} {\"h24_mae\":>10} {\"h24_skill\":>10}')
for r in runs:
    fs = r.data.tags.get('enercast.feature_set', '?')
    m = r.data.metrics
    print(f'{fs:<18} {m.get(\"h1_mae\", float(\"nan\")):>10.1f} {m.get(\"h6_mae\", float(\"nan\")):>10.1f} {m.get(\"h24_mae\", float(\"nan\")):>10.1f} {m.get(\"h24_skill_score\", float(\"nan\")):>10.3f}')
"
```
**Expected**: 3 rows. `demand_full` should have lower MAE than baseline at h≥6. Typical French demand MAE at h24 with NWP: 800-2000 MW.

---

### Task 21: CREATE `scripts/log_tso_baseline.py` and run it

**IMPLEMENT**:
```python
"""Log RTE TSO day-ahead forecast as a benchmark run in MLflow."""
from __future__ import annotations

import logging
from datetime import datetime

import mlflow
import polars as pl

from windcast.config import get_settings
from windcast.models.evaluation import compute_metrics
from windcast.tracking import setup_mlflow

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = get_settings()
    setup_mlflow(settings.mlflow_tracking_uri, "enercast-rte_france")

    df = pl.read_parquet("data/processed/rte_france.parquet")

    # Mirror the temporal split used by train.py
    ts = df["timestamp_utc"]
    start: datetime = ts.min()  # type: ignore[assignment]
    train_end = start.replace(year=start.year + settings.train_years)
    val_end = train_end.replace(year=train_end.year + settings.val_years)
    val = df.filter((pl.col("timestamp_utc") >= train_end) & (pl.col("timestamp_utc") < val_end))
    val = val.drop_nulls(subset=["load_mw", "tso_forecast_mw"])
    logger.info("Val set: %d rows (%s → %s)", len(val), train_end, val_end)

    y_true = val["load_mw"].to_numpy()
    y_pred = val["tso_forecast_mw"].to_numpy()
    metrics = compute_metrics(y_true, y_pred)

    with mlflow.start_run(run_name="rte_france-tso_baseline"):
        mlflow.set_tags({
            "enercast.stage": "dev",
            "enercast.domain": "demand",
            "enercast.purpose": "tso_baseline",
            "enercast.backend": "tso",
            "enercast.feature_set": "none",
            "enercast.run_type": "parent",
            "enercast.data_resolution_min": "60",
        })
        mlflow.log_params({
            "domain": "demand",
            "dataset": "rte_france",
            "n_val": len(val),
            "split.val_start": str(train_end),
            "split.val_end": str(val_end),
        })
        # Log against the day-ahead horizon (~h24 for hourly data)
        for k, v in metrics.items():
            mlflow.log_metric(f"h24_{k}", v)
        mlflow.set_tag(
            "mlflow.note.content",
            f"## RTE Day-Ahead Forecast Benchmark\n\n"
            f"The official D-1 forecast published by RTE, evaluated on the same "
            f"val split as our models.\n\n"
            f"**Val rows:** {len(val):,}\n"
            f"**MAE:** {metrics['mae']:.0f} MW\n"
            f"**RMSE:** {metrics['rmse']:.0f} MW\n\n"
            f"Our `demand_full` model should match or beat this at h24+.\n"
        )

    print("Logged TSO baseline:", metrics)


if __name__ == "__main__":
    main()
```

Run it:
```bash
uv run python scripts/log_tso_baseline.py
```

**PATTERN**: Thin MLflow logging, no training loop. Reuses `compute_metrics`.
**GOTCHA**:
- The RTE D-1 forecast is published ~noon J-1 and covers the full J day — treat it as a **day-ahead average** benchmark, not a pure h24. For slides, say "RTE day-ahead" not "h24".
- `compute_metrics` returns a dict with `mae`, `rmse`, `mape`, `bias`. If it also requires `y_persistence` for `skill_score`, pass `None` and the skill will be absent — that's fine for a benchmark.
**VALIDATE**:
```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('enercast-rte_france')
runs = client.search_runs([exp.experiment_id], filter_string=\"tags.enercast.purpose = 'tso_baseline'\")
assert runs
print('TSO baseline MAE:', runs[0].data.metrics.get('h24_mae'))
print('TSO baseline RMSE:', runs[0].data.metrics.get('h24_rmse'))
"
```
**This is the killer slide**: compare `demand_full h24_mae` to `rte_france-tso_baseline h24_mae`.

---

### Task 22: COMPARE runs + export PNGs

**IMPLEMENT**:
```bash
uv run python scripts/compare_runs.py --experiment enercast-rte_france
ls reports/comparison_enercast-rte_france_*.png
```

**PATTERN**: Same as Pass 6b.
**GOTCHA**: `compare_runs.py` reads flat `h{n}_mae` / `h{n}_skill_score` metrics from parent runs. The TSO baseline only has `h24_*` metrics (no h1/h6/h12/h48) — it will likely appear as a single-point run in the MAE chart. Check that the script handles partial-horizon runs gracefully; if not, skip the TSO run in `compare_runs.py` and mention the benchmark separately in STATUS.md.
**VALIDATE**: Two PNGs exist: `comparison_enercast-rte_france_mae.png`, `comparison_enercast-rte_france_skill.png`.

---

### Task 23: UPDATE `.claude/STATUS.md`

**IMPLEMENT**: Append a "Pass 7 — RTE France Demand (éCO2mix)" section. Include:
- The 3 feature-set MAE/skill table at h1/h6/h12/h24/h48 (in MW — demand, not kW)
- A line comparing `demand_full` h24 MAE to RTE TSO day-ahead MAE
- Note: "Spain Kaggle parser retained as 2nd demand reference implementation, not used for presentation"
- Note: "11 years of real French national load, RTE definitive files, offline ingestion (no API credentials)"
- Mark Pass 7 `[x]`

**PATTERN**: Mirror Pass 6b block at STATUS.md:116-139.
**VALIDATE**: `grep -n "Pass 7" .claude/STATUS.md`

---

### Task 24: REGRESSION CHECK

**IMPLEMENT**:
```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
uv run pytest tests/ -q
```

**Expected**: All green. Test count should be ~277+ (267 baseline + 8-10 new RTE tests). No Spain regressions.

---

## TESTING STRATEGY

### Unit Tests

- **Parser** (`test_rte_france.py`): fixture-based, no real ZIPs. 7-8 tests covering happy path, trailer dropping, ND/null handling, empty dir, hourly resampling correctness, DST edge cases (synthetic)
- **Schema** (`test_demand_schema.py`): column-count bump to 12
- **Spain parser** (`test_spain_demand.py`): column-count bump, still green
- **QC** (`test_demand_qc.py`): French holidays dispatched correctly, Spain holidays still work, empty-frame no crash
- **Features** (`test_demand.py`): NWP-based HDD/CDD, observed-weather fallback, removed price tests

### Integration Tests

None — the real integration is Tasks 19-22 running on real data.

### Edge Cases

- **Spring DST hour** (last Sunday of March, 02:00-03:00 Europe/Paris doesn't exist) → `non_existent="null"` → row dropped. Expected.
- **Fall DST hour** (last Sunday of October, 02:00-03:00 Europe/Paris happens twice) → `ambiguous="earliest"` → first occurrence kept.
- **Partial year** (2014 may start mid-year or 2024 may end mid-year) → handled by `.unique(subset=["timestamp_utc"])` at concat time.
- **Missing columns in older years** — use `pl.concat(..., how="vertical_relaxed")` to tolerate schema variation across years.
- **TSO forecast nulls at 15-min grain** — aggregated away by hourly mean; expect ~0% null rate after resampling.
- **Empty ZIPs or corrupt files** — caught by try/except in the `parse_rte_france` yearly loop, logged as warning, skipped.
- **Data range covers 11 years but `settings.train_years` default is 5** — override via env vars (`WINDCAST_TRAIN_YEARS=8 WINDCAST_VAL_YEARS=2`).

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
```
**Expected**: All clean.

### Level 2: Unit Tests

```bash
uv run pytest tests/ -q
uv run pytest tests/data/test_rte_france.py tests/data/test_demand_schema.py tests/features/test_demand.py tests/data/test_demand_qc.py -v
```
**Expected**: All green; ~277 total tests.

### Level 3: Integration — Pipeline Run

The 6 script invocations in Tasks 19-22 ARE the integration test. Their validation blocks double as assertions.

### Level 4: Manual Validation — MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db &
```
At http://localhost:5000:
1. Confirm **2 experiments**: `enercast-kelmarsh` + `enercast-rte_france`
2. In `enercast-rte_france`, confirm **4 parent runs**: baseline / enriched / full / tso_baseline
3. Compare `demand_full` vs `rte_france-tso_baseline` — the "we match the TSO" story
4. Charts tab: stepped `mae_by_horizon_min` overlays 3 training curves
5. Parent run Markdown descriptions render correctly

### Level 5: Cross-domain Comparison

```bash
uv run python scripts/compare_runs.py --experiment enercast-rte_france
uv run python scripts/compare_runs.py --experiment enercast-kelmarsh
ls reports/
```
**Expected**: 4 PNGs.

---

## ACCEPTANCE CRITERIA

- [ ] `DEMAND_SCHEMA` has 12 columns including `tso_forecast_mw`
- [ ] All Spain tests still green (no regression)
- [ ] `src/windcast/data/rte_france.py` exists with `parse_rte_france`, `_parse_one_year`, `_resample_hourly`
- [ ] `tests/data/test_rte_france.py` has ≥7 tests, all in-memory, no network
- [ ] `FRANCE_HOLIDAYS` has ≥88 entries (11 years × 8 fixed) and `HOLIDAYS_BY_DATASET` dispatcher works
- [ ] `DEMAND_FULL` references `nwp_temperature_2m` not `temperature_c`
- [ ] `DEMAND_ENRICHED` no longer references `temperature_c`
- [ ] `scripts/build_features.py` and `scripts/train.py` accept `--dataset`
- [ ] `scripts/ingest_rte_france.py` exists and runs successfully
- [ ] `DemandQCConfig.max_load_mw` raised to `100_000.0`
- [ ] `data/processed/rte_france.parquet` exists with ≥80,000 rows, load 29-85 GW, mostly non-null tso_forecast
- [ ] `data/weather.db` has Paris coverage for 2014-2024
- [ ] MLflow experiment `enercast-rte_france` has **4 parent runs**: 3 training + 1 `tso_baseline`
- [ ] Each training parent has stepped metrics (`mae_by_horizon_min`, `skill_score_by_horizon_min`)
- [ ] `demand_full` h24_mae ≤ `demand_enriched` h24_mae (NWP adds value)
- [ ] `demand_full` h24_mae within 30% of `tso_baseline` h24_mae (we're in RTE's ballpark)
- [ ] `reports/comparison_enercast-rte_france_mae.png` and `..._skill.png` exist
- [ ] `.claude/STATUS.md` updated with Pass 7 section and real metrics
- [ ] `uv run pytest tests/ -q` passes with count ≥ baseline + 8
- [ ] `uv run ruff check src/ tests/ scripts/` clean

---

## COMPLETION CHECKLIST

- [ ] Task 1-3: Schema extended, tests updated, Spain still passing
- [ ] Task 4-5: RTE dataset + weather configs registered
- [ ] Task 6: `rte_france.py` parser complete, pyright clean
- [ ] Task 7: Smoke test on real 2023 + 2014 files passes
- [ ] Task 8: Unit tests with in-memory fixtures passing
- [ ] Task 9-10: French holidays + dispatcher + tests
- [ ] Task 11-13: Feature set refactor + demand.py update + test updates
- [ ] Task 14-15: `build_features.py` and `train.py` accept `--dataset`
- [ ] Task 16: Paris NWP backfilled to `weather.db`
- [ ] Task 17-18: Ingest script created, QC config raised
- [ ] Task 19: Real ingest run, parquet created, ~90k rows verified
- [ ] Task 20: 3 training runs logged to MLflow
- [ ] Task 21: TSO baseline logged
- [ ] Task 22: PNG charts in `reports/`
- [ ] Task 23: STATUS.md updated
- [ ] Task 24: Regression check green
- [ ] Manual MLflow UI: 2 experiments, 4 demand parent runs, stepped charts
- [ ] Ruff + pytest + pyright all clean

---

## NOTES

### Why local files, not the RTE Data API

The user already downloaded 11 years of annual definitive files to `data/ecomix-rte/`. No API credentials, no OAuth, no network calls during ingest, no chunking, no rate limiting. This simplifies the parser from ~250 lines (async httpx client) to ~150 lines (zipfile + polars read_csv). Test coverage is higher because everything is deterministic and reproducible.

The trade-off: **we cannot refresh the data in real-time**. For the presentation this is fine — we're demonstrating a framework on historical data, not a live production system. If WN wants real-time, the RTE API client from wattcast is a known-good reference to port later.

### Why the `.xls` file is actually TSV

Verified by `file eCO2mix_RTE_Annuel-Definitif_2023.xls` → `ISO-8859 text`. Head byte inspection confirms plain tab-separated text. The `.xls` extension is presumably for users who double-click the file in Excel (Excel auto-detects TSV). Polars reads it directly after decoding Latin-1. No `openpyxl`, `xlrd`, or `calamine` needed.

The PDF specification (§3) says "fichier XLS zippé" which is technically wrong. Trust the file, not the spec.

### Why hourly resampling (not native 30-min or 15-min)

The éCO2mix file has:
- Load (`Consommation`) at native 30-min resolution (nulls at `:15` / `:45` slots)
- TSO forecasts (`Prévision J-1`, `Prévision J`) at native 15-min resolution

We aggregate to hourly in `_resample_hourly` because:
1. `DOMAIN_RESOLUTION[demand] = 60` is hardcoded in `config.py` and used in `train.py` for stepped-metric `minutes_ahead` computation. Changing to 30 would break wind/solar configs
2. Hourly is the industry standard for national load forecasting
3. Fewer features at 30-min means lower model complexity
4. Easier comparison against literature benchmarks

**Post-presentation polish**: make `DOMAIN_RESOLUTION` a per-dataset config (so RTE can be 30-min, Spain stays 60-min, wind stays 10-min) and surface the extra temporal granularity.

### Why single-point Paris NWP, not multi-city

Wattcast uses 8 weighted French cities (`TEMP_POINTS` at `../wattcast/src/wattcast/config.py:65-74`) for demand temperature. That's more accurate — France demand is continental-scale, with Paris carrying ~30% of the signal.

For Pass 7 we use Paris alone because:
- Single location fits the existing `WeatherConfig` schema (no code change)
- Paris alone correlates strongly with national (large population + heating-dominant)
- Multi-city aggregation would require a new component in the NWP join path
- Time budget

Post-presentation polish: add `MultiPointWeatherConfig` and port wattcast's `TEMP_POINTS`.

### Why HDD/CDD from h1 only, not per-horizon

Computing HDD/CDD for every forecast horizon (5 horizons × 2 features = 10 columns) would require either a deeper refactor of `_resolve_horizon_features` or 10 pre-computed columns. For the demo, using `nwp_temperature_2m_h1` as a static proxy captures ~85% of the value (h1 is strongly correlated with h24). Note as polish.

### Why drop prices from demand_full

RTE annual files contain no prices — only load + generation + flows. Adding EPEX spot prices would require an ENTSO-E or EPEX integration (= more credentials, more parsers). Out of scope for Pass 7. Wattcast does this if you need price-aware demand in the future.

### Expected French demand metrics (sanity band)

Based on RTE public benchmarks (RTE D-1 MAPE ~1.5% ≈ 800 MW on 55 GW mean) and literature:

| Horizon | baseline MAE | enriched MAE | full MAE | full skill |
|---------|-------------|--------------|----------|-----------|
| h1 (1h) | 400-700 MW | 350-600 MW | 300-550 MW | 0.10-0.25 |
| h6 (6h) | 1,200-2,000 MW | 1,000-1,600 MW | 700-1,300 MW | 0.25-0.40 |
| h24 (24h) | 2,000-3,500 MW | 1,500-2,500 MW | 800-1,600 MW | 0.40-0.60 |
| h48 (48h) | 2,500-4,500 MW | 2,000-3,200 MW | 1,000-2,000 MW | 0.45-0.65 |

**RTE TSO benchmark (day-ahead)**: ~700-900 MW MAE averaged over 24h. If `demand_full h24_mae` lands 800-1,500 MW, we're in the same league as the TSO. That's the headline slide.

If numbers are dramatically outside these bands:
- MAE << 300 MW anywhere: check NWP horizon alignment, look for leakage
- Skill < 0 at h1: persistence is usually strong at h1 for demand, 0 is plausible but negative suggests underfitting
- `tso_baseline` MAE >> 1,500 MW: data parsing issue (wrong column mapping, Europe/Paris → UTC error)

### Confidence score: 8.5/10

High confidence because:
- **+2** over the API plan: no credentials, no network, no rate limiting, no OAuth2 setup
- File format empirically verified on 2023 file (Polars parse succeeds in 100ms)
- Framework is 100% ready — only additive code changes needed
- 50 existing demand tests give a strong regression net
- The `demand_full` feature set refactor is the only risk surface and is small (2 files)

Remaining risks (-1.5):
- **-1**: Older years (2014-2018) may have different date format (`dd/mm/yyyy` vs `yyyy-mm-dd`) or different column sets. Task 7 smoke test catches this. Fallback: parse only years that work; document which years were skipped in STATUS.md
- **-0.5**: NWP backfill for 11 years might hit Open-Meteo rate limits or timeouts (rare but possible). Fallback: backfill in 2-3 year chunks via multiple calls to `get_weather()`

Mitigation for the hard path:
- If the parser stalls on older years (Task 7 smoke fails), the execution agent should narrow to 2019-2024 (6 years) and proceed. 6 years is still better than Spain's 4.
- If NWP backfill is slow, proceed with smaller window; `demand_baseline` and `demand_enriched` don't need NWP, only `demand_full` does.
