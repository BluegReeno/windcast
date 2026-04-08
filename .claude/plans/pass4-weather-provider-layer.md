# Feature: Weather Provider Layer (Pass 4)

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Build a pluggable weather data provider so the framework can fetch, cache, and serve NWP data for any domain/site. This is the "Wx sources" layer that WN has (ECMWF, ICON, AROME) — EnerCast uses Open-Meteo as free equivalent.

The layer provides: a `WeatherProvider` protocol, an Open-Meteo implementation, a SQLite data cache for temporal queries, and a declarative `WeatherConfig` registry per dataset.

## User Story

As an energy forecasting engineer
I want NWP weather data automatically fetched, cached, and queryable for any site
So that I can train models with future weather information without manual data wrangling

## Problem Statement

Current wind models only use SCADA lags (rearview mirror) — skill scores are 0.10-0.20. NWP data is the #1 lever for improvement, especially at longer horizons (h12-h48). The framework has no weather data layer — `open_meteo.py` exists but is a raw fetch function with HTTP-level caching only, not queryable by date range.

## Solution Statement

Create a `src/windcast/weather/` package with:
1. `WeatherProvider` protocol — abstract interface for any weather data source
2. `OpenMeteoProvider` — Open-Meteo Archive API implementation (reuses existing `open_meteo.py` patterns)
3. `WeatherStorage` — SQLite-based data cache with upsert, temporal range queries, gap detection
4. `WeatherConfig` registry — declarative per-dataset config (variables, locations)
5. Public API: `get_weather(dataset_id, start, end) -> pl.DataFrame`

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: `src/windcast/weather/` (new), `src/windcast/config.py` (minor update)
**Dependencies**: `openmeteo_requests`, `requests_cache`, `retry_requests`, `polars` (all already in pyproject.toml), Python stdlib `sqlite3`

---

## CONTEXT REFERENCES

### Relevant Codebase Files — MUST READ BEFORE IMPLEMENTING

- `src/windcast/data/open_meteo.py` (lines 1-99) — **Existing Open-Meteo client**. Reuse `build_client()`, `WIND_VARIABLES`, and `fetch_historical_weather()` patterns. The new provider wraps this.
- `src/windcast/config.py` (lines 10-104) — **Dataset configs with lat/lon**. `DatasetConfig`, `DemandDatasetConfig`, `SolarDatasetConfig` all have `latitude`/`longitude`. The `DATASETS` dict is the lookup. `WeatherConfig` will reference these coords.
- `src/windcast/features/registry.py` (lines 1-50) — **Pattern to follow for registry**. `FeatureSet` dataclass + `FEATURE_REGISTRY` dict + `get_feature_set()` lookup. Mirror this pattern for `WeatherConfig`.
- `tests/data/test_open_meteo.py` (lines 1-97) — **Test patterns**: mock `openmeteo_requests.Client`, mock response chain (`Variables(i).ValuesAsNumpy()`), verify Polars DataFrame output with UTC timestamps.
- `src/windcast/features/wind.py` (lines 46-58) — Shows where NWP columns are expected: `wind_full` feature set needs `nwp_wind_speed_100m`, `nwp_wind_direction_100m`, `nwp_temperature_2m` columns.

### New Files to Create

- `src/windcast/weather/__init__.py` — Public API: `get_weather()`, `WeatherConfig`
- `src/windcast/weather/provider.py` — `WeatherProvider` protocol + `OpenMeteoProvider`
- `src/windcast/weather/storage.py` — `WeatherStorage` class (SQLite cache)
- `src/windcast/weather/registry.py` — `WeatherConfig` dataclass + per-dataset configs + registry
- `tests/weather/__init__.py` — Test package
- `tests/weather/test_provider.py` — Provider tests (mocked API)
- `tests/weather/test_storage.py` — Storage tests (real SQLite, temp files)
- `tests/weather/test_registry.py` — Registry tests

### Relevant Documentation

**Open-Meteo Archive API:**
- Endpoint: `https://archive-api.open-meteo.com/v1/archive`
- Coverage: ERA5 back to 1940, higher-res from 2017
- Key wind vars: `wind_speed_100m`, `wind_direction_100m`, `temperature_2m`, `pressure_msl`
- ERA5 has ~5-day lag — guard against requesting near-today dates
- No API key needed. No documented rate limit but 429 possible.
- `openmeteo_requests.Client.weather_api()` returns list of responses. `.Variables(i)` is positional — order must match request.

**SQLite with Polars:**
- `pl.read_database(query, connection)` works with stdlib `sqlite3.Connection` — no SQLAlchemy needed
- Upsert: `INSERT OR REPLACE INTO ... VALUES (?,?,?,?)` on primary key

### Patterns to Follow

**Registry pattern** (from `features/registry.py`):
```python
@dataclass(frozen=True)
class WeatherConfig:
    name: str
    variables: list[str]
    ...

WEATHER_REGISTRY: dict[str, WeatherConfig] = { "kelmarsh": ..., }
def get_weather_config(name: str) -> WeatherConfig: ...
```

**Naming conventions:**
- Files: `snake_case.py`
- Classes: `PascalCase` (e.g., `WeatherProvider`, `WeatherStorage`, `OpenMeteoProvider`)
- Functions: `snake_case` (e.g., `get_weather`, `fetch_range`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `WEATHER_REGISTRY`)

**Error handling:** Fail fast with descriptive messages (project principle). Use `assert` for internal invariants, `ValueError` for bad user input.

**Logging:** `logger = logging.getLogger(__name__)` at module top. `logger.info()` for progress, `logger.warning()` for skipped data.

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — Storage + Config

Build the SQLite storage and declarative config first (no API calls yet).

### Phase 2: Provider — Open-Meteo Implementation

Wrap the existing `open_meteo.py` fetch function behind the `WeatherProvider` protocol. Add gap-detection logic to fetch only missing date ranges.

### Phase 3: Public API — `get_weather()`

Wire provider + storage + config into a single function that fetches, caches, and returns weather data.

### Phase 4: Tests

Unit tests for storage, provider (mocked), registry, and integration test for `get_weather()`.

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/windcast/weather/registry.py`

Declarative weather config per dataset. Mirrors `features/registry.py` pattern.

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class WeatherConfig:
    """Weather data configuration for a dataset/site."""
    name: str
    latitude: float
    longitude: float
    variables: list[str]
    description: str = ""

KELMARSH_WEATHER = WeatherConfig(
    name="kelmarsh",
    latitude=52.4016,
    longitude=-0.9436,
    variables=["wind_speed_100m", "wind_direction_100m", "wind_speed_10m",
               "wind_direction_10m", "temperature_2m", "pressure_msl"],
    description="Kelmarsh wind farm — 6 NWP variables for wind forecasting",
)

SPAIN_WEATHER = WeatherConfig(
    name="spain_demand",
    latitude=40.4168,
    longitude=-3.7038,
    variables=["temperature_2m", "wind_speed_10m", "relative_humidity_2m"],
    description="Spain (Madrid) — weather for demand forecasting",
)

PVDAQ_WEATHER = WeatherConfig(
    name="pvdaq_system4",
    latitude=39.7407,
    longitude=-105.1686,
    variables=["shortwave_radiation", "temperature_2m", "wind_speed_10m",
               "cloud_cover"],
    description="PVDAQ System 4 (Golden CO) — weather for solar forecasting",
)

WEATHER_REGISTRY: dict[str, WeatherConfig] = {
    "kelmarsh": KELMARSH_WEATHER,
    "spain_demand": SPAIN_WEATHER,
    "pvdaq_system4": PVDAQ_WEATHER,
}

def get_weather_config(name: str) -> WeatherConfig: ...
def list_weather_configs() -> list[str]: ...
```

- **PATTERN**: Mirror `features/registry.py:1-30` — frozen dataclass + dict + lookup function
- **IMPORTS**: `dataclasses`
- **GOTCHA**: Coordinates MUST match `config.py` KELMARSH/SPAIN_DEMAND/PVDAQ_SYSTEM4
- **VALIDATE**: `uv run pyright src/windcast/weather/registry.py`

### Task 2: CREATE `src/windcast/weather/storage.py`

SQLite data cache with upsert and temporal queries.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS weather_data (
    location_key TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,  -- ISO 8601
    variable TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (location_key, timestamp_utc, variable)
);
```

**Class:**
```python
class WeatherStorage:
    def __init__(self, db_path: Path):
        """Open or create SQLite database."""

    def upsert(self, location_key: str, df: pl.DataFrame) -> int:
        """Insert/replace weather data from a Polars DataFrame.
        df must have columns: timestamp_utc + variable columns (wide format).
        Melts to long format internally. Returns row count inserted."""

    def query(self, location_key: str, start: str, end: str,
              variables: list[str] | None = None) -> pl.DataFrame:
        """Query weather data for a location and date range.
        Returns wide-format DataFrame: timestamp_utc + one column per variable."""

    def get_coverage(self, location_key: str) -> tuple[str, str] | None:
        """Return (min_date, max_date) ISO strings for a location, or None if empty."""

    def close(self) -> None: ...
```

- **IMPORTS**: `sqlite3`, `pathlib.Path`, `polars`
- **GOTCHA**: Use `pl.DataFrame.unpivot()` (not `melt` — deprecated in Polars >=1.0) to go wide→long. Use `pl.DataFrame.pivot()` to go long→wide on query.
- **GOTCHA**: Timestamps must be stored as ISO strings (TEXT) for SQLite compatibility and human readability. Parse back to `pl.Datetime("us", "UTC")` on query.
- **GOTCHA**: `INSERT OR REPLACE` for upsert on the composite primary key.
- **VALIDATE**: `uv run pyright src/windcast/weather/storage.py`

### Task 3: CREATE `src/windcast/weather/provider.py`

`WeatherProvider` protocol + `OpenMeteoProvider` implementation.

```python
from typing import Protocol

class WeatherProvider(Protocol):
    """Protocol for weather data providers."""
    def fetch(self, latitude: float, longitude: float,
              start_date: str, end_date: str,
              variables: list[str]) -> pl.DataFrame:
        """Fetch hourly weather data. Returns DataFrame with timestamp_utc + variable columns."""
        ...

class OpenMeteoProvider:
    """Open-Meteo Archive API provider."""

    def __init__(self, cache_dir: str = ".cache"):
        self._client = build_client(cache_dir=cache_dir)

    def fetch(self, latitude: float, longitude: float,
              start_date: str, end_date: str,
              variables: list[str]) -> pl.DataFrame:
        """Fetch from Open-Meteo Archive. Delegates to existing fetch_historical_weather()."""
        return fetch_historical_weather(
            latitude=latitude, longitude=longitude,
            start_date=start_date, end_date=end_date,
            variables=variables, client=self._client,
        )
```

- **IMPORTS**: `from windcast.data.open_meteo import build_client, fetch_historical_weather`
- **PATTERN**: Reuse existing `open_meteo.py` entirely — provider is a thin wrapper
- **GOTCHA**: Don't duplicate fetch logic. The existing `fetch_historical_weather` already handles the full API dance.
- **VALIDATE**: `uv run pyright src/windcast/weather/provider.py`

### Task 4: CREATE `src/windcast/weather/__init__.py`

Public API that wires everything together.

```python
"""Weather data layer — fetch, cache, and serve NWP data for any site."""

import logging
from pathlib import Path

import polars as pl

from windcast.weather.provider import OpenMeteoProvider, WeatherProvider
from windcast.weather.registry import (
    WeatherConfig,
    get_weather_config,
    list_weather_configs,
)
from windcast.weather.storage import WeatherStorage

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/weather.db")


def get_weather(
    config_name: str,
    start_date: str,
    end_date: str,
    db_path: Path = DEFAULT_DB_PATH,
    provider: WeatherProvider | None = None,
) -> pl.DataFrame:
    """Fetch weather data for a registered config, with SQLite caching.

    1. Check SQLite cache for existing coverage
    2. Fetch only missing date ranges from provider
    3. Upsert new data into cache
    4. Return full requested range from cache

    Args:
        config_name: Registered weather config name (e.g., "kelmarsh")
        start_date: Start date ISO "YYYY-MM-DD"
        end_date: End date ISO "YYYY-MM-DD"
        db_path: Path to SQLite cache file
        provider: Weather data provider. Defaults to OpenMeteoProvider.

    Returns:
        Polars DataFrame: timestamp_utc + weather variable columns (hourly)
    """
```

**Logic for gap-detection:**
1. `storage.get_coverage(location_key)` → existing `(min, max)` or `None`
2. If `None` → fetch full range
3. If existing range covers request → query from cache only
4. If gaps exist (request starts before cache, or ends after cache) → fetch missing segments, upsert
5. Return `storage.query(location_key, start_date, end_date)`

The `location_key` is derived from config: `f"{config.latitude}_{config.longitude}"` (simple, deterministic).

- **IMPORTS**: All from submodules
- **GOTCHA**: Create `db_path.parent` directory if it doesn't exist (`db_path.parent.mkdir(parents=True, exist_ok=True)`)
- **GOTCHA**: ERA5 has ~5-day lag. If `end_date` is within 5 days of today, clamp to `today - 5 days` and log a warning.
- **VALIDATE**: `uv run pyright src/windcast/weather/__init__.py`

### Task 5: CREATE `tests/weather/__init__.py`

Empty `__init__.py` for test package.

### Task 6: CREATE `tests/weather/test_registry.py`

```python
def test_kelmarsh_weather_config_exists(): ...
def test_kelmarsh_coords_match_dataset_config(): ...  # Cross-check with config.py KELMARSH
def test_get_weather_config_unknown_raises(): ...
def test_list_weather_configs(): ...
```

- **PATTERN**: Follow `tests/data/test_open_meteo.py` style
- **VALIDATE**: `uv run pytest tests/weather/test_registry.py -v`

### Task 7: CREATE `tests/weather/test_storage.py`

```python
import tempfile
from pathlib import Path
import polars as pl
from windcast.weather.storage import WeatherStorage

def test_upsert_and_query_roundtrip():
    """Insert weather data, query it back, verify values match."""

def test_upsert_is_idempotent():
    """Inserting same data twice doesn't duplicate rows."""

def test_query_empty_returns_empty_df():
    """Query on empty DB returns empty DataFrame."""

def test_get_coverage_empty():
    """Coverage on empty DB returns None."""

def test_get_coverage_returns_range():
    """Coverage returns (min, max) dates after insert."""

def test_query_filters_by_date_range():
    """Query only returns rows within requested range."""
```

- **GOTCHA**: Use `tempfile.TemporaryDirectory()` for SQLite files — clean up after tests
- **VALIDATE**: `uv run pytest tests/weather/test_storage.py -v`

### Task 8: CREATE `tests/weather/test_provider.py`

```python
from unittest.mock import MagicMock
from windcast.weather.provider import OpenMeteoProvider

def test_open_meteo_provider_implements_protocol():
    """Verify OpenMeteoProvider satisfies WeatherProvider protocol."""

def test_open_meteo_provider_returns_dataframe():
    """Mock the underlying client and verify DataFrame output."""
    # Reuse mock pattern from tests/data/test_open_meteo.py
```

- **PATTERN**: Mirror mock setup from `tests/data/test_open_meteo.py:36-65`
- **VALIDATE**: `uv run pytest tests/weather/test_provider.py -v`

### Task 9: Run full validation

- **VALIDATE**: `uv run ruff check src/windcast/weather/ tests/weather/`
- **VALIDATE**: `uv run ruff format --check src/windcast/weather/ tests/weather/`
- **VALIDATE**: `uv run pyright src/windcast/weather/`
- **VALIDATE**: `uv run pytest tests/weather/ -v`
- **VALIDATE**: `uv run pytest tests/ -v` (full suite — no regressions)

---

## TESTING STRATEGY

### Unit Tests

| Test file | What it covers |
|-----------|---------------|
| `tests/weather/test_registry.py` | Config lookup, unknown key raises, coords consistency |
| `tests/weather/test_storage.py` | SQLite roundtrip, upsert idempotency, date range filtering, empty DB |
| `tests/weather/test_provider.py` | Protocol compliance, mocked API fetch |

### Integration Test (optional, requires network)

Not implemented in this pass. `get_weather("kelmarsh", "2020-01-01", "2020-01-07")` with real API = Pass 5 responsibility.

### Edge Cases

- Empty database → fetch full range
- Request fully covered by cache → no API call
- Request partially overlaps cache → fetch only missing segments
- ERA5 lag → clamp end_date, log warning
- Unknown config name → `ValueError`

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/windcast/weather/ tests/weather/
uv run ruff format --check src/windcast/weather/ tests/weather/
uv run pyright src/windcast/weather/
```

### Level 2: Unit Tests

```bash
uv run pytest tests/weather/ -v
```

### Level 3: Full Regression

```bash
uv run pytest tests/ -v
```

### Level 4: Manual Validation (Pass 5)

```bash
# This is for Pass 5, not Pass 4 — but confirms the layer works
uv run python -c "
from windcast.weather import get_weather
df = get_weather('kelmarsh', '2020-01-01', '2020-01-07')
print(df.shape, df.columns)
print(df.head())
"
```

**Expected**: All Level 1-3 pass with exit code 0. Level 4 deferred to Pass 5.

---

## ACCEPTANCE CRITERIA

- [ ] `WeatherProvider` protocol defined with `fetch()` method
- [ ] `OpenMeteoProvider` wraps existing `open_meteo.py` — no duplicate fetch logic
- [ ] `WeatherConfig` for Kelmarsh, Spain, PVDAQ registered with correct coordinates
- [ ] `WeatherStorage` SQLite works: upsert → query roundtrip preserves data
- [ ] `get_weather("kelmarsh", start, end)` returns hourly Polars DataFrame with correct columns
- [ ] Gap detection: only fetches missing date ranges, not full range every time
- [ ] ERA5 lag guard: clamps end_date if within 5 days of today
- [ ] Tests pass: `uv run pytest tests/weather/ -v`
- [ ] No regressions: `uv run pytest tests/ -v` (all 234+ tests pass)
- [ ] Lint + type clean: `ruff check`, `ruff format --check`, `pyright` all pass

---

## COMPLETION CHECKLIST

- [ ] Task 1: `weather/registry.py` created, pyright clean
- [ ] Task 2: `weather/storage.py` created, pyright clean
- [ ] Task 3: `weather/provider.py` created, pyright clean
- [ ] Task 4: `weather/__init__.py` created with `get_weather()`, pyright clean
- [ ] Task 5-8: All tests created and passing
- [ ] Task 9: Full validation suite green (ruff + pyright + pytest)
- [ ] All acceptance criteria met

---

## NOTES

**Design decisions:**
- **SQLite for data cache** (not just HTTP cache): Enables temporal queries, gap detection, and offline use. Same schema can migrate to PostgreSQL/Supabase for production.
- **Protocol for provider** (not ABC): Lighter, more Pythonic. Allows duck typing for testing.
- **Wide format in/out, long format in storage**: Wide format is what Polars users expect. Long format in SQLite is more flexible (variable list can change without schema migration).
- **Reuse existing `open_meteo.py`**: No code duplication. Provider delegates to the existing function.
- **`location_key` = `lat_lon`**: Simple, deterministic. No separate location table needed.

**What this does NOT do (deferred to Pass 5):**
- Fetch real weather data for Kelmarsh (requires network)
- Join weather data with SCADA features
- Update `build_features.py` with `--weather` flag
- Train `wind_full` with NWP features

**Confidence: 9/10** — All patterns are well-established in the codebase, no new dependencies, straightforward SQLite + Polars work.
