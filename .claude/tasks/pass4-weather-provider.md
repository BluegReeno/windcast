# Feature: Weather Provider Layer (Pass 4)

## Goal
Build a pluggable weather data provider with SQLite caching so the framework can fetch, cache, and serve NWP data for any site.

## Context
- **PRD Reference**: `.claude/PRD.md` — Wx sources layer
- **Plan**: `.claude/plans/pass4-weather-provider-layer.md`
- **Related Files**: `src/windcast/data/open_meteo.py`, `src/windcast/config.py`, `src/windcast/features/registry.py`

## Tasks

### Phase 1: Foundation — Storage + Config
- [x] Task 1: Create `src/windcast/weather/registry.py` — WeatherConfig dataclass + per-dataset configs ✓ 2026-04-08
- [x] Task 2: Create `src/windcast/weather/storage.py` — SQLite cache with upsert + temporal queries ✓ 2026-04-08

### Phase 2: Provider
- [x] Task 3: Create `src/windcast/weather/provider.py` — WeatherProvider protocol + OpenMeteoProvider ✓ 2026-04-08

### Phase 3: Public API
- [x] Task 4: Create `src/windcast/weather/__init__.py` — `get_weather()` with gap detection + caching ✓ 2026-04-08

### Phase 4: Tests
- [x] Task 5-8: Create tests (registry, storage, provider) ✓ 2026-04-08

### Validation
- [x] Task 9: Full validation — ruff + pyright + pytest (252 passed) ✓ 2026-04-08

## Files to Create

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/weather/__init__.py` | Create | Public API: `get_weather()` |
| `src/windcast/weather/registry.py` | Create | WeatherConfig + registry |
| `src/windcast/weather/storage.py` | Create | SQLite cache |
| `src/windcast/weather/provider.py` | Create | WeatherProvider protocol + OpenMeteo impl |
| `tests/weather/__init__.py` | Create | Test package |
| `tests/weather/test_registry.py` | Create | Registry tests |
| `tests/weather/test_storage.py` | Create | Storage tests |
| `tests/weather/test_provider.py` | Create | Provider tests |

## Notes
- Reuse `open_meteo.py` patterns — provider is a thin wrapper
- SQLite long format storage, wide format API
- ERA5 has ~5-day lag — clamp end_date

## Completion
- **Started**: 2026-04-08
- **Completed**: 2026-04-08
- **Commit**: (link to commit when done)
