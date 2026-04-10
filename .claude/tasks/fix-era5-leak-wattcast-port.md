# Feature: Fix ERA5 Leak — Port WattCast Historical Forecast Provider

## Goal
Add `HistoricalForecastProvider` + blend mode to EnerCast so val/test uses archived NWP
forecasts (Open-Meteo historical-forecast-api) while train still uses ERA5. Rebuild RTE
France + Kelmarsh features, retrain XGBoost/AutoGluon/TSO baseline, and publish honest
numbers in `docs/WNchallenge/post-era5-fix-results.md`.

## Context
- **Plan**: `.claude/plans/fix-era5-leak-wattcast-port.md`
- **Deadline**: 2026-04-14 (WeatherNews presentation)
- **WattCast reference**: `/Users/renaud/Projects/wattcast/src/wattcast/data/open_meteo.py:129-164`

## Tasks

### Phase 1 — Port weather code
- [x] 1.1 Read WattCast source files + delivery-time docs ✓ 2026-04-10
- [x] 1.2 Add `fetch_historical_forecast_weather` + URL constant to `src/windcast/data/open_meteo.py` ✓ 2026-04-10
- [x] 1.2b Add tests to `tests/data/test_open_meteo.py` (schema, URL guard, date range) ✓ 2026-04-10
- [x] 1.3 Add `HistoricalForecastProvider` class to `src/windcast/weather/provider.py` ✓ 2026-04-10
- [x] 1.3b Add tests to `tests/weather/test_provider.py` (protocol, distinctness) ✓ 2026-04-10
- [x] 1.4 Parameterise `WeatherStorage` db path + add `WEATHER_FORECAST_DB_PATH` constant ✓ 2026-04-10
- [x] 1.4b Update `tests/weather/test_storage.py` for db_path arg ✓ 2026-04-10
- [x] 1.5 Extend `src/windcast/weather/__init__.py` public API (`get_forecast_weather` helper) ✓ 2026-04-10
- [x] 1.6 Add `--weather-source {archive,historical_forecast,blend}` to `scripts/build_features.py` ✓ 2026-04-10
- [x] 1.6b Integration test for blend mode ✓ 2026-04-10
- [x] 1.7 Run full test suite + ruff + pyright — must be green ✓ 2026-04-10 (307 passing)

### Phase 2 — Fetch archived forecasts
- [x] 2.1 Populate `data/weather_forecast.db` for RTE 8 cities (2022-01-01 → 2024-12-31) ✓ 2026-04-10
- [x] 2.2 Populate `data/weather_forecast.db` for Kelmarsh (2022-01-01 → 2024-12-31) ✓ 2026-04-10
- [x] 2.3 Smoke test the new database (9 locations × 26,304 hours, 999K rows) ✓ 2026-04-10

### Phase 3 — Rebuild features, retrain, log
- [x] 3.1 Rebuild RTE France features with `--weather-source blend` ✓ 2026-04-10
- [x] 3.2 Rebuild Kelmarsh features with `--weather-source blend` ✓ 2026-04-10
- [x] 3.3 Retrain XGBoost on RTE (baseline/enriched/full) ✓ 2026-04-10
- [x] 3.4 Retrain XGBoost on Kelmarsh (baseline/enriched/full) ✓ 2026-04-10
- [x] 3.5 Re-run TSO baseline with new splits ✓ 2026-04-10
- [x] 3.6 Retrain AutoGluon on RTE `demand_full` (good_quality) ✓ 2026-04-10
- [x] 3.7 Retrain AutoGluon on Kelmarsh `wind_full` (best_quality) ✓ 2026-04-10
- [x] 3.8 Regenerate comparison charts ✓ 2026-04-10
- [x] 3.9 Write `docs/WNchallenge/post-era5-fix-results.md` ✓ 2026-04-10

### Phase 4 — Commit & handoff
- [x] 4.1 Final validation (pytest, ruff, pyright) ✓ 2026-04-10 (307 pass, all clean)
- [ ] @claude 4.2 Commits (atomic series) — ready for /commit
- [x] 4.3 Handoff note in results markdown ✓ 2026-04-10

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/windcast/data/open_meteo.py` | Modify | Add `fetch_historical_forecast_weather` + URL constant |
| `src/windcast/weather/provider.py` | Modify | Add `HistoricalForecastProvider` |
| `src/windcast/weather/storage.py` | Modify | (no change — already parameterised via `db_path`) |
| `src/windcast/weather/__init__.py` | Modify | Export new provider + forecast db path + `get_forecast_weather` |
| `scripts/build_features.py` | Modify | Add `--weather-source` flag + blend logic |
| `scripts/fetch_historical_forecasts.py` | Create | One-off script to populate `weather_forecast.db` |
| `tests/data/test_open_meteo.py` | Modify | Tests for new fetch function |
| `tests/weather/test_provider.py` | Modify | Tests for new provider |
| `tests/weather/test_storage.py` | Modify | Tests for db_path arg |
| `docs/WNchallenge/post-era5-fix-results.md` | Create | Results table + handoff |

## Notes
- Decision log fixed: two separate SQLite databases, verbatim port from WattCast
- Distribution shift (ERA5 train / forecast eval) is accepted and documented
- Kelmarsh single-location config; RTE uses `WeightedWeatherConfig` with 8 cities
- Existing `WeatherStorage.__init__` already accepts `db_path: Path` — Step 1.4 may be
  a no-op beyond exposing a constant for the forecast db path.

## Completion
- **Started**: 2026-04-10
- **Completed**: (fill when done)
- **Commit**: (link to commit when done)
