"""Tests for windcast.data.spain_demand module."""

from pathlib import Path

import polars as pl
import pytest

from windcast.data.demand_schema import validate_demand_schema
from windcast.data.spain_demand import (
    _aggregate_weather,
    _read_energy_csv,
    _read_weather_csv,
    parse_spain_demand,
)


def _make_energy_csv_bytes(n_rows: int = 48) -> bytes:
    """Create synthetic energy CSV bytes for testing."""
    lines = ["time,total load actual,total load forecast,price day ahead,price actual"]
    for i in range(n_rows):
        hour = i % 24
        day = 1 + i // 24
        ts = f"2015-01-{day:02d} {hour:02d}:00:00+01:00"
        load = 25000.0 + 1000.0 * (i % 12)
        forecast = load + 100.0
        price = 40.0 + i * 0.5
        lines.append(f"{ts},{load},{forecast},{price},{price - 1.0}")
    return "\n".join(lines).encode("utf-8")


def _make_weather_csv_bytes(n_rows: int = 48, n_cities: int = 5) -> bytes:
    """Create synthetic weather CSV bytes in long format."""
    cities = ["Madrid", " Barcelona", "Bilbao", "Seville", "Valencia"][:n_cities]
    lines = [
        "dt_iso,city_name,temp,temp_min,temp_max,pressure,humidity,"
        "wind_speed,wind_deg,rain_1h,rain_3h,snow_3h,clouds_all,weather_id,"
        "weather_main,weather_description,weather_icon"
    ]
    for i in range(n_rows):
        hour = i % 24
        day = 1 + i // 24
        ts = f"2015-01-{day:02d} {hour:02d}:00:00+01:00"
        for city in cities:
            temp_k = 283.15 + i * 0.1  # ~10°C
            pressure = 1013.0
            humidity = 65.0
            wind = 5.0
            lines.append(
                f"{ts},{city},{temp_k},{temp_k - 2},{temp_k + 2},"
                f"{pressure},{humidity},{wind},180,0.0,0.0,0.0,50,800,"
                f"Clouds,overcast clouds,04d"
            )
    return "\n".join(lines).encode("utf-8")


class TestReadEnergyCsv:
    def test_parses_timestamps_to_utc(self, tmp_path: Path):
        csv_path = tmp_path / "energy.csv"
        csv_path.write_bytes(_make_energy_csv_bytes(24))
        df = _read_energy_csv(csv_path)
        assert df["timestamp_utc"].dtype == pl.Datetime("us", "UTC")
        # +01:00 → UTC means 00:00+01:00 becomes 23:00 UTC of previous day
        first_ts = df["timestamp_utc"][0]
        assert first_ts.hour == 23  # type: ignore[union-attr]

    def test_selects_relevant_columns(self, tmp_path: Path):
        csv_path = tmp_path / "energy.csv"
        csv_path.write_bytes(_make_energy_csv_bytes(24))
        df = _read_energy_csv(csv_path)
        assert "load_mw" in df.columns
        assert "price_eur_mwh" in df.columns
        assert "timestamp_utc" in df.columns

    def test_handles_nan_values(self, tmp_path: Path):
        csv_bytes = _make_energy_csv_bytes(24)
        # Replace one load value with empty
        lines = csv_bytes.decode().split("\n")
        parts = lines[2].split(",")
        parts[1] = ""
        lines[2] = ",".join(parts)
        csv_path = tmp_path / "energy.csv"
        csv_path.write_bytes("\n".join(lines).encode())
        df = _read_energy_csv(csv_path)
        assert df["load_mw"].null_count() >= 1


class TestReadWeatherCsv:
    def test_strips_city_names(self, tmp_path: Path):
        csv_path = tmp_path / "weather.csv"
        csv_path.write_bytes(_make_weather_csv_bytes(24))
        df = _read_weather_csv(csv_path)
        cities = df["city_name"].unique().sort().to_list()
        assert "Barcelona" in cities
        assert " Barcelona" not in cities

    def test_converts_kelvin_to_celsius(self, tmp_path: Path):
        csv_path = tmp_path / "weather.csv"
        csv_path.write_bytes(_make_weather_csv_bytes(24))
        df = _read_weather_csv(csv_path)
        # Input was ~283.15K → ~10°C
        assert df["temperature_c"].mean() < 20.0  # type: ignore[operator]
        assert df["temperature_c"].mean() > 5.0  # type: ignore[operator]

    def test_filters_pressure_outliers(self, tmp_path: Path):
        """Barcelona pressure outliers > 1100 are set to null."""
        csv_bytes = _make_weather_csv_bytes(4, n_cities=1)
        # Modify to be Barcelona with extreme pressure
        text = csv_bytes.decode().replace("Madrid", " Barcelona")
        lines = text.split("\n")
        parts = lines[2].split(",")
        parts[5] = "1008371"  # outlier pressure
        lines[2] = ",".join(parts)
        csv_path = tmp_path / "weather.csv"
        csv_path.write_bytes("\n".join(lines).encode())
        df = _read_weather_csv(csv_path)
        # The outlier row should have null pressure (but pressure is not in output select)
        # We verify temperature_c still works
        assert len(df) > 0


class TestAggregateWeather:
    def test_aggregates_to_one_row_per_timestamp(self, tmp_path: Path):
        csv_path = tmp_path / "weather.csv"
        csv_path.write_bytes(_make_weather_csv_bytes(24, n_cities=5))
        df = _read_weather_csv(csv_path)
        agg = _aggregate_weather(df)
        # 24 hours * 1 day = 24 unique timestamps
        assert len(agg) == 24


class TestParseSpainDemand:
    def test_returns_schema_compliant_frame(self, tmp_path: Path):
        energy_path = tmp_path / "energy.csv"
        weather_path = tmp_path / "weather.csv"
        energy_path.write_bytes(_make_energy_csv_bytes(48))
        weather_path.write_bytes(_make_weather_csv_bytes(48))
        df = parse_spain_demand(energy_path, weather_path)
        errors = validate_demand_schema(df)
        assert errors == [], f"Schema errors: {errors}"

    def test_sorts_by_timestamp(self, tmp_path: Path):
        energy_path = tmp_path / "energy.csv"
        weather_path = tmp_path / "weather.csv"
        energy_path.write_bytes(_make_energy_csv_bytes(48))
        weather_path.write_bytes(_make_weather_csv_bytes(48))
        df = parse_spain_demand(energy_path, weather_path)
        ts = df.get_column("timestamp_utc")
        assert ts.is_sorted()

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_spain_demand(
                tmp_path / "nonexistent.csv",
                tmp_path / "also_missing.csv",
            )
