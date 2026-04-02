"""Tests for windcast.data.pvdaq module."""

from pathlib import Path

import polars as pl
import pytest

from windcast.data.pvdaq import parse_pvdaq
from windcast.data.solar_schema import validate_solar_schema


def _make_pvdaq_csv_bytes(n_rows: int = 30, start_hour: int = 10) -> bytes:
    """Create synthetic PVDAQ System 4 CSV for testing (1-min resolution).

    Args:
        n_rows: Number of 1-minute rows.
        start_hour: Local hour to start (Denver time).
    """
    lines = ["measured_on,ac_power__315,poa_irradiance__313,ambient_temp__320,module_temp_1__321"]
    for i in range(n_rows):
        minute = i % 60
        hour = start_hour + i // 60
        ts = f"2020-06-15 {hour:02d}:{minute:02d}:00"
        power_w = 1500.0 + 100.0 * (i % 10)  # Watts
        poa = 600.0 + 10.0 * (i % 5)
        ambient = 25.0 + 0.1 * i
        module = 35.0 + 0.2 * i
        lines.append(f"{ts},{power_w},{poa},{ambient},{module}")
    return "\n".join(lines).encode("utf-8")


def _write_pvdaq_csv(tmp_path: Path, n_rows: int = 30) -> Path:
    """Write synthetic CSV to tmp_path and return the directory."""
    data_dir = tmp_path / "pvdaq"
    data_dir.mkdir()
    csv_path = data_dir / "system_4__date_2020_06_15.csv"
    csv_path.write_bytes(_make_pvdaq_csv_bytes(n_rows))
    return data_dir


class TestParsePvdaq:
    def test_parses_timestamps_to_utc(self, tmp_path: Path):
        data_dir = _write_pvdaq_csv(tmp_path)
        df = parse_pvdaq(data_dir)
        assert df["timestamp_utc"].dtype == pl.Datetime("us", "UTC")
        # Denver is UTC-6 in summer, so 10:00 local → 16:00 UTC
        first_ts = df.sort("timestamp_utc")["timestamp_utc"][0]
        assert first_ts.hour == 16  # type: ignore[union-attr]

    def test_power_converted_to_kw(self, tmp_path: Path):
        data_dir = _write_pvdaq_csv(tmp_path)
        df = parse_pvdaq(data_dir)
        # Input was 1500-2400 W → 1.5-2.4 kW
        assert df["power_kw"].max() < 5.0  # type: ignore[operator]
        assert df["power_kw"].min() >= 0.0  # type: ignore[operator]

    def test_negative_power_clipped(self, tmp_path: Path):
        data_dir = tmp_path / "pvdaq"
        data_dir.mkdir()
        # Create CSV with negative power
        lines = [
            "measured_on,ac_power__315,poa_irradiance__313,ambient_temp__320,module_temp_1__321",
            "2020-06-15 10:00:00,-50.0,0.0,20.0,25.0",
            "2020-06-15 10:01:00,-10.0,0.0,20.0,25.0",
        ]
        csv_path = data_dir / "system_4__date_2020_06_15.csv"
        csv_path.write_text("\n".join(lines))
        df = parse_pvdaq(data_dir)
        assert df["power_kw"].min() >= 0.0  # type: ignore[operator]

    def test_returns_schema_compliant_frame(self, tmp_path: Path):
        data_dir = _write_pvdaq_csv(tmp_path)
        df = parse_pvdaq(data_dir)
        errors = validate_solar_schema(df)
        assert errors == [], f"Schema errors: {errors}"

    def test_sorts_by_timestamp(self, tmp_path: Path):
        data_dir = _write_pvdaq_csv(tmp_path)
        df = parse_pvdaq(data_dir)
        ts = df.get_column("timestamp_utc")
        assert ts.is_sorted()

    def test_aggregation_reduces_rows(self, tmp_path: Path):
        # 30 rows at 1-min → 2 intervals at 15-min
        data_dir = _write_pvdaq_csv(tmp_path, n_rows=30)
        df = parse_pvdaq(data_dir)
        assert len(df) == 2  # 30 min = 2 x 15-min intervals

    def test_missing_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_pvdaq(tmp_path / "nonexistent")

    def test_year_filter(self, tmp_path: Path):
        data_dir = tmp_path / "pvdaq"
        data_dir.mkdir()
        # Write 2020 file
        (data_dir / "system_4__date_2020_06_15.csv").write_bytes(_make_pvdaq_csv_bytes(30))
        # Write 2019 file
        lines = [
            "measured_on,ac_power__315,poa_irradiance__313,ambient_temp__320,module_temp_1__321",
            "2019-06-15 10:00:00,1000.0,500.0,22.0,30.0",
        ]
        (data_dir / "system_4__date_2019_06_15.csv").write_text("\n".join(lines))

        # Filter to 2020 only
        df = parse_pvdaq(data_dir, year=2020)
        years = df["timestamp_utc"].dt.year().unique().to_list()
        assert 2019 not in years
