"""Tests for windcast.data.kelmarsh parser."""

import io
import zipfile

import polars as pl
import pytest

from windcast.data.kelmarsh import (
    _extract_turbine_id,
    _read_turbine_csv,
    parse_kelmarsh,
)
from windcast.data.schema import validate_schema


def _make_csv_bytes(
    n_rows: int = 10,
    turbine_num: int = 1,
    power_scale: float = 1.0,
) -> bytes:
    """Create synthetic CSV bytes matching Kelmarsh format."""
    rows = []
    header = (
        "# Date and time,"
        "Wind speed (m/s),"
        "Wind direction (°),"
        "Power (kW),"
        "Nacelle position (°),"
        "Rotor speed (rpm),"
        "Blade angle (pitch position) A (°),"
        "Blade angle (pitch position) B (°),"
        "Blade angle (pitch position) C (°),"
        "Nacelle ambient temperature (°C),"
        "Nacelle temperature (°C)"
    )
    rows.append(header)

    for i in range(n_rows):
        ts = f"2021-01-01 {i:02d}:{(i * 10) % 60:02d}:00"
        ws = 8.0 + i * 0.5
        wd = 180.0 + i
        power = 1000.0 * power_scale + i * 50
        nacelle = 179.0 + i
        rpm = 12.0 + i * 0.1
        pitch_a = 2.0 + i * 0.1
        pitch_b = 2.1 + i * 0.1
        pitch_c = 2.2 + i * 0.1
        ambient = 10.0
        nacelle_temp = 35.0
        rows.append(
            f"{ts},{ws},{wd},{power},{nacelle},{rpm},"
            f"{pitch_a},{pitch_b},{pitch_c},{ambient},{nacelle_temp}"
        )

    return "\n".join(rows).encode("utf-8")


def _make_zip(csv_bytes_dict: dict[str, bytes]) -> bytes:
    """Create a ZIP archive from a dict of filename → bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in csv_bytes_dict.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_nested_zip(inner_zips: dict[str, bytes]) -> bytes:
    """Create an outer ZIP containing inner ZIPs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as outer:
        for name, inner_data in inner_zips.items():
            outer.writestr(name, inner_data)
    return buf.getvalue()


class TestExtractTurbineId:
    def test_standard_filename(self):
        fname = "Turbine_Data_Kelmarsh_1_2020-01-01_-_2020-12-31_1234.csv"
        assert _extract_turbine_id(fname) == "KWF1"

    def test_turbine_6(self):
        fname = "Turbine_Data_Kelmarsh_6_2021-01-01_-_2021-12-31_5678.csv"
        assert _extract_turbine_id(fname) == "KWF6"

    def test_invalid_filename(self):
        with pytest.raises(ValueError, match="Cannot extract"):
            _extract_turbine_id("random_file.csv")


class TestReadTurbineCsv:
    def test_produces_canonical_schema(self):
        csv_bytes = _make_csv_bytes(n_rows=5)
        df = _read_turbine_csv(csv_bytes, "KWF1")
        errors = validate_schema(df)
        assert errors == [], f"Schema errors: {errors}"

    def test_correct_row_count(self):
        csv_bytes = _make_csv_bytes(n_rows=10)
        df = _read_turbine_csv(csv_bytes, "KWF1")
        assert len(df) == 10

    def test_pitch_angle_averaged(self):
        csv_bytes = _make_csv_bytes(n_rows=3)
        df = _read_turbine_csv(csv_bytes, "KWF1")
        # Pitch A=2.0, B=2.1, C=2.2 for first row → avg=2.1
        assert abs(df["pitch_angle_deg"][0] - 2.1) < 0.01

    def test_timestamp_is_utc(self):
        csv_bytes = _make_csv_bytes(n_rows=1)
        df = _read_turbine_csv(csv_bytes, "KWF1")
        assert df["timestamp_utc"].dtype == pl.Datetime("us", "UTC")

    def test_turbine_id_set(self):
        csv_bytes = _make_csv_bytes(n_rows=1)
        df = _read_turbine_csv(csv_bytes, "KWF3")
        assert df["turbine_id"][0] == "KWF3"

    def test_dataset_id_set(self):
        csv_bytes = _make_csv_bytes(n_rows=1)
        df = _read_turbine_csv(csv_bytes, "KWF1")
        assert df["dataset_id"][0] == "kelmarsh"

    def test_power_conversion_when_low(self):
        """If max power < 500, should multiply by 6 (kWh → kW)."""
        csv_bytes = _make_csv_bytes(n_rows=5, power_scale=0.05)
        df = _read_turbine_csv(csv_bytes, "KWF1")
        # Original: 50 + i*50 → max=250. After *6: max=1500
        assert df["active_power_kw"].max() > 300


class TestParseKelmarsh:
    def test_parse_from_zip(self, tmp_path):
        """Parse a simple ZIP with one turbine CSV."""
        csv_bytes = _make_csv_bytes(n_rows=5, turbine_num=1)
        inner_zip = _make_zip(
            {
                "Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-12-31_001.csv": csv_bytes,
            }
        )
        outer_zip_bytes = _make_nested_zip(
            {
                "Kelmarsh_SCADA_2021.zip": inner_zip,
            }
        )

        zip_path = tmp_path / "kelmarsh.zip"
        zip_path.write_bytes(outer_zip_bytes)

        df = parse_kelmarsh(zip_path)
        assert len(df) == 5
        errors = validate_schema(df)
        assert errors == []

    def test_parse_multiple_turbines(self, tmp_path):
        """Parse ZIP with multiple turbine CSVs."""
        csv1 = _make_csv_bytes(n_rows=3, turbine_num=1)
        csv2 = _make_csv_bytes(n_rows=3, turbine_num=2)
        inner_zip = _make_zip(
            {
                "Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-12-31_001.csv": csv1,
                "Turbine_Data_Kelmarsh_2_2021-01-01_-_2021-12-31_002.csv": csv2,
            }
        )
        outer_zip_bytes = _make_nested_zip(
            {
                "Kelmarsh_SCADA_2021.zip": inner_zip,
            }
        )

        zip_path = tmp_path / "kelmarsh.zip"
        zip_path.write_bytes(outer_zip_bytes)

        df = parse_kelmarsh(zip_path)
        assert len(df) == 6
        assert df["turbine_id"].n_unique() == 2

    def test_parse_from_directory(self, tmp_path):
        """Parse from extracted CSV directory."""
        csv_bytes = _make_csv_bytes(n_rows=4)
        csv_path = tmp_path / "Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-12-31_001.csv"
        csv_path.write_bytes(csv_bytes)

        df = parse_kelmarsh(tmp_path)
        assert len(df) == 4

    def test_missing_path_raises(self, tmp_path):
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_kelmarsh(tmp_path / "nonexistent.zip")

    def test_empty_directory_raises(self, tmp_path):
        """Empty directory raises FileNotFoundError."""
        (tmp_path / "subdir").mkdir()
        with pytest.raises(FileNotFoundError):
            parse_kelmarsh(tmp_path / "subdir")
