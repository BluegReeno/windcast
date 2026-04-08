"""Weather data SQLite cache — upsert, temporal queries, gap detection."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS weather_data (
    location_key TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    variable TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (location_key, timestamp_utc, variable)
);
"""


class WeatherStorage:
    """SQLite-based weather data cache with upsert and temporal queries."""

    def __init__(self, db_path: Path) -> None:
        """Open or create SQLite database at db_path."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def upsert(self, location_key: str, df: pl.DataFrame) -> int:
        """Insert/replace weather data from a wide-format Polars DataFrame.

        Args:
            location_key: Unique identifier for the location (e.g., "52.4016_-0.9436").
            df: DataFrame with columns: timestamp_utc + variable columns (wide format).

        Returns:
            Number of rows inserted.
        """
        if df.is_empty():
            return 0

        variable_cols = [c for c in df.columns if c != "timestamp_utc"]
        if not variable_cols:
            return 0

        # Wide → long format for storage
        long_df = df.unpivot(
            on=variable_cols,
            index="timestamp_utc",
            variable_name="variable",
            value_name="value",
        )

        # Convert timestamps to ISO strings
        rows = [
            (location_key, ts.isoformat(), var, val)
            for ts, var, val in zip(
                long_df["timestamp_utc"].to_list(),
                long_df["variable"].to_list(),
                long_df["value"].to_list(),
                strict=True,
            )
        ]

        self._conn.executemany(
            "INSERT OR REPLACE INTO weather_data (location_key, timestamp_utc, variable, value) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        logger.info("Upserted %d rows for location %s", len(rows), location_key)
        return len(rows)

    def query(
        self,
        location_key: str,
        start: str,
        end: str,
        variables: list[str] | None = None,
    ) -> pl.DataFrame:
        """Query weather data for a location and date range.

        Args:
            location_key: Location identifier.
            start: Start date ISO "YYYY-MM-DD".
            end: End date ISO "YYYY-MM-DD".
            variables: Optional filter on variable names.

        Returns:
            Wide-format DataFrame: timestamp_utc + one column per variable.
        """
        sql = (
            "SELECT timestamp_utc, variable, value FROM weather_data "
            "WHERE location_key = ? AND timestamp_utc >= ? AND timestamp_utc <= ?"
        )
        params: list[str] = [location_key, start, end + "T23:59:59"]

        if variables:
            placeholders = ",".join("?" for _ in variables)
            sql += f" AND variable IN ({placeholders})"
            params.extend(variables)

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        if not rows:
            return pl.DataFrame(schema={"timestamp_utc": pl.Datetime("us", "UTC")})

        result = pl.DataFrame(
            rows,
            schema={"timestamp_utc": pl.Utf8, "variable": pl.Utf8, "value": pl.Float64},
            orient="row",
        )

        # Long → wide format
        wide = result.pivot(on="variable", index="timestamp_utc", values="value")

        # Parse ISO strings back to proper datetimes
        wide = wide.with_columns(
            pl.col("timestamp_utc").str.to_datetime(time_zone="UTC").alias("timestamp_utc")
        ).sort("timestamp_utc")

        return wide

    def get_coverage(self, location_key: str) -> tuple[str, str] | None:
        """Return (min_date, max_date) ISO strings for a location, or None if empty."""
        cursor = self._conn.execute(
            "SELECT MIN(timestamp_utc), MAX(timestamp_utc) FROM weather_data "
            "WHERE location_key = ?",
            (location_key,),
        )
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return (row[0], row[1])

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
