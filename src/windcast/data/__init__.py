"""Data ingestion and quality control modules."""

from windcast.data.schema import SCADA_SCHEMA, validate_schema

__all__ = ["SCADA_SCHEMA", "validate_schema"]
