# Polars Patterns — WindCast Reference

Polars version: 1.39.3 (as of project bootstrap). Python 3.12+.

---

## 1. Schema Definition

Define schemas as plain dicts mapping column names to Polars dtype objects. No special class needed.

```python
import polars as pl

# Canonical schema for a turbine SCADA file
SCADA_SCHEMA: dict[str, pl.PolarsDataType] = {
    "timestamp":     pl.String,      # parsed later to Datetime
    "wind_speed":    pl.Float64,
    "wind_direction": pl.Float64,
    "power_kw":      pl.Float64,
    "rotor_rpm":     pl.Float32,
    "ambient_temp":  pl.Float32,
    "status_code":   pl.Int32,
}
```

**Key rules:**
- Use `pl.String` for raw timestamp columns (parse explicitly after load).
- Use `pl.Float64` for any signal you will do math on.
- Use `pl.Float32` for lower-priority columns to halve memory.
- Use `pl.Int32` / `pl.Int16` for status/flag columns.

---

## 2. `pl.read_csv()` — Eager Load with Schema Enforcement

```python
df = pl.read_csv(
    "data/raw/kelmarsh/turbine_01.csv",
    schema=SCADA_SCHEMA,            # full schema: disables inference entirely
    null_values=["", "NA", "NaN", "N/A", "-999", "-9999"],
    infer_schema_length=None,       # ignored when schema= is set
)
```

**`schema` vs `schema_overrides`:**

| Parameter | Behaviour |
|-----------|-----------|
| `schema=` | Full schema. Pass all columns. Disables inference. Column order must match CSV. |
| `schema_overrides=` | Partial override. Inference runs for unspecified columns. Safer for wide CSVs. |

```python
# When you only want to pin a few columns
df = pl.read_csv(
    path,
    schema_overrides={"wind_speed": pl.Float64, "power_kw": pl.Float64},
    null_values=["", "NA", "N/A"],
)
```

**`infer_schema_length`:** Default is 100 rows. For noisy SCADA data, increase:
```python
pl.read_csv(path, infer_schema_length=10_000)
```
Set to `None` to scan entire file — slow but safest for mixed-type columns.

---

## 3. `pl.scan_csv()` — Lazy Load for Large Files (12+ GB)

```python
# Column rename mapping: native CSV name -> canonical name
COLUMN_MAP = {
    "ActivePower_kW": "power_kw",
    "WindSpeed_ms":   "wind_speed",
    "WindDir_deg":    "wind_direction",
    "AmbTemp_C":      "ambient_temp",
    "Timestamp":      "timestamp",
}

DTYPE_OVERRIDES = {
    "ActivePower_kW": pl.Float64,
    "WindSpeed_ms":   pl.Float64,
    "WindDir_deg":    pl.Float64,
    "AmbTemp_C":      pl.Float32,
}

lf: pl.LazyFrame = pl.scan_csv(
    "data/raw/kelmarsh/*.csv",       # glob supported
    schema_overrides=DTYPE_OVERRIDES,
    null_values=["", "NA", "NaN", "N/A", "-999", "-9999"],
    infer_schema_length=10_000,
    low_memory=False,                # False = faster; True = lower peak RAM
    rechunk=False,                   # rechunk at collect() instead
)

# All transformations are lazy until .collect()
lf = (
    lf
    .rename(COLUMN_MAP)
    .with_columns(
        pl.col("timestamp")
          .str.to_datetime("%Y-%m-%d %H:%M:%S")
          .alias("timestamp")
    )
    .filter(pl.col("power_kw") >= 0)   # drop negative power (sensor errors)
    .sort("timestamp")
)

df = lf.collect()
```

**`scan_csv` vs `read_csv` for large files:**
- `scan_csv` builds a lazy query plan — pushes down filters and projections before reading.
- Use `scan_csv` when the file is > 1 GB or you will filter to a subset of rows/columns.
- `low_memory=True` uses less peak RAM but is slower (processes in smaller batches).
- `rechunk=False` is faster; call `df.rechunk()` manually if you need contiguous memory.

**`scan_csv` extra parameters (v1.x):**
- `missing_columns="insert"` — tolerate CSV files with fewer columns than expected (useful for multi-turbine glob).
- `include_file_paths="source_file"` — adds a column with the source filename (good for multi-file glob).

```python
lf = pl.scan_csv(
    "data/raw/kelmarsh/*.csv",
    schema_overrides=DTYPE_OVERRIDES,
    missing_columns="insert",
    include_file_paths="source_file",
)
```

---

## 4. Timestamp Parsing and Timezone Handling

**Never use `try_parse_dates=True`** for production — it silently leaves unparseable columns as `String`.

Always parse timestamps explicitly:

```python
# SCADA timestamps are usually naive local time (e.g. Europe/London)
# Step 1: parse as naive datetime
# Step 2: attach source timezone
# Step 3: convert to UTC
df = df.with_columns(
    pl.col("timestamp")
      .str.to_datetime("%Y-%m-%d %H:%M:%S")          # format must match exactly
      .dt.replace_time_zone("Europe/London")           # declare source TZ (no conversion)
      .dt.convert_time_zone("UTC")                     # convert to UTC
      .alias("timestamp_utc")
)
```

**Hill of Towie — end-of-period shift:**
```python
# Timestamps are end-of-period: 00:10 = average of 00:00–00:10
# Shift back by 10 minutes to get period-start for NWP alignment
df = df.with_columns(
    (pl.col("timestamp_utc") - pl.duration(minutes=10)).alias("timestamp_utc")
)
```

**Common format strings:**
| CSV format | Polars format string |
|-----------|---------------------|
| `2024-01-15 14:30:00` | `"%Y-%m-%d %H:%M:%S"` |
| `15/01/2024 14:30` | `"%d/%m/%Y %H:%M"` |
| `2024-01-15T14:30:00` | `"%Y-%m-%dT%H:%M:%S"` |
| `2024-01-15T14:30:00Z` | `"%Y-%m-%dT%H:%M:%SZ"` (then replace_time_zone is not needed) |

---

## 5. Handling Missing Values in Numeric Columns

```python
# At read time: declare all sentinel values as null
df = pl.read_csv(
    path,
    null_values=["", "NA", "NaN", "N/A", "-999", "-9999", "9999"],
)

# Post-load: replace out-of-range values with null
df = df.with_columns(
    pl.when(pl.col("wind_speed").is_between(0, 50))
      .then(pl.col("wind_speed"))
      .otherwise(None)
      .alias("wind_speed"),
    
    pl.when(pl.col("power_kw").is_between(-100, 5000))
      .then(pl.col("power_kw"))
      .otherwise(None)
      .alias("power_kw"),
)

# Count nulls per column
null_counts = df.null_count()
```

---

## 6. Column Renaming

```python
# dict-based rename (preferred — explicit, works lazily)
df = df.rename({
    "ActivePower_kW": "power_kw",
    "WindSpeed_ms":   "wind_speed",
})

# In a scan_csv pipeline, rename before collect() — it's lazy:
lf = pl.scan_csv(path).rename(COLUMN_MAP)
```

**Pitfall:** `rename` raises if a key in the dict is not present in the DataFrame. Use `strict=False` in Polars ≥ 1.0:
```python
df = df.rename(COLUMN_MAP, strict=False)  # silently skips missing keys
```

---

## 7. Schema Validation

Polars does not have a built-in `validate_schema()`. Use this pattern:

```python
def validate_schema(
    df: pl.DataFrame,
    expected: dict[str, pl.PolarsDataType],
    *,
    strict: bool = True,
) -> list[str]:
    """Return list of schema violation messages. Empty = OK."""
    errors: list[str] = []
    actual = dict(zip(df.columns, df.dtypes))

    for col, dtype in expected.items():
        if col not in actual:
            errors.append(f"Missing column: {col!r}")
        elif actual[col] != dtype:
            errors.append(f"Column {col!r}: expected {dtype}, got {actual[col]}")

    if strict:
        extra = set(actual) - set(expected)
        for col in extra:
            errors.append(f"Unexpected column: {col!r}")

    return errors


# Usage
errors = validate_schema(df, SCADA_SCHEMA, strict=False)
if errors:
    raise ValueError(f"Schema violations:\n" + "\n".join(errors))
```

---

## 8. Writing to Parquet

```python
# Recommended: zstd compression, good ratio + fast decompression
df.write_parquet(
    "data/processed/kelmarsh_turbine_01.parquet",
    compression="zstd",
    compression_level=3,     # 1-22; 3 is fast with good ratio
    statistics=True,         # enables predicate pushdown when scanning later
)

# Read back
df = pl.read_parquet("data/processed/kelmarsh_turbine_01.parquet")

# Lazy scan of parquet (much faster than CSV — schema embedded in file)
lf = pl.scan_parquet("data/processed/*.parquet")
```

**Compression choices:**
| Option | Speed | Ratio | Use when |
|--------|-------|-------|----------|
| `zstd` (level 3) | Fast | Good | Default for SCADA data |
| `snappy` | Fastest | OK | Intermediate files, rapid iteration |
| `lz4` | Fastest | Low | Temporary files |
| `gzip` | Slow | Best | Long-term archival |
