# Kelmarsh Real Data — Parsing Patterns & Gotchas

## Reference project: sltzgs/OpenWindSCADA

The canonical open-source parser for Kelmarsh SCADA data.

### Standard pandas pattern

```python
df = pd.read_csv(path, skiprows=9, low_memory=False, index_col='# Date and time')
df.index = pd.to_datetime(df.index, utc=True)
df = df[df['Data Availability'] == 1]  # QC gate
df = df.dropna(axis=1, how='all')       # ~291 columns, many empty
```

### Polars equivalent

```python
df = pl.read_csv(
    io.BytesIO(csv_bytes_without_comments),
    infer_schema_length=10_000,
    null_values=["", "NaN", "NA", "N/A", "-999", "-9999"],
    truncate_ragged_lines=True,
    ignore_errors=True,
)
```

## CSV format details

- **9 comment lines** (lines 0-8) before the header on line 9
- **Header starts with `# Date and time`** — the `#` is literal, not a comment char
- **~291-299 columns** per CSV (varies by year and turbine)
- **Timestamps are UTC** (stated in comment line 4)
- **10-minute resolution** SCADA data

## Key column names (case-sensitive!)

| Signal | Exact column name |
|--------|-------------------|
| Timestamp | `# Date and time` |
| Active power | `Power (kW)` |
| Wind speed | `Wind speed (m/s)` |
| Wind direction | `Wind direction (°)` |
| Rotor speed | `Rotor speed (RPM)` — **not** `(rpm)` |
| Nacelle position | `Nacelle position (°)` |
| Pitch A | `Blade angle (pitch position) A (°)` |
| Pitch B | `Blade angle (pitch position) B (°)` |
| Pitch C | `Blade angle (pitch position) C (°)` |
| Ambient temp | `Nacelle ambient temperature (°C)` |
| Nacelle temp | `Nacelle temperature (°C)` |
| Quality flag | `Data Availability` (1=OK, 0=fault) |

## Gotchas

1. **`#` in column name is literal** — do NOT use `comment_prefix='#'`
2. **`Data Availability == 1`** is the standard QC filter (more reliable than status codes)
3. **Duplicate timestamps at year boundaries** — deduplicate after concat
4. **Power is in kW** (not kWh or MW). For kWh: `power_kw * 10/60`
5. **Mixed dtypes** in some columns — use `ignore_errors=True` or `low_memory=False`
6. **Entirely-empty columns are common** — not all 291 signals populated for all periods

## Published research using Kelmarsh

- **arXiv 2308.03472** — LightGBM, 58 features (lags + rolling + calendar), Kelmarsh + Penmanshiel
- **sltzgs/KernelCPD_WindSCADA** — Change-point detection on Kelmarsh
- No published XGBoost power forecasting on Kelmarsh found (gap we fill)

## Sources

- [OpenWindSCADA](https://github.com/sltzgs/OpenWindSCADA)
- [Zenodo v4](https://zenodo.org/records/16807551)
- [arXiv 2308.03472](https://arxiv.org/html/2308.03472v4)
