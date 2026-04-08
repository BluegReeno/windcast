# Feature: Ingest Kelmarsh Real Data

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Run the Kelmarsh ingestion pipeline on the real Kelmarsh v4 ZIP archive (3.8 GB, 6 turbines, 2016-2024 SCADA data). The current parser was built and tested against synthetic mock data. Real Greenbyte-exported CSVs have structural differences that must be fixed before the pipeline can succeed.

## User Story

As an ML engineer  
I want to ingest the real Kelmarsh v4 dataset into clean per-turbine Parquet files  
So that I can run feature engineering and training on actual wind farm data

## Problem Statement

The parser (`src/windcast/data/kelmarsh.py`) works on synthetic test data but will fail on real Kelmarsh CSVs due to 3 issues:

1. **Comment lines**: Real CSVs have 9 comment lines (lines 0-8) before the header on line 9. The mock test data has zero comment lines. Polars will fail or misparse.
2. **Signal map case mismatch**: Real CSV has `"Rotor speed (RPM)"` but signal map uses `"Rotor speed (rpm)"`. Result: rotor_rpm will be null.
3. **Ragged lines / dtype issues**: Real CSVs have ~299 columns with mixed types and trailing commas. Polars raises `ComputeError` without `truncate_ragged_lines=True` and `ignore_errors=True`.
4. **Duplicate timestamps**: When concatenating 9 years of data, year boundaries can produce duplicate timestamps per turbine (confirmed by OpenWindSCADA project).
5. **`Data Availability` column**: Real CSVs have a `Data Availability` column (1=OK, 0=fault/curtailed) that is the standard QC filter used by published Kelmarsh projects (sltzgs/OpenWindSCADA). More reliable than our status_code heuristic.

Additionally, the data lives at `data/KelmarshV4/16807551.zip` but the script defaults to `data/raw/kelmarsh/`. This is handled by `--raw-path` CLI arg (no code change needed).

## Solution Statement

Fix `_read_turbine_csv()` to handle real Greenbyte CSV format:
- Strip comment lines from bytes before Polars reads them
- Fix the RPM case mismatch in signal map
- Add resilience parameters for ragged/mixed-type CSVs
- Use `Data Availability` column for QC flagging if present
- Deduplicate timestamps after concat
- Update tests to cover the real data format

### Reference: OpenWindSCADA (sltzgs/OpenWindSCADA)

The canonical open-source parser for Kelmarsh. Their pattern:
```python
df = pd.read_csv(path, skiprows=9, low_memory=False, index_col='# Date and time')
df = df[df['Data Availability'] == 1]
df = df.dropna(axis=1, how='all')
```
Key learnings: `skiprows=9` is standard, `Data Availability` is the QC gate, timestamps are UTC.

## Feature Metadata

**Feature Type**: Bug Fix  
**Estimated Complexity**: Low  
**Primary Systems Affected**: `src/windcast/data/kelmarsh.py`  
**Dependencies**: None (Polars already installed)

---

## CONTEXT REFERENCES

### Relevant Codebase Files — YOU MUST READ THESE BEFORE IMPLEMENTING!

- `src/windcast/data/kelmarsh.py` (lines 130-209) — `_read_turbine_csv()` is the function to fix. The signal map at lines 16-28 has the RPM case bug.
- `src/windcast/data/schema.py` (lines 1-24) — SCADA_SCHEMA definition. The output must conform to these 15 columns.
- `src/windcast/data/qc.py` (lines 16-57) — `run_qc_pipeline()` runs downstream. Needs valid schema input.
- `scripts/ingest_kelmarsh.py` — The CLI script. No changes needed (has `--raw-path` arg).
- `tests/data/test_kelmarsh.py` — Tests to update. Mock CSV helper `_make_csv_bytes()` at line 17.

### New Files to Create

None. All changes are in existing files.

### Relevant Documentation

No external documentation needed. The fix is purely about matching real CSV format to the parser.

### Patterns to Follow

**CSV bytes preprocessing pattern** — Strip comment lines before Polars parses. Real format:
```
# This file was exported by Greenbyte at 2022-01-27 10:33:05...
#
# Turbine: Kelmarsh 1
# Turbine type: Senvion MM92
# Time zone: UTC
# Time interval: 2016-01-01 00:00:00 - ...
#
# Data that is missing or is erroneous has been marked with "NaN"
#
# Date and time,Wind speed (m/s),...    ← THIS is the header (line 9)
2016-01-03 00:00:00,NaN,...              ← Data starts at line 10
```

The header line ALSO starts with `#` (`# Date and time`), so `comment_prefix='#'` would eat it. The correct approach: strip lines that start with `# ` (note the space) AND don't contain `Date and time`.

---

## IMPLEMENTATION PLAN

### Phase 1: Fix Parser

Fix the 3 bugs in `_read_turbine_csv()`:
1. Add comment-line stripping to handle real Greenbyte CSV format
2. Fix signal map case: `"Rotor speed (rpm)"` → `"Rotor speed (RPM)"`
3. Add `truncate_ragged_lines=True` and `ignore_errors=True` to `pl.read_csv()`

### Phase 2: Update Tests

Add a test helper that generates realistic CSV bytes WITH comment lines. Add a test that verifies the parser handles both formats (with and without comments).

### Phase 3: Run and Validate

Run the actual ingestion on `data/KelmarshV4/16807551.zip` and verify Parquet output.

---

## STEP-BY-STEP TASKS

### Task 1: UPDATE `src/windcast/data/kelmarsh.py` — Fix signal map

- **IMPLEMENT**: Change signal map key from `"Rotor speed (rpm)"` to `"Rotor speed (RPM)"` at line 27
- **GOTCHA**: The test mock CSV also uses `"Rotor speed (rpm)"`. Must update test helper too (Task 3).
- **VALIDATE**: `uv run ruff check src/windcast/data/kelmarsh.py`

### Task 2: UPDATE `src/windcast/data/kelmarsh.py` — Fix `_read_turbine_csv()` for real CSV format

- **IMPLEMENT**: Add a helper function `_strip_comment_lines(csv_bytes: bytes) -> bytes` that:
  1. Decodes bytes to string
  2. Splits into lines
  3. Identifies comment lines: lines starting with `#` that do NOT contain `Date and time`
  4. Returns bytes WITHOUT those comment lines
  
  Then call it at the start of `_read_turbine_csv()` before `pl.read_csv()`.

- **IMPLEMENT**: Add `truncate_ragged_lines=True` and `ignore_errors=True` to the `pl.read_csv()` call at line 136.

- **PATTERN**: Keep the existing `_read_turbine_csv` signature unchanged. The preprocessing happens inside.

- **GOTCHA**: Do NOT use `comment_prefix='#'` — it would strip the header line too since it starts with `# Date and time`.

- **GOTCHA**: The existing test mock CSVs have NO comment lines (just header + data). The `_strip_comment_lines` function must handle this gracefully (no comment lines = no-op).

- **VALIDATE**: `uv run ruff check src/windcast/data/kelmarsh.py && uv run pyright src/windcast/data/kelmarsh.py`

### Task 3: UPDATE `src/windcast/data/kelmarsh.py` — Use `Data Availability` for QC

- **IMPLEMENT**: In `KELMARSH_SIGNAL_MAP`, add mapping: `"Data Availability": "data_availability"`. This column exists in real CSVs (value 1=OK, 0=fault/curtailed).
- **IMPLEMENT**: In `_read_turbine_csv()`, after reading and renaming, if `data_availability` column exists:
  - Set `status_code = 1` (non-operational) where `data_availability == 0`
  - Then drop the `data_availability` column (it's not in SCADA_SCHEMA)
- **GOTCHA**: This column may not exist in older exports or test mocks. Guard with `if "data_availability" in df.columns`.
- **VALIDATE**: `uv run ruff check src/windcast/data/kelmarsh.py && uv run pyright src/windcast/data/kelmarsh.py`

### Task 4: UPDATE `src/windcast/data/kelmarsh.py` — Deduplicate timestamps

- **IMPLEMENT**: In `parse_kelmarsh()`, after `pl.concat(frames).sort(...)`, add deduplication:
  ```python
  .unique(subset=["timestamp_utc", "turbine_id"], keep="first")
  ```
  This handles duplicate rows at year boundaries when concatenating annual ZIPs.
- **VALIDATE**: `uv run ruff check src/windcast/data/kelmarsh.py`

### Task 5: UPDATE `tests/data/test_kelmarsh.py` — Add real-format test

- **IMPLEMENT**: Update `_make_csv_bytes()` to accept an optional `include_comments: bool = False` parameter. When True, prepend the 9 Greenbyte comment lines before the header.

- **IMPLEMENT**: Also update the header in `_make_csv_bytes()`: change `"Rotor speed (rpm)"` to `"Rotor speed (RPM)"` to match the real format.

- **IMPLEMENT**: Add a new test `test_real_csv_format_with_comments` in `TestReadTurbineCsv` that calls `_make_csv_bytes(include_comments=True)` and verifies the parser still produces valid schema output.

- **VALIDATE**: `uv run pytest tests/data/test_kelmarsh.py -v`

### Task 6: RUN full ingestion on real data

- **IMPLEMENT**: Execute:
  ```bash
  uv run python scripts/ingest_kelmarsh.py --raw-path data/KelmarshV4/16807551.zip
  ```
  This will:
  1. Open the 3.8 GB outer ZIP
  2. Process 9 inner SCADA ZIPs (2016-2024), each containing 6 turbine CSVs
  3. Parse, validate schema, run QC
  4. Write 6 per-turbine Parquet files to `data/processed/`

- **GOTCHA**: The ZIP is 3.8 GB with nested ZIPs. Processing will take a few minutes. Expect ~54 CSV files (9 years × 6 turbines).

- **GOTCHA**: Non-SCADA ZIPs in the archive (Grid_Meter, PMU) will be opened but contain no `Turbine_Data_Kelmarsh_*.csv` files — the parser handles this (returns empty list from `_parse_zip_contents`).

- **GOTCHA**: The `.kmz` and `.csv` files at the root of the outer ZIP (signal mapping, static data) are not turbine CSVs — `_is_turbine_csv()` correctly filters them out.

- **VALIDATE**: 
  ```bash
  ls -la data/processed/kelmarsh_*.parquet  # Should see 6 files (kwf1-kwf6)
  python -c "import polars as pl; df=pl.read_parquet('data/processed/kelmarsh_kwf1.parquet'); print(f'Rows: {len(df)}, Cols: {df.columns}')"
  ```

### Task 7: RUN full validation suite

- **VALIDATE**: 
  ```bash
  uv run ruff check src/ tests/ scripts/
  uv run ruff format --check src/ tests/ scripts/
  uv run pyright src/
  uv run pytest tests/ -v
  ```

---

## TESTING STRATEGY

### Unit Tests

- Existing tests in `tests/data/test_kelmarsh.py` must continue passing (backward compatibility with clean CSV format)
- New test: verify parser handles comment-prefixed CSV (the real format)
- New test: verify `_strip_comment_lines` is a no-op for clean CSVs

### Integration Test

- Run `ingest_kelmarsh.py` on real data (Task 4) — this IS the integration test
- Verify: 6 Parquet files, schema-compliant, millions of rows, QC stats look reasonable

### Edge Cases

- CSV with 0 comment lines (backward compatibility with test format)
- CSV with 9 comment lines (real format)
- Missing `Rotor speed (RPM)` column in some years (should produce null, not crash)

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/
```

### Level 2: Unit Tests

```bash
uv run pytest tests/data/test_kelmarsh.py -v
uv run pytest tests/ -v
```

### Level 3: Integration (Real Data)

```bash
uv run python scripts/ingest_kelmarsh.py --raw-path data/KelmarshV4/16807551.zip
ls -la data/processed/kelmarsh_*.parquet
```

### Level 4: Data Sanity Check

```bash
python -c "
import polars as pl
for i in range(1, 7):
    df = pl.read_parquet(f'data/processed/kelmarsh_kwf{i}.parquet')
    ws = df['wind_speed_ms'].drop_nulls()
    pw = df['active_power_kw'].drop_nulls()
    print(f'KWF{i}: {len(df)} rows, wind [{ws.min():.1f}-{ws.max():.1f}] m/s, power [{pw.min():.0f}-{pw.max():.0f}] kW')
"
```

**Expected**: ~50k+ rows per turbine per year × 9 years = ~400k+ rows per turbine. Wind speed 0-25 m/s range. Power 0-2100 kW range.

---

## ACCEPTANCE CRITERIA

- [ ] Parser handles real Greenbyte CSV format (9 comment lines + header starting with `# Date and time`)
- [ ] Signal map correctly maps `Rotor speed (RPM)` → `rotor_rpm`
- [ ] Existing tests still pass (backward compatibility)
- [ ] New test covers comment-line CSV format
- [ ] 6 Parquet files written to `data/processed/` from real data
- [ ] Each Parquet file has valid SCADA_SCHEMA (15 columns, correct types)
- [ ] QC summary shows reasonable percentages (>80% OK)
- [ ] `ruff check`, `pyright`, `pytest` all pass

---

## COMPLETION CHECKLIST

- [ ] All tasks completed in order
- [ ] Level 1: ruff + pyright pass
- [ ] Level 2: pytest passes (all tests including new ones)
- [ ] Level 3: Real data ingestion succeeds
- [ ] Level 4: Data sanity check shows reasonable values
- [ ] All acceptance criteria met

---

## NOTES

**Performance**: The 3.8 GB ZIP contains nested ZIPs. Each inner SCADA ZIP is 100-700 MB. Polars reads from BytesIO so everything happens in memory. For 2024 (largest year), expect ~1 GB in memory. Should be fine on a modern Mac.

**Power unit conversion**: The parser has a heuristic at line 178 — if max power < 500, it multiplies by 6 (assuming kWh per 10-min → kW). Real data shows max power ~2072 kW, so this heuristic should NOT trigger. But watch for it in the logs.

**Downstream impact**: Once Parquets are in `data/processed/`, the next steps are:
1. `build_features.py` — reads `kelmarsh_*.parquet` glob pattern (already correct)
2. `train.py` — reads feature Parquets
3. `evaluate.py` — reads MLflow runs

No downstream script changes needed.
