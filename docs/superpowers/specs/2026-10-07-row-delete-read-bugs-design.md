# Row-Delete Read Bugs: `count(*)` over Deletes, and v3 `PUFFIN` Delete Files — Design / Handoff

**Date:** 2026-10-07
**Status:** Implemented on `fix/row-delete-read-bugs` (see "Implementation notes")
**Crates:** `iceberg-rust-spec`, `iceberg-rust`, `datafusion_iceberg`
**Found at:** `eeac8ab` (main) and still present on `feat/dv-exec-serialization` (`f71b8d3`)
**Upstream intent:** both are plain correctness bugs in reading Iceberg tables; the
fixes apply to JanKaul/iceberg-rust unchanged.

Two independent bugs, both on the single-process read path (no plan serialization
involved), both hit by tables written with row-level deletes by Trino (and, by
construction, any iceberg-java engine).

---

## Bug 1 — Queries that project no columns fail on tables with row deletes

### Symptom

```sql
SELECT count(*) FROM t;      -- t has v2 position deletes or v3 deletion vectors
SELECT 1 FROM t LIMIT 1;     -- likewise
```

fails with

```text
ArrowError(InvalidArgumentError("must either specify a row count or at least one column"))
```

raised from `DvFilterStream::finalize` (`datafusion_iceberg/src/table/dv_exec.rs:380`).
The same queries on a table without deletes succeed. `count(<column>)` succeeds,
which is the workaround in use today.

### Reproduction

With Trino against the same catalog:

```sql
CREATE TABLE iceberg.s.lineitem_v2 WITH (format_version = 2) AS SELECT * FROM tpch.sf1.lineitem;
DELETE FROM iceberg.s.lineitem_v2 WHERE l_orderkey % 5 = 0;   -- writes position-delete files
```

then `SELECT count(*) FROM s.lineitem_v2` through `datafusion_iceberg`.

### Root cause

When a scan has row deletes, `table_scan` forces two internal columns into the
Parquet projection — the `__data_file_path` partition column and the row-number
virtual column — and wraps the scan in `IcebergDvExec`, which strips them again
(`strip_indices`). For a query that needs no user columns (`count(*)`), the user
projection is empty, so after stripping **zero columns remain**.

`finalize` then rebuilds the batch with

```rust
RecordBatch::try_new(self.output_schema.clone(), projected.columns().to_vec())
```

Arrow cannot infer a row count from zero columns, so `try_new` fails.
`batch.project(&keep_cols)` itself is fine — it preserves the row count — the count is
lost only in the final `try_new`. The comment above it ("there is always at least one
column to strip") is true but does not imply a column remains.

### Fix

Carry the row count explicitly:

```rust
let projected = batch.project(&keep_cols)?;
Ok(RecordBatch::try_new_with_options(
    self.output_schema.clone(),
    projected.columns().to_vec(),
    &RecordBatchOptions::new().with_row_count(Some(projected.num_rows())),
)?)
```

`finalize` is the only production `RecordBatch::try_new` in `dv_exec.rs` (the others,
`:528` and `:544`, are test helpers). Any other place that rebuilds a batch after
removing internal columns needs the same treatment.

### Tests

- **Unit (`dv_exec.rs`):** an `IcebergDvExec` whose child emits only the path and
  row-number columns (empty user projection), with a deletion vector removing some
  rows: output batches have zero columns and `num_rows()` equal to the surviving rows.
- **Integration (`tests/position_delete.rs`):** `SELECT count(*)` over the existing
  position-delete fixture returns the post-delete count; also `SELECT 1 … LIMIT 1`.
- The same `count(*)` case over a deletion-vector table (see Bug 2 for a real one).

---

## Bug 2 — v3 tables with deletion vectors written by other engines cannot be read

### Symptom

Planning any scan of a format-version-3 table that has deletion vectors fails while
reading the manifest:

```text
Iceberg(Avro(Error { details: Failed to deserialize Avro value into value: Invalid data file format. }))
```

### Reproduction

```sql
CREATE TABLE iceberg.s.orders_v3 WITH (format_version = 3) AS SELECT * FROM tpch.sf1.orders;
DELETE FROM iceberg.s.orders_v3 WHERE o_orderkey % 3 = 0;   -- Trino writes deletion vectors
```

then any `SELECT … FROM s.orders_v3` through `datafusion_iceberg`.

### Root cause

The Iceberg v3 spec stores deletion vectors as blobs in **Puffin** files, and a DV's
manifest entry records `file_format = "PUFFIN"` (with `content = position deletes`,
`referenced_data_file`, `content_offset`, `content_size_in_bytes`).

`iceberg_rust_spec::spec::manifest::FileFormat` has only `Avro`, `Orc` and `Parquet`
(`iceberg-rust-spec/src/spec/manifest.rs:512`), and its `Deserialize` rejects every
other string with "Invalid data file format." (`:552`). So the manifest entry of a
real DV cannot be decoded, and the whole manifest read fails.

The crate already anticipates Puffin DVs: `load_deletion_vectors` documents its input
as entries "whose `file_format()` is `Puffin`"
(`iceberg-rust/src/table/deletion_vector.rs:32`) and loads them by byte range. But
because the variant doesn't exist, DVs are recognised only by their
offset/size/reference fields (`is_v3_deletion_vector`,
`datafusion_iceberg/src/table/mod.rs:155`), and the test fixtures label DV entries
`FileFormat::Parquet` (`deletion_vector.rs:129`, `table/mod.rs:2001`). That is why the
existing DV tests pass while real tables fail.

Deserialization is also case-sensitive (`"AVRO"`, `"ORC"`, `"PARQUET"` only), while
iceberg-java parses file formats case-insensitively.

### Fix

1. **`iceberg-rust-spec` — add the format.**
   - `FileFormat::Puffin = 3`.
   - Serialize as `"PUFFIN"`.
   - Deserialize all four names case-insensitively (`s.eq_ignore_ascii_case(…)`),
     matching iceberg-java.
   - Keep the error for unknown names, but include the offending value:
     `"Invalid data file format: {s}"`.
2. **Classify deletion vectors by format, and validate the rest.** In
   `is_v3_deletion_vector` (`datafusion_iceberg/src/table/mod.rs:155`), with
   `content = PositionDeletes`:
   - `file_format == Puffin` → must have `referenced_data_file`, `content_offset` and
     `content_size_in_bytes`, else `InvalidFormat` (as today);
   - `file_format == Parquet` (or Avro/ORC) → a position-delete file; the DV fields must
     be absent;
   - anything else → `InvalidFormat`.
   Format-first classification follows the spec and catches inconsistent metadata in
   both directions.
3. **Reject Puffin where a data file is expected.** Adding a variant surfaces every
   `match` on `FileFormat` at compile time; the ones that read data files
   (`iceberg-rust/src/arrow/read.rs:95`) must return a clear `NotSupported` /
   `InvalidFormat` error for `Puffin` rather than a catch-all. The position-delete
   loader already rejects non-Parquet files (`position_delete.rs:66`), which stays
   correct.
4. **Make the fixtures realistic.** DV entries in unit tests use
   `FileFormat::Puffin`, so the tests exercise what real tables contain.

### Tests

- **Spec unit:** `"PUFFIN"`, `"puffin"`, `"Parquet"` round-trip through
  `FileFormat` serde; an unknown name errors and names the value.
- **Classification unit:** Puffin + all DV fields → DV; Puffin missing a field →
  error; Parquet with DV fields → error; Parquet without → position-delete file.
- **Read path unit:** a data-file read of a Puffin entry fails with the explicit
  error.
- **Integration, real writer:** a v3 table written with deletes by an iceberg-java
  engine (the existing `tests/integration_trino.rs` / `integration_spark.rs`
  harnesses): scan returns exactly the non-deleted rows; `count(*)` (with Bug 1 fixed)
  returns the post-delete count. This is the test that would have caught the bug.

---

## Bug 3 (found in review) — `count(*)` overcounts with dangling deletes

`statistics_from_datafiles` (`datafusion_iceberg/src/statistics.rs`) summed
`record_count` over **every** live manifest entry, delete files included, and
reported it as `Precision::Exact`. Usually hidden: `IcebergDvExec` reports unknown
statistics, and equality-delete plans go through joins. But when every position
delete applies to no live data file (dangling deletes left by a data-file rewrite),
the delete index is empty, the scan is not wrapped, and DataFusion's
`AggregateStatistics` answered `count(*)` from the statistic: data rows **plus**
delete rows (reproduced: 4 data rows + 3 dangling deletes → `count(*) = 7`).

Fix: sum only `Content::Data` entries, and mark the statistics inexact while any
delete file is live.

## Implementation notes (deviations from the plan above)

- **Bug 2 step 3:** adding the variant flagged nothing at compile time — the only
  `match` (`arrow/read.rs`) has a catch-all, and `read()` is not on the DataFusion
  scan path. The guard went into `table_scan` classification instead
  (`check_data_file_format`): data entries in any format but Parquet are rejected
  with an error naming the file (Puffin as `InvalidFormat`, Avro/ORC as
  `NotSupported`). `read()`'s error now names the file and format.
- **Bug 2 step 2:** a non-Puffin position-delete file may carry
  `referenced_data_file` (the spec allows it for file-scoped position deletes); only
  `content_offset` / `content_size_in_bytes` are rejected on it.
- **Breaking change:** `FileFormat` is a public enum without `#[non_exhaustive]`;
  the new `Puffin` variant breaks downstream exhaustive matches.
- **Tests:** besides the unit tests, `tests/position_delete.rs` has an in-process v3
  test that commits a Puffin DV and reads it back through the manifest, and a
  dangling-delete `count(*)` test; `tests/integration_trino.rs` checks Trino-written
  v2 (Parquet position deletes) and v3 (Puffin DV) tables differentially against
  the delete-free `orders` copy.

## Order of work

Independent; Bug 1 is a one-line fix plus tests. Bug 2's step 1 alone already makes
real v3 tables readable (DVs are detected by their fields today); steps 2–4 harden it.
Together they unblock `SELECT count(*)` over any table with row deletes and reading
v3 tables written by Trino, Spark, Flink and other iceberg-java engines.

## Checklist

- [x] Bug 1: `try_new_with_options` with row count in `DvFilterStream::finalize`.
- [x] Bug 1 tests (unit + `position_delete.rs` `count(*)`).
- [x] Bug 2: `FileFormat::Puffin`, case-insensitive serde, error names the value.
- [x] Bug 2: format-first DV classification in `is_v3_deletion_vector`.
- [x] Bug 2: explicit errors for Puffin on data-file read paths (in `table_scan`; see notes).
- [x] Bug 2: fixtures use `FileFormat::Puffin`; tests above, including a real-writer v3 table.
- [x] Bug 3: statistics sum data files only; inexact while delete files are live.
- [x] `make test`; `cargo clippy --all-targets --all-features -- -D warnings`.
