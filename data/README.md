# Published corpus snapshot

Every measurement behind [the atlas](../docs/silent-fallback.md), as Parquet and CSV.
Regenerate with `uv run edgefit corpus export --out data`.

**This is a snapshot, not the live corpus.** Measurements are immutable, so rows here
never change — but new rows are added by later runs and this file is only refreshed
when it is re-exported.

## What is in it

| file | contents |
|---|---|
| `measurements.*` | One row per measurement, successes and failures alike |
| `recipes.*` | The deployment recipe each measurement used |
| `graph_fingerprints.*` | Structural summary per exported graph — no weights |

## Reading it honestly

- **Failures are rows.** Filter on `outcome` — `success`, `lowering_failure`,
  `runtime_failure`, `gate_refused`. An analysis that drops them is not describing
  what happened.
- **`measurement_source` separates ours from third-party.** Rows marked `third_party`
  were measured on Qualcomm AI Hub's hardware under their harness. Their thermal state
  is unknown, their timing excludes model load and host framework overhead, and
  `source_detail` says so per row. Do not average them together with ours.
- **`latency_cv` is the row's own quality.** Every timing is a distribution, not a
  point. A row with a high cv is weaker evidence and says so.
- **Null means unmeasured, never zero.** The reason is recorded alongside.
- **`harness_version` matters.** A re-measurement is a new row under a new version,
  never an edit to the old one.

```python
import duckdb
duckdb.sql("SELECT * FROM 'data/measurements.parquet' WHERE outcome = 'success'")
```
