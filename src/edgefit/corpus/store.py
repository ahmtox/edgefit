"""Insert-only measurement corpus (PROJECT.md §14.3).

The corpus is the asset — it trains the cost model that is the moat (§12). So the
store deliberately has no mutation surface at all: no ``update``, no ``delete``,
no ``execute``. Re-measuring inserts a new row carrying a new ``harness_version``.
Correcting a bad measurement means recording a *new* measurement, never editing
the old one.

Every write also records failures. §5.8: "Failures are as valuable as successes:
they train the tier-1 static filter."
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb

from edgefit.corpus.ddl import DDL
from edgefit.schema.common import MeasurementSource, canonical_json, content_hash
from edgefit.schema.fingerprint import GraphFingerprint
from edgefit.schema.measurement import (
    MEASUREMENT_SCHEMA_VERSION,
    MeasurementRecord,
)
from edgefit.schema.recipe import Recipe
from edgefit.schema.vendor import provider_vendor, soc_vendor

DEFAULT_CORPUS_PATH = Path("corpus/measurements.duckdb")

# query() is for reading. This is a guard rail, not a security boundary — but it
# means an accidental mutating statement fails loudly instead of quietly
# violating the one rule the whole corpus depends on.
_READ_ONLY_STATEMENT = re.compile(
    r"^\s*(select|with|describe|explain|pragma|show)\b", re.IGNORECASE
)


class CorpusSchemaMismatch(Exception):
    """The corpus file was written by a different schema version.

    Raised on open rather than letting the first insert fail with DuckDB's
    "table has 44 columns but 48 values were supplied", which tells an operator
    nothing about what to do. Measurements are immutable (PROJECT.md §14.3), so the
    resolution is a migration that rewrites rows into the new schema — or, for a
    throwaway dev corpus, deleting the file.
    """


class DuplicateRecordError(Exception):
    """Raised when an id already exists.

    Not an error to paper over: identical content hash means identical content,
    so a duplicate insert is either a retry bug or a hash collision, and both
    deserve a stack trace.
    """


def _vendor_columns(record: MeasurementRecord) -> dict[str, object]:
    """Vendor attribution for the query columns, preferring the as-run report.

    Rows measured before schema v7 carry no vendor fields, so those are **derived** from
    the SoC string and execution provider already recorded on the same row. That is
    deterministic re-reading of measured data, not synthesis: nothing here invents a
    number, and hard rule #1 governs measurement values. The immutable ``payload`` is
    left exactly as measured — per :func:`migrate`, the columns are a query convenience
    and the payload is the record.

    Deriving rather than backfilling matters because the alternative is worse in both
    directions: leaving 374 existing rows as "vendor unknown" would make every one of
    them non-diagnostic and silently drop the neutral Qualcomm-internal comparison, while
    editing their payloads to add the fields would violate immutability for metadata that
    was always recoverable from what they already say.
    """
    report = record.fallback_as_run or record.fallback
    if report is None:
        return {
            "toolchain_vendor": None,
            "device_soc_vendor": None,
            "cross_vendor": None,
            "fallback_is_diagnostic": False,
        }

    toolchain = report.toolchain_vendor
    if toolchain is None:
        if record.measurement_source is MeasurementSource.THIRD_PARTY:
            # The only third-party path is Qualcomm AI Hub, which compiles and runs the
            # artifact itself. Recorded on new rows; derived here for older ones.
            toolchain = "qualcomm"
        else:
            toolchain = provider_vendor(report.intended_provider)

    device = report.device_soc_vendor or soc_vendor(record.device.soc)

    resolved = report.model_copy(
        update={"toolchain_vendor": toolchain, "device_soc_vendor": device}
    )
    return {
        "toolchain_vendor": resolved.toolchain_vendor,
        "device_soc_vendor": resolved.device_soc_vendor,
        "cross_vendor": resolved.cross_vendor,
        "fallback_is_diagnostic": resolved.fallback_is_diagnostic,
    }


class CorpusStore:
    """Append-only DuckDB corpus.

    Deliberately absent: any method that can change or remove a stored row.
    ``tests/test_corpus_immutable.py`` asserts that this stays true.
    """

    def __init__(self, path: Path | str = DEFAULT_CORPUS_PATH) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._conn.execute(DDL)
        self._check_schema_version()

    @classmethod
    def from_export(cls, directory: Path | str) -> Self:
        """A read-only store backed by a published Parquet snapshot.

        Exists because a clean clone has no corpus — it is dev state and gitignored —
        so a stranger following the README got a **six-page atlas reporting zero
        measurements**. The data was sitting in `data/*.parquet` the whole time with
        nothing able to load it. Hard rule #5 asks that every published number be
        independently checkable, and "clone it and get an empty site" does not clear
        that bar.

        Parquet is read through views rather than copied into tables, so the snapshot
        stays the single source of truth and nothing can accidentally write back to it.
        """
        directory = Path(directory)
        store = cls(":memory:")
        for table in ("measurements", "recipes", "graph_fingerprints"):
            path = directory / f"{table}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing. A published snapshot needs measurements, "
                    "recipes and graph_fingerprints; regenerate one with "
                    "`edgefit corpus export --out <dir>`."
                )
            store._conn.execute(f"DROP TABLE IF EXISTS {table}")
            store._conn.execute(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}')"
            )
        return store

    def _check_schema_version(self) -> None:
        """Reject a corpus whose record schema *or* table shape differs from this build.

        Both are checked because they move independently: adding a denormalised column
        changes the tables without touching the record version, and that slipped
        through as DuckDB's "Referenced column not found" on the first query.
        """
        self._check_one(
            "measurement_schema_version",
            str(MEASUREMENT_SCHEMA_VERSION),
            "record schema",
        )
        self._check_one("table_shape", content_hash(DDL), "table shape")

    def _check_one(self, key: str, expected: str, description: str) -> None:
        row = self._conn.execute(
            "SELECT value FROM corpus_meta WHERE key = $key", {"key": key}
        ).fetchone()
        if row is None:
            rows = self._conn.execute("SELECT count(*) FROM measurements").fetchone()
            if rows and rows[0]:
                raise CorpusSchemaMismatch(
                    f"{self.path} holds {rows[0]} measurements but records no "
                    f"{description}, so it predates this check. Export what you need and "
                    "rebuild, or write a migration."
                )
            self._conn.execute(
                "INSERT INTO corpus_meta VALUES ($key, $v)", {"key": key, "v": expected}
            )
            return
        if row[0] != expected:
            raise CorpusSchemaMismatch(
                f"{self.path} was written with a different {description} "
                f"({row[0]}; this build expects {expected}). Measurements are immutable, "
                "so migrate the rows or start a new corpus file."
            )

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- writes (insert only) ---------------------------------------------

    def insert_recipe(self, recipe: Recipe) -> str:
        """Idempotent: re-inserting an identical recipe is a no-op, not an error.

        Recipes are pure descriptions, so the same point in the search space
        legitimately recurs across jobs. Measurements are observations and are
        held to a stricter rule.
        """
        recipe_id = recipe.recipe_id
        if self._exists("recipes", "recipe_id", recipe_id):
            return recipe_id

        quant = recipe.quantization
        self._conn.execute(
            """
            INSERT INTO recipes VALUES (
                $recipe_id, $schema_version, $model_ref, $task, $runtime_kind,
                $intended_provider, $providers, $weight_dtype, $weight_granularity,
                $activation_quant, $quant_algorithm, $num_threads, $label,
                $payload, $first_seen_at
            )
            """,
            {
                "recipe_id": recipe_id,
                "schema_version": recipe.schema_version,
                "model_ref": recipe.model.ref,
                "task": str(recipe.model.task),
                "runtime_kind": str(recipe.runtime.kind),
                "intended_provider": recipe.intended_provider,
                "providers": recipe.provider_chain,
                "weight_dtype": str(quant.weight_dtype) if quant else None,
                "weight_granularity": str(quant.weight_granularity) if quant else None,
                "activation_quant": str(quant.activation_quant) if quant else None,
                "quant_algorithm": str(quant.algorithm) if quant and quant.algorithm else None,
                "num_threads": recipe.execution.num_threads,
                "label": recipe.label,
                "payload": canonical_json(recipe.model_dump(mode="json")),
                "first_seen_at": datetime.now(UTC),
            },
        )
        return recipe_id

    def insert_fingerprint(self, fingerprint: GraphFingerprint) -> str:
        fingerprint_id = fingerprint.fingerprint_id
        if self._exists("graph_fingerprints", "fingerprint_id", fingerprint_id):
            return fingerprint_id

        self._conn.execute(
            """
            INSERT INTO graph_fingerprints VALUES (
                $fingerprint_id, $n_nodes, $n_parameters, $attention_variant,
                $norm_type, $payload, $first_seen_at
            )
            """,
            {
                "fingerprint_id": fingerprint_id,
                "n_nodes": fingerprint.n_nodes,
                "n_parameters": fingerprint.n_parameters,
                "attention_variant": str(fingerprint.attention_variant),
                "norm_type": str(fingerprint.norm_type),
                "payload": canonical_json(fingerprint.model_dump(mode="json")),
                "first_seen_at": datetime.now(UTC),
            },
        )
        return fingerprint_id

    def insert_measurement(self, record: MeasurementRecord) -> str:
        """Insert one observation. The only way data enters the corpus."""
        measurement_id = record.measurement_id
        if self._exists("measurements", "measurement_id", measurement_id):
            raise DuplicateRecordError(
                f"measurement {measurement_id} already exists. Measurements are "
                "immutable (PROJECT.md §14.3) — record a new measurement instead."
            )

        metrics = record.metrics
        latency = metrics.latency_ms if metrics else None
        ttft = metrics.ttft_ms if metrics else None
        decode = metrics.decode_tok_s if metrics else None
        fallback = record.fallback
        as_run = record.fallback_as_run
        probe = record.calibration_probe

        self._conn.execute(
            """
            INSERT INTO measurements VALUES (
                $measurement_id, $schema_version, $harness_version, $created_at,
                $recipe_id, $model_ref, $graph_fingerprint_id,
                $device_id, $sku_id, $device_model, $soc, $os_name, $os_version, $os_build,
                $measurement_source, $source_detail, $reported_latency_ms,
                $stress_profile, $outcome, $failure_reason, $run_count, $warmup_count,
                $latency_p50_ms, $latency_p95_ms, $latency_cv,
                $ttft_p50_ms, $ttft_cv, $decode_tok_s_p50, $sustained_tok_s_5min,
                $peak_rss_bytes, $artifact_bytes, $lowering_ms,
                $cold_load_ms, $warm_load_ms, $first_inference_ms,
                $accuracy, $accuracy_delta_vs_fp16, $output_cosine_vs_reference,
                $token_agreement, $power_mw,
                $fallback_node_pct, $fallback_flops_pct, $fallback_time_pct,
                $as_run_node_pct, $as_run_time_pct, $as_run_partitions,
                $toolchain_vendor, $device_soc_vendor, $cross_vendor,
                $fallback_is_diagnostic,
                $power_source, $low_power_mode, $thermal_state, $load_avg_1m,
                $calibration_ratio, $notes, $payload
            )
            """,
            {
                "measurement_id": measurement_id,
                "schema_version": record.schema_version,
                "harness_version": record.harness_version,
                "created_at": record.created_at,
                "recipe_id": record.recipe_id,
                "model_ref": record.model_ref,
                "graph_fingerprint_id": record.graph_fingerprint_id,
                "device_id": record.device.device_id,
                "sku_id": record.device.sku_id,
                "device_model": record.device.model,
                "soc": record.device.soc,
                "os_name": record.device.os_name,
                "os_version": record.device.os_version,
                "os_build": record.device.os_build,
                "measurement_source": str(record.measurement_source),
                "source_detail": record.source_detail,
                "reported_latency_ms": (
                    metrics.reported_latency_ms if metrics else None
                ),
                "stress_profile": str(record.stress_profile),
                "outcome": str(record.outcome),
                "failure_reason": record.failure_reason,
                "run_count": record.run_count,
                "warmup_count": record.warmup_count,
                "latency_p50_ms": latency.p50 if latency else None,
                "latency_p95_ms": latency.p95 if latency else None,
                "latency_cv": latency.cv if latency else None,
                "ttft_p50_ms": ttft.p50 if ttft else None,
                "ttft_cv": ttft.cv if ttft else None,
                "decode_tok_s_p50": decode.p50 if decode else None,
                "sustained_tok_s_5min": metrics.sustained_tok_s_5min if metrics else None,
                "peak_rss_bytes": metrics.peak_rss_bytes if metrics else None,
                "artifact_bytes": metrics.artifact_bytes if metrics else None,
                "lowering_ms": metrics.lowering_ms if metrics else None,
                "cold_load_ms": metrics.cold_load_ms if metrics else None,
                "warm_load_ms": metrics.warm_load_ms if metrics else None,
                "first_inference_ms": metrics.first_inference_ms if metrics else None,
                "accuracy": metrics.accuracy if metrics else None,
                "accuracy_delta_vs_fp16": metrics.accuracy_delta_vs_fp16 if metrics else None,
                "output_cosine_vs_reference": (
                    metrics.output_cosine_vs_reference if metrics else None
                ),
                "token_agreement": metrics.token_agreement if metrics else None,
                "power_mw": metrics.power_mw if metrics else None,
                "fallback_node_pct": fallback.fallback_node_pct if fallback else None,
                "fallback_flops_pct": fallback.fallback_flops_pct if fallback else None,
                "fallback_time_pct": fallback.fallback_time_pct if fallback else None,
                "as_run_node_pct": as_run.fallback_node_pct if as_run else None,
                "as_run_time_pct": as_run.fallback_time_pct if as_run else None,
                "as_run_partitions": as_run.partition_count if as_run else None,
                **_vendor_columns(record),
                "power_source": str(record.host_state.power_source),
                "low_power_mode": record.host_state.low_power_mode,
                "thermal_state": str(record.host_state.thermal_state),
                "load_avg_1m": record.host_state.load_avg_1m,
                "calibration_ratio": probe.ratio_to_baseline if probe else None,
                "notes": record.notes,
                "payload": canonical_json(record.model_dump(mode="json")),
            },
        )
        return measurement_id

    # -- reads -------------------------------------------------------------

    def query(self, sql: str, params: dict[str, Any] | None = None) -> duckdb.DuckDBPyRelation:
        """Run a read-only statement.

        Rejects anything that isn't obviously a read. The corpus has exactly one
        writer path and this is not it.
        """
        if not _READ_ONLY_STATEMENT.match(sql):
            raise ValueError(
                "CorpusStore.query accepts read-only statements only — the corpus "
                "is append-only (PROJECT.md §14.3)"
            )
        return self._conn.sql(sql, params=params)

    def get_measurement(self, measurement_id: str) -> MeasurementRecord | None:
        row = self._conn.execute(
            "SELECT payload FROM measurements WHERE measurement_id = $id",
            {"id": measurement_id},
        ).fetchone()
        return MeasurementRecord.model_validate_json(row[0]) if row else None

    def get_recipe(self, recipe_id: str) -> Recipe | None:
        row = self._conn.execute(
            "SELECT payload FROM recipes WHERE recipe_id = $id", {"id": recipe_id}
        ).fetchone()
        return Recipe.model_validate_json(row[0]) if row else None

    def has_measurement(
        self,
        recipe_id: str,
        device_id: str,
        harness_version: str,
        *,
        stress_profile: str = "clean",
    ) -> bool:
        """Has this cell already been measured under these exact conditions?

        Powers sweep resumption. ``gate_refused`` rows deliberately do not count:
        they record that we *could not* measure, so the cell is still outstanding.
        """
        row = self._conn.execute(
            """
            SELECT 1 FROM measurements
            WHERE recipe_id = $recipe_id
              AND device_id = $device_id
              AND harness_version = $harness_version
              AND stress_profile = $stress_profile
              AND outcome != 'gate_refused'
            LIMIT 1
            """,
            {
                "recipe_id": recipe_id,
                "device_id": device_id,
                "harness_version": harness_version,
                "stress_profile": stress_profile,
            },
        ).fetchone()
        return row is not None

    def count(self, table: str = "measurements") -> int:
        if table not in {"measurements", "recipes", "graph_fingerprints"}:
            raise ValueError(f"unknown table {table!r}")
        row = self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
        return int(row[0]) if row else 0

    # -- internals ---------------------------------------------------------

    def _exists(self, table: str, column: str, value: str) -> bool:
        row = self._conn.execute(
            f"SELECT 1 FROM {table} WHERE {column} = $value LIMIT 1",  # noqa: S608
            {"value": value},
        ).fetchone()
        return row is not None


def migrate(source: Path | str, destination: Path | str) -> dict[str, int]:
    """Copy an older corpus into the current schema, losslessly.

    Every table stores the record's canonical JSON in ``payload``, which exists for
    exactly this: the denormalised columns are a query convenience, and the payload is
    the record. Migration re-validates each payload against today's models and inserts
    it, so a schema that gained a column picks up ``NULL`` for the rows that predate it
    without anything being invented.

    **This does not violate immutability.** Hard rule #3 forbids `UPDATE`, and nothing
    here updates: records are copied unchanged into a new container, keeping their
    original ``harness_version``. A row measured under 0.3.6 stays a 0.3.6 row, which
    is what makes it honest to keep — and what tells a later reader why its newer
    columns are empty.
    """
    import json  # noqa: PLC0415

    source, destination = Path(source), Path(destination)
    if not source.exists():
        raise FileNotFoundError(f"no corpus at {source}")

    old = duckdb.connect(str(source), read_only=True)
    counts = {"recipes": 0, "graph_fingerprints": 0, "measurements": 0}
    try:
        with CorpusStore(destination) as new:
            for table, model, insert in (
                ("recipes", Recipe, lambda r: new.insert_recipe(r)),
                ("graph_fingerprints", GraphFingerprint, lambda f: new.insert_fingerprint(f)),
                ("measurements", MeasurementRecord, lambda m: new.insert_measurement(m)),
            ):
                for (payload,) in old.execute(f"SELECT payload FROM {table}").fetchall():
                    insert(model.model_validate(json.loads(payload)))
                    counts[table] += 1
    finally:
        old.close()
    return counts


def parse_harness_version(value: str) -> tuple[int, ...]:
    """Version as a comparable tuple.

    String comparison is wrong and quietly so: ``"0.3.10" < "0.3.9"`` is True, which
    would mark the newest generation of rows as the superseded one and publish the
    figures we had just corrected.
    """
    parts: list[int] = []
    for chunk in value.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def superseded_ids(store: CorpusStore) -> set[str]:
    """Measurements a later harness version has re-measured on the same cell.

    Nothing is deleted — hard rule #3 — and nothing needs to be. A cell measured
    again under a newer harness produces a new row, and the old one stays as an
    immutable record of what that instrument reported. This just says which rows are
    no longer the current answer, so the atlas and the published snapshot can show one
    generation instead of silently averaging four.

    Supersession is keyed on harness version, never on time. Repeats *within* a
    version are the repeatability evidence the atlas deliberately shows, so they must
    not supersede each other.
    """
    rows = store.query(
        "SELECT measurement_id, recipe_id, device_id, harness_version FROM measurements"
    ).fetchall()
    newest: dict[tuple[str, str], tuple[int, ...]] = {}
    for _, recipe_id, device_id, version in rows:
        cell = (recipe_id, device_id)
        parsed = parse_harness_version(version)
        if parsed > newest.get(cell, ()):
            newest[cell] = parsed
    return {
        mid
        for mid, recipe_id, device_id, version in rows
        if parse_harness_version(version) < newest[(recipe_id, device_id)]
    }
