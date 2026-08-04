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
from edgefit.schema.common import canonical_json
from edgefit.schema.fingerprint import GraphFingerprint
from edgefit.schema.measurement import MeasurementRecord
from edgefit.schema.recipe import Recipe

DEFAULT_CORPUS_PATH = Path("corpus/measurements.duckdb")

# query() is for reading. This is a guard rail, not a security boundary — but it
# means an accidental mutating statement fails loudly instead of quietly
# violating the one rule the whole corpus depends on.
_READ_ONLY_STATEMENT = re.compile(
    r"^\s*(select|with|describe|explain|pragma|show)\b", re.IGNORECASE
)


class DuplicateRecordError(Exception):
    """Raised when an id already exists.

    Not an error to paper over: identical content hash means identical content,
    so a duplicate insert is either a retry bug or a hash collision, and both
    deserve a stack trace.
    """


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
                "providers": ",".join(str(p) for p in recipe.runtime.providers),
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
        probe = record.calibration_probe

        self._conn.execute(
            """
            INSERT INTO measurements VALUES (
                $measurement_id, $schema_version, $harness_version, $created_at,
                $recipe_id, $model_ref, $graph_fingerprint_id,
                $device_id, $sku_id, $device_model, $soc, $os_name, $os_version, $os_build,
                $stress_profile, $outcome, $failure_reason, $run_count, $warmup_count,
                $latency_p50_ms, $latency_p95_ms, $latency_cv,
                $ttft_p50_ms, $decode_tok_s_p50, $sustained_tok_s_5min,
                $peak_rss_bytes, $artifact_bytes, $lowering_ms,
                $accuracy, $accuracy_delta_vs_fp16, $power_mw,
                $fallback_node_pct, $fallback_flops_pct, $fallback_time_pct,
                $power_source, $low_power_mode, $thermal_state, $load_avg_1m,
                $calibration_ratio, $payload
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
                "stress_profile": str(record.stress_profile),
                "outcome": str(record.outcome),
                "failure_reason": record.failure_reason,
                "run_count": record.run_count,
                "warmup_count": record.warmup_count,
                "latency_p50_ms": latency.p50 if latency else None,
                "latency_p95_ms": latency.p95 if latency else None,
                "latency_cv": latency.cv if latency else None,
                "ttft_p50_ms": ttft.p50 if ttft else None,
                "decode_tok_s_p50": decode.p50 if decode else None,
                "sustained_tok_s_5min": metrics.sustained_tok_s_5min if metrics else None,
                "peak_rss_bytes": metrics.peak_rss_bytes if metrics else None,
                "artifact_bytes": metrics.artifact_bytes if metrics else None,
                "lowering_ms": metrics.lowering_ms if metrics else None,
                "accuracy": metrics.accuracy if metrics else None,
                "accuracy_delta_vs_fp16": metrics.accuracy_delta_vs_fp16 if metrics else None,
                "power_mw": metrics.power_mw if metrics else None,
                "fallback_node_pct": fallback.fallback_node_pct if fallback else None,
                "fallback_flops_pct": fallback.fallback_flops_pct if fallback else None,
                "fallback_time_pct": fallback.fallback_time_pct if fallback else None,
                "power_source": str(record.host_state.power_source),
                "low_power_mode": record.host_state.low_power_mode,
                "thermal_state": str(record.host_state.thermal_state),
                "load_avg_1m": record.host_state.load_avg_1m,
                "calibration_ratio": probe.ratio_to_baseline if probe else None,
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
