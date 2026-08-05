"""Corpus immutability (PROJECT.md §14.3) and lossless round-trip.

The store having no mutation surface is a structural property, so it is tested
structurally — not just "we didn't call update", but "update is not callable".
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pytest

from edgefit.corpus import CorpusStore, DuplicateRecordError, export_parquet
from edgefit.schema import (
    AttentionVariant,
    DeviceFingerprint,
    GraphFingerprint,
    HostState,
    MeasurementRecord,
    Metrics,
    ModelRef,
    NormType,
    Outcome,
    QaiHubComputeUnit,
    QaiHubRuntimeConfig,
    Recipe,
    RunStats,
    TaskType,
)

MINILM = "hf:sentence-transformers/all-MiniLM-L6-v2"
SAMPLES = [4.81, 4.92, 5.03, 4.88, 5.21, 4.95]


@pytest.fixture
def store(tmp_path):
    with CorpusStore(tmp_path / "corpus.duckdb") as corpus:
        yield corpus


def _measurement(
    device: DeviceFingerprint, host_state: HostState, recipe: Recipe, **overrides
) -> MeasurementRecord:
    base = {
        "harness_version": "0.1.0",
        "recipe_id": recipe.recipe_id,
        "model_ref": recipe.model.ref,
        "device": device,
        "host_state": host_state,
        "outcome": Outcome.SUCCESS,
        "run_count": len(SAMPLES),
        "warmup_count": 3,
        "metrics": Metrics(
            latency_ms=RunStats.from_samples(SAMPLES),
            peak_rss_bytes=214 * 1024**2,
            unavailable={"power_mw": "no power instrumentation on this host"},
        ),
    }
    return MeasurementRecord(**(base | overrides))


class TestNoMutationSurface:
    def test_store_exposes_no_update_or_delete(self) -> None:
        public = {
            name
            for name, _ in inspect.getmembers(CorpusStore, callable)
            if not name.startswith("_")
        }
        forbidden = {
            name
            for name in public
            if any(word in name.lower() for word in ("update", "delete", "drop", "truncate", "set"))
        }
        assert not forbidden, f"corpus must be append-only; found {sorted(forbidden)}"

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM measurements",
            "UPDATE measurements SET outcome = 'success'",
            "DROP TABLE measurements",
            "  update measurements set run_count = 5",
        ],
    )
    def test_query_rejects_mutating_statements(self, store: CorpusStore, sql: str) -> None:
        with pytest.raises(ValueError, match="read-only"):
            store.query(sql)

    def test_query_allows_reads(self, store: CorpusStore) -> None:
        assert store.query("SELECT count(*) FROM measurements").fetchone() == (0,)


class TestInsert:
    def test_measurement_round_trips_losslessly(
        self, store: CorpusStore, device, host_state, cpu_recipe
    ) -> None:
        store.insert_recipe(cpu_recipe)
        record = _measurement(device, host_state, cpu_recipe)
        measurement_id = store.insert_measurement(record)

        restored = store.get_measurement(measurement_id)
        assert restored == record

    def test_duplicate_measurement_is_rejected(
        self, store: CorpusStore, device, host_state, cpu_recipe
    ) -> None:
        record = _measurement(device, host_state, cpu_recipe)
        store.insert_measurement(record)
        with pytest.raises(DuplicateRecordError, match="immutable"):
            store.insert_measurement(record)

    def test_repeat_measurement_of_same_config_is_allowed(
        self, store: CorpusStore, device, host_state, cpu_recipe
    ) -> None:
        """Repeatability data is the point — the same recipe measured twice is two rows."""
        first = _measurement(device, host_state, cpu_recipe)
        second = _measurement(
            device, host_state, cpu_recipe, created_at=first.created_at + timedelta(seconds=30)
        )
        store.insert_measurement(first)
        store.insert_measurement(second)
        assert store.count("measurements") == 2

    def test_duplicate_config_is_idempotent(self, store: CorpusStore, cpu_recipe) -> None:
        assert store.insert_recipe(cpu_recipe) == store.insert_recipe(cpu_recipe)
        assert store.count("recipes") == 1

    def test_failures_are_stored(
        self, store: CorpusStore, device, host_state, coreml_recipe
    ) -> None:
        """§5.8: failures train the tier-1 static filter, so they must persist."""
        record = _measurement(
            device,
            host_state,
            coreml_recipe,
            outcome=Outcome.LOWERING_FAILURE,
            failure_reason="CoreML EP rejected dynamic sequence dimension",
            metrics=None,
            run_count=0,
            warmup_count=0,
        )
        store.insert_measurement(record)
        rows = store.query(
            "SELECT outcome, failure_reason FROM measurements WHERE outcome != 'success'"
        ).fetchall()
        assert rows == [("lowering_failure", "CoreML EP rejected dynamic sequence dimension")]

    def test_denormalised_columns_are_queryable(
        self, store: CorpusStore, device, host_state, cpu_recipe
    ) -> None:
        """The atlas scans columns, not JSON."""
        store.insert_recipe(cpu_recipe)
        store.insert_measurement(_measurement(device, host_state, cpu_recipe))

        row = store.query(
            """
            SELECT m.soc, m.os_build, c.intended_provider, m.latency_p50_ms, m.latency_cv
            FROM measurements m JOIN recipes c USING (recipe_id)
            """
        ).fetchone()
        assert row is not None
        soc, os_build, provider, p50, cv = row
        assert (soc, os_build, provider) == ("Apple M2", "24C101", "CPUExecutionProvider")
        assert p50 == pytest.approx(4.935)
        assert 0 < cv < 0.1

    def test_every_runtime_variant_can_be_stored(self, store: CorpusStore) -> None:
        """The store must not know what an ONNX Runtime provider list is.

        It did: `insert_recipe` read `recipe.runtime.providers` directly, so the very
        first hosted recipe crashed with `'QaiHubRuntimeConfig' object has no
        attribute 'providers'` — after the device had been provisioned and the job
        had run, which is the expensive place to find out. §9.5 exists because a
        second runtime falsifies fields that look universal, and it did so on first
        contact. Adding ExecuTorch must not require touching this file.
        """
        hosted = Recipe(
            model=ModelRef(ref=MINILM, task=TaskType.EMBED),
            runtime=QaiHubRuntimeConfig(
                device_name="Samsung Galaxy S24 (Family)",
                compute_unit=QaiHubComputeUnit.NPU,
            ),
        )
        assert store.insert_recipe(hosted) == hosted.recipe_id

        row = store.query(
            "SELECT runtime_kind, intended_provider, providers FROM recipes"
        ).fetchone()
        assert row is not None
        kind, intended, providers = row
        assert (kind, intended) == ("qai_hub", "NPU")
        # Null, not the compute unit: we never chose a provider order, and a query
        # must not be able to read one as though we had.
        assert providers is None


class TestFingerprints:
    def test_round_trips(self, store: CorpusStore) -> None:
        fingerprint = GraphFingerprint(
            n_nodes=47,
            n_parameters=22_713_216,
            n_initializers=100,
            op_histogram={"MatMul": 24, "Add": 12, "LayerNormalization": 6},
            attention_variant=AttentionVariant.MHA,
            norm_type=NormType.LAYERNORM,
        )
        assert store.insert_fingerprint(fingerprint) == fingerprint.fingerprint_id
        assert store.count("graph_fingerprints") == 1


def test_export_parquet_is_readable(store: CorpusStore, tmp_path, device, host_state, cpu_recipe):
    store.insert_recipe(cpu_recipe)
    store.insert_measurement(_measurement(device, host_state, cpu_recipe))

    written = export_parquet(store, tmp_path / "export")
    assert written["measurements"].exists()

    reread = store.query(
        "SELECT count(*) FROM read_parquet($path)", {"path": str(written["measurements"])}
    ).fetchone()
    assert reread == (1,)


def test_created_at_survives_the_round_trip(store: CorpusStore, device, host_state, cpu_recipe):
    """Timezone handling silently corrupting timestamps is a classic corpus poisoner."""
    stamped = datetime(2026, 8, 2, 12, 34, 56, tzinfo=UTC)
    record = _measurement(device, host_state, cpu_recipe, created_at=stamped)
    restored = store.get_measurement(store.insert_measurement(record))
    assert restored is not None
    assert restored.created_at == stamped


def test_timestamp_column_is_readable_from_python(
    store: CorpusStore, device, host_state, cpu_recipe
):
    """Reading TIMESTAMPTZ back through SQL needs pytz installed.

    The JSON payload round-trip above passes without it, so this gap stayed
    invisible until `corpus list` blew up in the terminal.
    """
    stamped = datetime(2026, 8, 2, 12, 34, 56, tzinfo=UTC)
    store.insert_measurement(_measurement(device, host_state, cpu_recipe, created_at=stamped))

    row = store.query("SELECT created_at FROM measurements").fetchone()
    assert row is not None
    assert row[0] == stamped
