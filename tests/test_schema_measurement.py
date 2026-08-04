"""Measurement record invariants — the mechanical form of PROJECT.md §14.

These tests are the actual enforcement of the hard rules. If one of them starts
failing, the corpus is no longer trustworthy, which is the only Critical risk in
§13. Treat a failure here as a stop-the-line event, not a flaky test.
"""

from __future__ import annotations

import statistics

import pytest
from pydantic import ValidationError

from edgefit.schema import (
    MIN_RUNS,
    DeviceFingerprint,
    FallbackReport,
    HostState,
    MeasurementRecord,
    Metrics,
    Outcome,
    RunStats,
    StressProfile,
)

SAMPLES = [4.81, 4.92, 5.03, 4.88, 5.21, 4.95]


def _record(device: DeviceFingerprint, host_state: HostState, **overrides: object):
    base = {
        "harness_version": "0.1.0",
        "recipe_id": "abc123",
        "model_ref": "hf:sentence-transformers/all-MiniLM-L6-v2",
        "device": device,
        "host_state": host_state,
        "outcome": Outcome.SUCCESS,
        "run_count": len(SAMPLES),
        "warmup_count": 3,
        "metrics": Metrics(latency_ms=RunStats.from_samples(SAMPLES)),
    }
    return MeasurementRecord(**(base | overrides))  # type: ignore[arg-type]


class TestRunStats:
    def test_rejects_fewer_than_five_runs(self) -> None:
        """PROJECT.md §14.2. Not configurable downwards."""
        with pytest.raises(ValueError, match="below the mandatory minimum"):
            RunStats.from_samples([1.0, 2.0, 3.0, 4.0])

    def test_accepts_exactly_five(self) -> None:
        assert RunStats.from_samples([1.0, 2.0, 3.0, 4.0, 5.0]).n == MIN_RUNS

    def test_derives_aggregates_from_samples(self) -> None:
        stats = RunStats.from_samples(SAMPLES)
        assert stats.minimum == min(SAMPLES)
        assert stats.maximum == max(SAMPLES)
        assert stats.mean == pytest.approx(statistics.fmean(SAMPLES))
        assert stats.stddev == pytest.approx(statistics.stdev(SAMPLES))
        assert stats.cv == pytest.approx(stats.stddev / stats.mean)

    def test_p95_matches_numpy_linear_interpolation(self) -> None:
        """Documented methodology: numpy's default percentile method."""
        numpy = pytest.importorskip("numpy")
        stats = RunStats.from_samples(SAMPLES)
        assert stats.p95 == pytest.approx(float(numpy.percentile(SAMPLES, 95)))
        assert stats.p50 == pytest.approx(float(numpy.percentile(SAMPLES, 50)))

    def test_rejects_variance_that_did_not_come_from_the_samples(self) -> None:
        """Hard rule #1: a fabricated number must not be representable."""
        honest = RunStats.from_samples(SAMPLES)
        payload = honest.model_dump()
        payload["stddev"] = 0.0001  # a suspiciously tidy result
        with pytest.raises(ValidationError, match="not derivable from the samples"):
            RunStats.model_validate(payload)

    def test_rejects_mismatched_n(self) -> None:
        payload = RunStats.from_samples(SAMPLES).model_dump()
        payload["n"] = 100
        with pytest.raises(ValidationError, match="does not match"):
            RunStats.model_validate(payload)


class TestMetricsUnavailability:
    def test_requires_a_reason_field_to_name_a_real_metric(self) -> None:
        with pytest.raises(ValidationError, match="unknown metric"):
            Metrics(unavailable={"gpu_temperature": "no sensor"})

    def test_rejects_explaining_away_a_populated_metric(self) -> None:
        with pytest.raises(ValidationError, match="populated but listed as unavailable"):
            Metrics(peak_rss_bytes=1024, unavailable={"peak_rss_bytes": "not measured"})

    def test_accepts_an_honest_absence(self) -> None:
        metrics = Metrics(
            latency_ms=RunStats.from_samples(SAMPLES),
            unavailable={"power_mw": "no power instrumentation on this host"},
        )
        assert metrics.power_mw is None

    def test_primary_stats_prefers_latency_then_ttft(self) -> None:
        stats = RunStats.from_samples(SAMPLES)
        assert Metrics(latency_ms=stats).primary_stats is stats
        assert Metrics(ttft_ms=stats).primary_stats is stats
        assert Metrics().primary_stats is None


class TestMeasurementRecord:
    def test_accepts_a_well_formed_success(
        self, device: DeviceFingerprint, host_state: HostState
    ) -> None:
        record = _record(device, host_state)
        assert record.outcome is Outcome.SUCCESS
        assert len(record.measurement_id) == 16

    def test_success_requires_metrics(
        self, device: DeviceFingerprint, host_state: HostState
    ) -> None:
        with pytest.raises(ValidationError, match="must carry metrics"):
            _record(device, host_state, metrics=None)

    def test_success_requires_a_timing_distribution(
        self, device: DeviceFingerprint, host_state: HostState
    ) -> None:
        """A "successful" run with no variance data is not a measurement."""
        with pytest.raises(ValidationError, match="timing distribution"):
            _record(device, host_state, metrics=Metrics(peak_rss_bytes=1024), run_count=0)

    def test_run_count_must_agree_with_samples(
        self, device: DeviceFingerprint, host_state: HostState
    ) -> None:
        with pytest.raises(ValidationError, match="disagrees with"):
            _record(device, host_state, run_count=99)

    @pytest.mark.parametrize(
        "outcome",
        [
            Outcome.LOWERING_FAILURE,
            Outcome.RUNTIME_FAILURE,
            Outcome.ACCURACY_FAILURE,
            Outcome.GATE_REFUSED,
        ],
    )
    def test_failure_requires_a_reason(
        self, device: DeviceFingerprint, host_state: HostState, outcome: Outcome
    ) -> None:
        with pytest.raises(ValidationError, match="requires a failure_reason"):
            _record(device, host_state, outcome=outcome, metrics=None, run_count=0)

    def test_failure_is_a_first_class_record(
        self, device: DeviceFingerprint, host_state: HostState
    ) -> None:
        """§5.8: failures train the tier-1 static filter, so they must be storable."""
        record = _record(
            device,
            host_state,
            outcome=Outcome.LOWERING_FAILURE,
            failure_reason="CoreML EP rejected dynamic sequence dimension",
            metrics=None,
            run_count=0,
            warmup_count=0,
        )
        assert record.outcome is Outcome.LOWERING_FAILURE

    def test_id_distinguishes_repeat_measurements(
        self, device: DeviceFingerprint, host_state: HostState
    ) -> None:
        """Repeatability data is the point; two runs must not collapse to one id."""
        first = _record(device, host_state)
        second = _record(device, host_state, created_at=first.created_at.replace(microsecond=1))
        assert first.measurement_id != second.measurement_id

    def test_defaults_to_the_clean_bench(
        self, device: DeviceFingerprint, host_state: HostState
    ) -> None:
        """Only the clean rung of the §5.6 ladder exists today, so it is the default."""
        assert _record(device, host_state).stress_profile is StressProfile.CLEAN

    @pytest.mark.parametrize("profile", list(StressProfile))
    def test_records_every_rung_of_the_validation_ladder(
        self, device: DeviceFingerprint, host_state: HostState, profile: StressProfile
    ) -> None:
        """§2.2 puts the benchmark-to-production P99 gap at 3-5x.

        A corpus that cannot distinguish a clean run from a thermally soaked one
        can never quantify that gap, which is why the field lands before the
        stress bench that populates it.
        """
        assert _record(device, host_state, stress_profile=profile).stress_profile is profile

    def test_stress_profile_changes_the_measurement_id(
        self, device: DeviceFingerprint, host_state: HostState
    ) -> None:
        """Same recipe under different conditions is a different observation."""
        clean = _record(device, host_state)
        soaked = _record(
            device, host_state, stress_profile=StressProfile.THERMAL_SOAK,
            created_at=clean.created_at,
        )
        assert clean.measurement_id != soaked.measurement_id

    def test_id_reflects_os_build(
        self, device: DeviceFingerprint, host_state: HostState
    ) -> None:
        """Stage 3 exists because an OS update changes the answer."""
        upgraded = device.model_copy(update={"os_build": "24D60"})
        assert device.device_id != upgraded.device_id
        assert device.sku_id == upgraded.sku_id


class TestFallbackReport:
    def test_rejects_more_intended_nodes_than_total(self) -> None:
        with pytest.raises(ValidationError, match="exceeds"):
            FallbackReport(
                intended_provider="CoreMLExecutionProvider",
                nodes_total=10,
                nodes_on_intended=11,
                fallback_node_pct=0.0,
            )

    def test_carries_the_actionable_diagnostic(self) -> None:
        report = FallbackReport(
            intended_provider="CoreMLExecutionProvider",
            nodes_total=47,
            nodes_on_intended=42,
            fallback_node_pct=10.6,
            unclaimed_op_types={"LayerNormalization": 4, "Gather": 1},
        )
        assert report.unclaimed_op_types["LayerNormalization"] == 4
