"""Preflight gate behaviour.

Host conditions are constructed rather than probed, so these tests assert the
*policy* deterministically on any machine, in any state.
"""

from __future__ import annotations

import multiprocessing
import time

import pytest

from edgefit.harness.gate import (
    BaselineStore,
    DeviceBusyError,
    GateThresholds,
    device_lock,
    evaluate_gate,
    run_calibration_probe,
)
from edgefit.schema import CalibrationProbe, HostState, PowerSource, ThermalState


def _state(**overrides) -> HostState:
    base = {
        "power_source": PowerSource.AC,
        "battery_percent": 100.0,
        "low_power_mode": False,
        "thermal_state": ThermalState.NOMINAL,
        "load_avg_1m": 0.4,
        "load_avg_5m": 0.5,
        "available_ram_bytes": 9 * 1024**3,
    }
    return HostState(**(base | overrides))


def _failed_check_names(state: HostState, **kwargs) -> set[str]:
    return {check.name for check in evaluate_gate(state, **kwargs).failures}


class TestGatePolicy:
    def test_idle_plugged_in_host_passes(self) -> None:
        assert evaluate_gate(_state()).passed

    def test_refuses_on_battery(self) -> None:
        """Battery changes DVFS and thermal headroom; the number would not compare."""
        assert "power source" in _failed_check_names(
            _state(power_source=PowerSource.BATTERY, battery_percent=8.0)
        )

    def test_refuses_low_power_mode(self) -> None:
        assert "low power mode" in _failed_check_names(_state(low_power_mode=True))

    def test_refuses_unknown_low_power_mode(self) -> None:
        """Unverifiable is not the same as fine — an unknown power policy can halve clocks."""
        assert "low power mode" in _failed_check_names(_state(low_power_mode=None))

    @pytest.mark.parametrize(
        "thermal", [ThermalState.FAIR, ThermalState.SERIOUS, ThermalState.CRITICAL]
    )
    def test_refuses_above_nominal_thermal(self, thermal: ThermalState) -> None:
        assert "thermal state" in _failed_check_names(_state(thermal_state=thermal))

    def test_skips_thermal_when_unavailable(self) -> None:
        """Gating on a missing API would make the harness unusable off macOS.

        The calibration probe is what covers that platform instead.
        """
        report = evaluate_gate(_state(thermal_state=ThermalState.UNAVAILABLE))
        assert report.passed
        assert "thermal state" not in {check.name for check in report.checks}

    def test_refuses_busy_host(self) -> None:
        assert "load average (1m)" in _failed_check_names(_state(load_avg_1m=5.98))

    def test_refuses_memory_pressure(self) -> None:
        assert "available memory" in _failed_check_names(
            _state(available_ram_bytes=512 * 1024**2)
        )

    def test_reports_every_failure_not_just_the_first(self) -> None:
        """An operator fixing one problem at a time is an operator wasting a morning."""
        failures = _failed_check_names(
            _state(
                power_source=PowerSource.BATTERY,
                low_power_mode=True,
                load_avg_1m=9.0,
                thermal_state=ThermalState.SERIOUS,
            )
        )
        assert failures == {
            "power source",
            "low power mode",
            "thermal state",
            "load average (1m)",
        }

    def test_reason_names_observed_and_required(self) -> None:
        reason = evaluate_gate(_state(load_avg_1m=5.98)).reason()
        assert "load average (1m) is 5.98" in reason
        assert "need <= 2.00" in reason

    def test_passing_gate_has_empty_reason(self) -> None:
        assert evaluate_gate(_state()).reason() == ""

    def test_thresholds_are_overridable(self) -> None:
        busy = _state(load_avg_1m=5.0)
        assert not evaluate_gate(busy).passed
        assert evaluate_gate(busy, GateThresholds(max_load_avg_1m=8.0)).passed


class TestCalibrationGate:
    def test_refuses_a_throttled_host(self) -> None:
        probe = CalibrationProbe(
            kernel="matmul_f32_1024_x5_v1",
            elapsed_ms=48.0,
            baseline_ms=12.0,
            ratio_to_baseline=4.0,
        )
        assert "calibration probe" in _failed_check_names(_state(), calibration_probe=probe)

    def test_accepts_a_host_at_baseline(self) -> None:
        probe = CalibrationProbe(
            kernel="matmul_f32_1024_x5_v1",
            elapsed_ms=12.4,
            baseline_ms=12.0,
            ratio_to_baseline=12.4 / 12.0,
        )
        assert evaluate_gate(_state(), calibration_probe=probe).passed

    def test_skips_the_check_without_a_baseline(self) -> None:
        """Nothing to compare against is not a failure; it is a first run."""
        probe = CalibrationProbe(kernel="matmul_f32_1024_x5_v1", elapsed_ms=12.4)
        report = evaluate_gate(_state(), calibration_probe=probe)
        assert report.passed
        assert "calibration probe" not in {check.name for check in report.checks}


class TestLoadAverageIsSupersededByTheProbe:
    """Load average lags by a minute and is inflated by our own measurements.

    Observed 2026-08-03: a golden session read load 1.71 at entry and 2.02 by its
    third fixture, purely from the harness's own subprocess activity, while the
    probe held at 1.09x baseline. The probe was right, so it wins.
    """

    def _probe(self, ratio: float) -> CalibrationProbe:
        return CalibrationProbe(
            kernel="matmul_f32_1024_x5_v1",
            elapsed_ms=12.0 * ratio,
            baseline_ms=12.0,
            ratio_to_baseline=ratio,
        )

    def test_high_load_does_not_veto_when_the_probe_is_healthy(self) -> None:
        report = evaluate_gate(_state(load_avg_1m=2.02), calibration_probe=self._probe(1.09))
        assert report.passed
        assert report.reason() == ""

    def test_the_reading_is_still_reported_as_an_advisory(self) -> None:
        """Superseded is not hidden — an operator should still see the number."""
        report = evaluate_gate(_state(load_avg_1m=2.02), calibration_probe=self._probe(1.09))
        advisory = next(c for c in report.checks if c.name == "load average (1m)")
        assert advisory.advisory
        assert not advisory.passed
        assert advisory.observed == "2.02"
        assert report.advisories == (advisory,)

    def test_probe_still_vetoes_a_genuinely_degraded_host(self) -> None:
        """Superseding load average must not mean nothing checks contention."""
        report = evaluate_gate(_state(load_avg_1m=0.4), calibration_probe=self._probe(3.0))
        assert not report.passed
        assert "calibration probe" in {c.name for c in report.failures}

    def test_load_average_still_gates_without_a_baseline(self) -> None:
        """With nothing better available it is all we have, so it must still bite."""
        unbaselined = CalibrationProbe(kernel="matmul_f32_1024_x5_v1", elapsed_ms=12.4)
        report = evaluate_gate(_state(load_avg_1m=5.98), calibration_probe=unbaselined)
        assert not report.passed
        assert "load average (1m)" in {c.name for c in report.failures}

    def test_load_average_still_gates_with_no_probe_at_all(self) -> None:
        report = evaluate_gate(_state(load_avg_1m=5.98))
        assert not report.passed
        assert "load average (1m)" in {c.name for c in report.failures}

    def test_probe_produces_a_plausible_measurement(self) -> None:
        probe = run_calibration_probe(baseline_ms=10.0)
        assert probe.kernel == "matmul_f32_1024_x5_v1"
        assert 0.1 < probe.elapsed_ms < 60_000
        assert probe.ratio_to_baseline == pytest.approx(probe.elapsed_ms / 10.0)


class TestBaselineStore:
    def test_returns_none_before_anything_is_recorded(self, tmp_path, device) -> None:
        assert BaselineStore(tmp_path / "baselines.json").get(device) is None

    def test_round_trips(self, tmp_path, device) -> None:
        store = BaselineStore(tmp_path / "baselines.json")
        store.record(device, 12.5)
        assert store.get(device) == pytest.approx(12.5)

    def test_keeps_the_fastest_time_seen(self, tmp_path, device) -> None:
        """A baseline captured on a busy machine would permanently soften the gate."""
        store = BaselineStore(tmp_path / "baselines.json")
        store.record(device, 12.5)
        store.record(device, 40.0)
        assert store.get(device) == pytest.approx(12.5)
        store.record(device, 11.0)
        assert store.get(device) == pytest.approx(11.0)

    def test_separates_two_units_of_the_same_sku(self, tmp_path, device) -> None:
        """The two-unit test is meaningless if both units share a baseline."""
        store = BaselineStore(tmp_path / "baselines.json")
        other_unit = device.model_copy(update={"unit_serial_hash": "ffffffffffff"})
        store.record(device, 12.5)
        assert store.get(other_unit) is None

    def test_survives_a_corrupt_file(self, tmp_path, device) -> None:
        path = tmp_path / "baselines.json"
        path.write_text("{ this is not json")
        store = BaselineStore(path)
        assert store.get(device) is None
        store.record(device, 12.5)
        assert store.get(device) == pytest.approx(12.5)


def _hold_lock(state_dir: str, device_json: str, ready, release) -> None:
    """Child-process helper: flock is per-process, so contention needs a real one."""
    import os

    os.environ["EDGEFIT_STATE_DIR"] = state_dir
    import importlib

    from edgefit.harness import gate as gate_module

    importlib.reload(gate_module)
    from edgefit.schema import DeviceFingerprint

    held = DeviceFingerprint.model_validate_json(device_json)
    with gate_module.device_lock(held):
        ready.set()
        release.wait(timeout=30)


class TestDeviceLock:
    def test_acquires_and_releases(self, tmp_path, device, monkeypatch) -> None:
        monkeypatch.setattr("edgefit.harness.gate.STATE_DIR", tmp_path)
        with device_lock(device):
            pass
        with device_lock(device):
            pass  # released, so re-acquirable

    def test_refuses_while_another_process_holds_it(self, tmp_path, device, monkeypatch) -> None:
        """PROJECT.md §5.6: exclusive access. Concurrent runs contend for cores."""
        monkeypatch.setattr("edgefit.harness.gate.STATE_DIR", tmp_path)
        ctx = multiprocessing.get_context("spawn")
        ready, release = ctx.Event(), ctx.Event()
        holder = ctx.Process(
            target=_hold_lock,
            args=(str(tmp_path), device.model_dump_json(), ready, release),
        )
        holder.start()
        try:
            assert ready.wait(timeout=30), "helper process never acquired the lock"
            started = time.monotonic()
            with (
                pytest.raises(DeviceBusyError, match="another measurement holds"),
                device_lock(device, timeout_s=0.5),
            ):
                pass
            assert time.monotonic() - started >= 0.5, "should have waited out the timeout"
        finally:
            release.set()
            holder.join(timeout=30)
