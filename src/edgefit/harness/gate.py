"""Preflight gate — refuse to measure an unfit host.

PROJECT.md §5.6 specifies exclusive device access and a thermal gate. Apple
Silicon exposes no unprivileged temperature, so the gate is built from what can
actually be known:

* **Categorical facts** — AC power, low-power mode, load average, free memory,
  ``NSProcessInfo.thermalState``. Cheap, exact, and each one independently
  capable of ruining a measurement.
* **A measured throttle probe** — a fixed deterministic kernel timed against this
  host's recorded baseline. If the machine has got slower, it is throttled or
  contended, whatever the sensors claim. This is measured, not estimated, and it
  is stored on every record so the corpus can be re-filtered if the threshold
  later proves wrong.

The gate *refuses* rather than annotating, because hard rule #1 says a missing
number beats a wrong one, and a run on a hot, busy, battery-powered laptop is a
wrong number that looks completely reasonable.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgefit.schema.common import PowerSource, ThermalState
from edgefit.schema.host import DeviceFingerprint, HostState
from edgefit.schema.measurement import CalibrationProbe, _percentile

STATE_DIR = Path(os.environ.get("EDGEFIT_STATE_DIR", ".edgefit"))

# Identity of the probe workload. Changing the kernel invalidates every stored
# baseline, so the name is versioned and comparisons are keyed on it.
CALIBRATION_KERNEL = "matmul_f32_1024_x5_v1"
_MATRIX_DIM = 1024
_MATMULS_PER_ROUND = 5
_PROBE_ROUNDS = 3

# Name of the probe check, so other code can reason about it without matching strings.
_PROBE_CHECK = "calibration probe"


class DeviceBusyError(Exception):
    """Another measurement holds this device. Concurrency invalidates timings."""


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------

_THERMAL_ORDER = {
    ThermalState.NOMINAL: 0,
    ThermalState.FAIR: 1,
    ThermalState.SERIOUS: 2,
    ThermalState.CRITICAL: 3,
}


@dataclass(frozen=True)
class GateThresholds:
    """Deliberately strict. Loosening these is a methodology change and belongs
    in `/methodology`, not in a call site."""

    require_ac_power: bool = True
    forbid_low_power_mode: bool = True
    max_thermal_state: ThermalState = ThermalState.NOMINAL
    max_load_avg_1m: float = 2.0
    min_available_ram_bytes: int = 2 * 1024**3
    # A host >15% slower than its own best is throttled or contended.
    max_calibration_ratio: float = 1.15


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    observed: str
    required: str
    advisory: bool = False
    """Reported but not gating.

    Used when a stronger check supersedes a weaker one measuring the same thing.
    Keeping the weak reading visible is useful to an operator; letting it veto a
    run it is not qualified to judge is not.
    """


@dataclass(frozen=True)
class GateReport:
    checks: tuple[GateCheck, ...]
    host_state: HostState
    calibration_probe: CalibrationProbe | None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks if not check.advisory)

    @property
    def failures(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed and not check.advisory)

    @property
    def passed_ignoring_probe(self) -> bool:
        """True when every gating check except the calibration probe passed.

        The categorical checks (AC power, low-power mode, thermal state, memory)
        cannot be gamed by the baseline, so a probe sample taken under these
        conditions is a legitimate observation of healthy throughput even if the
        ratio check itself failed. That is what lets the baseline self-correct
        instead of ratcheting.
        """
        return all(
            check.passed
            for check in self.checks
            if not check.advisory and check.name != _PROBE_CHECK
        )

    @property
    def advisories(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if check.advisory and not check.passed)

    def reason(self) -> str:
        """One-line summary, suitable for ``MeasurementRecord.failure_reason``."""
        if self.passed:
            return ""
        return "host unfit to measure: " + "; ".join(
            f"{check.name} is {check.observed} (need {check.required})" for check in self.failures
        )


# --------------------------------------------------------------------------
# Calibration probe
# --------------------------------------------------------------------------


def run_calibration_probe(baseline_ms: float | None = None) -> CalibrationProbe:
    """Time a fixed deterministic kernel; compare against this host's baseline.

    Best-of-N, because the question is "how fast *can* this machine go right
    now" — the minimum is the least contaminated estimate of available compute.
    """
    rng = np.random.default_rng(seed=0)
    left = rng.standard_normal((_MATRIX_DIM, _MATRIX_DIM), dtype=np.float32)
    right = rng.standard_normal((_MATRIX_DIM, _MATRIX_DIM), dtype=np.float32)

    best_ns = None
    for _ in range(_PROBE_ROUNDS):
        started = time.perf_counter_ns()
        for _ in range(_MATMULS_PER_ROUND):
            left @ right
        elapsed = time.perf_counter_ns() - started
        best_ns = elapsed if best_ns is None else min(best_ns, elapsed)

    elapsed_ms = (best_ns or 0) / 1e6
    return CalibrationProbe(
        kernel=CALIBRATION_KERNEL,
        elapsed_ms=elapsed_ms,
        baseline_ms=baseline_ms,
        ratio_to_baseline=(elapsed_ms / baseline_ms) if baseline_ms else None,
    )


class BaselineStore:
    """Per-unit calibration baselines, on disk.

    Keyed on the physical unit rather than the SKU: two Mac minis of one model do
    not necessarily deliver the same sustained clocks, and pretending they do is
    how a fleet quietly develops a systematic bias.

    The baseline is a **low percentile of recent healthy samples**, not the fastest
    time ever seen. An all-time minimum is a ratchet: one unusually quiet moment
    records 7.6 ms, the machine's normal healthy throughput is 8.4 ms, and from
    then on nothing passes a 1.15x threshold. Observed exactly that way — a sweep
    lost 16 of 45 cells to a baseline it could no longer reach.

    A low percentile over a window keeps the original protection (a few slow
    samples cannot soften the gate much) without the ratchet, and samples are only
    recorded when the categorical checks pass, so a genuinely throttled host
    cannot quietly relax its own threshold.
    """

    #: Enough history to make a percentile meaningful without tracking a whole day.
    WINDOW = 32
    #: Below this, fall back to the minimum. Conservative, and self-corrects as
    #: samples accumulate, because samples are recorded even when the ratio fails.
    MIN_SAMPLES_FOR_PERCENTILE = 5
    PERCENTILE = 0.10

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (STATE_DIR / "baselines.json")

    def _key(self, device: DeviceFingerprint) -> str:
        return f"{device.sku_id}:{device.unit_serial_hash or 'unknown'}:{CALIBRATION_KERNEL}"

    def _load(self) -> dict[str, list[float]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        # Migrate the original scalar format by treating it as a single sample.
        migrated: dict[str, list[float]] = {}
        for key, value in raw.items():
            if isinstance(value, int | float):
                migrated[key] = [float(value)]
            elif isinstance(value, dict) and isinstance(value.get("samples"), list):
                migrated[key] = [float(v) for v in value["samples"]]
            elif isinstance(value, list):
                migrated[key] = [float(v) for v in value]
        return migrated

    def _write(self, data: dict[str, list[float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: {"samples": samples} for key, samples in data.items()}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def samples(self, device: DeviceFingerprint) -> list[float]:
        return self._load().get(self._key(device), [])

    def get(self, device: DeviceFingerprint) -> float | None:
        """Healthy-throughput estimate for this unit, or None if never measured."""
        samples = self.samples(device)
        if not samples:
            return None
        if len(samples) < self.MIN_SAMPLES_FOR_PERCENTILE:
            return min(samples)
        return _percentile(sorted(samples), self.PERCENTILE)

    def record(self, device: DeviceFingerprint, elapsed_ms: float) -> float:
        """Add a sample and return the resulting baseline.

        Only call this when the categorical checks passed — see
        ``GateReport.passed_ignoring_probe``.
        """
        data = self._load()
        key = self._key(device)
        samples = [*data.get(key, []), float(elapsed_ms)][-self.WINDOW :]
        data[key] = samples
        self._write(data)
        return self.get(device) or float(elapsed_ms)


# --------------------------------------------------------------------------
# Exclusive device access (PROJECT.md §5.6)
# --------------------------------------------------------------------------


@contextmanager
def device_lock(device: DeviceFingerprint, timeout_s: float = 0.0) -> Iterator[None]:
    """Hold an exclusive lock on this device for the duration of a measurement."""
    lock_dir = STATE_DIR / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{device.device_id}.lock"

    deadline = time.monotonic() + timeout_s
    handle = lock_path.open("w")
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise DeviceBusyError(
                        f"another measurement holds {device.model} ({lock_path}). "
                        "Concurrent runs contend for the same cores and invalidate timings."
                    ) from exc
                time.sleep(0.25)
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def wait_until_fit(
    thresholds: GateThresholds | None = None,
    *,
    device: DeviceFingerprint | None = None,
    timeout_s: float = 600.0,
    poll_s: float = 20.0,
    on_wait: Callable[[GateReport, float], None] | None = None,
) -> GateReport:
    """Poll until the host is fit, or give up after ``timeout_s``.

    PROJECT.md §5.6 specifies the thermal gate as "idle until temp below
    threshold" — waiting, not refusing. A single measurement is right to refuse
    immediately, but a sweep that refuses is a sweep that writes fifty
    ``gate_refused`` rows and measures nothing. Waiting is what lets a laptop
    produce a usable corpus overnight.
    """
    from edgefit.harness.hostinfo import probe_device, probe_state  # noqa: PLC0415

    device = device or probe_device()
    baselines = BaselineStore()
    deadline = time.monotonic() + timeout_s

    while True:
        probe = run_calibration_probe(baselines.get(device))
        report = evaluate_gate(probe_state(), thresholds, probe)
        if report.passed_ignoring_probe:
            # A legitimate observation of healthy throughput even if the ratio
            # check failed — this is what lets the baseline self-correct.
            baselines.record(device, probe.elapsed_ms)
        if report.passed:
            return report
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return report
        if on_wait is not None:
            on_wait(report, remaining)
        time.sleep(min(poll_s, remaining))


def _format_bytes(value: int | None) -> str:
    return f"{value / 1024**3:.1f} GiB" if value is not None else "unknown"


def evaluate_gate(
    host_state: HostState,
    thresholds: GateThresholds | None = None,
    calibration_probe: CalibrationProbe | None = None,
) -> GateReport:
    """Decide whether this host may be measured on. Pure — takes probes, no I/O."""
    limits = thresholds or GateThresholds()
    checks: list[GateCheck] = []

    if limits.require_ac_power:
        checks.append(
            GateCheck(
                name="power source",
                passed=host_state.power_source is PowerSource.AC,
                observed=str(host_state.power_source)
                + (
                    f" ({host_state.battery_percent:.0f}%)"
                    if host_state.battery_percent is not None
                    else ""
                ),
                required="ac",
            )
        )

    if limits.forbid_low_power_mode:
        # Unknown is not a pass: an unverifiable power policy can halve clocks.
        checks.append(
            GateCheck(
                name="low power mode",
                passed=host_state.low_power_mode is False,
                observed=(
                    "unknown"
                    if host_state.low_power_mode is None
                    else str(host_state.low_power_mode).lower()
                ),
                required="off",
            )
        )

    # Thermal is only gated when it is actually known. Failing on UNAVAILABLE
    # would make the gate unusable on every platform lacking the API; the
    # calibration probe is what covers that case.
    if host_state.thermal_state is not ThermalState.UNAVAILABLE:
        checks.append(
            GateCheck(
                name="thermal state",
                passed=_THERMAL_ORDER[host_state.thermal_state]
                <= _THERMAL_ORDER[limits.max_thermal_state],
                observed=str(host_state.thermal_state),
                required=f"<= {limits.max_thermal_state}",
            )
        )

    if host_state.load_avg_1m is not None:
        # Load average and the calibration probe measure the same thing —
        # contention — but the probe measures it directly and instantaneously,
        # whereas load average is a one-minute decayed mean that *our own*
        # preceding measurements inflate. Observed on 2026-08-03: `doctor` read
        # 1.71, and by the third golden fixture load had drifted to 2.02 purely
        # from this harness's subprocess activity, while the probe held steady at
        # 1.09x baseline. The probe was right; load average vetoed a run it was
        # not qualified to judge.
        #
        # So when a baseline exists the probe is authoritative and this reading
        # becomes advisory. Without a baseline there is nothing better, and it
        # gates as normal.
        superseded = (
            calibration_probe is not None and calibration_probe.ratio_to_baseline is not None
        )
        checks.append(
            GateCheck(
                name="load average (1m)",
                passed=host_state.load_avg_1m <= limits.max_load_avg_1m,
                observed=f"{host_state.load_avg_1m:.2f}",
                required=(
                    "advisory — superseded by the calibration probe"
                    if superseded
                    else f"<= {limits.max_load_avg_1m:.2f}"
                ),
                advisory=superseded,
            )
        )

    if host_state.available_ram_bytes is not None:
        checks.append(
            GateCheck(
                name="available memory",
                passed=host_state.available_ram_bytes >= limits.min_available_ram_bytes,
                observed=_format_bytes(host_state.available_ram_bytes),
                required=f">= {_format_bytes(limits.min_available_ram_bytes)}",
            )
        )

    if calibration_probe is not None and calibration_probe.ratio_to_baseline is not None:
        checks.append(
            GateCheck(
                name=_PROBE_CHECK,
                passed=calibration_probe.ratio_to_baseline <= limits.max_calibration_ratio,
                observed=f"{calibration_probe.ratio_to_baseline:.2f}x baseline "
                f"({calibration_probe.elapsed_ms:.1f} ms)",
                required=f"<= {limits.max_calibration_ratio:.2f}x",
            )
        )

    return GateReport(
        checks=tuple(checks), host_state=host_state, calibration_probe=calibration_probe
    )
