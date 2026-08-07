"""Controlled stress for the second rung of the validation ladder (PROJECT.md §5.6).

> Everyone benchmarks clean cold devices; users don't have those.

The ladder is about *trust*, distinct from the cost cascade which is about search
economics. Only its first rung exists: every local row so far says
``stress_profile = clean``, because the harness refuses to measure a contended host
at all. That refusal is correct for a clean-bench number and it is precisely what
makes the second rung possible — a stress measurement is only interpretable if the
machine was verifiably quiet *before* a known load was applied.

So the protocol is deliberately two-phase:

1. **Wait for the gate to pass.** Proves the host is otherwise idle. Without this a
   "stress" measurement is indistinguishable from measuring on a busy laptop, which
   is the thing §13 calls confident garbage.
2. **Apply a known, described stress**, then measure through it.

The stress parameters are recorded on the row alongside the profile, because "under
memory pressure" means nothing without saying how much pressure.

This module deliberately does not try to reproduce a phone's conditions. It produces
*a* controlled perturbation on this host and labels it honestly. Quantifying the
real benchmark-to-production gap (§17.7, anecdotally 3–5x at P99) needs the canary
rung and real user devices; this rung is what makes the question askable at all.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass

from edgefit.schema.common import StressProfile

#: A CPU burner as a standalone program. Deliberately `subprocess` rather than
#: `multiprocessing`: macOS defaults to the *spawn* start method, which re-imports
#: the parent's `__main__` in every child. Started from anywhere that lacks an
#: importable `__main__` — a REPL, `python -c`, pytest in some configurations — the
#: workers die on launch and the stressor silently applies **no load at all**. That
#: was observed: the calibration probe read 1.00x under a nominal four-worker load.
#: A stress profile that quietly applies nothing is worse than not having one, since
#: it labels an ordinary measurement as a stressed one.
_BURNER = (
    # Raise the worker to USER_INTERACTIVE before spinning. Without this the stressor
    # is nearly inert on Apple Silicon: eight burners saturated the machine (611% CPU
    # across 8 cores) and slowed the calibration probe by **4%**, because the darwin
    # scheduler parks default-QoS background work on efficiency cores and leaves the
    # foreground measurement on the performance cores. Competing for the cores the
    # measurement actually uses requires asking for them.
    "import ctypes\n"
    "try:\n"
    "    ctypes.CDLL(None).pthread_set_qos_class_self_np(0x21, 0)\n"
    "except Exception:\n"
    "    pass\n"
    "x = 0\n"
    "while True:\n"
    "    for _ in range(100000):\n"
    "        x = (x * 1103515245 + 12345) & 0x7FFFFFFF\n"
)


@dataclass(frozen=True)
class StressSpec:
    """A described perturbation. The description travels with the measurement."""

    profile: StressProfile
    workers: int = 0
    balloon_mib: int = 0
    soak_s: float = 0.0

    @property
    def description(self) -> str:
        """Human-readable, recorded on the row. Vague labels are unfalsifiable."""
        if self.profile is StressProfile.CLEAN:
            return "no applied load; host gate-verified idle"
        if self.profile is StressProfile.CONCURRENT_LOAD:
            return f"{self.workers} CPU-bound worker processes running throughout"
        if self.profile is StressProfile.MEMORY_PRESSURE:
            return f"{self.balloon_mib} MiB resident balloon held throughout"
        if self.profile is StressProfile.THERMAL_SOAK:
            return f"{self.soak_s:.0f}s of sustained load immediately before measuring"
        return "unknown"


class Stressor:
    """Applies a StressSpec for the duration of a `with` block.

    Cleanup is unconditional. A leaked worker process would silently contend with
    every later measurement on this host, which would be a much worse bug than the
    one this module exists to explore — the corpus would fill with rows labelled
    ``clean`` that were not.
    """

    def __init__(self, spec: StressSpec) -> None:
        self.spec = spec
        self._workers: list[subprocess.Popen] = []
        self._balloon: bytearray | None = None

    def __enter__(self) -> Stressor:
        spec = self.spec
        for _ in range(spec.workers):
            self._workers.append(
                subprocess.Popen(  # noqa: S603
                    [sys.executable, "-c", _BURNER],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        if spec.balloon_mib:
            # Touched every page: an untouched allocation is virtual and applies no
            # real pressure, which would make the profile a lie.
            self._balloon = bytearray(spec.balloon_mib * 1024 * 1024)
            self._balloon[::4096] = b"\x01" * len(self._balloon[::4096])
        if spec.workers:
            # Let the burners actually reach the CPU before anything is timed.
            time.sleep(0.5)
        if spec.soak_s:
            time.sleep(spec.soak_s)
        return self

    def __exit__(self, *exc: object) -> None:
        for proc in self._workers:
            proc.terminate()
        for proc in self._workers:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - belt and braces
                proc.kill()
                proc.wait(timeout=5)
        self._workers.clear()
        self._balloon = None

    def verify_applied(self, baseline_ms: float) -> float:
        """Ratio of delivered compute under stress against an idle baseline.

        Called so the row can record that the load was *observed*, not merely
        requested. A worker that failed to start is otherwise invisible.
        """
        from edgefit.harness.gate import run_calibration_probe  # noqa: PLC0415

        return run_calibration_probe().elapsed_ms / baseline_ms
