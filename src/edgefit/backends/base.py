"""Backend protocol.

Shaped around the tiered evaluation cascade (PROJECT.md §5.5), which is the
economic core of the whole system:

* ``analyze`` — tier 1. Free-ish, seconds. Does it lower? How big? What did the
  partitioner actually claim? Expected to kill 60–80% of candidates.
* ``measure`` — tier 2. Cheap, ~1 min. Real timings on real hardware.

Tier 3 (accuracy) is not a backend concern; it consumes artifacts these produce.

Keeping the surface this small is deliberate. Backend #2 (ExecuTorch) exists to
stress-test this abstraction, and an abstraction with four methods survives that
better than one with twenty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from edgefit.schema.common import RuntimeKind
from edgefit.schema.fingerprint import GraphFingerprint
from edgefit.schema.measurement import FallbackReport
from edgefit.schema.recipe import Recipe


@dataclass(frozen=True)
class StaticAnalysis:
    """Tier-1 result: what we can learn without timing anything."""

    lowered: bool
    artifact_bytes: int
    failure_reason: str | None = None
    fingerprint: GraphFingerprint | None = None
    fallback: FallbackReport | None = None
    fallback_as_run: FallbackReport | None = None
    lowering_ms: float | None = None


@dataclass(frozen=True)
class DeviceRun:
    """Tier-2 result: raw timings from one measurement session.

    Deliberately raw. Aggregation into ``RunStats`` happens in one place so the
    variance policy cannot drift between backends.
    """

    samples_ms: list[float] = field(default_factory=list)
    peak_rss_bytes: int | None = None
    warmup_count: int = 0
    outputs: dict[str, list] | None = None
    failure_reason: str | None = None

    # Generative tasks report two distributions instead of one, because prefill and
    # decode are different workloads (PROJECT.md §2.3): prefill is compute-bound and
    # yields TTFT, decode is bandwidth-bound and yields tok/s. Averaging them into a
    # single "latency" would describe neither.
    ttft_samples_ms: list[float] = field(default_factory=list)
    decode_samples_tok_s: list[float] = field(default_factory=list)
    generated_tokens: list[int] | None = None

    @property
    def succeeded(self) -> bool:
        if self.failure_reason is not None:
            return False
        return bool(self.samples_ms) or bool(self.ttft_samples_ms)

    @property
    def is_generative(self) -> bool:
        return bool(self.ttft_samples_ms)


@runtime_checkable
class Backend(Protocol):
    """A runtime we can lower to and measure on."""

    kind: RuntimeKind

    def analyze(self, artifact_dir: Path, recipe: Recipe) -> StaticAnalysis:
        """Tier 1: lower, inspect, and report the partition decision."""
        ...

    def measure(
        self, artifact_dir: Path, recipe: Recipe, runs: int, warmup: int
    ) -> DeviceRun:
        """Tier 2: timed runs on real hardware. Called inside the measurement subprocess."""
        ...
