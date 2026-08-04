"""Measurement policy — run counts, warmup, and aggregation.

One place, so the policy cannot quietly diverge between backends. Anything here
that changes a measured number must be accompanied by a ``HARNESS_VERSION`` bump,
because the corpus is immutable and old rows must stay interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

from edgefit.schema.measurement import MIN_RUNS, RunStats


@dataclass(frozen=True)
class MeasurementPolicy:
    """How a single measurement is taken."""

    runs: int = 10
    warmup: int = 3
    # Above this coefficient of variation the host was not actually quiet and the
    # number should not be trusted. Recorded, not silently discarded — a noisy
    # result is itself evidence about the device.
    max_acceptable_cv: float = 0.10
    decode_tokens: int = 32
    """Decode steps per generative run.

    Long enough for tok/s to stabilise past the first-step transient, short enough
    that a sweep finishes. Sustained throughput under thermal soak is a different
    measurement and belongs to the stress bench (§5.6), not here."""

    def __post_init__(self) -> None:
        if self.runs < MIN_RUNS:
            raise ValueError(
                f"runs={self.runs} is below the mandatory minimum of {MIN_RUNS} "
                "(PROJECT.md §14.2)"
            )
        if self.warmup < 1:
            raise ValueError(
                "at least one warmup run is required: the first inference includes "
                "lazy kernel compilation and delegate model caching, which is a real "
                "cost but not steady-state latency"
            )


def aggregate(samples_ms: list[float]) -> RunStats:
    """Turn raw timings into a validated distribution."""
    return RunStats.from_samples(samples_ms)


def is_noisy(stats: RunStats, policy: MeasurementPolicy) -> bool:
    return stats.cv > policy.max_acceptable_cv
