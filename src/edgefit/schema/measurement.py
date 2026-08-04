"""Measurement record (PROJECT.md §6.2) — immutable, variance-mandatory.

The hard rules from §14 are enforced here rather than in prose, because a rule
that lives only in a document gets violated at 2am in month four:

* **#1 never synthesize a value** — ``RunStats`` is only constructible from raw
  samples and revalidates its own aggregates; ``Metrics.unavailable`` forces a
  written reason for every absent number.
* **#2 n>=5 with variance** — a ``success`` record without ``RunStats`` of at
  least ``MIN_RUNS`` samples fails validation outright.
* **#3 immutable** — enforced at the store layer; here we simply carry
  ``harness_version`` so a re-measure is a new row rather than an edit.
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgefit.schema.common import Outcome, StressProfile, content_hash
from edgefit.schema.host import DeviceFingerprint, HostState

# v2 adds stress_profile (PROJECT.md §6.2, §9 step 2: "from day one").
MEASUREMENT_SCHEMA_VERSION = 2

# PROJECT.md §14.2. Not a suggestion, and not configurable downwards.
MIN_RUNS = 5

# Aggregates are recomputed from samples on validation; this is float noise only.
_REL_TOL = 1e-9


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


def _percentile(sorted_samples: list[float], q: float) -> float:
    """Linear-interpolated percentile, matching numpy's default method.

    Spelled out rather than imported because it is a published methodology
    detail — anyone reproducing our numbers must be able to see exactly how a
    p95 was derived from five samples.
    """
    if not sorted_samples:
        raise ValueError("no samples")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    position = (len(sorted_samples) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_samples) - 1)
    weight = position - lower
    return sorted_samples[lower] * (1 - weight) + sorted_samples[upper] * weight


class RunStats(_Frozen):
    """Timing distribution over n>=5 real runs.

    Aggregates are derived, never asserted: the validator recomputes them from
    ``samples`` and rejects any mismatch. It is therefore impossible to write a
    variance figure into the corpus that did not come from measured data.
    """

    samples: tuple[float, ...] = Field(min_length=MIN_RUNS)
    n: int
    mean: float
    p50: float
    p95: float
    minimum: float
    maximum: float
    stddev: float = Field(description="Sample standard deviation (n-1 denominator)")
    cv: float = Field(description="Coefficient of variation, stddev/mean. The trust signal.")

    @classmethod
    def from_samples(cls, samples: list[float]) -> Self:
        if len(samples) < MIN_RUNS:
            raise ValueError(
                f"{len(samples)} runs is below the mandatory minimum of {MIN_RUNS} "
                "(PROJECT.md §14.2) — a measurement without variance data is invalid"
            )
        ordered = sorted(samples)
        mean = statistics.fmean(samples)
        stddev = statistics.stdev(samples)
        return cls(
            samples=tuple(samples),
            n=len(samples),
            mean=mean,
            p50=_percentile(ordered, 0.50),
            p95=_percentile(ordered, 0.95),
            minimum=ordered[0],
            maximum=ordered[-1],
            stddev=stddev,
            cv=stddev / mean if mean else 0.0,
        )

    @model_validator(mode="after")
    def _aggregates_match_samples(self) -> RunStats:
        if self.n != len(self.samples):
            raise ValueError(f"n={self.n} does not match {len(self.samples)} samples")
        recomputed = {
            "mean": statistics.fmean(self.samples),
            "p50": _percentile(sorted(self.samples), 0.50),
            "p95": _percentile(sorted(self.samples), 0.95),
            "minimum": min(self.samples),
            "maximum": max(self.samples),
            "stddev": statistics.stdev(self.samples),
        }
        for name, expected in recomputed.items():
            actual = getattr(self, name)
            if abs(actual - expected) > max(_REL_TOL * abs(expected), _REL_TOL):
                raise ValueError(
                    f"{name}={actual!r} is not derivable from the samples "
                    f"(expected {expected!r}) — aggregates must come from measured data"
                )
        return self


class FallbackReport(_Frozen):
    """Silent CPU fallback (PROJECT.md §2.2) — the lead diagnostic.

    Three independent estimates of the same quantity, because we do not yet know
    which is honest. Node share is easy and misleading (one unclaimed matmul
    outweighs fifty unclaimed reshapes); FLOP share is what the atlas spec asks
    for but is static; time share is measured but includes dispatch overhead.
    Recording all three lets us later publish which proxy to trust.
    """

    intended_provider: str
    nodes_total: int = Field(ge=0)
    nodes_on_intended: int = Field(ge=0)
    fallback_node_pct: float = Field(ge=0.0, le=100.0)

    flops_total: int | None = None
    flops_on_intended: int | None = None
    fallback_flops_pct: float | None = Field(default=None, ge=0.0, le=100.0)

    time_total_us: float | None = None
    time_on_intended_us: float | None = None
    fallback_time_pct: float | None = Field(default=None, ge=0.0, le=100.0)

    nodes_per_provider: dict[str, int] = Field(default_factory=dict)
    unclaimed_op_types: dict[str, int] = Field(
        default_factory=dict,
        description="op type -> count that missed the intended accelerator. The actionable part.",
    )
    partition_count: int | None = Field(
        default=None,
        description=(
            "Contiguous subgraphs handed to the accelerator. Fragmentation is its own "
            "performance story: 31 partitions means 31 round trips, however good the ratio looks."
        ),
    )

    # Methodology provenance — these numbers are meaningless without it.
    analysis_graph_optimization: str = Field(
        default="disabled",
        description=(
            "ORT graph optimisation level during partition analysis. Disabled by default so "
            "node names still map to the as-authored graph, which is what makes the "
            "unclaimed-op list actionable."
        ),
    )
    flops_estimator_version: int | None = None

    @model_validator(mode="after")
    def _check_coherent(self) -> FallbackReport:
        if self.nodes_on_intended > self.nodes_total:
            raise ValueError("nodes_on_intended exceeds nodes_total")
        return self


class CalibrationProbe(_Frozen):
    """Measured throttle proxy, run immediately before the timed runs.

    Apple Silicon exposes no unprivileged temperature, so instead of inventing
    one we time a fixed deterministic kernel and compare against this host's
    recorded baseline. A host that has slowed down is a host that is throttled or
    contended. This is measured, not estimated — and it is stored on every record
    so the corpus can be re-filtered later if the threshold turns out wrong.
    """

    kernel: str
    elapsed_ms: float
    baseline_ms: float | None = None
    ratio_to_baseline: float | None = None


class Metrics(_Frozen):
    """What we measured. Absent fields must be explained in ``unavailable``."""

    # Generic
    latency_ms: RunStats | None = None
    peak_rss_bytes: int | None = None
    artifact_bytes: int | None = None
    lowering_ms: float | None = Field(
        default=None,
        description="Hard rule #4: pipeline cost counts, not just kernel time.",
    )

    # Generative (PROJECT.md §4 Stage 1 atlas columns)
    ttft_ms: RunStats | None = None
    decode_tok_s: RunStats | None = None
    sustained_tok_s_5min: float | None = None

    # Accuracy
    accuracy: float | None = None
    accuracy_delta_vs_fp16: float | None = None

    # Power — null on every laptop host; needs Phase-2 instrumentation.
    power_mw: float | None = None

    unavailable: dict[str, str] = Field(
        default_factory=dict, description="metric field name -> why it is absent"
    )

    @model_validator(mode="after")
    def _unavailable_is_truthful(self) -> Metrics:
        """An explanation must name a real, actually-absent metric.

        Guards both directions: no explaining away a field that is populated, and
        no inventing field names that drift from the schema.
        """
        for name in self.unavailable:
            if name not in type(self).model_fields:
                raise ValueError(f"unavailable names unknown metric {name!r}")
            if name == "unavailable":
                raise ValueError("'unavailable' cannot explain itself")
            if getattr(self, name) is not None:
                raise ValueError(f"metric {name!r} is populated but listed as unavailable")
        return self

    @property
    def primary_stats(self) -> RunStats | None:
        """The timing distribution this measurement is really about."""
        return self.latency_ms or self.ttft_ms


class MeasurementRecord(_Frozen):
    """One (config, device) measurement. Immutable; never updated in place."""

    schema_version: int = MEASUREMENT_SCHEMA_VERSION
    harness_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    config_id: str
    model_ref: str
    graph_fingerprint_id: str | None = None

    device: DeviceFingerprint
    host_state: HostState
    calibration_probe: CalibrationProbe | None = None

    stress_profile: StressProfile = Field(
        default=StressProfile.CLEAN,
        description=(
            "Which rung of the §5.6 validation ladder produced this row. Only the "
            "clean bench exists today; soak/pressure/concurrent arrive with the "
            "stress bench. Recorded now so old rows stay interpretable then."
        ),
    )

    outcome: Outcome
    failure_reason: str | None = None
    run_count: int = Field(ge=0)
    warmup_count: int = Field(ge=0)

    metrics: Metrics | None = None
    fallback: FallbackReport | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _enforce_hard_rules(self) -> MeasurementRecord:
        if self.outcome is Outcome.SUCCESS:
            if self.metrics is None:
                raise ValueError("a successful measurement must carry metrics")
            stats = self.metrics.primary_stats
            if stats is None:
                raise ValueError(
                    "a successful measurement must carry a timing distribution "
                    "(PROJECT.md §14.2: without variance the record is invalid)"
                )
            if stats.n < MIN_RUNS:
                raise ValueError(f"run_count {stats.n} is below the minimum of {MIN_RUNS}")
            if self.run_count != stats.n:
                raise ValueError(
                    f"run_count={self.run_count} disagrees with {stats.n} timing samples"
                )
        elif not (self.failure_reason or "").strip():
            raise ValueError(f"outcome {self.outcome!r} requires a failure_reason")
        return self

    @property
    def measurement_id(self) -> str:
        """Content hash including ``created_at``.

        Repeat measurements of the same config on the same device are genuinely
        different observations and must not collapse onto one id — repeatability
        data is exactly what the two-unit and drift checks consume.
        """
        return content_hash(self.model_dump(mode="json"))
