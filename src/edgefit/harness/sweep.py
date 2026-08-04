"""Sweep runner — the cross product of models and recipes (PROJECT.md §9 step 6).

One measurement at a time is a debugging tool. The corpus is the asset (§12), the
research finding in §17.6 is a *rate* across models and delegates, and the atlas
needs a matrix to render — all of which need breadth, which needs this.

Three behaviours matter more than the loop itself:

**It waits rather than refusing.** §5.6 specifies the thermal gate as "idle until
temp below threshold". A sweep that refuses on a warm host writes fifty
``gate_refused`` rows and measures nothing; a sweep that waits produces a usable
corpus overnight on a laptop.

**It resumes.** Laptops get closed and sweeps get interrupted. A cell already
measured for this device and harness version is skipped, so re-running is cheap
and safe. ``gate_refused`` rows do not count as done — they record that we could
not measure, so the cell is still outstanding.

**It records everything.** An unsupported quantization scheme, a delegate that
aborts the interpreter, a lowering failure: all become rows. §5.9 is explicit that
failures train the tier-1 static filter, and a sweep is precisely where they are
cheapest to collect.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from edgefit import HARNESS_VERSION
from edgefit.corpus.store import CorpusStore
from edgefit.harness.gate import GateReport, GateThresholds, wait_until_fit
from edgefit.harness.hostinfo import probe_device
from edgefit.harness.runner import MeasurementOutcome, measure
from edgefit.harness.timing import MeasurementPolicy
from edgefit.schema.common import Outcome
from edgefit.schema.host import DeviceFingerprint
from edgefit.schema.measurement import MeasurementRecord, Metrics


@dataclass(frozen=True)
class SweepCell:
    """One (model, recipe file) pair to measure."""

    model_ref: str
    recipe_path: Path

    @property
    def label(self) -> str:
        return f"{self.model_ref.split('/')[-1]} × {self.recipe_path.stem}"


@dataclass
class SweepReport:
    """What a sweep did. Counts are outcomes, not attempts."""

    cells: int = 0
    measured: int = 0
    resumed: int = 0
    not_applicable: int = 0
    lowering_failures: int = 0
    runtime_failures: int = 0
    gate_refused: int = 0
    aborted_reason: str | None = None
    elapsed_s: float = 0.0
    outcomes: list[tuple[SweepCell, str]] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return self.cells - self.resumed


# Progress events, so the CLI owns presentation and the sweep owns logic.
SweepEvent = Callable[[str, SweepCell, str], None]


def expand(model_refs: Iterable[str], recipe_paths: Iterable[Path]) -> list[SweepCell]:
    """The cross product, in a stable order so interrupted runs resume predictably."""
    return [
        SweepCell(model_ref=ref, recipe_path=Path(path))
        for ref in sorted(model_refs)
        for path in sorted(recipe_paths, key=str)
    ]


def _lowering_failure_record(
    recipe, device: DeviceFingerprint, gate: GateReport, reason: str
) -> MeasurementRecord:
    """A recipe this backend cannot express is a lowering failure, not a crash.

    Recorded rather than skipped: "ORT cannot do blockwise int4" is exactly the
    kind of fact the tier-1 static filter should learn once and never re-attempt.
    """
    return MeasurementRecord(
        harness_version=HARNESS_VERSION,
        recipe_id=recipe.recipe_id,
        model_ref=recipe.model.ref,
        device=device,
        host_state=gate.host_state,
        calibration_probe=gate.calibration_probe,
        outcome=Outcome.LOWERING_FAILURE,
        failure_reason=reason,
        run_count=0,
        warmup_count=0,
        metrics=Metrics(
            unavailable={
                "power_mw": "no power instrumentation on this host",
                "accuracy": "recipe never lowered, so nothing was evaluated",
            }
        ),
    )


def _describe(record: MeasurementRecord) -> str:
    """One line summarising what was measured.

    Dispatches on which distribution arrived, because a generative recipe has no
    single "latency" — it has TTFT and a decode rate.
    """
    metrics = record.metrics
    if metrics is None:
        return ""
    if metrics.ttft_ms is not None:
        line = f"ttft {metrics.ttft_ms.p50:.0f} ms · cv {metrics.ttft_ms.cv:.1%}"
        if metrics.decode_tok_s is not None:
            line += f" · {metrics.decode_tok_s.p50:.2f} tok/s"
        return line
    if metrics.latency_ms is not None:
        return f"p50 {metrics.latency_ms.p50:.2f} ms · cv {metrics.latency_ms.cv:.1%}"
    return ""


def run_sweep(
    model_refs: Iterable[str],
    recipe_paths: Iterable[Path],
    *,
    store: CorpusStore,
    policy: MeasurementPolicy | None = None,
    thresholds: GateThresholds | None = None,
    resume: bool = True,
    wait_for_fit_s: float = 600.0,
    on_event: SweepEvent | None = None,
) -> SweepReport:
    """Measure every (model, recipe) cell, recording every outcome."""
    from edgefit.backends.artifacts import (  # noqa: PLC0415 - keeps torch out of import time
        UnsupportedDecoderLowering,
        UnsupportedQuantizationError,
        recipe_applicability,
        resolve_artifact,
    )
    from edgefit.cli.recipes import load_recipe  # noqa: PLC0415
    from edgefit.models.registry import resolve

    policy = policy or MeasurementPolicy()
    device = probe_device()
    cells = expand(model_refs, recipe_paths)
    report = SweepReport(cells=len(cells))
    started = time.monotonic()

    def emit(kind: str, cell: SweepCell, detail: str = "") -> None:
        if on_event is not None:
            on_event(kind, cell, detail)

    for cell in cells:
        recipe = load_recipe(cell.recipe_path, cell.model_ref)

        # Tier 0: legality. An illegal pair was never a candidate (§5.4), so it is
        # skipped rather than written as a failure row.
        reason = recipe_applicability(resolve(cell.model_ref), recipe)
        if reason is not None:
            report.not_applicable += 1
            emit("skipped", cell, reason)
            continue

        if resume and store.has_measurement(recipe.recipe_id, device.device_id, HARNESS_VERSION):
            report.resumed += 1
            emit("resumed", cell)
            continue

        # Wait for a fit host before spending anything on this cell.
        # `current` binds the loop variable explicitly; a late-binding closure here
        # would attribute a wait to whichever cell happened to be current later.
        def report_wait(rep: GateReport, left: float, current: SweepCell = cell) -> None:
            emit("waiting", current, f"{rep.reason()} · {left:.0f}s left")

        gate = wait_until_fit(
            thresholds, device=device, timeout_s=wait_for_fit_s, on_wait=report_wait
        )
        if not gate.passed:
            # Persistently unfit. Stop rather than grind out refusals — the
            # operator needs to fix the host, and forty identical rows say nothing
            # the first one did not.
            report.aborted_reason = (
                f"host stayed unfit for {wait_for_fit_s:.0f}s: {gate.reason()}"
            )
            emit("aborted", cell, report.aborted_reason)
            break

        # Tier 0: lowering. An unsupported recipe is a recorded failure.
        try:
            artifact = resolve_artifact(resolve(cell.model_ref), recipe)
        except (UnsupportedQuantizationError, UnsupportedDecoderLowering) as exc:
            record = _lowering_failure_record(recipe, device, gate, str(exc))
            store.insert_recipe(recipe)
            store.insert_measurement(record)
            report.lowering_failures += 1
            report.outcomes.append((cell, "lowering_failure"))
            emit("failed", cell, f"unsupported recipe: {exc}")
            continue

        outcome: MeasurementOutcome = measure(
            artifact.directory,
            recipe,
            store=store,
            policy=policy,
            thresholds=thresholds,
            # Reuse the decision from wait_until_fit; a second probe here is a
            # second noisy draw against the same threshold, not more rigour.
            gate=gate,
        )
        result = str(outcome.record.outcome)
        report.outcomes.append((cell, result))

        if outcome.record.outcome is Outcome.SUCCESS:
            report.measured += 1
            emit("measured", cell, _describe(outcome.record))
        elif outcome.record.outcome is Outcome.LOWERING_FAILURE:
            report.lowering_failures += 1
            emit("failed", cell, outcome.record.failure_reason or "")
        elif outcome.record.outcome is Outcome.GATE_REFUSED:
            report.gate_refused += 1
            emit("refused", cell, outcome.record.failure_reason or "")
        else:
            report.runtime_failures += 1
            emit("failed", cell, outcome.record.failure_reason or "")

    report.elapsed_s = time.monotonic() - started
    return report
