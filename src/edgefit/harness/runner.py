"""Measurement orchestration — the device execution protocol of PROJECT.md §5.6.

    acquire exclusive lock
      -> thermal gate (idle until conditions are met)
      -> push artifact
      -> warmup runs (discarded)
      -> N timed runs (N >= 5)
      -> collect metrics + variance
      -> release lock

Every path through this function produces a ``MeasurementRecord``. A refused
gate, a failed lowering, and a crashed runtime are all recorded outcomes rather
than exceptions that leave nothing behind — §5.8 is explicit that failures are as
valuable as successes because they are what train the tier-1 static filter.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path

from edgefit import HARNESS_VERSION
from edgefit.backends.base import DeviceRun, StaticAnalysis
from edgefit.corpus.store import CorpusStore
from edgefit.harness.gate import (
    BaselineStore,
    GateReport,
    GateThresholds,
    device_lock,
    evaluate_gate,
    run_calibration_probe,
)
from edgefit.harness.hostinfo import probe_device, probe_state
from edgefit.harness.timing import MeasurementPolicy, aggregate, is_noisy
from edgefit.schema.fingerprint import GraphFingerprint
from edgefit.schema.host import DeviceFingerprint
from edgefit.schema.measurement import FallbackReport, MeasurementRecord, Metrics, Outcome
from edgefit.schema.recipe import Recipe

_WORKER_TIMEOUT_S = 900.0
_SIGNALS = {member.value for member in signal.Signals}


@dataclass(frozen=True)
class MeasurementOutcome:
    """A record plus the transient detail that does not belong in the corpus."""

    record: MeasurementRecord
    analysis: StaticAnalysis | None = None
    gate: GateReport | None = None
    # Kept out of the record deliberately: 512 floats per row would bloat the
    # corpus for something only the golden numerics check consumes.
    outputs: dict[str, list[float]] | None = None

    @property
    def succeeded(self) -> bool:
        return self.record.outcome is Outcome.SUCCESS


def _unavailable_on_this_host(*, generative: bool = False) -> dict[str, str]:
    reasons = {
        "power_mw": "no power instrumentation on this host (needs Phase-2 hardware)",
        "sustained_tok_s_5min": (
            "sustained throughput is a thermal-soak measurement and belongs to the "
            "stress bench (PROJECT.md §5.6), which is not built yet"
        ),
        "accuracy": "tier-3 eval-set accuracy not implemented yet "
        "(numerics checks are not task accuracy)",
    }
    if generative:
        reasons["latency_ms"] = (
            "a generative recipe reports ttft_ms and decode_tok_s separately; a single "
            "averaged latency would describe neither phase"
        )
        reasons["output_cosine_vs_reference"] = (
            "generative recipes are compared by token agreement, not output cosine"
        )
    else:
        reasons["ttft_ms"] = "not a generative task"
        reasons["decode_tok_s"] = "not a generative task"
        reasons["token_agreement"] = "not a generative task"
    return reasons


def _describe_exit(completed: subprocess.CompletedProcess[str]) -> str:
    """Explain why a worker produced nothing, naming the signal if it was killed.

    A delegate that aborts the interpreter is a specific, recurring failure mode
    worth distinguishing from a Python-level exception, because the two call for
    completely different fixes.
    """
    if completed.returncode < 0:
        signal_number = -completed.returncode
        name = signal.Signals(signal_number).name if signal_number in _SIGNALS else "unknown"
        detail = completed.stderr.strip()[-400:]
        return (
            f"the runtime aborted the process ({name}). This is a delegate-level crash, "
            f"not a Python exception.{f' Last output: {detail}' if detail else ''}"
        )
    return completed.stderr.strip()[-500:] or f"exit code {completed.returncode}, no output"


def _token_agreement(tokens: list[int] | None, reference_path: Path) -> float | None:
    """Fraction of greedily-decoded tokens matching the fp32 PyTorch reference.

    Stricter than a float tolerance: the graph either decodes the same text or it
    does not. Still a numerics check rather than a quality measure — identical greedy
    tokens say nothing about behaviour under sampling or on a real prompt mix.
    """
    if not tokens or not reference_path.exists():
        return None
    try:
        import numpy as np  # noqa: PLC0415

        with np.load(reference_path) as data:
            if "tokens" not in data:
                return None
            reference = [int(v) for v in data["tokens"]]
    except Exception:  # noqa: BLE001 - a missing comparison is a null, not a failed run
        return None
    if not reference:
        return None
    compared = min(len(reference), len(tokens))
    matches = sum(1 for a, b in zip(tokens[:compared], reference[:compared], strict=True) if a == b)
    return matches / compared


def _output_cosine(outputs: dict[str, list[float]] | None, reference_path: Path) -> float | None:
    """Cosine similarity of the measured output against the fp32 reference.

    Deliberately cheap and deliberately *not* called accuracy. It catches a
    quantization scheme that destroyed the model, which is the failure a latency
    table alone would present as a win. It does not catch a scheme that degrades
    one task slice while holding global similarity — that needs tier 3.
    """
    if not outputs or not reference_path.exists():
        return None
    try:
        import numpy as np  # noqa: PLC0415

        with np.load(reference_path) as data:
            reference = next(iter(data.values())).ravel()
        measured = np.asarray(next(iter(outputs.values())), dtype=np.float64)
        reference = reference[: measured.size].astype(np.float64)
        if measured.size == 0 or not measured.any() or not reference.any():
            return None
        denominator = float(np.linalg.norm(measured) * np.linalg.norm(reference))
        return float(np.dot(measured, reference) / denominator) if denominator else None
    except Exception:  # noqa: BLE001 - a missing comparison is a null, not a failed run
        return None


def _run_worker(job: dict, timeout_s: float = _WORKER_TIMEOUT_S) -> dict:
    """Run one worker job out of process. Never raises; always returns a payload."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "edgefit.harness.worker"],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"failure_reason": f"exceeded the {timeout_s:.0f}s worker timeout"}

    if not completed.stdout.strip():
        return {"failure_reason": _describe_exit(completed)}

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "failure_reason": f"worker returned unparseable output: {_describe_exit(completed)}"
        }


def _analyze_in_subprocess(artifact_dir: Path, recipe: Recipe) -> StaticAnalysis:
    """Tier 1, isolated. Partitioning is where delegates crash hardest."""
    payload = _run_worker(
        {
            "mode": "analyze",
            "artifact_dir": str(artifact_dir),
            "recipe": recipe.model_dump(mode="json"),
        }
    )
    if payload.get("lowered") is not True:
        return StaticAnalysis(
            lowered=False,
            artifact_bytes=int(payload.get("artifact_bytes") or 0),
            failure_reason=payload.get("failure_reason") or "analysis produced no result",
            lowering_ms=payload.get("lowering_ms"),
        )
    return StaticAnalysis(
        lowered=True,
        artifact_bytes=int(payload["artifact_bytes"]),
        lowering_ms=payload.get("lowering_ms"),
        fingerprint=(
            GraphFingerprint.model_validate(payload["fingerprint"])
            if payload.get("fingerprint")
            else None
        ),
        fallback=(
            FallbackReport.model_validate(payload["fallback"]) if payload.get("fallback") else None
        ),
        fallback_as_run=(
            FallbackReport.model_validate(payload["fallback_as_run"])
            if payload.get("fallback_as_run")
            else None
        ),
    )


def _measure_in_subprocess(
    artifact_dir: Path, recipe: Recipe, policy: MeasurementPolicy
) -> DeviceRun:
    """Tier 2, isolated. See ``worker`` for why this is not optional."""
    payload = _run_worker(
        {
            "mode": "measure",
            "artifact_dir": str(artifact_dir),
            "recipe": recipe.model_dump(mode="json"),
            "runs": policy.runs,
            "warmup": policy.warmup,
            "decode_tokens": policy.decode_tokens,
        }
    )
    # Rebuilt from the dataclass's own fields rather than a hand-written list. The
    # worker already serialises with `asdict`, so listing them here made the boundary
    # asymmetric — and a field added to DeviceRun was silently dropped in transit.
    # That is exactly what happened to the cold-load metrics: measured correctly in
    # the child, absent from every local row, with nothing failing to say so.
    known = {f.name for f in fields(DeviceRun)}
    return DeviceRun(**{k: v for k, v in payload.items() if k in known and v is not None})


def _failure_record(
    recipe: Recipe,
    device: DeviceFingerprint,
    gate: GateReport,
    outcome: Outcome,
    reason: str,
    *,
    analysis: StaticAnalysis | None = None,
    warmup: int = 0,
) -> MeasurementRecord:
    metrics = None
    if analysis is not None:
        metrics = Metrics(
            artifact_bytes=analysis.artifact_bytes,
            lowering_ms=analysis.lowering_ms,
            unavailable=_unavailable_on_this_host(),
        )
    return MeasurementRecord(
        harness_version=HARNESS_VERSION,
        recipe_id=recipe.recipe_id,
        model_ref=recipe.model.ref,
        graph_fingerprint_id=(
            analysis.fingerprint.fingerprint_id
            if analysis and analysis.fingerprint
            else None
        ),
        device=device,
        host_state=gate.host_state,
        calibration_probe=gate.calibration_probe,
        outcome=outcome,
        failure_reason=reason,
        run_count=0,
        warmup_count=warmup,
        metrics=metrics,
        fallback=analysis.fallback if analysis else None,
        fallback_as_run=analysis.fallback_as_run if analysis else None,
    )


def measure(
    artifact_dir: Path,
    recipe: Recipe,
    *,
    store: CorpusStore | None = None,
    policy: MeasurementPolicy | None = None,
    thresholds: GateThresholds | None = None,
    lock_timeout_s: float = 0.0,
    calibrate: bool = True,
    gate: GateReport | None = None,
) -> MeasurementOutcome:
    """Measure one (recipe, device) pair. Always produces a record.

    ``gate`` accepts an already-evaluated gate decision. A sweep waits for a fit
    host and then passes that result in, because re-probing here would take a
    second independent noisy draw against a tight threshold — which cost 16 of 45
    cells in the first full sweep, on a host that was fine both times.
    """
    policy = policy or MeasurementPolicy()
    device = probe_device()
    baselines = BaselineStore()

    with device_lock(device, timeout_s=lock_timeout_s):
        if gate is None:
            probe = run_calibration_probe(baselines.get(device)) if calibrate else None
            gate = evaluate_gate(probe_state(), thresholds, probe)
            if gate.passed_ignoring_probe and probe is not None:
                baselines.record(device, probe.elapsed_ms)

        if not gate.passed:
            record = _failure_record(
                recipe, device, gate, Outcome.GATE_REFUSED, gate.reason()
            )
            if store is not None:
                store.insert_recipe(recipe)
                store.insert_measurement(record)
            return MeasurementOutcome(record=record, gate=gate)

        # --- tier 1: static ---
        analysis = _analyze_in_subprocess(artifact_dir, recipe)
        if not analysis.lowered:
            record = _failure_record(
                recipe,
                device,
                gate,
                Outcome.LOWERING_FAILURE,
                analysis.failure_reason or "lowering failed for an unreported reason",
                analysis=analysis,
            )
            if store is not None:
                store.insert_recipe(recipe)
                if analysis.fingerprint:
                    store.insert_fingerprint(analysis.fingerprint)
                store.insert_measurement(record)
            return MeasurementOutcome(record=record, analysis=analysis, gate=gate)

        # --- tier 2: device ---
        run = _measure_in_subprocess(artifact_dir, recipe, policy)

    if not run.succeeded:
        record = _failure_record(
            recipe,
            device,
            gate,
            Outcome.RUNTIME_FAILURE,
            run.failure_reason or "no timing samples were produced",
            analysis=analysis,
            warmup=run.warmup_count,
        )
    else:
        generative = run.is_generative
        stats = aggregate(run.ttft_samples_ms if generative else run.samples_ms)
        notes = None
        if is_noisy(stats, policy):
            notes = (
                f"high variance: cv={stats.cv:.3f} exceeds {policy.max_acceptable_cv:.3f}. "
                "The host was not as quiet as the gate believed; treat this row with suspicion."
            )
        record = MeasurementRecord(
            harness_version=HARNESS_VERSION,
            recipe_id=recipe.recipe_id,
            model_ref=recipe.model.ref,
            graph_fingerprint_id=(
                analysis.fingerprint.fingerprint_id if analysis.fingerprint else None
            ),
            device=device,
            host_state=gate.host_state,
            calibration_probe=gate.calibration_probe,
            outcome=Outcome.SUCCESS,
            run_count=stats.n,
            warmup_count=run.warmup_count,
            metrics=Metrics(
                # TTFT is the primary distribution for a generative recipe; a single
                # averaged "latency" would describe neither prefill nor decode.
                latency_ms=None if generative else stats,
                ttft_ms=stats if generative else None,
                decode_tok_s=(
                    aggregate(run.decode_samples_tok_s)
                    if generative and run.decode_samples_tok_s
                    else None
                ),
                peak_rss_bytes=run.peak_rss_bytes,
                artifact_bytes=analysis.artifact_bytes,
                lowering_ms=analysis.lowering_ms,
                cold_load_ms=run.cold_load_ms,
                warm_load_ms=run.warm_load_ms,
                first_inference_ms=run.first_inference_ms,
                output_cosine_vs_reference=(
                    None if generative else _output_cosine(
                        run.outputs, artifact_dir / "reference.npz"
                    )
                ),
                token_agreement=(
                    _token_agreement(run.generated_tokens, artifact_dir / "reference.npz")
                    if generative
                    else None
                ),
                unavailable=_unavailable_on_this_host(generative=generative),
            ),
            fallback=analysis.fallback,
            fallback_as_run=analysis.fallback_as_run,
            notes=notes,
        )

    if store is not None:
        store.insert_recipe(recipe)
        if analysis.fingerprint:
            store.insert_fingerprint(analysis.fingerprint)
        store.insert_measurement(record)

    return MeasurementOutcome(
        record=record, analysis=analysis, gate=gate, outputs=run.outputs
    )
