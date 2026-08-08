"""Measurement on hosted hardware — Qualcomm AI Hub (PROJECT.md §7).

> **Qualcomm AI Hub as a virtual device backend** — same interface, someone else's
> hardware, free. Covers Snapdragon at zero capital.

This is the pass that makes the corpus cross-vendor, which §12 names as the moat
layer no silicon vendor can copy: a vendor can out-measure us on their own silicon
and will never publish a comparison where a competitor wins.

The remote path is deliberately *not* routed through ``harness.runner``. Almost every
guarantee that module provides is about our own machine — the preflight gate, the
exclusive device lock, the throttle probe, the out-of-process worker — and none of
them apply to a device in someone else's rack. Pretending otherwise would mean
recording a gate verdict about the wrong computer.

What survives, and matters:

* **Raw samples.** ``all_inference_times`` returns 100 per-run figures, not an
  aggregate, so hard rule #2 is satisfied natively and ``RunStats`` — constructible
  only from real samples — takes them directly. A service reporting a single number
  would have had to go in ``reported_latency_ms`` instead.
* **Measured per-node placement.** ``execution_detail`` carries a ``compute_unit`` of
  NPU/GPU/CPU per node. That is the same question our CoreML fallback proxies answer
  on Apple silicon, asked of a second vendor.

What does not survive is honestly recorded rather than papered over: the rows are
``measurement_source = third_party`` because the harness is theirs, and
``stress_profile = unknown`` because we cannot see the device's thermal or power
state.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from onnx import TensorProto

from edgefit import HARNESS_VERSION
from edgefit.corpus.store import CorpusStore
from edgefit.schema.common import (
    MeasurementSource,
    Outcome,
    PowerSource,
    StressProfile,
    ThermalState,
)
from edgefit.schema.fingerprint import GraphFingerprint
from edgefit.schema.host import DeviceFingerprint, HostState
from edgefit.schema.measurement import FallbackReport, MeasurementRecord, Metrics, RunStats
from edgefit.schema.recipe import (
    ModelRef,
    QaiHubComputeUnit,
    QaiHubRuntimeConfig,
    Recipe,
)
from edgefit.schema.vendor import soc_vendor as _soc_vendor

#: Leading samples discarded before aggregation.
#:
#: AI Hub returns every run including the cold ones, and they are dramatic: an
#: observed series began [1865, 109, 82, 55, 57, 55] microseconds. Feeding that whole
#: series to RunStats lets one cold outlier inflate the coefficient of variation by
#: more than an order of magnitude. Three matches our own local warmup policy, so the
#: two sources stay comparable, and the count is recorded on every row.
REMOTE_WARMUP_SAMPLES = 3

#: ONNX integer dtypes. An input of one of these is an index into something, and a
#: random value for it is very unlikely to be in bounds.
_INDEX_DTYPES = frozenset(
    {
        TensorProto.INT8,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.INT64,
        TensorProto.UINT8,
        TensorProto.UINT16,
        TensorProto.UINT32,
        TensorProto.UINT64,
    }
)

#: Device facts a hosted farm does not expose. Recorded as reasons, never guessed.
_UNKNOWN_HOST = {
    "thermal_state": "hosted device; AI Hub does not report thermal state",
    "low_power_mode": "hosted device; power policy not visible",
    "load_avg_1m": "hosted device; system load not visible",
    "available_ram_bytes": "hosted device; free memory not reported",
    "cpu_temperature_c": "hosted device; no temperature exposed",
}


class RemoteMeasurementError(Exception):
    """The hosted job could not be submitted or its result could not be read."""


class RemoteJobFailed(RemoteMeasurementError):
    """The hosted job ran and failed. Carries the vendor's own diagnosis.

    Distinguished from RemoteMeasurementError because this is *data*: §5.9 makes
    failures first-class, and a vendor telling us exactly why their NPU refused a
    graph is worth more than most successes. Swallowing it — which an earlier version
    did, by parsing a failed job's empty profile and complaining about missing samples
    — throws away the most useful thing the service said.
    """

    def __init__(self, message: str, *, job_id: str, job_url: str) -> None:
        super().__init__(message)
        self.job_id = job_id
        self.job_url = job_url


@dataclass(frozen=True)
class RemoteProfile:
    """The parts of an AI Hub profile we record."""

    job_id: str
    job_url: str
    samples_us: list[float]
    peak_memory_bytes: int | None
    cold_load_ms: float | None
    warm_load_ms: float | None
    compile_ms: float | None
    nodes_per_unit: dict[str, int]
    time_per_unit_us: dict[str, float]
    device_name: str
    device_os: str
    device_attributes: tuple[str, ...]
    client_version: str

    @property
    def timed_samples_ms(self) -> list[float]:
        """Steady-state samples in milliseconds, warmup discarded."""
        kept = self.samples_us[REMOTE_WARMUP_SAMPLES:]
        return [value / 1000.0 for value in kept]


def _first_number(summary: dict, *keys: str) -> float | None:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, list) and value and isinstance(value[0], int | float):
            return float(statistics.median(value))
    return None


def submit_profile(recipe: Recipe, artifact_dir: Path, *, wait: bool = True) -> RemoteProfile:
    """Upload the artifact and profile it on the recipe's hosted device."""
    try:
        import qai_hub as hub  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RemoteMeasurementError(
            "qai-hub is not installed. Install the extra: uv sync --extra qai-hub"
        ) from exc

    runtime = recipe.runtime
    if not isinstance(runtime, QaiHubRuntimeConfig):
        raise RemoteMeasurementError(f"recipe runtime {runtime.kind!r} is not a hosted device")

    model_path = artifact_dir / "model.onnx"
    # Before the upload, not after: an 86 MiB push and a provisioned device are both
    # wasted on a job that cannot produce a valid input.
    check_inputs_are_synthesizable(model_path)

    device = _resolve_device(hub, runtime)
    options = _profile_options(runtime)

    try:
        model = hub.upload_model(str(model_path))
        # Quantize then compile, when the recipe asks. Each stage is a separate hosted
        # job and each can fail on its own terms, so failures are attributed to the
        # stage that produced them rather than surfacing as "profiling failed".
        if runtime.quantize is not None:
            model = _quantize(hub, model, model_path, runtime)
        if runtime.target_runtime is not None:
            model = _compile(hub, model, device, runtime)
        job = hub.submit_profile_job(
            model=model,
            device=device,
            name=f"edgefit {recipe.model.ref}",
            **({"options": options} if options else {}),
        )
        if not wait:
            raise RemoteMeasurementError("non-blocking submission is not implemented")
        status = job.wait()
        _raise_if_failed(job, status)
        profile = job.download_profile()
    except RemoteMeasurementError:
        raise
    except Exception as exc:  # noqa: BLE001 - a vendor-side failure is data, not a crash
        raise RemoteMeasurementError(f"{type(exc).__name__}: {exc}") from exc

    return _parse_profile(profile, job, device, getattr(hub, "__version__", "unknown"))




def _pipeline_detail(runtime: QaiHubRuntimeConfig) -> str:
    """What was actually measured, when it is not the ONNX we uploaded.

    A row profiling a quantized, compiled artifact must not read like a row profiling
    our own export. The quantized numbers are only interpretable alongside how they were
    produced — uncalibrated int8 cost ViT-base 8.4x on hardware that accelerated it, so
    "int8" alone is not a description.
    """
    parts = []
    if runtime.quantize is not None:
        q = runtime.quantize
        parts.append(
            f"quantized on AI Hub to weights={q.weights_dtype}/activations="
            f"{q.activations_dtype} using {q.calibration_samples} calibration samples "
            "derived from the artifact's pinned inputs (a thin set; real calibration "
            "uses hundreds of representative examples)"
        )
    if runtime.target_runtime is not None:
        parts.append(f"compiled by AI Hub to {runtime.target_runtime}")
    if not parts:
        return ""
    return "; " + "; ".join(parts) + "; the profiled artifact is therefore not the ONNX we uploaded"


def _calibration_data(model_path: Path, samples: int) -> dict:
    """Real activations for the quantizer, built from the artifact's pinned inputs.

    Perturbations of one pinned input, not photographs: the harness ships no image
    assets and every number has to be reproducible by anyone from the repo alone. It is
    a thin calibration set and the recipe records how thin — `calibration_samples` is on
    the row precisely so nobody reads these as production-calibrated.
    """
    import numpy as np  # noqa: PLC0415

    with np.load(model_path.parent / "inputs.npz") as data:
        pinned = {name: data[name] for name in data.files}
    if not pinned:
        raise RemoteMeasurementError(
            f"no inputs.npz beside {model_path.name}, so there is nothing to calibrate on"
        )
    rng = np.random.default_rng(0)  # fixed: quantization must be reproducible
    out = {}
    for name, value in pinned.items():
        base = value.astype(np.float32)
        series = [base]
        for _ in range(max(samples - 1, 0)):
            series.append(base + rng.normal(0, 0.25, base.shape).astype(np.float32))
        out[name] = series
    return out


def _quantize(hub, model, model_path: Path, runtime: QaiHubRuntimeConfig):
    """Run a quantize job and return the quantized model."""
    spec = runtime.quantize
    assert spec is not None
    job = hub.submit_quantize_job(
        model=model,
        calibration_data=_calibration_data(model_path, spec.calibration_samples),
        weights_dtype=_dtype(hub, spec.weights_dtype),
        activations_dtype=_dtype(hub, spec.activations_dtype),
        name=f"edgefit quantize {spec.weights_dtype}/{spec.activations_dtype}",
    )
    _raise_if_failed(job, job.wait())
    return job.get_target_model()


def _compile(hub, model, device, runtime: QaiHubRuntimeConfig):
    """Compile to the recipe's target runtime and return the compiled model."""
    job = hub.submit_compile_job(
        model=model,
        device=device,
        options=f"--target_runtime {runtime.target_runtime}",
        name=f"edgefit compile {runtime.target_runtime}",
    )
    _raise_if_failed(job, job.wait())
    return job.get_target_model()


def _dtype(hub, name: str):
    """Map our dtype string onto the client's enum, refusing rather than guessing."""
    try:
        return getattr(hub.QuantizeDtype, name.upper())
    except AttributeError as exc:
        available = [d for d in dir(hub.QuantizeDtype) if d.isupper()]
        raise RemoteMeasurementError(
            f"qai-hub has no quantize dtype {name!r}; available: {available}"
        ) from exc


def _profile_options(runtime: QaiHubRuntimeConfig) -> str | None:
    """Translate the recipe into AI Hub's own flag vocabulary.

    Only flags a *profile* job accepts appear here. Compile-job flags are rejected
    with ``unrecognized arguments`` rather than ignored, which is the better
    failure — but it means the recipe must not carry them at all.
    """
    if runtime.compute_unit is QaiHubComputeUnit.ALL:
        # The service default. Sending it explicitly would claim we constrained
        # something we did not.
        return None
    return f"--compute_unit {runtime.compute_unit.value}"


class UnprofilableOnHostedService(RemoteMeasurementError):
    """The service cannot generate valid inputs for this model, so no job is worth running."""


def check_inputs_are_synthesizable(model_path: Path) -> None:
    """Refuse models whose inputs a random-input profiler cannot fabricate validly.

    AI Hub profile jobs synthesize their own inputs — there is no parameter to supply
    them, unlike inference jobs. For a float input that is harmless: any value is
    legal. For an *index* input it is not. MiniLM failed on a Snapdragon CPU with
    ``indices element out of data bounds, idx=3 must be within the inclusive range
    [-2,1]`` on the `token_type_embeddings` Gather, because a random int64 is not a
    token type.

    That failure says nothing about the device and nothing about the model. It is a
    property of submitting an index-input graph to a service that fabricates inputs —
    our choice. Recording it as the device refusing the model would be the same error
    the decisions log already names: a true observation about one thing, reported as a
    fact about another. So we refuse before spending a device, and record no row.
    """
    import onnx  # noqa: PLC0415

    model = onnx.load(str(model_path), load_external_data=False)
    integer_inputs = [
        value.name
        for value in model.graph.input
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type in _INDEX_DTYPES
    ]
    if integer_inputs:
        raise UnprofilableOnHostedService(
            f"AI Hub profile jobs synthesize random inputs, and {', '.join(integer_inputs)} "
            "are integer indices — random values fall outside embedding bounds and the "
            "job fails for a reason that is about neither the device nor the model. "
            "Use a float-input model, or an inference job that accepts real inputs."
        )


def _raise_if_failed(job, status) -> None:
    """Turn a failed hosted job into its vendor-supplied reason.

    Checked explicitly because ``download_profile()`` on a failed job returns a
    payload with no samples, which is indistinguishable from a malformed response
    unless the status is consulted first.
    """
    code = getattr(status, "code", None)
    name = getattr(code, "name", str(code)) if code is not None else ""
    if str(name).upper() == "SUCCESS":
        return
    message = getattr(status, "message", None) or getattr(status, "failure_reason", None)
    raise RemoteJobFailed(
        str(message or f"hosted job ended in state {name!r} without a message"),
        job_id=job.job_id,
        job_url=job.url,
    )


def _resolve_device(hub, runtime: QaiHubRuntimeConfig):
    """Find the catalogued device this recipe names."""
    wanted = runtime.device_name
    for candidate in hub.get_devices():
        if candidate.name != wanted:
            continue
        if runtime.device_os and str(candidate.os) != runtime.device_os:
            continue
        return candidate
    raise RemoteMeasurementError(
        f"{wanted!r} is not in the AI Hub catalogue for this account. "
        "Run `edgefit devices refresh` and `edgefit devices list` to see what is."
    )


def _parse_profile(profile: dict, job, device, client_version: str) -> RemoteProfile:
    summary = profile.get("execution_summary", {}) or {}
    samples = summary.get("all_inference_times") or profile.get("all_inference_times") or []
    if not samples:
        raise RemoteMeasurementError(
            "profile returned no per-run samples; without them there is no variance and "
            "the record would be invalid (PROJECT.md §14.2)"
        )

    detail = profile.get("execution_detail") or []
    nodes_per_unit: Counter[str] = Counter()
    time_per_unit: Counter[str] = Counter()
    for node in detail:
        unit = str(node.get("compute_unit") or "UNKNOWN").upper()
        nodes_per_unit[unit] += 1
        with contextlib.suppress(TypeError, ValueError):
            time_per_unit[unit] += float(node.get("execution_time") or 0.0)

    return RemoteProfile(
        job_id=job.job_id,
        job_url=job.url,
        samples_us=[float(v) for v in samples],
        peak_memory_bytes=(
            int(
                _first_number(
                    summary,
                    "inference_memory_peak_range",
                    "estimated_inference_peak_memory",
                )
                or 0
            )
            or None
        ),
        cold_load_ms=_scale_ms(_first_number(summary, "all_first_load_times", "first_load_time")),
        warm_load_ms=_scale_ms(_first_number(summary, "all_warm_load_times", "warm_load_time")),
        compile_ms=_scale_ms(_first_number(summary, "all_compile_times", "compile_time")),
        nodes_per_unit=dict(nodes_per_unit),
        time_per_unit_us=dict(time_per_unit),
        device_name=device.name,
        device_os=str(device.os),
        device_attributes=tuple(sorted(device.attributes)),
        client_version=client_version,
    )


def _scale_ms(microseconds: float | None) -> float | None:
    return microseconds / 1000.0 if microseconds is not None else None


def remote_device_fingerprint(profile: RemoteProfile) -> DeviceFingerprint:
    """Identify a device we do not own.

    ``kind='hosted'`` because it is real physical hardware accessed as a service —
    not a simulator, and not ours. Core count and RAM stay ``None``: AI Hub does not
    report them, and §9.5's whole point is that a second device class falsifies fields
    that only looked mandatory.
    """
    attributes = {
        a.split(":", 1)[0]: a.split(":", 1)[1]
        for a in profile.device_attributes
        if ":" in a
    }
    chipsets = [
        a.split(":", 1)[1] for a in profile.device_attributes if a.startswith("chipset:")
    ]
    soc = next(
        (c for c in chipsets if "snapdragon" not in c),
        chipsets[0] if chipsets else "unknown",
    )
    return DeviceFingerprint(
        kind="hosted",
        model=profile.device_name,
        soc=soc,
        arch=attributes.get("abi", "unknown"),
        os_name=attributes.get("os", "unknown"),
        os_version=profile.device_os,
        # AI Hub does not expose the build. Saying "unknown" is the honest answer, and
        # it matters: Stage 3 exists because an OS build changes delegate behaviour.
        os_build="unknown",
    )


def remote_host_state() -> HostState:
    """Conditions on a device we cannot inspect."""
    return HostState(
        power_source=PowerSource.UNKNOWN,
        thermal_state=ThermalState.UNAVAILABLE,
        unavailable=dict(_UNKNOWN_HOST) | {"power_source": "hosted device; not reported"},
    )


def _on_intended(per_unit: dict[str, float], intended: str) -> float:
    """How much of the graph landed where the recipe wanted it.

    ``ALL`` needs its own arm and not having one was a real defect: it is our sentinel
    for "we did not constrain the unit", not a unit AI Hub ever reports. A plain
    ``per_unit.get("ALL", 0)`` therefore matched nothing and every unconstrained row
    reported **100% fallback while running entirely on the NPU** — an inverted answer on
    the one metric this project exists to measure.

    When the unit is unconstrained the delegate's job is simply to get work off the
    CPU, so anything not on the CPU counts as intended.
    """
    if intended == QaiHubComputeUnit.ALL.value.upper():
        return sum(amount for unit, amount in per_unit.items() if unit.upper() != "CPU")
    return per_unit.get(intended, 0)


#: The toolchain doing the compiling and running on this path. Not inferred — it is
#: simply which service we submitted to.
TOOLCHAIN_VENDOR = "qualcomm"

#: Re-exported so the hosted path reads in one place. Lives in the schema package
#: because the local ORT path needs the same answer for the host's own SoC.
soc_vendor = _soc_vendor

def remote_fallback_report(profile: RemoteProfile, intended: str) -> FallbackReport | None:
    """Per-node compute-unit placement, as the vendor measured it.

    The cross-vendor twin of our CoreML analysis. Node and time share come straight
    from the service's own per-node report, so unlike our local path there is no
    as-authored versus as-run ambiguity — this *is* the executed graph. FLOP share is
    withheld: the graph AI Hub compiled is not the ONNX we submitted, so our static
    estimate would be attributing arithmetic to nodes that no longer exist.
    """
    total_nodes = sum(profile.nodes_per_unit.values())
    if not total_nodes:
        return None

    on_intended = _on_intended(profile.nodes_per_unit, intended)
    total_time = sum(profile.time_per_unit_us.values())
    intended_time = _on_intended(profile.time_per_unit_us, intended)

    unit_names = ", ".join(
        f"{unit}:{count}" for unit, count in sorted(profile.nodes_per_unit.items())
    )
    return FallbackReport(
        intended_provider=intended,
        nodes_total=total_nodes,
        nodes_on_intended=on_intended,
        fallback_node_pct=round(100.0 * (total_nodes - on_intended) / total_nodes, 4),
        time_total_us=round(total_time, 3) or None,
        time_on_intended_us=round(intended_time, 3) if total_time else None,
        fallback_time_pct=(
            round(100.0 * (total_time - intended_time) / total_time, 4) if total_time else None
        ),
        nodes_per_provider=dict(profile.nodes_per_unit),
        node_basis="as_executed",
        analysis_graph_optimization=f"qai_hub compiled ({unit_names})",
        toolchain_vendor=TOOLCHAIN_VENDOR,
        device_soc_vendor=soc_vendor(remote_device_fingerprint(profile).soc),
    )


def build_record(
    recipe: Recipe, profile: RemoteProfile, fingerprint_id: str | None
) -> MeasurementRecord:
    """Turn a hosted profile into a corpus row that does not overclaim."""
    runtime = recipe.runtime
    assert isinstance(runtime, QaiHubRuntimeConfig)
    stats = RunStats.from_samples(profile.timed_samples_ms)

    return MeasurementRecord(
        harness_version=HARNESS_VERSION,
        recipe_id=recipe.recipe_id,
        model_ref=recipe.model.ref,
        graph_fingerprint_id=fingerprint_id,
        device=remote_device_fingerprint(profile),
        host_state=remote_host_state(),
        # Their harness on their hardware. The samples are raw, so latency is a real
        # distribution — but the conditions are not ours to vouch for.
        measurement_source=MeasurementSource.THIRD_PARTY,
        source_detail=(
            f"Qualcomm AI Hub profile job {profile.job_id} ({profile.job_url}); "
            f"qai-hub client {profile.client_version}; "
            f"{len(profile.samples_us)} raw samples reported, first "
            f"{REMOTE_WARMUP_SAMPLES} discarded as warmup; "
            # Hard rule #4 says measure end-to-end. This does not, and the difference
            # has to travel with the number: the atlas will print it beside our own
            # rows, which include the host-side framework overhead that AI Hub's
            # inference time excludes. Load is reported separately, in notes.
            "timing is AI Hub's on-device inference time, which excludes model load "
            "and host-side framework overhead, so it is not end-to-end in our sense "
            "and reads faster than our own rows by that margin"
            + _pipeline_detail(runtime)
        ),
        stress_profile=StressProfile.UNKNOWN,
        outcome=Outcome.SUCCESS,
        run_count=stats.n,
        warmup_count=REMOTE_WARMUP_SAMPLES,
        metrics=Metrics(
            latency_ms=stats,
            peak_rss_bytes=profile.peak_memory_bytes,
            lowering_ms=profile.compile_ms,
            # Promoted out of `notes`: these were always reported, and on a Galaxy
            # S24 the cold load is 1100x the inference it enables. A number that
            # large does not belong in a free-text field.
            cold_load_ms=profile.cold_load_ms,
            warm_load_ms=profile.warm_load_ms,
            unavailable={
                "artifact_bytes": (
                    "AI Hub compiles the model server-side and compile jobs are broken "
                    "for this account, so the deployed artifact size is unknown"
                ),
                "power_mw": "hosted device; no power instrumentation exposed",
                "accuracy": "tier-3 eval-set accuracy not implemented yet",
                "output_cosine_vs_reference": (
                    "a profile job returns timings only; use an inference job to compare outputs"
                ),
                "first_inference_ms": (
                    "AI Hub reports load and steady-state inference times but not the "
                    "duration of the first run specifically, so it is not derivable "
                    "from what the service exposes"
                ),
                "sustained_tok_s_5min": "not a generative measurement",
                "ttft_ms": "not a generative measurement",
                "decode_tok_s": "not a generative measurement",
                "token_agreement": "not a generative measurement",
            },
        ),
        fallback_as_run=remote_fallback_report(profile, runtime.intended_provider),
        notes=(
            f"cold load {profile.cold_load_ms:.1f} ms, warm load {profile.warm_load_ms:.1f} ms"
            if profile.cold_load_ms and profile.warm_load_ms
            else None
        ),
    )


def failure_record(
    recipe: Recipe, failure: RemoteJobFailed, fingerprint_id: str | None
) -> MeasurementRecord:
    """A hosted job that ran and failed, with the vendor's reason attached.

    This is the cross-vendor counterpart of our local ``lowering_failure`` rows. A
    delegate refusing a graph is the finding, not an inconvenience.
    """
    runtime = recipe.runtime
    assert isinstance(runtime, QaiHubRuntimeConfig)
    return MeasurementRecord(
        harness_version=HARNESS_VERSION,
        recipe_id=recipe.recipe_id,
        model_ref=recipe.model.ref,
        graph_fingerprint_id=fingerprint_id,
        device=DeviceFingerprint(
            kind="hosted",
            model=runtime.device_name,
            soc="unknown",
            arch="unknown",
            os_name="unknown",
            os_version=runtime.device_os or "unknown",
            os_build="unknown",
        ),
        host_state=remote_host_state(),
        measurement_source=MeasurementSource.THIRD_PARTY,
        source_detail=(
            f"Qualcomm AI Hub profile job {failure.job_id} ({failure.job_url}) "
            f"ended in FAILED"
        ),
        stress_profile=StressProfile.UNKNOWN,
        outcome=Outcome.RUNTIME_FAILURE,
        failure_reason=str(failure),
        run_count=0,
        warmup_count=0,
        metrics=Metrics(
            unavailable={
                "latency_ms": "the hosted job failed, so nothing was timed",
                "power_mw": "hosted device; no power instrumentation exposed",
                "accuracy": "the hosted job failed",
            }
        ),
    )


def graph_fingerprint(artifact_dir: Path) -> GraphFingerprint | None:
    """Fingerprint the artifact about to be uploaded.

    Hosted rows carried no fingerprint at all — 120 of them — because the field was
    threaded through the record and nothing ever populated it. §5.2 calls the
    fingerprint the key a cost model indexes on, so a corpus whose every cross-vendor
    row lacks one cannot answer the question those rows exist to raise: *does the graph
    predict whether a delegate claims it?* Nine models fall back on a Pixel 9 and one
    does not, and the graphs were not recorded beside the outcomes.

    Fingerprinted from the ONNX we submit, not from whatever AI Hub compiles it into:
    that is the object we can inspect, and the one a user would hand to any other
    runtime.
    """
    from edgefit.backends.analysis.graph import fingerprint_onnx  # noqa: PLC0415

    model_path = artifact_dir / "model.onnx"
    try:
        import onnx  # noqa: PLC0415

        return fingerprint_onnx(onnx.load(str(model_path), load_external_data=False))
    except Exception:  # noqa: BLE001 - a gap in the corpus beats a wasted provisioning
        return None


def measure_remote(
    recipe: Recipe,
    artifact_dir: Path,
    *,
    store: CorpusStore | None = None,
    fingerprint_id: str | None = None,
) -> MeasurementRecord:
    """Profile one recipe on its hosted device and record the result.

    A job that runs and fails produces a ``runtime_failure`` row carrying the
    vendor's message, rather than an exception the caller has to interpret.
    """
    fingerprint = graph_fingerprint(artifact_dir) if fingerprint_id is None else None
    if fingerprint is not None:
        fingerprint_id = fingerprint.fingerprint_id

    try:
        profile = submit_profile(recipe, artifact_dir)
    except RemoteJobFailed as failure:
        record = failure_record(recipe, failure, fingerprint_id)
    else:
        record = build_record(recipe, profile, fingerprint_id)

    if store is not None:
        store.insert_recipe(recipe)
        if fingerprint is not None:
            store.insert_fingerprint(fingerprint)
        store.insert_measurement(record)
    return record


@dataclass
class RemoteSweepReport:
    """Outcome of a hosted sweep. Mirrors the local one so the two read alike."""

    cells: int = 0
    measured: int = 0
    failed: int = 0
    resumed: int = 0
    refused: int = 0
    elapsed_s: float = 0.0

    @property
    def attempted(self) -> int:
        return self.cells - self.resumed


def _already_measured(store: CorpusStore, recipe_id: str) -> bool:
    """Whether this exact cell already has a row at this harness version.

    Keyed on ``recipe_id`` alone, which is enough here and is not for local rows: a
    hosted recipe carries its own ``device_name`` and ``compute_unit``, so the hash
    already identifies the (model, device, unit) cell. The local path keys on
    ``device_id`` because one recipe is measured on many machines.

    ``device_id`` could not be used even if we wanted to — it hashes the fingerprint,
    which for a hosted device only exists after the job has run, which is the cost
    resuming is meant to avoid.
    """
    row = store.query(
        "SELECT 1 FROM measurements WHERE recipe_id = ? AND harness_version = ? "
        "AND outcome <> 'gate_refused' LIMIT 1",
        [recipe_id, HARNESS_VERSION],
    ).fetchone()
    return row is not None


def sweep_remote(
    model_refs: Sequence[str],
    device_names: Sequence[str],
    *,
    store: CorpusStore,
    compute_unit: QaiHubComputeUnit = QaiHubComputeUnit.ALL,
    resume: bool = True,
    on_event: Callable[[str, str, str], None] | None = None,
) -> RemoteSweepReport:
    """Profile every (model, device) cell on hosted hardware, recording every outcome.

    The hosted twin of ``run_sweep``, and deliberately not the same function: there is
    no gate, no thermal wait and no host lock, because none of those describe someone
    else's rack. What it keeps is the part that matters — resumption, and a row for
    every outcome including the failures.

    Models are checked once before any device is spent. A model a hosted profiler
    cannot feed is a property of the graph, not of the hardware, so discovering it
    thirty jobs in would waste thirty provisions to learn one fact.
    """
    from edgefit.backends.artifacts import resolve_artifact  # noqa: PLC0415
    from edgefit.models.registry import resolve  # noqa: PLC0415

    report = RemoteSweepReport()
    started = time.monotonic()

    def emit(kind: str, cell: str, detail: str = "") -> None:
        if on_event is not None:
            on_event(kind, cell, detail)

    def lowering_for(spec) -> dict:
        """Text models need their token ids frozen to be profilable here at all.

        A hosted profiler invents its own inputs and cannot invent a valid token id,
        so an index-input graph is refused. Freezing the ids into the graph leaves a
        single float input, which is always in range. Applied automatically because
        the alternative is not "a normal text row" but no row at all.
        """
        return {
            "static_shapes": spec.exporter != "decoder",
            "frozen_token_inputs": spec.exporter == "text",
        }

    usable: list[tuple[str, object, Path]] = []
    for ref in model_refs:
        spec = resolve(ref)
        recipe = Recipe(
            model=ModelRef(ref=spec.ref, task=spec.task),
            runtime=QaiHubRuntimeConfig(device_name=device_names[0], compute_unit=compute_unit),
            lowering=lowering_for(spec),
        )
        artifact = resolve_artifact(spec, recipe)
        try:
            check_inputs_are_synthesizable(artifact.model_path)
        except UnprofilableOnHostedService as exc:
            # Not a row: nothing about any device was learned. See the refusal's own
            # docstring for why this must not be recorded against the hardware.
            report.refused += 1
            emit("unprofilable", ref, str(exc))
            continue
        usable.append((ref, spec, artifact.directory))

    for ref, spec, artifact_dir in usable:
        for name in device_names:
            cell = f"{ref.removeprefix('hf:')} × {name}"
            report.cells += 1
            recipe = Recipe(
                model=ModelRef(ref=spec.ref, task=spec.task),
                runtime=QaiHubRuntimeConfig(device_name=name, compute_unit=compute_unit),
                lowering=lowering_for(spec),
            )
            if resume and _already_measured(store, recipe.recipe_id):
                report.resumed += 1
                emit("resumed", cell)
                continue

            emit("submitting", cell)
            try:
                record = measure_remote(recipe, artifact_dir, store=store)
            except RemoteMeasurementError as exc:
                # Never submitted — no row, because no device reported anything.
                report.failed += 1
                emit("not-submitted", cell, str(exc))
                continue

            if record.outcome is Outcome.SUCCESS:
                report.measured += 1
                stats = record.metrics.latency_ms  # type: ignore[union-attr]
                fb = record.fallback_as_run
                placement = (
                    " · ".join(f"{u}:{n}" for u, n in sorted((fb.nodes_per_provider or {}).items()))
                    if fb
                    else "—"
                )
                emit("measured", cell, f"p50 {stats.p50:.2f} ms · cv {stats.cv:.1%} · {placement}")
            else:
                report.failed += 1
                emit("failed", cell, (record.failure_reason or "")[:160])

    report.elapsed_s = time.monotonic() - started
    return report
