"""ONNX Runtime backend — CPU and CoreML execution providers.

Backend #1 because it buys two real measurement targets on one machine and,
unlike llama.cpp, has a genuine partitioner whose decisions can be inspected.
That makes silent CPU fallback (PROJECT.md §2.2) measurable on day one with no
hardware spend.

Two sessions are built per measurement, on purpose:

* the **analysis** session runs with graph optimisation disabled, so profile node
  names still correspond to the graph the user authored and the unclaimed-op list
  is actionable rather than a list of ORT-internal fusion names;
* the **measurement** session runs exactly as configured, because that is the
  latency a user would actually experience (hard rule #4).

The distinction is recorded on every fallback report rather than left implicit.
"""

from __future__ import annotations

import contextlib
import glob
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from edgefit.backends.analysis.ep_placement import (
    build_as_run_report,
    build_fallback_report,
    parse_profile,
)
from edgefit.backends.analysis.flops import FLOPS_ESTIMATOR_VERSION, estimate_flops
from edgefit.backends.analysis.graph import fingerprint_onnx
from edgefit.backends.base import DeviceRun, StaticAnalysis
from edgefit.backends.export_decoder import decoder_shape
from edgefit.backends.export_onnx import artifact_size_bytes
from edgefit.harness.memory import peak_rss_bytes
from edgefit.schema.common import RuntimeKind, TaskType
from edgefit.schema.recipe import GraphOptLevel, OrtProvider, Recipe

_OPT_LEVEL = {
    GraphOptLevel.DISABLED: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    GraphOptLevel.BASIC: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    GraphOptLevel.EXTENDED: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    GraphOptLevel.ALL: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

_ANALYSIS_OPT_LEVEL = "disabled"


class OrtBackend:
    """ONNX Runtime, CPU and CoreML EPs."""

    kind = RuntimeKind.ONNXRUNTIME

    # -- session construction ---------------------------------------------

    def _provider_spec(self, recipe: Recipe) -> list[tuple[str, dict[str, str]]]:
        """Providers with their options, in ORT's priority order."""
        runtime = recipe.runtime
        spec: list[tuple[str, dict[str, str]]] = []
        for provider in runtime.providers:
            options: dict[str, str] = {}
            if provider is OrtProvider.COREML:
                if runtime.coreml_compute_units is not None:
                    options["MLComputeUnits"] = str(runtime.coreml_compute_units.value)
                if runtime.coreml_model_format is not None:
                    options["ModelFormat"] = runtime.coreml_model_format
                if runtime.coreml_require_static_shapes is not None:
                    options["RequireStaticInputShapes"] = (
                        "1" if runtime.coreml_require_static_shapes else "0"
                    )
                if runtime.coreml_allow_low_precision is not None:
                    options["AllowLowPrecisionAccumulationOnGPU"] = (
                        "1" if runtime.coreml_allow_low_precision else "0"
                    )
            spec.append((str(provider), options))
        return spec

    def _session_options(
        self, recipe: Recipe, *, opt_level: GraphOptLevel, profile_prefix: str | None
    ) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.graph_optimization_level = _OPT_LEVEL[opt_level]

        if recipe.execution.num_threads is not None:
            options.intra_op_num_threads = recipe.execution.num_threads
        if recipe.runtime.inter_op_num_threads is not None:
            options.inter_op_num_threads = recipe.runtime.inter_op_num_threads
        options.execution_mode = (
            ort.ExecutionMode.ORT_PARALLEL
            if recipe.runtime.parallel_execution
            else ort.ExecutionMode.ORT_SEQUENTIAL
        )
        if profile_prefix is not None:
            options.enable_profiling = True
            options.profile_file_prefix = profile_prefix
        return options

    def _create_session(
        self,
        model_path: Path,
        recipe: Recipe,
        *,
        opt_level: GraphOptLevel,
        profile_prefix: str | None = None,
    ) -> ort.InferenceSession:
        providers = self._provider_spec(recipe)
        return ort.InferenceSession(
            str(model_path),
            self._session_options(recipe, opt_level=opt_level, profile_prefix=profile_prefix),
            providers=[name for name, _ in providers],
            provider_options=[options for _, options in providers],
        )

    def _feeds(self, artifact_dir: Path, recipe: Recipe) -> dict[str, np.ndarray]:
        """The input feed for one forward pass.

        A decoder needs its past-key/value inputs present even for a single pass —
        prefill is a step whose past has zero length — so the feed is the pinned
        prompt plus empty cache tensors. Omitting them makes ORT reject the run with
        32 missing inputs, which is how this was found.
        """
        with np.load(artifact_dir / "inputs.npz") as data:
            feeds = {name: data[name] for name in data.files}
        if recipe.model.task is TaskType.GENERATE:
            import json  # noqa: PLC0415

            meta = json.loads((artifact_dir / "meta.json").read_text())
            feeds |= decoder_shape(meta).empty_past()
        return feeds

    # -- tier 1: static analysis ------------------------------------------

    def analyze(self, artifact_dir: Path, recipe: Recipe) -> StaticAnalysis:
        """Lower, fingerprint, and determine what the partitioner actually claimed."""
        model_path = artifact_dir / "model.onnx"
        artifact_bytes = artifact_size_bytes(artifact_dir)

        try:
            model = onnx.load(str(model_path))
        except Exception as exc:  # noqa: BLE001
            return StaticAnalysis(
                lowered=False,
                artifact_bytes=artifact_bytes,
                failure_reason=f"could not load ONNX model: {exc}",
            )

        # A decoder's meta.json records its head layout, so the attention variant is
        # arithmetic rather than guessed. Without it the detector cannot tell GQA from
        # MHA and correctly says UNKNOWN.
        architecture: dict[str, int] = {}
        meta_path = artifact_dir / "meta.json"
        if meta_path.exists():
            import json  # noqa: PLC0415

            meta = json.loads(meta_path.read_text())
            for key, field in (("layers", "n_layers"), ("kv_heads", "n_kv_heads")):
                if key in meta:
                    architecture[field] = int(meta[key])
            if "n_heads" in meta:
                architecture["n_heads"] = int(meta["n_heads"])
        fingerprint = fingerprint_onnx(model, **architecture)
        flops = estimate_flops(model)

        feeds = self._feeds(artifact_dir, recipe)

        with tempfile.TemporaryDirectory() as scratch:
            prefix = os.path.join(scratch, "analysis")
            started = time.perf_counter()
            try:
                session = self._create_session(
                    model_path,
                    recipe,
                    opt_level=GraphOptLevel.DISABLED,
                    profile_prefix=prefix,
                )
            except Exception as exc:  # noqa: BLE001 - a refused partition is data, not a crash
                return StaticAnalysis(
                    lowered=False,
                    artifact_bytes=artifact_bytes,
                    failure_reason=f"session creation failed: {exc}",
                    fingerprint=fingerprint,
                    lowering_ms=(time.perf_counter() - started) * 1000.0,
                )
            lowering_ms = (time.perf_counter() - started) * 1000.0

            try:
                session.run(None, feeds)
            except Exception as exc:  # noqa: BLE001
                return StaticAnalysis(
                    lowered=False,
                    artifact_bytes=artifact_bytes,
                    failure_reason=f"analysis run failed: {exc}",
                    fingerprint=fingerprint,
                    lowering_ms=lowering_ms,
                )

            profile_path = session.end_profiling()
            events = parse_profile(profile_path)
            del session
            for stale in glob.glob(f"{prefix}*"):
                with contextlib.suppress(OSError):
                    os.remove(stale)

        fallback = build_fallback_report(
            model,
            events,
            intended_provider=recipe.intended_provider,
            flops=flops,
            runs=1,
        )
        fallback = fallback.model_copy(
            update={
                "analysis_graph_optimization": _ANALYSIS_OPT_LEVEL,
                "flops_estimator_version": FLOPS_ESTIMATOR_VERSION if flops.is_complete else None,
                "partition_count": fallback.nodes_per_provider.get(
                    recipe.intended_provider + " (fused partitions)"
                ),
            }
        )

        return StaticAnalysis(
            lowered=True,
            artifact_bytes=artifact_bytes,
            fingerprint=fingerprint,
            fallback=fallback,
            fallback_as_run=self._as_run_report(model_path, recipe, feeds),
            lowering_ms=lowering_ms,
        )

    def _as_run_report(self, model_path: Path, recipe: Recipe, feeds: dict):
        """Profile once more at the recipe's own optimisation level.

        The partition analysed with optimisation disabled is measurably not the one
        executed at level `all` — on ViT-base, 244 CPU nodes became 86 — so a
        fallback figure from the first pass does not describe the timed run. Cheap
        enough to just measure both rather than argue about which is right.
        """
        level = recipe.runtime.graph_optimization_level
        if level is GraphOptLevel.DISABLED:
            return None  # identical to the as-authored pass; a duplicate row is noise

        with tempfile.TemporaryDirectory() as scratch:
            prefix = os.path.join(scratch, "asrun")
            try:
                session = self._create_session(
                    model_path, recipe, opt_level=level, profile_prefix=prefix
                )
                session.run(None, feeds)
                events = parse_profile(session.end_profiling())
                del session
            except Exception:  # noqa: BLE001 - the primary report already landed
                return None
            for stale in glob.glob(f"{prefix}*"):
                with contextlib.suppress(OSError):
                    os.remove(stale)

        return build_as_run_report(
            events,
            intended_provider=recipe.intended_provider,
            graph_optimization=str(level),
            runs=1,
        )

    # -- tier 2: device measurement ---------------------------------------

    def measure(
        self, artifact_dir: Path, recipe: Recipe, runs: int, warmup: int, decode_tokens: int = 32
    ) -> DeviceRun:
        """Timed runs at the configured optimisation level.

        Runs inside the measurement subprocess, so ``peak_rss_bytes`` here is this
        process's own high-water mark and is attributable to this recipe alone.
        """
        if recipe.model.task is TaskType.GENERATE:
            return self._measure_generative(artifact_dir, recipe, runs, warmup, decode_tokens)

        model_path = artifact_dir / "model.onnx"
        feeds = self._feeds(artifact_dir, recipe)

        try:
            session = self._create_session(
                model_path, recipe, opt_level=recipe.runtime.graph_optimization_level
            )
        except Exception as exc:  # noqa: BLE001
            return DeviceRun(failure_reason=f"session creation failed: {exc}")

        try:
            # Warmup runs are discarded (PROJECT.md §5.6): first-call cost includes
            # lazy kernel compilation and CoreML model caching, which is a real
            # cost but a different one from steady-state latency.
            for _ in range(warmup):
                session.run(None, feeds)

            samples_ms: list[float] = []
            outputs = None
            for _ in range(runs):
                started = time.perf_counter_ns()
                result = session.run(None, feeds)
                samples_ms.append((time.perf_counter_ns() - started) / 1e6)
                outputs = result
        except Exception as exc:  # noqa: BLE001
            return DeviceRun(failure_reason=f"inference failed: {exc}", warmup_count=warmup)

        output_names = [o.name for o in session.get_outputs()]
        return DeviceRun(
            samples_ms=samples_ms,
            peak_rss_bytes=peak_rss_bytes(),
            warmup_count=warmup,
            outputs={
                name: np.asarray(value).ravel()[:512].tolist()
                for name, value in zip(output_names, outputs or [], strict=False)
            },
        )

    # -- tier 2, generative: prefill then decode -----------------------------

    def _measure_generative(
        self, artifact_dir: Path, recipe: Recipe, runs: int, warmup: int, decode_tokens: int
    ) -> DeviceRun:
        """Measure TTFT and decode throughput separately.

        Two distributions, not one, because prefill and decode are different
        workloads (PROJECT.md §2.3). A single averaged "latency" for a generative
        model describes neither phase and is the number vendors most often quote.

        The KV cache is threaded through explicitly: each step feeds the previous
        step's ``present`` tensors back in as ``past``. Without that the graph would
        reprocess the whole sequence every step and the tok/s figure would be
        fiction.
        """
        import json  # noqa: PLC0415

        model_path = artifact_dir / "model.onnx"
        meta = json.loads((artifact_dir / "meta.json").read_text())
        shape = decoder_shape(meta)

        with np.load(artifact_dir / "inputs.npz") as data:
            prompt_ids = data["input_ids"]
            prompt_mask = data["attention_mask"]
            prompt_positions = data["position_ids"]

        try:
            session = self._create_session(
                model_path, recipe, opt_level=recipe.runtime.graph_optimization_level
            )
        except Exception as exc:  # noqa: BLE001
            return DeviceRun(failure_reason=f"session creation failed: {exc}")

        past_names, present_names = shape.past_names, shape.present_names

        def one_run() -> tuple[float, float, list[int]]:
            """Returns (ttft_ms, decode_tok_s, tokens)."""
            past = shape.empty_past()
            total = int(prompt_ids.shape[1])
            tokens: list[int] = []

            # --- prefill: the whole prompt in one pass. This is TTFT. ---
            started = time.perf_counter_ns()
            outputs = session.run(
                None,
                {
                    "input_ids": prompt_ids,
                    "attention_mask": prompt_mask,
                    "position_ids": prompt_positions,
                    **past,
                },
            )
            ttft_ms = (time.perf_counter_ns() - started) / 1e6
            tokens.append(int(np.asarray(outputs[0])[0, -1].argmax()))
            past = dict(zip(past_names, outputs[1:], strict=True))
            total += 1

            # --- decode: one token at a time, re-feeding the cache. ---
            decode_started = time.perf_counter_ns()
            for _ in range(decode_tokens):
                feeds = {
                    "input_ids": np.array([[tokens[-1]]], dtype=np.int64),
                    "attention_mask": np.ones((1, total), dtype=np.int64),
                    # Rotary position of the token being generated. Must advance, or
                    # every step re-uses prefill's positions and the text diverges.
                    "position_ids": np.array([[total - 1]], dtype=np.int64),
                    **past,
                }
                outputs = session.run(None, feeds)
                tokens.append(int(np.asarray(outputs[0])[0, -1].argmax()))
                past = dict(zip(past_names, outputs[1:], strict=True))
                total += 1
            decode_s = (time.perf_counter_ns() - decode_started) / 1e9
            return ttft_ms, (decode_tokens / decode_s if decode_s else 0.0), tokens

        try:
            for _ in range(warmup):
                one_run()
            ttft_samples: list[float] = []
            decode_samples: list[float] = []
            tokens: list[int] = []
            for _ in range(runs):
                ttft, rate, tokens = one_run()
                ttft_samples.append(ttft)
                decode_samples.append(rate)
        except Exception as exc:  # noqa: BLE001
            return DeviceRun(
                failure_reason=f"generative inference failed: {exc}", warmup_count=warmup
            )

        assert present_names  # names are paired with outputs above
        return DeviceRun(
            peak_rss_bytes=peak_rss_bytes(),
            warmup_count=warmup,
            ttft_samples_ms=ttft_samples,
            decode_samples_tok_s=decode_samples,
            generated_tokens=tokens,
        )
