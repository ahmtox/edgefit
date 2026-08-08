"""Hosted measurement on Qualcomm AI Hub.

Network calls are stubbed; what is tested is the part that decides what enters the
corpus — warmup handling, provenance honesty, and per-node placement parsing.
"""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from edgefit.harness.remote import (
    REMOTE_WARMUP_SAMPLES,
    RemoteMeasurementError,
    RemoteProfile,
    UnprofilableOnHostedService,
    _parse_profile,
    _profile_options,
    build_record,
    check_inputs_are_synthesizable,
    remote_device_fingerprint,
    remote_fallback_report,
    remote_host_state,
)
from edgefit.schema import (
    MeasurementSource,
    ModelRef,
    Outcome,
    QaiHubComputeUnit,
    QaiHubRuntimeConfig,
    Recipe,
    StressProfile,
    TaskType,
    ThermalState,
)

#: The shape AI Hub actually returns: cold first runs, then steady state.
RAW_US = [1865.0, 109.0, 82.0, 55.0, 57.0, 55.0, 56.0, 55.0, 54.0, 56.0]

ATTRS = (
    "abi:aarch64-android",
    "chipset:qualcomm-snapdragon-8gen3",
    "chipset:sm8650",
    "format:phone",
    "framework:onnx",
    "hexagon:v75",
    "os:android",
)


def _profile(**overrides) -> RemoteProfile:
    base = {
        "job_id": "j5797x6lg",
        "job_url": "https://workbench.aihub.qualcomm.com/jobs/j5797x6lg/",
        "samples_us": list(RAW_US),
        "peak_memory_bytes": 123 * 1024**2,
        "cold_load_ms": 40.0,
        "warm_load_ms": 12.0,
        "compile_ms": 900.0,
        "nodes_per_unit": {"NPU": 8, "CPU": 2},
        "time_per_unit_us": {"NPU": 40.0, "CPU": 10.0},
        "device_name": "Samsung Galaxy S24 (Family)",
        "device_os": "14",
        "device_attributes": ATTRS,
        "client_version": "0.54.0",
    }
    return RemoteProfile(**(base | overrides))


def _recipe(unit: QaiHubComputeUnit = QaiHubComputeUnit.NPU) -> Recipe:
    return Recipe(
        model=ModelRef(ref="hf:sentence-transformers/all-MiniLM-L6-v2", task=TaskType.EMBED),
        runtime=QaiHubRuntimeConfig(device_name="Samsung Galaxy S24 (Family)", compute_unit=unit),
    )


class TestWarmupHandling:
    def test_discards_the_cold_leading_samples(self) -> None:
        """An observed series began [1865, 109, 82, 55, ...] microseconds.

        Keeping the 1865 would inflate the coefficient of variation by more than an
        order of magnitude and misreport steady-state latency.
        """
        kept = _profile().timed_samples_ms
        assert len(kept) == len(RAW_US) - REMOTE_WARMUP_SAMPLES
        assert max(kept) < 0.1, "a cold outlier survived into the timed samples"

    def test_converts_microseconds_to_milliseconds(self) -> None:
        assert _profile().timed_samples_ms[0] == pytest.approx(0.055)

    def test_warmup_count_matches_the_local_policy(self) -> None:
        """Both sources discard three, so their variance figures stay comparable."""
        from edgefit.harness.timing import MeasurementPolicy

        assert MeasurementPolicy().warmup == REMOTE_WARMUP_SAMPLES


class TestProvenance:
    def test_recorded_as_third_party(self) -> None:
        """Their harness, their device. The samples are raw; the conditions are not ours."""
        record = build_record(_recipe(), _profile(), None)
        assert record.measurement_source is MeasurementSource.THIRD_PARTY
        assert record.source_detail is not None

    def test_source_detail_names_the_job_and_the_client(self) -> None:
        detail = build_record(_recipe(), _profile(), None).source_detail or ""
        assert "j5797x6lg" in detail
        assert "0.54.0" in detail
        assert "discarded as warmup" in detail

    def test_source_detail_says_the_timing_is_not_end_to_end(self) -> None:
        """Hard rule #4 is measure end-to-end; AI Hub's inference time is not.

        The caveat has to travel on the row, because the atlas prints these beside our
        own numbers — ViT reads 10.9 ms on a Snapdragon NPU against 108.7 ms on our M2
        CPU, and part of that gap is a difference in what the two harnesses time.
        """
        detail = build_record(_recipe(), _profile(), None).source_detail or ""
        assert "not end-to-end" in detail
        assert "framework overhead" in detail

    def test_stress_profile_is_unknown_not_clean(self) -> None:
        """We cannot see the device's thermal state, so we must not claim it was quiet."""
        assert build_record(_recipe(), _profile(), None).stress_profile is StressProfile.UNKNOWN

    def test_latency_is_a_real_distribution(self) -> None:
        """Raw samples mean hard rule #2 is satisfied natively, not waived."""
        record = build_record(_recipe(), _profile(), None)
        stats = record.metrics.latency_ms
        assert stats is not None
        assert stats.n == len(RAW_US) - REMOTE_WARMUP_SAMPLES
        assert record.run_count == stats.n
        assert record.outcome is Outcome.SUCCESS

    def test_unmeasurable_fields_carry_reasons(self) -> None:
        metrics = build_record(_recipe(), _profile(), None).metrics
        assert metrics is not None
        for field in ("artifact_bytes", "power_mw", "output_cosine_vs_reference"):
            assert field in metrics.unavailable, f"{field} absent without explanation"


class TestDeviceFingerprint:
    def test_prefers_the_silicon_code_over_the_marketing_name(self) -> None:
        assert remote_device_fingerprint(_profile()).soc == "sm8650"

    def test_marked_hosted_not_virtual(self) -> None:
        """Real hardware, accessed as a service, and not ours."""
        assert remote_device_fingerprint(_profile()).kind == "hosted"

    def test_core_count_and_ram_stay_absent(self) -> None:
        """AI Hub does not report them; §9.5 predicted a second device class would
        falsify fields that only looked mandatory."""
        fingerprint = remote_device_fingerprint(_profile())
        assert fingerprint.cpu_cores_total is None
        assert fingerprint.ram_bytes is None

    def test_os_build_is_unknown_and_says_so(self) -> None:
        assert remote_device_fingerprint(_profile()).os_build == "unknown"

    def test_host_state_admits_what_it_cannot_see(self) -> None:
        state = remote_host_state()
        assert state.thermal_state is ThermalState.UNAVAILABLE
        assert "hosted device" in state.unavailable["thermal_state"]


class TestFallbackFromComputeUnits:
    def test_computes_node_and_time_share(self) -> None:
        report = remote_fallback_report(_profile(), "NPU")
        assert report is not None
        assert (report.nodes_total, report.nodes_on_intended) == (10, 8)
        assert report.fallback_node_pct == pytest.approx(20.0)
        assert report.fallback_time_pct == pytest.approx(20.0)

    def test_withholds_flop_share(self) -> None:
        """AI Hub compiled a different graph than the ONNX we submitted, so attributing
        our static FLOP estimate to its nodes would be a guess."""
        report = remote_fallback_report(_profile(), "NPU")
        assert report is not None
        assert report.fallback_flops_pct is None
        assert report.node_basis == "as_executed"

    def test_all_on_cpu_is_total_fallback(self) -> None:
        report = remote_fallback_report(
            _profile(nodes_per_unit={"CPU": 3}, time_per_unit_us={"CPU": 9.0}), "NPU"
        )
        assert report is not None
        assert report.fallback_node_pct == pytest.approx(100.0)

    def test_an_unconstrained_unit_counts_anything_off_the_cpu(self) -> None:
        """`ALL` is our sentinel for "we did not choose", not a unit AI Hub reports.

        Looking it up in `nodes_per_unit` matched nothing, so every unconstrained row
        reported **100% fallback while running entirely on the NPU** — inverted, on the
        one metric this project exists to measure. Published, it would have said
        Qualcomm's NPU never claims a graph.
        """
        report = remote_fallback_report(_profile(), QaiHubComputeUnit.ALL.value.upper())
        assert report is not None
        # 8 of 10 nodes on NPU, 2 on CPU: the CPU share is the fallback.
        assert report.nodes_on_intended == 8
        assert report.fallback_node_pct == pytest.approx(20.0)
        assert report.fallback_time_pct == pytest.approx(20.0)

    def test_unconstrained_with_everything_on_the_npu_is_zero_fallback(self) -> None:
        """The real shape of the ViT rows: 429 of 429 nodes accelerated."""
        report = remote_fallback_report(
            _profile(nodes_per_unit={"NPU": 429}, time_per_unit_us={"NPU": 7678.0}),
            QaiHubComputeUnit.ALL.value.upper(),
        )
        assert report is not None
        assert report.fallback_node_pct == pytest.approx(0.0)
        assert report.fallback_time_pct == pytest.approx(0.0)

    def test_unconstrained_with_everything_on_the_cpu_is_total_fallback(self) -> None:
        report = remote_fallback_report(
            _profile(nodes_per_unit={"CPU": 401}, time_per_unit_us={"CPU": 271288.0}),
            QaiHubComputeUnit.ALL.value.upper(),
        )
        assert report is not None
        assert report.fallback_node_pct == pytest.approx(100.0)

    def test_gpu_counts_as_accelerated_when_unconstrained(self) -> None:
        """Off the CPU is the point; which accelerator claimed it is a separate question."""
        report = remote_fallback_report(
            _profile(
                nodes_per_unit={"GPU": 30, "CPU": 10},
                time_per_unit_us={"GPU": 90.0, "CPU": 10.0},
            ),
            QaiHubComputeUnit.ALL.value.upper(),
        )
        assert report is not None
        assert report.nodes_on_intended == 30
        assert report.fallback_time_pct == pytest.approx(10.0)

    def test_no_detail_yields_no_report(self) -> None:
        assert remote_fallback_report(_profile(nodes_per_unit={}), "NPU") is None


class TestParseProfile:
    class _Job:
        job_id = "jabc"
        url = "https://example/jobs/jabc/"

    class _Device:
        name = "Samsung Galaxy S24 (Family)"
        os = "14"
        attributes = list(ATTRS)

    def test_reads_samples_and_placement(self) -> None:
        parsed = _parse_profile(
            {
                "execution_summary": {"all_inference_times": [100, 60, 55, 54]},
                "execution_detail": [
                    {"compute_unit": "NPU", "execution_time": 30},
                    {"compute_unit": "cpu", "execution_time": 10},
                ],
            },
            self._Job(),
            self._Device(),
            "0.54.0",
        )
        assert parsed.samples_us == [100.0, 60.0, 55.0, 54.0]
        # lower-case from the service is normalised, so counts do not split
        assert parsed.nodes_per_unit == {"NPU": 1, "CPU": 1}

    def test_a_profile_without_samples_is_refused(self) -> None:
        """No raw samples means no variance, and a record without variance is invalid."""
        with pytest.raises(RemoteMeasurementError, match="no per-run samples"):
            _parse_profile({"execution_summary": {}}, self._Job(), self._Device(), "0.54.0")


class TestProfileOptions:
    """Every recipe axis must actually reach the job.

    The first version of this backend passed ``--target_runtime`` — a *compile*-job
    flag that profile jobs reject outright — while ``compute_unit``, the axis profile
    jobs do honour, was never sent at all. The recipe therefore recorded a constraint
    the run never applied, which is the same class of defect as quoting a FLOP share
    for a graph that never ran.
    """

    def test_a_requested_unit_becomes_the_flag_the_service_accepts(self) -> None:
        for unit in (QaiHubComputeUnit.NPU, QaiHubComputeUnit.GPU, QaiHubComputeUnit.CPU):
            options = _profile_options(_recipe(unit).runtime)
            assert options == f"--compute_unit {unit.value}"

    def test_the_default_sends_nothing(self) -> None:
        """`all` is the service default; sending it would claim a constraint we did not set."""
        assert _profile_options(_recipe(QaiHubComputeUnit.ALL).runtime) is None

    def test_target_runtime_is_not_sent_to_a_profile_job(self) -> None:
        """It is a compile-job flag, and profile jobs reject it outright.

        `target_runtime` is expressible again — compile works, so the field is honoured
        by compiling before profiling. But it must never reach the *profile* options, or
        the job fails with `unrecognized arguments`. That is the invariant this replaces
        the old "the field cannot exist" test with.
        """
        recipe = QaiHubRuntimeConfig(
            device_name="Samsung Galaxy S24 (Family)", target_runtime="tflite"
        )
        assert _profile_options(recipe) is None

    def test_quantization_requires_a_target_runtime(self) -> None:
        """A quantized model still has to be lowered to something runnable."""
        with pytest.raises(ValidationError, match="needs a target_runtime"):
            QaiHubRuntimeConfig(
                device_name="Samsung Galaxy S24 (Family)",
                quantize={"weights_dtype": "int8"},
            )


class TestInputSynthesis:
    """A hosted profiler fabricates its own inputs, and cannot fabricate an index.

    MiniLM failed on a Snapdragon CPU with `indices element out of data bounds, idx=3
    must be within the inclusive range [-2,1]` on the `token_type_embeddings` Gather.
    A random int64 is not a token type. That failure is about neither the device nor
    the model — it is about submitting an index-input graph to a service that invents
    inputs — so it must not become a `runtime_failure` row against the device.
    """

    @staticmethod
    def _model(tmp_path, inputs):
        import onnx
        from onnx import helper

        graph = helper.make_graph(
            nodes=[helper.make_node("Identity", [inputs[0][0]], ["out"])],
            name="g",
            inputs=[helper.make_tensor_value_info(n, t, [1, 8]) for n, t in inputs],
            outputs=[helper.make_tensor_value_info("out", inputs[0][1], [1, 8])],
        )
        path = tmp_path / "model.onnx"
        onnx.save_model(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), str(path)
        )
        return path

    def test_an_index_input_is_refused_before_a_device_is_spent(self, tmp_path) -> None:
        from onnx import TensorProto

        path = self._model(
            tmp_path,
            [
                ("input_ids", TensorProto.INT64),
                ("token_type_ids", TensorProto.INT64),
            ],
        )

        with pytest.raises(UnprofilableOnHostedService) as caught:
            check_inputs_are_synthesizable(path)
        # Names the offending inputs, so the message is actionable rather than a verdict.
        assert "input_ids" in str(caught.value)
        assert "token_type_ids" in str(caught.value)

    def test_a_float_input_model_is_allowed(self, tmp_path) -> None:
        """ViT takes only `pixel_values`: every random value is a legal image."""
        from onnx import TensorProto

        path = self._model(tmp_path, [("pixel_values", TensorProto.FLOAT)])
        check_inputs_are_synthesizable(path)  # must not raise


def test_a_local_recipe_is_rejected_by_the_remote_path(cpu_recipe) -> None:
    from edgefit.harness.remote import submit_profile

    with pytest.raises(RemoteMeasurementError, match="not a hosted device"):
        submit_profile(cpu_recipe, __import__("pathlib").Path("."))


class TestFirstInferenceCost:
    """Hard rule #4: the cost of getting to a runnable state is part of the number.

    Measured on a Galaxy S24: an 8.5 s cold load in front of a 7.68 ms inference —
    1100x. And it is not a constant offset, so it cannot be waved away: on the same
    device the CPU-only path loaded 14x faster than the NPU path, which means the
    accelerator's headline win is partly bought with startup cost.
    """

    def test_load_times_are_metrics_not_prose(self) -> None:
        metrics = build_record(_recipe(), _profile(), None).metrics
        assert metrics is not None
        assert metrics.cold_load_ms == pytest.approx(40.0)
        assert metrics.warm_load_ms == pytest.approx(12.0)

    def test_first_inference_is_refused_rather_than_inferred(self) -> None:
        """The service reports load and steady state, but not the first run itself."""
        metrics = build_record(_recipe(), _profile(), None).metrics
        assert metrics is not None
        assert metrics.first_inference_ms is None
        assert "first run" in metrics.unavailable["first_inference_ms"]


class TestQuantizeAndCompile:
    """Hosted recipes that quantize and compile before profiling.

    This axis was removed once and is back, so the tests pin the reasons it is safe
    now: the compile-only flag never reaches profile options, quantization cannot be
    requested without somewhere to compile it to, and the row says what was actually
    measured.
    """

    def test_a_quantized_recipe_has_its_own_identity(self) -> None:
        """Same model, different artifact, therefore a different recipe.

        Without this a quantized run and an fp32 run would collide in the corpus and
        the second would look like a repeat of the first.
        """
        base = _recipe()
        quantized = Recipe(
            model=base.model,
            runtime=QaiHubRuntimeConfig(
                device_name="Samsung Galaxy S24 (Family)",
                target_runtime="tflite",
                quantize={"weights_dtype": "int8", "activations_dtype": "int8"},
            ),
        )
        assert quantized.recipe_id != base.recipe_id

    def test_calibration_sample_count_changes_identity(self) -> None:
        """Eight samples and eight hundred are not the same measurement."""
        def build(n: int) -> str:
            return Recipe(
                model=_recipe().model,
                runtime=QaiHubRuntimeConfig(
                    device_name="Samsung Galaxy S24 (Family)",
                    target_runtime="tflite",
                    quantize={"calibration_samples": n},
                ),
            ).recipe_id

        assert build(8) != build(64)

    def test_provenance_states_the_pipeline_and_its_thinness(self) -> None:
        """A quantized row must not read like a row profiling our own export."""
        from edgefit.harness.remote import _pipeline_detail

        runtime = QaiHubRuntimeConfig(
            device_name="Samsung Galaxy S24 (Family)",
            target_runtime="tflite",
            quantize={"weights_dtype": "int8", "calibration_samples": 8},
        )
        detail = _pipeline_detail(runtime)
        assert "int8" in detail
        assert "8 calibration samples" in detail
        assert "thin set" in detail, "the calibration set's weakness must travel with the row"
        assert "not the ONNX we uploaded" in detail

    def test_an_unquantized_recipe_adds_no_pipeline_prose(self) -> None:
        from edgefit.harness.remote import _pipeline_detail

        assert _pipeline_detail(_recipe().runtime) == ""

    def test_calibration_data_is_deterministic(self) -> None:
        """Quantization must be reproducible, so the perturbations are seeded."""
        import tempfile

        import numpy as np

        from edgefit.harness.remote import _calibration_data
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "model.onnx"
            np.savez(path.parent / "inputs.npz", pixel_values=np.zeros((1, 3, 4, 4), np.float32))
            first = _calibration_data(path, 4)
            second = _calibration_data(path, 4)
        assert len(first["pixel_values"]) == 4
        for a, b in zip(first["pixel_values"], second["pixel_values"], strict=True):
            assert np.array_equal(a, b), "calibration data differed between runs"
