"""Hosted measurement on Qualcomm AI Hub.

Network calls are stubbed; what is tested is the part that decides what enters the
corpus — warmup handling, provenance honesty, and per-node placement parsing.
"""

from __future__ import annotations

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

    def test_no_compile_only_flag_can_be_expressed(self) -> None:
        """`target_runtime` is gone from the schema, not merely unsent."""
        with pytest.raises(ValidationError):
            QaiHubRuntimeConfig(device_name="Samsung Galaxy S24 (Family)", target_runtime="tflite")


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
