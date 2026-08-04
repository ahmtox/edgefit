"""Generative harness — decoder export, KV cache plumbing, and the token check.

The load-bearing test here is token agreement. It caught a real correctness bug that
every float-tolerance check would have passed: without an explicit ``position_ids``
input, the tracer bakes prefill's rotary positions into the graph, so every decode
step computes RoPE at the wrong position. The model still runs and still emits
fluent text — just different text. Agreement read 25%.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from edgefit.backends.artifacts import recipe_applicability
from edgefit.backends.export_decoder import (
    DecoderShape,
    UnsupportedDecoderLowering,
    decoder_shape,
    export_decoder,
)
from edgefit.backends.export_onnx import HARNESS_SIDECARS, artifact_size_bytes
from edgefit.harness.runner import _token_agreement
from edgefit.harness.timing import MeasurementPolicy
from edgefit.models.registry import resolve
from edgefit.schema import (
    Dtype,
    MeasurementRecord,
    Metrics,
    Outcome,
    QuantizationConfig,
    RunStats,
)

TTFT = [120.5, 121.0, 120.1, 122.0, 120.8]
DECODE = [12.2, 12.3, 12.25, 12.1, 12.28]


class TestDecoderShape:
    def test_names_are_paired_per_layer(self) -> None:
        shape = DecoderShape(layers=3, kv_heads=8, head_dim=64, prompt_length=12)
        assert shape.past_names == [
            "past.0.key", "past.0.value", "past.1.key", "past.1.value",
            "past.2.key", "past.2.value",
        ]
        assert len(shape.present_names) == len(shape.past_names)

    def test_empty_past_has_zero_sequence_length(self) -> None:
        """Prefill is a decode step whose past is empty — that is what makes one graph
        serve both phases."""
        shape = DecoderShape(layers=2, kv_heads=4, head_dim=16, prompt_length=5)
        past = shape.empty_past()
        assert set(past) == set(shape.past_names)
        assert past["past.0.key"].shape == (1, 4, 0, 16)
        assert past["past.0.key"].dtype == np.float32

    def test_round_trips_through_meta(self) -> None:
        shape = DecoderShape(layers=16, kv_heads=8, head_dim=64, prompt_length=13)
        meta = {"layers": 16, "kv_heads": 8, "head_dim": 64, "prompt_length": 13}
        assert decoder_shape(meta) == shape


class TestStaticShapesRefused:
    def test_export_refuses_static_shapes(self) -> None:
        """Refused, not silently ignored: the cache grows one token per step."""
        with pytest.raises(UnsupportedDecoderLowering, match="dynamic sequence axis"):
            export_decoder(resolve("hf:meta-llama/Llama-3.2-1B-Instruct"), static_shapes=True)

    def test_reason_names_the_real_world_alternative(self) -> None:
        """The honest framing: NPU stacks use a fixed cache buffer, and we do not."""
        try:
            export_decoder(resolve("hf:meta-llama/Llama-3.2-1B-Instruct"), static_shapes=True)
        except UnsupportedDecoderLowering as exc:
            assert "position index" in str(exc)


class TestApplicability:
    def test_static_shape_recipe_is_illegal_for_a_decoder(self, cpu_recipe) -> None:
        """§5.4 generates only legal recipes, so this is skipped, not failed."""
        spec = resolve("hf:meta-llama/Llama-3.2-1B-Instruct")
        reason = recipe_applicability(spec, cpu_recipe)
        assert reason is not None and "static shapes" in reason

    def test_quantized_recipe_is_illegal_for_a_decoder(self, cpu_recipe) -> None:
        spec = resolve("hf:meta-llama/Llama-3.2-1B-Instruct")
        quantized = cpu_recipe.derive(
            lowering={"static_shapes": False},
            quantization=QuantizationConfig(
                weight_dtype=Dtype.INT8, activation_quant="dynamic", activation_dtype=Dtype.INT8
            ).model_dump(mode="json"),
        )
        reason = recipe_applicability(spec, quantized)
        assert reason is not None and "precision policy" in reason

    def test_a_dynamic_unquantized_recipe_is_legal(self, cpu_recipe) -> None:
        spec = resolve("hf:meta-llama/Llama-3.2-1B-Instruct")
        legal = cpu_recipe.derive(lowering={"static_shapes": False})
        assert recipe_applicability(spec, legal) is None

    def test_encoder_recipes_stay_legal(self, cpu_recipe) -> None:
        assert recipe_applicability(resolve("hf:facebook/bart-base"), cpu_recipe) is None


class TestTokenAgreement:
    def _reference(self, tmp_path, tokens: list[int]):
        path = tmp_path / "reference.npz"
        np.savez(path, tokens=np.asarray(tokens, dtype=np.int64))
        return path

    def test_exact_match_is_one(self, tmp_path) -> None:
        ref = self._reference(tmp_path, [1, 2, 3, 4])
        assert _token_agreement([1, 2, 3, 4], ref) == pytest.approx(1.0)

    def test_partial_divergence_is_reported_as_a_fraction(self, tmp_path) -> None:
        """The RoPE bug read 25%, which is what made it visible."""
        ref = self._reference(tmp_path, [1, 2, 3, 4])
        assert _token_agreement([1, 9, 9, 9], ref) == pytest.approx(0.25)

    def test_compares_only_the_overlapping_prefix(self, tmp_path) -> None:
        """The harness decodes more tokens than the reference records."""
        ref = self._reference(tmp_path, [1, 2])
        assert _token_agreement([1, 2, 7, 8, 9], ref) == pytest.approx(1.0)

    def test_missing_reference_is_a_null_not_a_failure(self, tmp_path) -> None:
        assert _token_agreement([1, 2], tmp_path / "absent.npz") is None

    def test_a_tensor_reference_is_not_mistaken_for_tokens(self, tmp_path) -> None:
        """An encoder artifact stores logits under a different key."""
        path = tmp_path / "reference.npz"
        np.savez(path, last_hidden_state=np.zeros((1, 4)))
        assert _token_agreement([1, 2], path) is None


class TestGenerativeRecord:
    def _record(self, device, host_state, **overrides) -> MeasurementRecord:
        base = {
            "harness_version": "0.2.0",
            "recipe_id": "r1",
            "model_ref": "hf:meta-llama/Llama-3.2-1B-Instruct",
            "device": device,
            "host_state": host_state,
            "outcome": Outcome.SUCCESS,
            "run_count": len(TTFT),
            "warmup_count": 1,
            "metrics": Metrics(
                ttft_ms=RunStats.from_samples(TTFT),
                decode_tok_s=RunStats.from_samples(DECODE),
                token_agreement=1.0,
                unavailable={
                    "latency_ms": "generative recipes report ttft and decode separately"
                },
            ),
        }
        return MeasurementRecord(**(base | overrides))

    def test_ttft_satisfies_the_variance_requirement(self, device, host_state) -> None:
        """Hard rule #2 is met by TTFT; there is no single 'latency' for a decoder."""
        record = self._record(device, host_state)
        assert record.metrics.primary_stats is record.metrics.ttft_ms
        assert record.metrics.latency_ms is None

    def test_both_phases_are_recorded(self, device, host_state) -> None:
        metrics = self._record(device, host_state).metrics
        assert metrics.ttft_ms.n == len(TTFT)
        assert metrics.decode_tok_s.p50 == pytest.approx(12.25)

    def test_absent_latency_must_be_explained(self, device, host_state) -> None:
        with pytest.raises(ValueError, match="populated but listed as unavailable"):
            Metrics(
                latency_ms=RunStats.from_samples(TTFT),
                unavailable={"latency_ms": "should not be claimed absent"},
            )


class TestArtifactSize:
    def test_excludes_harness_sidecars(self, tmp_path) -> None:
        (tmp_path / "model.onnx").write_bytes(b"x" * 100)
        (tmp_path / "model.onnx.data").write_bytes(b"y" * 900)
        for name in HARNESS_SIDECARS:
            (tmp_path / name).write_bytes(b"z" * 50)
        assert artifact_size_bytes(tmp_path) == 1000

    def test_counts_external_weight_files(self, tmp_path) -> None:
        """A 5.6 GB Llama artifact measured 0.7 MiB when the glob was `*.onnx*`.

        Weights over the 2 GB protobuf limit spill into files named after tensors, so
        the glob captured the graph and none of its weights.
        """
        (tmp_path / "model.onnx").write_bytes(b"x" * 10)
        (tmp_path / "inner.model.layers.0.weight").write_bytes(b"y" * 5000)
        assert artifact_size_bytes(tmp_path) == 5010


def test_policy_carries_a_decode_length() -> None:
    assert MeasurementPolicy().decode_tokens >= 8
    assert MeasurementPolicy(decode_tokens=64).decode_tokens == 64


def test_exported_llama_graph_takes_explicit_positions() -> None:
    """Regression guard for the RoPE bug, checked against the real artifact if present."""
    from pathlib import Path

    roots = sorted(Path("artifacts/onnx").glob("meta-llama*/meta.json"))
    if not roots:
        pytest.skip("Llama artifact not exported on this machine")
    meta = json.loads(roots[0].read_text())
    assert meta["layers"] > 0 and meta["kv_heads"] > 0
    import onnx

    model = onnx.load(str(roots[0].parent / "model.onnx"), load_external_data=False)
    names = {i.name for i in model.graph.input}
    assert "position_ids" in names, "rotary positions must be an explicit graph input"
