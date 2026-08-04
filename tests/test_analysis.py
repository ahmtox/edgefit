"""Static analysis: FLOP estimation, EP attribution, graph fingerprinting.

Built on synthetic graphs so the arithmetic is checkable by hand and the tests
run anywhere without hardware or model downloads.
"""

from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from edgefit.backends.analysis import (
    KernelEvent,
    build_fallback_report,
    estimate_flops,
    fingerprint_onnx,
    parse_profile,
)
from edgefit.schema.common import AttentionVariant, NormType

BATCH, SEQ, HIDDEN = 1, 128, 384


def _matmul_graph() -> onnx.ModelProto:
    """x[1,128,384] @ W[384,384] -> y, then a free Reshape."""
    weight = numpy_helper.from_array(
        np.zeros((HIDDEN, HIDDEN), dtype=np.float32), name="W"
    )
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["h"], name="dense"),
        helper.make_node("Reshape", ["h", "shape"], ["y"], name="reshape"),
    ]
    graph = helper.make_graph(
        nodes,
        "matmul_graph",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [BATCH, SEQ, HIDDEN])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [BATCH * SEQ, HIDDEN])],
        initializer=[
            weight,
            numpy_helper.from_array(np.array([BATCH * SEQ, HIDDEN], dtype=np.int64), name="shape"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestFlops:
    def test_counts_matmul_exactly(self) -> None:
        table = estimate_flops(_matmul_graph())
        assert table.per_node["dense"] == 2 * BATCH * SEQ * HIDDEN * HIDDEN

    def test_shape_ops_cost_nothing(self) -> None:
        """Reshape moves memory, not arithmetic. Conflating the two hides the real story."""
        assert estimate_flops(_matmul_graph()).per_node["reshape"] == 0

    def test_reports_completeness(self) -> None:
        table = estimate_flops(_matmul_graph())
        assert table.is_complete
        assert table.unresolved == ()
        assert table.total == table.per_node["dense"]

    def test_subtotal_selects_named_nodes(self) -> None:
        table = estimate_flops(_matmul_graph())
        assert table.subtotal({"dense"}) == table.total
        assert table.subtotal({"reshape"}) == 0
        assert table.subtotal({"nonexistent"}) == 0

    def test_dynamic_shapes_are_reported_not_guessed(self) -> None:
        """An unresolved denominator must not silently produce a confident ratio."""
        graph = helper.make_graph(
            [helper.make_node("Softmax", ["x"], ["y"], name="softmax")],
            "dynamic",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "seq"])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", "seq"])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        table = estimate_flops(model)
        assert not table.is_complete
        assert "softmax" in table.unresolved


class TestFingerprint:
    def test_summarises_structure(self) -> None:
        fingerprint = fingerprint_onnx(_matmul_graph())
        assert fingerprint.n_nodes == 2
        assert fingerprint.op_histogram == {"MatMul": 1, "Reshape": 1}
        assert fingerprint.n_parameters == HIDDEN * HIDDEN + 2
        assert fingerprint.opset == {"ai.onnx": 17}

    def test_keeps_symbolic_dims_as_names(self) -> None:
        """Dynamic shape is a fact delegates care about enormously."""
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"], name="id")],
            "dyn",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 128])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 128])],
        )
        fingerprint = fingerprint_onnx(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        )
        assert fingerprint.input_shapes["x"] == ["batch", 128]

    @pytest.mark.parametrize(
        ("op_type", "expected"),
        [
            ("LayerNormalization", NormType.LAYERNORM),
            ("SimplifiedLayerNormalization", NormType.RMSNORM),
            ("BatchNormalization", NormType.BATCHNORM),
        ],
    )
    def test_detects_norm_type(self, op_type: str, expected: NormType) -> None:
        graph = helper.make_graph(
            [helper.make_node(op_type, ["x"], ["y"], name="norm")],
            "norm",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        assert fingerprint_onnx(model).norm_type is expected

    def test_reports_no_attention_when_there_is_none(self) -> None:
        assert fingerprint_onnx(_matmul_graph()).attention_variant is AttentionVariant.NONE


def _events(pairs: list[tuple[str, str, float]], runs: int = 1) -> list[KernelEvent]:
    return [
        KernelEvent(node_name=name, op_type=None, provider=provider, duration_us=duration)
        for _ in range(runs)
        for name, provider, duration in pairs
    ]


class TestFallbackReport:
    def test_node_and_flop_share_can_disagree_wildly(self) -> None:
        """The whole reason three proxies exist.

        One unclaimed MatMul outweighs a Reshape the accelerator did claim, so
        node share reads 50% while FLOP share reads ~100%. A team reading only
        the node number would conclude the delegate was working.
        """
        model = _matmul_graph()
        report = build_fallback_report(
            model,
            _events([("dense", "CPUExecutionProvider", 900.0),
                     ("CoreMLExecutionProvider_x_CoreML_x_0_0", "CoreMLExecutionProvider", 100.0)]),
            intended_provider="CoreMLExecutionProvider",
            flops=estimate_flops(model),
        )
        assert report.fallback_node_pct == pytest.approx(50.0)
        assert report.fallback_flops_pct == pytest.approx(100.0)
        assert report.fallback_time_pct == pytest.approx(90.0)
        assert report.unclaimed_op_types == {"MatMul": 1}

    def test_full_claim_reports_zero_fallback(self) -> None:
        model = _matmul_graph()
        report = build_fallback_report(
            model,
            _events([("CoreMLExecutionProvider_x_CoreML_x_0_0", "CoreMLExecutionProvider", 100.0)]),
            intended_provider="CoreMLExecutionProvider",
            flops=estimate_flops(model),
        )
        assert report.fallback_node_pct == 0.0
        assert report.fallback_flops_pct == 0.0
        assert report.nodes_on_intended == report.nodes_total

    def test_counts_fused_partitions(self) -> None:
        """Fragmentation is its own performance story: N partitions, N round trips."""
        model = _matmul_graph()
        report = build_fallback_report(
            model,
            _events(
                [
                    ("CoreMLExecutionProvider_a_CoreML_a_0_0", "CoreMLExecutionProvider", 50.0),
                    ("CoreMLExecutionProvider_a_CoreML_a_1_1", "CoreMLExecutionProvider", 50.0),
                    ("dense", "CPUExecutionProvider", 900.0),
                ]
            ),
            intended_provider="CoreMLExecutionProvider",
        )
        assert report.nodes_per_provider["CoreMLExecutionProvider (fused partitions)"] == 2

    def test_deduplicates_across_repeated_runs(self) -> None:
        """Profile events repeat once per inference; nodes must not be double counted."""
        model = _matmul_graph()
        pairs = [("dense", "CPUExecutionProvider", 900.0),
                 ("CoreMLExecutionProvider_x_CoreML_x_0_0", "CoreMLExecutionProvider", 100.0)]
        single = build_fallback_report(
            model, _events(pairs), intended_provider="CoreMLExecutionProvider"
        )
        repeated = build_fallback_report(
            model, _events(pairs, runs=5), intended_provider="CoreMLExecutionProvider", runs=5
        )
        assert repeated.fallback_node_pct == single.fallback_node_pct
        assert repeated.nodes_total == single.nodes_total
        assert repeated.time_total_us == pytest.approx(single.time_total_us)

    def test_omits_flop_share_when_shapes_are_unresolved(self) -> None:
        """Better a missing column than a confident wrong one (hard rule #1)."""
        graph = helper.make_graph(
            [helper.make_node("Softmax", ["x"], ["y"], name="softmax")],
            "dynamic",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "seq"])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", "seq"])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        report = build_fallback_report(
            model,
            _events([("softmax", "CPUExecutionProvider", 10.0)]),
            intended_provider="CoreMLExecutionProvider",
            flops=estimate_flops(model),
        )
        assert report.fallback_flops_pct is None
        assert report.fallback_node_pct == pytest.approx(100.0)


class TestParseProfile:
    def test_extracts_kernel_events(self, tmp_path) -> None:
        path = tmp_path / "profile.json"
        path.write_text(
            json.dumps(
                [
                    {"cat": "Session", "name": "model_loading", "dur": 5},
                    {
                        "cat": "Node",
                        "name": "dense_kernel_time",
                        "dur": 900,
                        "args": {"provider": "CPUExecutionProvider", "op_name": "MatMul"},
                    },
                    {"cat": "Node", "name": "dense_fence_before", "dur": 1, "args": {}},
                ]
            )
        )
        events = parse_profile(path)
        assert len(events) == 1
        assert events[0].node_name == "dense"
        assert events[0].op_type == "MatMul"
        assert events[0].duration_us == 900.0


class TestAttentionVariant:
    """The fingerprint is the key the cost model indexes on (PROJECT.md §5.2).

    A confidently wrong label there is worse than a blank, because it makes knowledge
    transfer between models silently incorrect. This class exists because the detector
    used to report MHA for Llama-3.2-1B, which is GQA with 32 query heads over 8 KV
    heads.
    """

    def _decomposed_attention(self) -> onnx.ModelProto:
        """Softmax over a MatMul chain — attention with no explicit op to read."""
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["q", "k"], ["scores"], name="qk"),
                helper.make_node("Softmax", ["scores"], ["probs"], name="softmax"),
                helper.make_node("MatMul", ["probs", "v"], ["out"], name="av"),
            ],
            "attn",
            inputs=[
                helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 4, 8, 8])
                for n in ("q", "k", "v")
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 8, 8])],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def test_undecidable_reports_unknown_not_mha(self) -> None:
        """The regression. Softmax+MatMul proves attention exists, not how KV is grouped."""
        variant = fingerprint_onnx(self._decomposed_attention()).attention_variant
        assert variant is AttentionVariant.UNKNOWN

    def test_head_counts_make_it_exact(self) -> None:
        model = self._decomposed_attention()
        assert (
            fingerprint_onnx(model, n_heads=32, n_kv_heads=8).attention_variant
            is AttentionVariant.GQA
        )
        assert (
            fingerprint_onnx(model, n_heads=32, n_kv_heads=32).attention_variant
            is AttentionVariant.MHA
        )
        assert (
            fingerprint_onnx(model, n_heads=32, n_kv_heads=1).attention_variant
            is AttentionVariant.MQA
        )

    def test_head_counts_are_recorded_for_the_cost_model(self) -> None:
        fingerprint = fingerprint_onnx(
            self._decomposed_attention(), n_heads=32, n_kv_heads=8, n_layers=16
        )
        assert (fingerprint.n_heads, fingerprint.n_kv_heads, fingerprint.n_layers) == (32, 8, 16)

    def test_no_softmax_still_means_no_attention(self) -> None:
        assert fingerprint_onnx(_matmul_graph()).attention_variant is AttentionVariant.NONE

    def test_head_counts_change_the_fingerprint_id(self) -> None:
        """Two architectures must not collide on one key."""
        model = self._decomposed_attention()
        gqa = fingerprint_onnx(model, n_heads=32, n_kv_heads=8)
        mha = fingerprint_onnx(model, n_heads=32, n_kv_heads=32)
        assert gqa.fingerprint_id != mha.fingerprint_id
