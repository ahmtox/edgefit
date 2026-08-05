"""Duplicate-initializer detection.

Found by hand on Llama-3.2-1B, where `torch.onnx.export` un-tied a tied embedding and
shipped 128256x2048 twice — ~1.05 GB of a 5.6 GB fp32 artifact. These tests pin the
general detector, including the cases where it must stay quiet.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from edgefit.backends.analysis.weights import (
    MIN_REPORTED_BYTES,
    find_duplicate_initializers,
)


def _model(tensors: list[TensorProto], path, *, external: bool = False):
    """A graph that merely holds initializers — the detector never runs it."""
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["x"], ["y"])],
        name="held",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializer=tensors,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    if external:
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
    else:
        onnx.save_model(model, str(path))
    return path


def _big(name: str, seed: int, rows: int = 512) -> TensorProto:
    """A tensor large enough to clear the reporting floor (512x512 fp32 = 1 MiB)."""
    rng = np.random.default_rng(seed)
    return numpy_helper.from_array(rng.standard_normal((rows, 512), dtype=np.float32), name)


def _copy_as(tensor: TensorProto, name: str) -> TensorProto:
    clone = TensorProto()
    clone.CopyFrom(tensor)
    clone.name = name
    return clone


class TestDetection:
    def test_finds_a_tensor_shipped_twice_under_two_names(self, tmp_path) -> None:
        """The Llama case: same bytes, different names, both in the file."""
        weight = _big("embed_tokens.weight", seed=0)
        path = _model(
            [weight, _copy_as(weight, "onnx::MatMul_1"), _big("other", seed=1)],
            tmp_path / "model.onnx",
        )

        report = find_duplicate_initializers(path)

        assert len(report.groups) == 1
        group = report.groups[0]
        assert group.names == ("embed_tokens.weight", "onnx::MatMul_1")
        assert group.copies == 2
        assert group.dims == (512, 512)
        assert group.dtype == "float"
        assert group.bytes_each == 512 * 512 * 4
        assert group.wasted_bytes == 512 * 512 * 4

    def test_the_wasted_fraction_is_of_weight_bytes(self, tmp_path) -> None:
        """Three equal tensors, one duplicated: a third of the weights is waste."""
        weight = _big("shared", seed=0)
        path = _model(
            [weight, _copy_as(weight, "shared_again"), _big("other", seed=1)],
            tmp_path / "model.onnx",
        )

        report = find_duplicate_initializers(path)

        assert report.initializer_bytes == 3 * 512 * 512 * 4
        assert report.wasted_fraction == pytest.approx(1 / 3)

    def test_three_copies_waste_two_of_them(self, tmp_path) -> None:
        weight = _big("w", seed=0)
        path = _model(
            [weight, _copy_as(weight, "w2"), _copy_as(weight, "w3")],
            tmp_path / "model.onnx",
        )

        group = find_duplicate_initializers(path).groups[0]
        assert (group.copies, group.wasted_bytes) == (3, 2 * 512 * 512 * 4)

    def test_reads_tensors_held_in_external_data(self, tmp_path) -> None:
        """Any artifact big enough to have this problem stores weights externally.

        The duplicate is two slices of one weights file at different offsets, which is
        exactly the on-disk shape of the Llama case — so hashing must follow the
        external reference rather than the (now empty) raw_data field.
        """
        weight = _big("embed_tokens.weight", seed=0)
        path = _model(
            [weight, _copy_as(weight, "onnx::MatMul_1"), _big("other", seed=1)],
            tmp_path / "model.onnx",
            external=True,
        )
        assert (tmp_path / "weights.bin").exists()

        report = find_duplicate_initializers(path)

        assert [group.names for group in report.groups] == [
            ("embed_tokens.weight", "onnx::MatMul_1")
        ]
        assert report.initializer_bytes == 3 * 512 * 512 * 4


class TestTranspose:
    """The case that motivated the detector, which hashing alone cannot see.

    Llama-3.2-1B ships `(128256, 2048)` and `(2048, 128256)` — the embedding and its
    transpose, materialised because `torch.onnx.export` un-tied a tied weight. Written
    as a pure content hash, this detector returned *clean* on that artifact. Same
    values, different bytes.
    """

    def test_finds_a_weight_shipped_alongside_its_transpose(self, tmp_path) -> None:
        rng = np.random.default_rng(7)
        values = rng.standard_normal((1024, 256), dtype=np.float32)
        path = _model(
            [
                numpy_helper.from_array(values, "embed_tokens.weight"),
                numpy_helper.from_array(np.ascontiguousarray(values.T), "onnx::MatMul_1"),
            ],
            tmp_path / "model.onnx",
        )

        report = find_duplicate_initializers(path)

        assert len(report.groups) == 1
        group = report.groups[0]
        assert group.relation == "transpose"
        assert group.names == ("embed_tokens.weight", "onnx::MatMul_1")
        assert group.bytes_each == 1024 * 256 * 4
        assert report.wasted_fraction == pytest.approx(0.5)

    def test_finds_it_through_external_data(self, tmp_path) -> None:
        """The real artifact is 5.6 GB, so the pair is two slices of one weights file."""
        rng = np.random.default_rng(7)
        values = rng.standard_normal((1024, 256), dtype=np.float32)
        path = _model(
            [
                numpy_helper.from_array(values, "embed_tokens.weight"),
                numpy_helper.from_array(np.ascontiguousarray(values.T), "lm_head"),
            ],
            tmp_path / "model.onnx",
            external=True,
        )

        group = find_duplicate_initializers(path).groups[0]
        assert group.relation == "transpose"
        assert group.dims == (1024, 256)

    def test_complementary_shapes_alone_are_not_enough(self, tmp_path) -> None:
        """Two unrelated weights of mirrored shape must not be reported."""
        rng = np.random.default_rng(0)
        path = _model(
            [
                numpy_helper.from_array(rng.standard_normal((1024, 256), dtype=np.float32), "a"),
                numpy_helper.from_array(rng.standard_normal((256, 1024), dtype=np.float32), "b"),
            ],
            tmp_path / "model.onnx",
        )

        assert find_duplicate_initializers(path).groups == ()

    def test_a_near_transpose_is_not_a_transpose(self, tmp_path) -> None:
        """One altered element disqualifies the pair — the walk proves it, not the probe."""
        rng = np.random.default_rng(7)
        values = rng.standard_normal((1024, 256), dtype=np.float32)
        transposed = np.ascontiguousarray(values.T)
        transposed[100, 900] += 1.0
        path = _model(
            [
                numpy_helper.from_array(values, "a"),
                numpy_helper.from_array(transposed, "b"),
            ],
            tmp_path / "model.onnx",
        )

        assert find_duplicate_initializers(path).groups == ()

    def test_a_square_weight_is_not_reported_against_itself(self, tmp_path) -> None:
        """A square tensor's shape is its own mirror; that must not self-match."""
        rng = np.random.default_rng(3)
        path = _model(
            [numpy_helper.from_array(rng.standard_normal((512, 512), dtype=np.float32), "w")],
            tmp_path / "model.onnx",
        )

        assert find_duplicate_initializers(path).groups == ()


class TestQuiet:
    """A detector that cries wolf gets ignored, so the negative cases are the test."""

    def test_a_model_without_duplicates_reports_nothing(self, tmp_path) -> None:
        path = _model([_big("a", seed=1), _big("b", seed=2)], tmp_path / "model.onnx")

        report = find_duplicate_initializers(path)

        assert not report.has_findings
        assert report.wasted_bytes == 0

    def test_same_shape_different_contents_is_not_a_duplicate(self, tmp_path) -> None:
        """Bucketing by shape is only a prefilter; the answer comes from the bytes."""
        path = _model([_big("a", seed=1), _big("b", seed=2)], tmp_path / "model.onnx")
        assert find_duplicate_initializers(path).groups == ()

    def test_same_contents_different_shape_is_not_a_duplicate(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        values = rng.standard_normal(512 * 512, dtype=np.float32)
        flat = numpy_helper.from_array(values, "flat")
        square = numpy_helper.from_array(values.reshape(512, 512), "square")
        path = _model([flat, square], tmp_path / "model.onnx")

        # Identical bytes, but they are not interchangeable weights: dropping one
        # would change the graph, so this is not shippable waste.
        assert find_duplicate_initializers(path).groups == ()

    def test_small_duplicates_are_below_the_floor(self, tmp_path) -> None:
        """A shared scalar or a repeated bias costs nothing and would bury the finding."""
        tiny = numpy_helper.from_array(np.zeros((8, 8), dtype=np.float32), "bias")
        path = _model([tiny, _copy_as(tiny, "bias_again")], tmp_path / "model.onnx")

        assert find_duplicate_initializers(path).groups == ()
        # …and are found when the caller asks for everything.
        assert find_duplicate_initializers(path, min_bytes=0).groups != ()

    def test_an_empty_model_has_no_fraction_rather_than_zero(self, tmp_path) -> None:
        """No weights means the question is unanswerable, not answered with 0%."""
        path = _model([], tmp_path / "model.onnx")

        report = find_duplicate_initializers(path)
        assert report.initializer_bytes == 0
        assert report.wasted_fraction is None


def test_the_reporting_floor_is_a_megabyte() -> None:
    assert MIN_REPORTED_BYTES == 1024 * 1024
