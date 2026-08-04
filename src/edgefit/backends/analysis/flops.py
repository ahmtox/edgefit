"""Static per-node FLOP estimation over an ONNX graph.

Why this exists: node-count fallback is a badly misleading proxy. A graph can
hand 54% of its *nodes* to an accelerator and still run 99% of its *arithmetic*
on CPU, because the nodes that matter are a handful of large MatMuls. PROJECT.md
§4 specifies the atlas report fallback as "% of FLOPs on intended accelerator",
and this module is what makes that column computable.

**This is an estimate from graph structure, not a measurement**, and is labelled
as such everywhere it surfaces. Hard rule #1 governs measured values; a static
analysis is allowed to be approximate so long as it never pretends otherwise.

Accuracy policy:

* MatMul / Gemm / Conv / ConvTranspose — counted exactly (2*MACs). These dominate
  every model we care about.
* Elementwise and normalisation ops — counted as a small multiple of output
  elements. Approximate, and small enough not to move the ratio.
* Shape-manipulation ops (Reshape, Transpose, …) — zero arithmetic. They cost
  memory traffic, not FLOPs, and conflating the two hides the real story.

Any node whose shapes cannot be resolved is reported in ``unresolved`` rather
than guessed, and the caller decides whether the total is trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import onnx
from onnx import ModelProto, shape_inference

# Bump when the estimator changes: stored measurements record which version
# produced their FLOP attribution so old and new rows stay comparable.
FLOPS_ESTIMATOR_VERSION = 1

# Ops that move or reinterpret data without arithmetic.
_ZERO_FLOP_OPS = frozenset(
    {
        "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Identity",
        "Constant", "ConstantOfShape", "Shape", "Gather", "GatherElements",
        "Concat", "Split", "Slice", "Expand", "Cast", "Pad", "Tile", "Range",
        "DequantizeLinear", "QuantizeLinear",
    }
)

# Rough per-output-element arithmetic cost. Order-of-magnitude only.
_ELEMENTWISE_COST = {
    "Add": 1, "Sub": 1, "Mul": 1, "Div": 1, "Pow": 2, "Sqrt": 1, "Abs": 1,
    "Neg": 1, "Min": 1, "Max": 1, "Where": 1, "And": 1, "Or": 1, "Not": 1,
    "Equal": 1, "Greater": 1, "Less": 1, "IsNaN": 1, "Relu": 1, "Clip": 2,
    "Sigmoid": 4, "Tanh": 4, "Exp": 4, "Log": 4, "Erf": 8, "Gelu": 10,
    "Softmax": 5, "LogSoftmax": 6, "ReduceMean": 1, "ReduceSum": 1,
    "LayerNormalization": 8, "SkipLayerNormalization": 9,
    "BatchNormalization": 4, "InstanceNormalization": 8, "RMSNormalization": 6,
}


@dataclass(frozen=True)
class FlopsTable:
    """Per-node FLOP estimates for one graph."""

    per_node: dict[str, int]
    total: int
    unresolved: tuple[str, ...] = field(default=())
    estimator_version: int = FLOPS_ESTIMATOR_VERSION

    @property
    def is_complete(self) -> bool:
        """True when every node's shapes resolved.

        When False the caller must not publish a FLOP percentage — an incomplete
        denominator produces a confident, wrong ratio.
        """
        return not self.unresolved

    def subtotal(self, node_names: set[str]) -> int:
        return sum(self.per_node.get(name, 0) for name in node_names)


def _shapes_by_value(model: ModelProto) -> dict[str, list[int | None]]:
    """Resolve every tensor's shape. Symbolic dims stay ``None``."""
    shapes: dict[str, list[int | None]] = {}

    def record(value: onnx.ValueInfoProto) -> None:
        if not value.type.HasField("tensor_type"):
            return
        dims: list[int | None] = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") and dim.dim_value > 0 else None)
        shapes[value.name] = dims

    graph = model.graph
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        record(value)
    for initializer in graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def _numel(shape: list[int | None] | None) -> int | None:
    if shape is None or any(dim is None for dim in shape):
        return None
    return math.prod(shape)  # type: ignore[arg-type]


def _matmul_flops(a: list[int | None], b: list[int | None]) -> int | None:
    """2*M*N*K, broadcasting leading batch dims."""
    if len(a) < 1 or len(b) < 1 or any(d is None for d in a) or any(d is None for d in b):
        return None
    lhs = [1] + a if len(a) == 1 else list(a)
    rhs = list(b) + [1] if len(b) == 1 else list(b)
    m, k = lhs[-2], lhs[-1]
    k2, n = rhs[-2], rhs[-1]
    if k != k2:
        return None
    # Leading dims broadcast, so the batch count is the larger of the two.
    batch = max(math.prod(lhs[:-2]), math.prod(rhs[:-2]), 1)  # type: ignore[arg-type]
    return 2 * int(batch) * int(m) * int(n) * int(k)


def _conv_flops(output: list[int | None], weight: list[int | None]) -> int | None:
    """2 * output_elements * (in_channels/groups * kernel_size).

    Grouping is already folded into the weight's second dimension, so it needs no
    separate divisor here.
    """
    out_elems = _numel(output)
    if out_elems is None or any(d is None for d in weight) or len(weight) < 2:
        return None
    macs_per_output = math.prod(weight[1:])  # type: ignore[arg-type]
    return 2 * out_elems * int(macs_per_output)


def _infer_shapes(model: ModelProto) -> ModelProto:
    """Resolve tensor shapes, preferring ORT's symbolic inferencer.

    ONNX's built-in pass gives up inside attention mask logic (Where/IsNaN chains),
    leaving the attention-score MatMuls unresolved — which is to say, it fails on
    exactly the nodes that dominate a transformer's arithmetic. ORT's symbolic
    inferencer resolves them, and it ships with a dependency we already require.
    """
    try:
        from onnxruntime.tools.symbolic_shape_infer import (  # noqa: PLC0415
            SymbolicShapeInference,
        )

        return SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=True)
    except Exception:  # noqa: BLE001 - fall through to the weaker but more forgiving pass
        pass

    try:
        return shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception:  # noqa: BLE001 - a malformed graph is the caller's problem, not ours
        return model


def estimate_flops(model: ModelProto) -> FlopsTable:
    """Estimate FLOPs per node. Nodes with unresolvable shapes are reported, not guessed."""
    inferred = _infer_shapes(model)

    shapes = _shapes_by_value(inferred)
    per_node: dict[str, int] = {}
    unresolved: list[str] = []

    for node in inferred.graph.node:
        name = node.name or f"{node.op_type}_{len(per_node)}"
        op = node.op_type

        if op in _ZERO_FLOP_OPS:
            per_node[name] = 0
            continue

        flops: int | None
        if op == "MatMul" or op in ("Gemm", "FusedMatMul"):
            flops = _matmul_flops(shapes.get(node.input[0], []), shapes.get(node.input[1], []))
        elif op in ("Conv", "ConvTranspose"):
            flops = _conv_flops(shapes.get(node.output[0], []), shapes.get(node.input[1], []))
        else:
            elems = _numel(shapes.get(node.output[0])) if node.output else None
            flops = elems * _ELEMENTWISE_COST.get(op, 1) if elems is not None else None

        if flops is None:
            unresolved.append(name)
            per_node[name] = 0
        else:
            per_node[name] = flops

    return FlopsTable(
        per_node=per_node,
        total=sum(per_node.values()),
        unresolved=tuple(unresolved),
    )
