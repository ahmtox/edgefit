"""Extract a GraphFingerprint from an ONNX model (PROJECT.md §5.2).

Detection is deliberately conservative. An architecture trait we cannot
establish from the graph is reported as ``UNKNOWN`` rather than guessed: the
fingerprint is the key the cost model will index on, and a confidently wrong
label there poisons transfer between models in a way that is very hard to debug
later.
"""

from __future__ import annotations

import math
from collections import Counter

from onnx import ModelProto, TensorProto

from edgefit.schema.common import AttentionVariant, NormType
from edgefit.schema.fingerprint import GraphFingerprint

_ACTIVATIONS = frozenset(
    {
        "Relu", "LeakyRelu", "PRelu", "Elu", "Selu", "Gelu", "Erf", "Sigmoid",
        "Tanh", "Softplus", "HardSigmoid", "HardSwish", "Swish", "Silu", "Mish",
    }
)

_NORM_BY_OP = {
    "SkipLayerNormalization": NormType.LAYERNORM,
    "LayerNormalization": NormType.LAYERNORM,
    "SimplifiedLayerNormalization": NormType.RMSNORM,
    "RMSNormalization": NormType.RMSNORM,
    "BatchNormalization": NormType.BATCHNORM,
    "GroupNormalization": NormType.GROUPNORM,
}

_ATTENTION_BY_OP = {
    "GroupQueryAttention": AttentionVariant.GQA,
    "MultiHeadAttention": AttentionVariant.MHA,
    "Attention": AttentionVariant.MHA,
}


def _tensor_dtype_name(data_type: int) -> str:
    return TensorProto.DataType.Name(data_type).lower()


def _shape_of(value) -> list[int | str]:
    if not value.type.HasField("tensor_type"):
        return []
    dims: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            # Symbolic dimension. Kept as a name because dynamic shape is a fact
            # about the model that delegates care about enormously.
            dims.append(dim.dim_param or "dynamic")
    return dims


def _detect_attention(ops: Counter[str]) -> AttentionVariant:
    for op_type, variant in _ATTENTION_BY_OP.items():
        if ops.get(op_type):
            return variant
    if not ops.get("Softmax"):
        return AttentionVariant.NONE
    # Softmax over a MatMul chain is the decomposed attention pattern. We can see
    # that attention exists but not how the KV heads are grouped, so MHA is the
    # honest floor rather than a claim about grouping.
    return AttentionVariant.MHA if ops.get("MatMul") else AttentionVariant.UNKNOWN


def _detect_norm(ops: Counter[str]) -> NormType:
    for op_type, norm in _NORM_BY_OP.items():
        if ops.get(op_type):
            return norm
    return NormType.UNKNOWN


def fingerprint_onnx(model: ModelProto) -> GraphFingerprint:
    """Summarise a graph's structure. Contains no weights and no customer data."""
    graph = model.graph
    ops: Counter[str] = Counter(node.op_type for node in graph.node)

    dtypes: Counter[str] = Counter()
    n_parameters = 0
    for initializer in graph.initializer:
        dtypes[_tensor_dtype_name(initializer.data_type)] += 1
        n_parameters += math.prod(initializer.dims) if initializer.dims else 1

    return GraphFingerprint(
        n_nodes=len(graph.node),
        n_parameters=n_parameters,
        n_initializers=len(graph.initializer),
        op_histogram=dict(ops.most_common()),
        dtype_histogram=dict(dtypes),
        input_shapes={value.name: _shape_of(value) for value in graph.input},
        output_shapes={value.name: _shape_of(value) for value in graph.output},
        opset={(entry.domain or "ai.onnx"): entry.version for entry in model.opset_import},
        attention_variant=_detect_attention(ops),
        norm_type=_detect_norm(ops),
        activation_fns=tuple(sorted(op for op in ops if op in _ACTIVATIONS)),
    )
