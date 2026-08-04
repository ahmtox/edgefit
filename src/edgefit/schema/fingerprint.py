"""Graph fingerprint — a structural summary of a model (PROJECT.md §5.2).

> "The graph fingerprint is quietly the most important object in the system. It
> is the key the future cost model indexes on, and how knowledge transfers from
> one customer's model to the next."

So it is deliberately *structural*, never identifying: an op histogram and a
shape distribution describe the workload without describing the weights. That is
what lets a Tier-3 self-hosted runner (§8) return something useful about a model
we are contractually forbidden from ever seeing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from edgefit.schema.common import AttentionVariant, NormType, content_hash


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class GraphFingerprint(_Frozen):
    """Structural summary. Contains no weights and no customer data."""

    # Bulk structure
    n_nodes: int = Field(ge=0)
    n_parameters: int = Field(ge=0)
    n_initializers: int = Field(ge=0)
    op_histogram: dict[str, int] = Field(default_factory=dict)
    dtype_histogram: dict[str, int] = Field(default_factory=dict)

    # Interface
    input_shapes: dict[str, list[int | str]] = Field(
        default_factory=dict, description="Symbolic dims kept as strings — dynamic shape is a fact"
    )
    output_shapes: dict[str, list[int | str]] = Field(default_factory=dict)
    opset: dict[str, int] = Field(default_factory=dict, description="domain -> version")

    # Architecture traits the cost model cares about
    attention_variant: AttentionVariant = AttentionVariant.UNKNOWN
    norm_type: NormType = NormType.UNKNOWN
    activation_fns: tuple[str, ...] = ()
    n_layers: int | None = None
    hidden_size: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    kv_cache_layout: str | None = None

    @property
    def fingerprint_id(self) -> str:
        return content_hash(self.model_dump(mode="json"))

    @property
    def total_ops(self) -> int:
        return sum(self.op_histogram.values())
