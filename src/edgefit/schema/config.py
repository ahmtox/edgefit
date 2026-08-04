"""Configuration record — one point in the search space (PROJECT.md §6.1).

Design notes that matter later:

* **The target device is deliberately absent.** One config is measured on many
  devices; that product is exactly the atlas matrix. Device identity lives on the
  measurement record.
* **Frozen and content-addressed.** ``config_id`` is a hash of the canonical form,
  so the same config always lands on the same id and a duplicate insert into the
  insert-only corpus is detectable rather than silently duplicated.
* **Runtime-specific settings live in a discriminated variant**, not in a flat
  god-object. Adding ExecuTorch adds a variant; it does not mutate the core.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgefit.schema.common import (
    ActivationQuant,
    Dtype,
    Granularity,
    QuantAlgorithm,
    RuntimeKind,
    TaskType,
    content_hash,
)

# Bump on any change to the meaning or shape of these fields. Records are never
# edited in place; a schema change requires a migration (PROJECT.md §6.1).
CONFIG_SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class ModelRef(_Frozen):
    """Where the model came from, pinned tightly enough to reproduce."""

    ref: str = Field(description="'hf:<repo-id>' or 'file:<path>'")
    task: TaskType
    revision: str | None = Field(
        default=None,
        description="Upstream commit sha. Unpinned models make numbers irreproducible.",
    )


class QuantizationConfig(_Frozen):
    """PROJECT.md §6.1 quantization axis."""

    weight_dtype: Dtype
    weight_granularity: Granularity = Granularity.PER_TENSOR
    block_size: int | None = Field(
        default=None, description="Required iff granularity is blockwise"
    )
    symmetric: bool = True
    activation_quant: ActivationQuant = ActivationQuant.NONE
    activation_dtype: Dtype | None = None
    algorithm: QuantAlgorithm | None = None
    skip_op_types: tuple[str, ...] = ()
    kv_cache_dtype: Dtype | None = None

    @model_validator(mode="after")
    def _check_coherent(self) -> QuantizationConfig:
        if self.weight_granularity is Granularity.BLOCKWISE and self.block_size is None:
            raise ValueError("blockwise quantization requires block_size")
        if self.weight_granularity is not Granularity.BLOCKWISE and self.block_size is not None:
            raise ValueError("block_size is only meaningful for blockwise quantization")
        if self.activation_quant is ActivationQuant.NONE and self.activation_dtype is not None:
            raise ValueError("activation_dtype set but activation quantization is disabled")
        if self.activation_quant is not ActivationQuant.NONE and self.activation_dtype is None:
            raise ValueError("activation quantization enabled but no activation_dtype given")
        return self


class CalibrationConfig(_Frozen):
    """PROJECT.md §6.1 calibration axis.

    ``dataset_ref`` records *which* data was used. When a customer supplies none we
    use a generic corpus and say so explicitly (Stage 2 input #6) — that honesty is
    recorded here rather than left implicit.
    """

    dataset_ref: str
    sample_count: int = Field(gt=0)
    observer: str = "min_max"
    seed: int = 0
    is_generic_corpus: bool = False


class PartitionConfig(_Frozen):
    """PROJECT.md §6.1 partition boundaries — forced placement and exclusions."""

    forced_placement: dict[str, str] = Field(
        default_factory=dict, description="node name or pattern -> provider/backend"
    )
    excluded_op_types: tuple[str, ...] = ()
    attention_boundary_split: bool = False


class ExecutionConfig(_Frozen):
    """PROJECT.md §6.1 runtime config axis. Runtime-agnostic knobs only."""

    num_threads: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=1, gt=0)
    memory_planning: str | None = None
    kv_cache_layout: str | None = None
    prefill_chunk_size: int | None = Field(default=None, gt=0)
    sequence_buckets: tuple[int, ...] | None = None


class OrtProvider(StrEnum):
    """ONNX Runtime execution providers. Values are ORT's own literals."""

    CPU = "CPUExecutionProvider"
    COREML = "CoreMLExecutionProvider"
    XNNPACK = "XnnpackExecutionProvider"
    NNAPI = "NnapiExecutionProvider"
    QNN = "QNNExecutionProvider"
    CUDA = "CUDAExecutionProvider"


class GraphOptLevel(StrEnum):
    DISABLED = "disabled"
    BASIC = "basic"
    EXTENDED = "extended"
    ALL = "all"


class CoreMLComputeUnits(StrEnum):
    ALL = "ALL"
    CPU_ONLY = "CPUOnly"
    CPU_AND_GPU = "CPUAndGPU"
    CPU_AND_NE = "CPUAndNeuralEngine"


class OrtRuntimeConfig(_Frozen):
    """ONNX Runtime session configuration, including CoreML EP vendor flags."""

    kind: Literal[RuntimeKind.ONNXRUNTIME] = RuntimeKind.ONNXRUNTIME
    providers: tuple[OrtProvider, ...] = Field(
        default=(OrtProvider.CPU,),
        min_length=1,
        description="Priority order, as ORT consumes it. CPU is the implicit last resort.",
    )
    graph_optimization_level: GraphOptLevel = GraphOptLevel.ALL
    inter_op_num_threads: int | None = Field(default=None, gt=0)
    parallel_execution: bool = False

    # --- CoreML EP vendor flags (PROJECT.md §6.1 "vendor flags") ---
    coreml_compute_units: CoreMLComputeUnits | None = None
    coreml_model_format: Literal["MLProgram", "NeuralNetwork"] | None = None
    coreml_require_static_shapes: bool | None = None
    coreml_allow_low_precision: bool | None = None

    @model_validator(mode="after")
    def _check_coherent(self) -> OrtRuntimeConfig:
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("duplicate execution provider in priority list")
        coreml_flags_set = any(
            v is not None
            for v in (
                self.coreml_compute_units,
                self.coreml_model_format,
                self.coreml_require_static_shapes,
                self.coreml_allow_low_precision,
            )
        )
        if coreml_flags_set and OrtProvider.COREML not in self.providers:
            raise ValueError("CoreML EP flags set but CoreMLExecutionProvider is not in providers")
        return self

    @property
    def intended_provider(self) -> OrtProvider:
        """The accelerator this config is *trying* to reach.

        Fallback is measured against this. If the whole point of a config is the
        Neural Engine and half the graph lands on CPU, that gap is the finding.
        """
        for provider in self.providers:
            if provider is not OrtProvider.CPU:
                return provider
        return OrtProvider.CPU


# One variant today. When ExecuTorch lands this becomes
#   Annotated[OrtRuntimeConfig | ExecutorchRuntimeConfig, Field(discriminator="kind")]
# and previously-serialised records still deserialise, because `kind` is stored.
RuntimeConfig = OrtRuntimeConfig


class ConfigRecord(_Frozen):
    """One point in the search space. Immutable, versioned, content-addressed."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    model: ModelRef
    runtime: RuntimeConfig
    quantization: QuantizationConfig | None = None
    calibration: CalibrationConfig | None = None
    partition: PartitionConfig = PartitionConfig()
    execution: ExecutionConfig = ExecutionConfig()
    label: str | None = Field(
        default=None,
        description="Human tag. Excluded from config_id — it carries no semantics.",
    )

    @model_validator(mode="after")
    def _check_coherent(self) -> ConfigRecord:
        needs_calibration = (
            self.quantization is not None
            and self.quantization.activation_quant is ActivationQuant.STATIC
        )
        if needs_calibration and self.calibration is None:
            raise ValueError("static activation quantization requires calibration data")
        return self

    @property
    def config_id(self) -> str:
        """Content hash over everything semantically meaningful."""
        payload = self.model_dump(mode="json", exclude={"label"})
        return content_hash(payload)

    @property
    def intended_provider(self) -> str:
        return str(self.runtime.intended_provider)
