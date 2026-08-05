"""Configuration record — one point in the search space (PROJECT.md §6.1).

Design notes that matter later:

* **The target device is deliberately absent.** One recipe is measured on many
  devices; that product is exactly the atlas matrix. Device identity lives on the
  measurement record.
* **Frozen and content-addressed.** ``recipe_id`` is a hash of the canonical form,
  so the same recipe always lands on the same id and a duplicate insert into the
  insert-only corpus is detectable rather than silently duplicated.
* **Runtime-specific settings live in a discriminated variant**, not in a flat
  god-object. Adding ExecuTorch adds a variant; it does not mutate the core.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

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
# v2 adds the `lowering` section, so a recipe fully determines its artifact.
RECIPE_SCHEMA_VERSION = 2


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


class LoweringConfig(_Frozen):
    """How the model is exported, before any runtime sees it.

    Separate from ``ExecutionConfig`` because lowering and execution are genuinely
    different stages (PROJECT.md §2.1 steps 2 and 6), and because putting these
    here is what makes a recipe *fully determine* its artifact — the invariant the
    sweep runner relies on to cache and resume.
    """

    opset: int = Field(default=17, gt=0)
    static_shapes: bool = Field(
        default=True,
        description=(
            "False marks batch and sequence dynamic. Not a convenience knob: dynamic "
            "shape is the single biggest reason a delegate declines a subgraph, so the "
            "two variants are a controlled experiment in silent fallback."
        ),
    )


class ExecutionConfig(_Frozen):
    """PROJECT.md §6.1 runtime recipe axis. Runtime-agnostic knobs only."""

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
        """The accelerator this recipe is *trying* to reach.

        Fallback is measured against this. If the whole point of a recipe is the
        Neural Engine and half the graph lands on CPU, that gap is the finding.
        """
        for provider in self.providers:
            if provider is not OrtProvider.CPU:
                return provider
        return OrtProvider.CPU


class QaiHubComputeUnit(StrEnum):
    """Compute units a Qualcomm AI Hub job may target, in its own vocabulary."""

    NPU = "npu"
    GPU = "gpu"
    CPU = "cpu"
    ALL = "all"


class QaiHubRuntimeConfig(_Frozen):
    """A job run on Qualcomm AI Hub's hosted devices.

    Deliberately not expressed as ONNX Runtime with a provider list: on AI Hub we do
    not choose the execution provider, the service compiles and schedules the model
    itself. Pretending otherwise would put a provider in the recipe that nothing
    honours, and a recipe field that does not affect the run is how recipes start
    lying.

    That rule is why there is no ``target_runtime`` here. It reads like the obvious
    axis — tflite vs onnx vs qnn_context_binary — but it is a *compile*-job option,
    and profile jobs reject it outright (``unrecognized arguments:
    --target_runtime``). Compile jobs are broken server-side, so the field could
    only ever have been recorded and ignored. It returns when compile does.
    """

    kind: Literal[RuntimeKind.QAI_HUB] = RuntimeKind.QAI_HUB
    device_name: str = Field(description="Device as AI Hub names it, e.g. 'Samsung Galaxy S24'")
    device_os: str | None = Field(
        default=None, description="Pin the OS version; AI Hub picks one when omitted."
    )
    compute_unit: QaiHubComputeUnit = Field(
        default=QaiHubComputeUnit.ALL,
        description=(
            "Passed to the job as --compute_unit. What we ask for; AI Hub still "
            "decides per node, and the measured split is what gets recorded."
        ),
    )

    @property
    def intended_provider(self) -> str:
        """The unit this recipe is aiming at, for the fallback report."""
        return self.compute_unit.value.upper()


# Discriminated on `kind`, so previously-serialised records still deserialise and
# adding ExecuTorch later adds a variant rather than mutating the core.
RuntimeConfig = Annotated[
    OrtRuntimeConfig | QaiHubRuntimeConfig,
    Field(discriminator="kind"),
]


class Recipe(_Frozen):
    """One point in the search space. Immutable, versioned, content-addressed."""

    schema_version: int = RECIPE_SCHEMA_VERSION
    model: ModelRef
    runtime: RuntimeConfig
    lowering: LoweringConfig = LoweringConfig()
    quantization: QuantizationConfig | None = None
    calibration: CalibrationConfig | None = None
    partition: PartitionConfig = PartitionConfig()
    execution: ExecutionConfig = ExecutionConfig()
    label: str | None = Field(
        default=None,
        description="Human tag. Excluded from recipe_id — it carries no semantics.",
    )

    @model_validator(mode="after")
    def _check_coherent(self) -> Recipe:
        needs_calibration = (
            self.quantization is not None
            and self.quantization.activation_quant is ActivationQuant.STATIC
        )
        if needs_calibration and self.calibration is None:
            raise ValueError("static activation quantization requires calibration data")
        return self

    @property
    def recipe_id(self) -> str:
        """Content hash over everything semantically meaningful."""
        payload = self.model_dump(mode="json", exclude={"label"})
        return content_hash(payload)

    @property
    def intended_provider(self) -> str:
        return str(self.runtime.intended_provider)

    @property
    def is_remote(self) -> bool:
        """True when the run happens on hardware we do not own."""
        return self.runtime.kind is RuntimeKind.QAI_HUB

    def derive(self, *, label: str | None = None, **sections: object) -> Recipe:
        """A new recipe with sections deep-merged over this one.

        PROJECT.md §6.1 requires recipes to *compose*: to inherit from
        expert-vetted defaults and "swap cleanly so re-compiling with a different
        config is one field change." That is the difference between a recipe object
        and a dict, and it is what makes a known-good baseline usable as
        warm-start material for the search rather than something to copy-paste.

        Pass a dict to merge into a section, or a model instance to replace it:

            fast = baseline.derive(execution={"num_threads": 4})
            ane = baseline.derive(runtime={"coreml_compute_units": "CPUAndNeuralEngine"})

        Merging is one level deep, which matches the shape of the object: sections
        are flat bags of knobs, so deeper recursion would buy nothing and make the
        result harder to predict.
        """
        payload = self.model_dump(mode="json")
        for name, value in sections.items():
            if name not in type(self).model_fields:
                raise ValueError(f"unknown recipe section {name!r}")
            if isinstance(value, dict) and isinstance(payload.get(name), dict):
                payload[name] = payload[name] | value
            elif isinstance(value, BaseModel):
                payload[name] = value.model_dump(mode="json")
            else:
                payload[name] = value
        if label is not None:
            payload["label"] = label
        return type(self).model_validate(payload)
