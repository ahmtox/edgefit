"""Shared enums and the canonical-hash helper used by every record type.

Every value here ends up in the corpus and therefore in the cost model's feature
space, so names are chosen to be stable and vendor-neutral.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any


class TaskType(StrEnum):
    """PROJECT.md §4 Stage 2, input #3. Determines harness and eval methodology."""

    GENERATE = "generate"
    CLASSIFY = "classify"
    EMBED = "embed"
    VISION = "vision"
    ASR = "asr"


class Outcome(StrEnum):
    """PROJECT.md §6.2. A run that did not produce numbers still produces a record."""

    SUCCESS = "success"
    LOWERING_FAILURE = "lowering_failure"
    RUNTIME_FAILURE = "runtime_failure"
    ACCURACY_FAILURE = "accuracy_failure"
    # Not in §6.2, but required by hard rule #1: the harness refused to measure
    # because the host was unfit. Recording this is how we avoid silently
    # substituting a bad number for a missing one.
    GATE_REFUSED = "gate_refused"


class Dtype(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"


class Granularity(StrEnum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    BLOCKWISE = "blockwise"


class ActivationQuant(StrEnum):
    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"


class QuantAlgorithm(StrEnum):
    MIN_MAX = "min_max"
    ENTROPY = "entropy"
    PERCENTILE = "percentile"
    GPTQ = "gptq"
    AWQ = "awq"
    SMOOTHQUANT = "smoothquant"
    SPINQUANT = "spinquant"


class RuntimeKind(StrEnum):
    """Layer-1 runtimes (PROJECT.md §3.1). We build on these, never compete."""

    ONNXRUNTIME = "onnxruntime"
    EXECUTORCH = "executorch"
    LITERT = "litert"
    LLAMA_CPP = "llama_cpp"
    COREML = "coreml"


class AttentionVariant(StrEnum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"
    NONE = "none"
    UNKNOWN = "unknown"


class NormType(StrEnum):
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    BATCHNORM = "batchnorm"
    GROUPNORM = "groupnorm"
    NONE = "none"
    UNKNOWN = "unknown"


class StressProfile(StrEnum):
    """Condition a measurement was taken under (PROJECT.md §5.6, §6.2).

    The staged validation ladder is about *trust*, distinct from the cost cascade
    which is about *search economics*. Everyone benchmarks clean cold devices; no
    user has one, and §2.2 puts the resulting P99 gap at 3–5x. Recording the
    profile on every row is what makes that gap measurable later — a corpus that
    cannot distinguish clean from soaked can never quantify it.
    """

    CLEAN = "clean"
    THERMAL_SOAK = "thermal_soak"
    MEMORY_PRESSURE = "memory_pressure"
    CONCURRENT_LOAD = "concurrent_load"


class PowerSource(StrEnum):
    AC = "ac"
    BATTERY = "battery"
    UNKNOWN = "unknown"


class ThermalState(StrEnum):
    """Mirrors NSProcessInfo.thermalState. Coarse, but real and unprivileged.

    There is no unprivileged temperature reading on Apple Silicon, so this plus
    the measured calibration probe is the honest ceiling of what we can know.
    """

    NOMINAL = "nominal"
    FAIR = "fair"
    SERIOUS = "serious"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    The basis of every content hash in the system, so its output must not drift
    between Python versions or dict insertion orders.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(payload: Any, *, length: int = 16) -> str:
    """Stable short digest of a record's canonical form.

    Content addressing is what makes the corpus insert-only workable: the same
    recipe always lands on the same id, so a duplicate insert is detectable
    rather than silently creating a second row.
    """
    digest = hashlib.blake2b(canonical_json(payload).encode("utf-8"), digest_size=32)
    return digest.hexdigest()[:length]
