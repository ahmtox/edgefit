"""Known model subjects.

Every entry here is already in the local HuggingFace cache, so breadth costs zero
downloads and zero disk beyond the ONNX exports themselves. That matters: the
whole capital argument in PROJECT.md §7 is that phase 0 and phase 1 cost $0.

Adding a model is adding a row to ``REGISTRY``. Nothing else. Hard rule #7 says
every engagement must add a *platform* capability rather than a bespoke script, and
the way that rule is kept honest is by making the platform path the easy one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from edgefit.schema.common import TaskType


@dataclass(frozen=True)
class ModelSpec:
    """How to obtain a model and how to exercise it."""

    ref: str
    hf_id: str
    task: TaskType

    #: Which input harness to use. "text" tokenizes a fixed prompt; "vision" feeds a
    #: fixed pseudo-random image. Both are deterministic so the numbers reproduce.
    exporter: str

    #: transformers class to load with, e.g. AutoModel or CLIPVisionModel.
    hf_class: str = "AutoModel"

    #: Attribute of the forward output to export. Classifiers emit `logits`;
    #: encoders emit `last_hidden_state`.
    output_attr: str = "last_hidden_state"

    #: Optional submodule to export instead of the whole model, e.g. bart's encoder.
    #: A seq2seq model's decoder needs a different harness, so the encoder is the
    #: honest unit to measure until the generative harness exists.
    submodule: str | None = None

    static_shape: dict[str, int] = field(default_factory=dict)
    description: str = ""

    @property
    def slug(self) -> str:
        return self.hf_id.replace("/", "__")


_TEXT_SHAPE = {"batch": 1, "sequence": 128}
_IMAGE_SHAPE = {"batch": 1, "height": 224, "width": 224, "channels": 3}


REGISTRY: dict[str, ModelSpec] = {
    "hf:sentence-transformers/all-MiniLM-L6-v2": ModelSpec(
        ref="hf:sentence-transformers/all-MiniLM-L6-v2",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        task=TaskType.EMBED,
        exporter="text",
        static_shape=_TEXT_SHAPE,
        description="22M MiniLM encoder. Small, fast, and the first golden fixture.",
    ),
    "hf:martin-ha/toxic-comment-model": ModelSpec(
        ref="hf:martin-ha/toxic-comment-model",
        hf_id="martin-ha/toxic-comment-model",
        task=TaskType.CLASSIFY,
        exporter="text",
        hf_class="AutoModelForSequenceClassification",
        output_attr="logits",
        static_shape=_TEXT_SHAPE,
        description="DistilBERT classifier. No token_type_ids, and a logits head.",
    ),
    "hf:facebook/bart-base": ModelSpec(
        ref="hf:facebook/bart-base",
        hf_id="facebook/bart-base",
        task=TaskType.EMBED,
        exporter="text",
        submodule="encoder",
        static_shape=_TEXT_SHAPE,
        description="BART encoder. Learned positional embeddings and a wider hidden size.",
    ),
    "hf:meta-llama/Llama-3.2-1B-Instruct": ModelSpec(
        ref="hf:meta-llama/Llama-3.2-1B-Instruct",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        task=TaskType.GENERATE,
        exporter="decoder",
        hf_class="AutoModelForCausalLM",
        static_shape={"batch": 1},
        description=(
            "1.2B decoder-only, GQA 32q/8kv, RMSNorm — the architecture PROJECT.md's "
            "worked example describes. Exported with KV cache I/O so decode is O(1) "
            "per token rather than O(n)."
        ),
    ),
    "hf:google/vit-base-patch16-224-in21k": ModelSpec(
        ref="hf:google/vit-base-patch16-224-in21k",
        hf_id="google/vit-base-patch16-224-in21k",
        task=TaskType.VISION,
        exporter="vision",
        static_shape=_IMAGE_SHAPE,
        description="86M ViT. Conv stem plus transformer — a different partitioner story.",
    ),
    "hf:openai/clip-vit-base-patch32": ModelSpec(
        ref="hf:openai/clip-vit-base-patch32",
        hf_id="openai/clip-vit-base-patch32",
        task=TaskType.VISION,
        exporter="vision",
        hf_class="CLIPVisionModel",
        static_shape=_IMAGE_SHAPE,
        description="CLIP vision tower. Patch 32, so a quarter of ViT-16's token count.",
    ),
}


class UnknownModelError(KeyError):
    """Raised for a ref that is not in the registry."""


def resolve(ref: str) -> ModelSpec:
    """Look up a model spec by ref, e.g. ``hf:google/vit-base-patch16-224-in21k``."""
    try:
        return REGISTRY[ref]
    except KeyError as exc:
        known = "\n  ".join(sorted(REGISTRY))
        raise UnknownModelError(f"unknown model {ref!r}. Known models:\n  {known}") from exc


def known_refs() -> list[str]:
    return sorted(REGISTRY)
