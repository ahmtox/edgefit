"""Known model subjects.

Pass 1 deliberately starts with models already in the local HuggingFace cache, so
the harness can be validated without spending a single byte of the ~19 GB of free
disk on this host. Both entries are small enough to measure quickly and real
enough to exercise the CoreML partitioner properly.

Adding a model is adding a registry entry — per hard rule #7, every engagement
adds a *platform* capability, never a bespoke script.
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
    exporter: str = field(metadata={"choices": ["encoder", "vision"]})
    static_shape: dict[str, int] = field(default_factory=dict)
    description: str = ""

    @property
    def slug(self) -> str:
        return self.hf_id.replace("/", "__")


REGISTRY: dict[str, ModelSpec] = {
    "hf:sentence-transformers/all-MiniLM-L6-v2": ModelSpec(
        ref="hf:sentence-transformers/all-MiniLM-L6-v2",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        task=TaskType.EMBED,
        exporter="encoder",
        static_shape={"batch": 1, "sequence": 128},
        description="22M-param MiniLM encoder. Small, fast, and the first golden fixture.",
    ),
    "hf:google/vit-base-patch16-224-in21k": ModelSpec(
        ref="hf:google/vit-base-patch16-224-in21k",
        hf_id="google/vit-base-patch16-224-in21k",
        task=TaskType.VISION,
        exporter="vision",
        static_shape={"batch": 1, "height": 224, "width": 224, "channels": 3},
        description="86M-param ViT. Conv stem plus transformer — a different partitioner story.",
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
        raise UnknownModelError(
            f"unknown model {ref!r}. Known models:\n  {known}"
        ) from exc


def known_refs() -> list[str]:
    return sorted(REGISTRY)
