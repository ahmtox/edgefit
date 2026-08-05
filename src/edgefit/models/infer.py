"""Infer a :class:`ModelSpec` from a HuggingFace ``config.json``.

Until this existed, :func:`edgefit.models.registry.resolve` was a dict lookup over six
hand-written specs and everything else raised. That made the harness unable to measure
a model we had not personally added — including, decisively, a customer's. PROJECT.md
§11a's funnel is *inbound → sends a model → sends an eval set → pays*, and step two was
a ``KeyError``.

**The rule that shapes this module: infer or refuse, never approximate.** A model
measured through the wrong input harness does not fail. It produces a plausible number
for a workload nobody asked about — a text encoder fed a 224x224 image tensor, a
seq2seq model measured as though it were an encoder. That is hard rule #1's failure
mode wearing a different hat, and it is worse here than elsewhere because the result
looks entirely reasonable. So every branch below either establishes what a model is
from evidence in its config, or raises with the specific thing that defeated it.

The decision table is a pure function of a config dict. No network, no torch, no
filesystem — so the whole of it is unit-testable, which is the only way a table this
full of special cases stays honest.
"""

from __future__ import annotations

from typing import Any

from edgefit.models.registry import ModelSpec
from edgefit.schema.common import TaskType

#: Bumped when inference could assign a *different* spec to an unchanged config.
#: The spec determines the artifact, and the artifact is part of what gets measured —
#: the same reason the exporters carry a version.
SPEC_INFERENCE_VERSION = 1

#: Fixed so measurements stay comparable across models. Deliberately not the model's
#: own maximum: comparing a 128-token encoder against a 512-token one measures the
#: sequence length, not the model.
DEFAULT_SEQUENCE = 128
DEFAULT_IMAGE = 224

#: Architecture suffixes we can place exactly. Suffix rather than whole name, because
#: `BertForSequenceClassification` and `DistilBertForSequenceClassification` are the
#: same measurement problem.
_BY_SUFFIX: dict[str, tuple[TaskType, str, str, str]] = {
    # suffix: (task, exporter, hf_class, output_attr)
    "ForCausalLM": (TaskType.GENERATE, "decoder", "AutoModelForCausalLM", "logits"),
    "ForSequenceClassification": (
        TaskType.CLASSIFY,
        "text",
        "AutoModelForSequenceClassification",
        "logits",
    ),
    "ForImageClassification": (
        TaskType.VISION,
        "vision",
        "AutoModelForImageClassification",
        "logits",
    ),
    "ForTokenClassification": (
        TaskType.CLASSIFY,
        "text",
        "AutoModelForTokenClassification",
        "logits",
    ),
}

#: Bare-encoder architectures, where the head is absent and the task is the embedding.
_ENCODER_SUFFIXES = ("Model", "ForMaskedLM", "ForPreTraining")


class UninferableModelError(ValueError):
    """We cannot establish what this model is, so we decline to measure it.

    Carries the specific obstacle. "Unsupported" on its own sends someone hunting
    through our source; naming the field that defeated us does not.
    """


def _architecture(config: dict[str, Any]) -> str | None:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        first = architectures[0]
        if isinstance(first, str) and first:
            return first
    return None


def _vision_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """The sub-config describing an image tower, if there is one."""
    nested = config.get("vision_config")
    return nested if isinstance(nested, dict) else None


def _image_size(config: dict[str, Any]) -> int | None:
    for source in (config, _vision_config(config) or {}):
        size = source.get("image_size")
        if isinstance(size, int) and size > 0:
            return size
    return None


def _channels(config: dict[str, Any]) -> int:
    for source in (config, _vision_config(config) or {}):
        channels = source.get("num_channels")
        if isinstance(channels, int) and channels > 0:
            return channels
    return 3


def _sequence(config: dict[str, Any]) -> int:
    """A fixed sequence length, shortened only if the model genuinely cannot take it."""
    limit = config.get("max_position_embeddings")
    if isinstance(limit, int) and 0 < limit < DEFAULT_SEQUENCE:
        return limit
    return DEFAULT_SEQUENCE


def _refuse_unmeasurable_shapes(config: dict[str, Any], hf_id: str) -> None:
    """Reject model shapes whose measurement would silently mean something else."""
    if config.get("is_encoder_decoder"):
        raise UninferableModelError(
            f"{hf_id} is encoder-decoder (`is_encoder_decoder: true`). Prefill and decode "
            "run through different stacks, so measuring it whole would report a number "
            "belonging to neither. Add a registry override naming the submodule to "
            "measure, the way `facebook/bart-base` pins its encoder."
        )
    if _vision_config(config) and config.get("text_config"):
        raise UninferableModelError(
            f"{hf_id} carries both a vision and a text tower, so there is no single "
            "graph to measure and no honest way to choose one for you. Add a registry "
            "override naming the tower, the way `openai/clip-vit-base-patch32` pins "
            "CLIPVisionModel."
        )


def infer_spec(ref: str, config: dict[str, Any]) -> ModelSpec:
    """Work out how to obtain and exercise a model, or refuse and say why.

    Pure: the config dict is the only input, so every branch is testable offline.
    """
    hf_id = ref.removeprefix("hf:")
    if not isinstance(config, dict) or not config:
        raise UninferableModelError(
            f"{hf_id} has no readable config.json, so nothing establishes its task, "
            "input shape or output tensor. All three would have to be guessed."
        )

    _refuse_unmeasurable_shapes(config, hf_id)

    architecture = _architecture(config)
    model_type = config.get("model_type") or "unknown"
    if architecture is None:
        raise UninferableModelError(
            f"{hf_id} lists no `architectures` in its config (model_type "
            f"{model_type!r}), which is the only field that says what head the model "
            "has. Without it the output tensor is a guess."
        )

    image_size = _image_size(config)

    for suffix, (task, exporter, hf_class, output_attr) in _BY_SUFFIX.items():
        if architecture.endswith(suffix):
            return _build(ref, hf_id, task, exporter, hf_class, output_attr, config, image_size)

    if architecture.endswith(_ENCODER_SUFFIXES):
        # A bare encoder: no head, so the embedding *is* the output. Modality comes
        # from whether the config describes an image at all.
        if image_size is not None:
            return _build(
                ref, hf_id, TaskType.VISION, "vision", "AutoModel",
                "last_hidden_state", config, image_size,
            )
        return _build(
            ref, hf_id, TaskType.EMBED, "text", "AutoModel",
            "last_hidden_state", config, None,
        )

    raise UninferableModelError(
        f"{hf_id} has architecture {architecture!r}, which does not match any head we "
        "know how to exercise. Recognised suffixes: "
        f"{', '.join(sorted([*_BY_SUFFIX, *_ENCODER_SUFFIXES]))}. Add a registry entry "
        "if this model should be measurable."
    )


def _build(
    ref: str,
    hf_id: str,
    task: TaskType,
    exporter: str,
    hf_class: str,
    output_attr: str,
    config: dict[str, Any],
    image_size: int | None,
) -> ModelSpec:
    if exporter == "vision":
        size = image_size or DEFAULT_IMAGE
        shape = {
            "batch": 1,
            "height": size,
            "width": size,
            "channels": _channels(config),
        }
    elif exporter == "decoder":
        # A KV-cached decoder needs a dynamic sequence axis, so only batch is pinned.
        shape = {"batch": 1}
    else:
        shape = {"batch": 1, "sequence": _sequence(config)}

    return ModelSpec(
        ref=ref,
        hf_id=hf_id,
        task=task,
        exporter=exporter,
        hf_class=hf_class,
        output_attr=output_attr,
        static_shape=shape,
        description=(
            f"inferred from config.json (architecture "
            f"{_architecture(config)}, model_type {config.get('model_type', 'unknown')})"
        ),
    )


def load_hf_config(hf_id: str) -> dict[str, Any]:
    """Fetch just ``config.json`` — kilobytes, not gigabytes.

    Deliberately not `AutoConfig.from_pretrained`: that pulls in transformers, and the
    point of probing is to answer "can we measure this" before committing to a
    multi-gigabyte download.
    """
    import json  # noqa: PLC0415

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - export extra
        raise UninferableModelError(
            "huggingface_hub is not installed, so an unregistered model cannot be "
            "inspected. Install the extra: uv sync --extra export"
        ) from exc

    try:
        path = hf_hub_download(hf_id, "config.json")
    except Exception as exc:  # noqa: BLE001 - hub errors are many and all mean the same here
        raise UninferableModelError(
            f"could not fetch config.json for {hf_id}: {type(exc).__name__}: {exc}"
        ) from exc
    return json.loads(open(path).read())  # noqa: SIM115
