"""HuggingFace -> ONNX ingestion (PROJECT.md §5.2).

Three things are produced together and cached as one unit:

* ``model.onnx``    — the artifact under test
* ``inputs.npz``    — the exact tensors every run is fed
* ``reference.npz`` — the fp32 PyTorch output for those inputs

Pinning the inputs matters more than it looks. If each run generated its own
random tensors, latency would vary with data and the known-answer test would
have nothing to compare against. Storing the reference alongside means the
numerics check (§9 "known-answer test") needs no torch at measurement time —
which is what keeps the Tier-3 self-hosted runner shippable.

This module needs the ``export`` extra. Nothing at measurement time does.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx

from edgefit.models.registry import ModelSpec
from edgefit.schema.common import content_hash


class ExportError(RuntimeError):
    """The export produced something the harness cannot honestly measure."""


DEFAULT_ARTIFACT_ROOT = Path("artifacts/onnx")
DEFAULT_OPSET = 17

#: Bump whenever this exporter would produce a different **artifact** for the same
#: inputs — which includes meta.json, not only the graph. It is part of the artifact
#: key, because otherwise changing the export code silently reuses a stale cached
#: artifact, and two measurements of two different artifacts end up sharing one
#: identity in the corpus.
#:
#: v3 records head and layer counts in meta.json. Missing this bump left dynamic-shape
#: and quantized artifacts carrying pre-fix sidecars, so the same model fingerprinted
#: as both `mha` and `unknown` depending on which artifact a row happened to use.
#:
#: v4 filters harness inputs by what the model's forward accepts and reconciles
#: inputs.npz against the exported graph, so a tokenizer emitting more than the model
#: takes can no longer put an unusable tensor in the artifact.
#:
#: v5 adds the frozen-token text export, which changes the input surface of an
#: artifact and therefore its identity.
EXPORTER_VERSION = 5

#: Files the harness writes alongside the model; not part of the shipped artifact.
HARNESS_SIDECARS = frozenset({"inputs.npz", "reference.npz", "meta.json"})


def artifact_size_bytes(directory: Path) -> int:
    """Bytes of the model artifact in ``directory``, excluding harness sidecars."""
    return sum(
        path.stat().st_size
        for path in directory.iterdir()
        if path.is_file() and path.name not in HARNESS_SIDECARS
    )

# One fixed prompt, so the exported reference is reproducible by anyone.
_CALIBRATION_TEXT = "EdgeFit measures what actually runs on the device."


@dataclass(frozen=True)
class ExportedArtifact:
    """A lowered model plus everything needed to exercise and verify it."""

    directory: Path
    model_path: Path
    inputs_path: Path
    reference_path: Path
    lowering_ms: float
    was_cached: bool

    @property
    def size_bytes(self) -> int:
        """Total on-disk size of the model artifact.

        Every file except our own harness sidecars. Globbing ``*.onnx*`` was wrong:
        a model over the 2 GB protobuf limit spills its weights into external files
        named after the tensors, so a 5.6 GB Llama artifact measured 0.7 MiB — the
        graph without any of its weights. Hard rule #4 says measure what a user
        actually ships, and they ship the whole directory.
        """
        return artifact_size_bytes(self.directory)

    def load_inputs(self) -> dict[str, np.ndarray]:
        with np.load(self.inputs_path) as data:
            return {name: data[name] for name in data.files}

    def load_reference(self) -> dict[str, np.ndarray]:
        with np.load(self.reference_path) as data:
            return {name: data[name] for name in data.files}


def artifact_key(
    spec: ModelSpec, opset: int, static_shapes: bool, frozen_tokens: bool = False
) -> str:
    return content_hash(
        {
            "hf_id": spec.hf_id,
            "exporter": spec.exporter,
            "exporter_version": EXPORTER_VERSION,
            "hf_class": spec.hf_class,
            "output_attr": spec.output_attr,
            "submodule": spec.submodule,
            "opset": opset,
            "static_shapes": static_shapes,
            "frozen_tokens": frozen_tokens,
            "shape": spec.static_shape,
        }
    )


def architecture_from_config(config) -> dict[str, int]:
    """Head and layer counts, for an exact attention label in the fingerprint.

    Recorded because the fingerprint is the key the cost model indexes on (§5.2), and
    a graph alone cannot distinguish MHA from GQA — the detector correctly reports
    UNKNOWN without these. Encoders without a `num_key_value_heads` field are
    multi-head by definition, so the query-head count stands in for both.
    """
    heads = getattr(config, "num_attention_heads", None)
    layers = getattr(config, "num_hidden_layers", None)
    if heads is None:
        # Some vision/dual-tower configs nest the real settings one level down.
        for attribute in ("vision_config", "text_config", "encoder"):
            nested = getattr(config, attribute, None)
            if nested is not None and getattr(nested, "num_attention_heads", None):
                heads = nested.num_attention_heads
                layers = getattr(nested, "num_hidden_layers", layers)
                config = nested
                break
    if heads is None:
        return {}
    # Hierarchical models describe themselves per stage: Swin carries
    # num_heads=[3,6,12,24] and depths=[2,2,6,2]. `int()` on a list raises, which is
    # how this surfaced — a crash mid-export rather than a bad number, which is the
    # better failure but still a failure.
    #
    # There is no honest scalar here. Collapsing four stages to a max or a sum would
    # put a head count in the fingerprint that no part of the model has, and the
    # fingerprint is what a cost model indexes on. So the fields stay absent, exactly
    # as they do when the attention variant cannot be established: reported exactly,
    # or reported not at all.
    if isinstance(heads, (list, tuple)) or isinstance(layers, (list, tuple)):
        return {"stages": len(heads) if isinstance(heads, (list, tuple)) else len(layers)}
    architecture = {
        "n_heads": int(heads),
        "kv_heads": int(getattr(config, "num_key_value_heads", heads) or heads),
    }
    if layers:
        architecture["layers"] = int(layers)
    return architecture


def _load_hf_model(spec: ModelSpec):
    """Instantiate the model class the spec names, optionally a submodule of it."""
    import transformers  # noqa: PLC0415

    try:
        cls = getattr(transformers, spec.hf_class)
    except AttributeError as exc:
        raise ValueError(f"transformers has no class {spec.hf_class!r} for {spec.ref}") from exc

    model = cls.from_pretrained(spec.hf_id).eval()
    if spec.submodule is not None:
        model = getattr(model, spec.submodule)
    return model.eval()


def _accepted_arguments(model) -> frozenset[str]:
    """Keyword arguments this model's forward actually takes.

    A model whose forward accepts ``**kwargs`` would swallow anything, so that case
    reports "everything" and leaves the post-export reconciliation to catch mistakes.
    """
    import inspect  # noqa: PLC0415

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return frozenset({"input_ids", "attention_mask", "token_type_ids"})
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return frozenset({"input_ids", "attention_mask", "token_type_ids"})
    return frozenset(parameters)


def _build_text(spec: ModelSpec):
    """Text model fed a fixed prompt padded to the static sequence length.

    transformers>=5 rejects the dict-of-kwargs form the TorchScript tracer emits
    ("got multiple values for argument 'use_cache'"), so the positional wrapper is
    load bearing, not cosmetic. It also pins the input names in the exported graph.

    Input names come from what the tokenizer actually returns, because that varies
    by architecture — DistilBERT has no ``token_type_ids`` and BERT does, and
    exporting an input the model never receives produces a graph that fails at
    session creation rather than at export.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = _load_hf_model(spec)
    encoded = tokenizer(
        _CALIBRATION_TEXT,
        return_tensors="pt",
        padding="max_length",
        max_length=spec.static_shape["sequence"],
        truncation=True,
    )
    # Two filters, because the tokenizer and the model disagree more often than they
    # look like they should. `distilbert-base-uncased`'s tokenizer emits
    # `token_type_ids`; `DistilBertModel.forward` does not accept it. Keying only on
    # the tokenizer put a tensor in inputs.npz that the traced graph had pruned, and
    # every recipe then failed at session creation with `Invalid input name:
    # token_type_ids`. The registered toxic-comment DistilBERT never showed it —
    # its tokenizer config omits the field — so six curated models hid the bug.
    accepted = _accepted_arguments(model)
    names = [
        name
        for name in ("input_ids", "attention_mask", "token_type_ids")
        if name in encoded and name in accepted
    ]
    args = tuple(encoded[name] for name in names)
    output_attr = spec.output_attr

    class TextWrapper(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *tensors):
            outputs = self.inner(**dict(zip(names, tensors, strict=True)))
            return getattr(outputs, output_attr)

    return TextWrapper(model).eval(), names, args, [output_attr], model.config


def _build_text_frozen(spec: ModelSpec):
    """Text model with its token ids baked in, exposing only a float mask.

    A hosted profiler generates its own inputs and cannot be given real ones. For a
    float tensor that is harmless; for an *index* it is not, because a random int64 is
    not a token id and the embedding Gather goes out of bounds. Every text model was
    therefore unprofilable on hosted hardware — which is why the fleet finding covers
    vision only.

    The fix is to stop asking the profiler for indices. Token ids become constants in
    the graph, and the one remaining input is the attention mask as float32, which
    genuinely feeds attention (so the exporter cannot prune it) and has no valid range
    to fall outside of.

    **What this changes and does not change.** The work is identical: the same
    embedding lookup runs, over the same sequence length, through the same layers —
    latency does not depend on *which* tokens. What changes is that the mask holds
    random values rather than ones, so the output is numerically meaningless. That
    costs nothing here, because hosted rows already record
    `output_cosine_vs_reference` as unavailable: a profile job returns timings only.

    Kept as a separate artifact and a separate recipe axis rather than a quiet
    substitution, so a frozen-token row can never be mistaken for a normal one.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = _load_hf_model(spec)
    accepted = _accepted_arguments(model)
    if "attention_mask" not in accepted:
        raise ExportError(
            f"{spec.hf_id} does not accept `attention_mask`, so there is no float input "
            "to expose once token ids are frozen. A graph with no inputs cannot be "
            "profiled by a service that generates its own."
        )

    encoded = tokenizer(
        _CALIBRATION_TEXT,
        return_tensors="pt",
        padding="max_length",
        max_length=spec.static_shape["sequence"],
        truncation=True,
    )
    frozen = [
        name
        for name in ("input_ids", "token_type_ids")
        if name in encoded and name in accepted
    ]
    output_attr = spec.output_attr

    class FrozenTextWrapper(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner
            for name in frozen:
                # A buffer, so torch.onnx.export emits it as an initializer rather
                # than as a graph input.
                self.register_buffer(f"frozen__{name}", encoded[name])

        def forward(self, attention_mask):
            kwargs = {name: getattr(self, f"frozen__{name}") for name in frozen}
            kwargs["attention_mask"] = attention_mask
            return getattr(self.inner(**kwargs), output_attr)

    mask = encoded["attention_mask"].to(torch.float32)
    return FrozenTextWrapper(model).eval(), ["attention_mask"], (mask,), [output_attr], model.config


def _build_vision(spec: ModelSpec):
    """Vision model fed a deterministic synthetic image.

    A fixed pseudo-random image rather than a photograph: the numbers must be
    reproducible by anyone without shipping image assets, and a vision
    transformer's latency does not depend on image content.
    """
    import torch  # noqa: PLC0415

    model = _load_hf_model(spec)
    generator = torch.Generator().manual_seed(0)
    pixel_values = torch.rand(
        (
            spec.static_shape["batch"],
            spec.static_shape["channels"],
            spec.static_shape["height"],
            spec.static_shape["width"],
        ),
        generator=generator,
    )
    output_attr = spec.output_attr

    class VisionWrapper(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return getattr(self.inner(pixel_values=pixel_values), output_attr)

    return (
        VisionWrapper(model).eval(),
        ["pixel_values"],
        (pixel_values,),
        [output_attr],
        model.config,
    )


_BUILDERS = {"text": _build_text, "vision": _build_vision}


def export_onnx(
    spec: ModelSpec,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    opset: int = DEFAULT_OPSET,
    static_shapes: bool = True,
    frozen_tokens: bool = False,
    force: bool = False,
) -> ExportedArtifact:
    """Export to ONNX, capturing inputs and the fp32 reference output.

    ``static_shapes=False`` marks batch and sequence as dynamic. That is not just
    a convenience knob: dynamic shape is the single biggest reason a delegate
    declines to claim a subgraph, so the two variants are a controlled experiment
    in silent fallback.

    torch is imported only on a cache miss, so replaying an already-exported
    artifact needs nothing but the thin runtime dependency set.
    """
    key = artifact_key(spec, opset, static_shapes, frozen_tokens)
    suffix = "__frozen" if frozen_tokens else ""
    directory = Path(artifact_root) / f"{spec.slug}__{key}{suffix}"
    model_path = directory / "model.onnx"
    inputs_path = directory / "inputs.npz"
    reference_path = directory / "reference.npz"
    meta_path = directory / "meta.json"

    if not force and all(p.exists() for p in (model_path, inputs_path, reference_path, meta_path)):
        meta = json.loads(meta_path.read_text())
        return ExportedArtifact(
            directory=directory,
            model_path=model_path,
            inputs_path=inputs_path,
            reference_path=reference_path,
            lowering_ms=float(meta["lowering_ms"]),
            was_cached=True,
        )

    import torch

    if frozen_tokens and spec.exporter != "text":
        raise ExportError(
            f"frozen token inputs only apply to text models; {spec.hf_id} uses the "
            f"{spec.exporter!r} harness, whose inputs are already float."
        )
    builder = _build_text_frozen if frozen_tokens else _BUILDERS.get(spec.exporter)
    if builder is None:
        raise ValueError(f"no exporter named {spec.exporter!r} for {spec.ref}")

    directory.mkdir(parents=True, exist_ok=True)
    wrapper, input_names, args, output_names, hf_config = builder(spec)

    dynamic_axes = None
    if not static_shapes:
        dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
        for name in input_names:
            if name != "pixel_values":
                dynamic_axes[name][1] = "sequence"

    started = time.perf_counter()
    torch.onnx.export(
        wrapper,
        args,
        str(model_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamo=False,
        dynamic_axes=dynamic_axes,
    )
    lowering_ms = (time.perf_counter() - started) * 1000.0

    # The graph is the authority on what a session will accept, not our intent.
    # torch.onnx.export prunes inputs the traced module never used, so a name we
    # asked for can silently vanish — and every recipe then dies at session creation
    # with `Invalid input name`, eleven identical failures for one export defect.
    # Reconciling here turns that into one error, at the place that caused it.
    tensors = dict(zip(input_names, args, strict=True))
    graph_inputs = [value.name for value in onnx.load(str(model_path)).graph.input]
    missing = [name for name in graph_inputs if name not in tensors]
    if missing:
        raise ExportError(
            f"{spec.hf_id} exported a graph expecting {missing}, which the input "
            f"harness did not build. Built: {sorted(tensors)}."
        )
    np.savez(inputs_path, **{name: tensors[name].numpy() for name in graph_inputs})
    input_names = graph_inputs
    with torch.no_grad():
        reference = wrapper(*args)
    np.savez(reference_path, **{output_names[0]: reference.numpy()})

    meta_path.write_text(
        json.dumps(
            {
                "ref": spec.ref,
                "hf_id": spec.hf_id,
                "task": str(spec.task),
                "opset": opset,
                "static_shapes": static_shapes,
                "input_names": input_names,
                "output_names": output_names,
                "lowering_ms": lowering_ms,
                "torch_version": torch.__version__,
                **architecture_from_config(hf_config),
            },
            indent=2,
        )
    )

    return ExportedArtifact(
        directory=directory,
        model_path=model_path,
        inputs_path=inputs_path,
        reference_path=reference_path,
        lowering_ms=lowering_ms,
        was_cached=False,
    )
