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

from edgefit.models.registry import ModelSpec
from edgefit.schema.common import content_hash

DEFAULT_ARTIFACT_ROOT = Path("artifacts/onnx")
DEFAULT_OPSET = 17

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
        """Total on-disk size — external weight files count (hard rule #4)."""
        return sum(p.stat().st_size for p in self.directory.glob("*.onnx*"))

    def load_inputs(self) -> dict[str, np.ndarray]:
        with np.load(self.inputs_path) as data:
            return {name: data[name] for name in data.files}

    def load_reference(self) -> dict[str, np.ndarray]:
        with np.load(self.reference_path) as data:
            return {name: data[name] for name in data.files}


def artifact_key(spec: ModelSpec, opset: int, static_shapes: bool) -> str:
    return content_hash(
        {
            "hf_id": spec.hf_id,
            "exporter": spec.exporter,
            "opset": opset,
            "static_shapes": static_shapes,
            "shape": spec.static_shape,
        }
    )


def _build_encoder(spec: ModelSpec):
    """Text encoder: wrap in an explicit positional-arg module.

    transformers>=5 rejects the dict-of-kwargs form the TorchScript tracer emits
    ("got multiple values for argument 'use_cache'"), so the wrapper is load
    bearing, not cosmetic. It also pins the input names in the exported graph.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModel.from_pretrained(spec.hf_id).eval()
    encoded = tokenizer(
        _CALIBRATION_TEXT,
        return_tensors="pt",
        padding="max_length",
        max_length=spec.static_shape["sequence"],
        truncation=True,
    )
    names = [name for name in ("input_ids", "attention_mask", "token_type_ids") if name in encoded]
    args = tuple(encoded[name] for name in names)

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *tensors):
            outputs = self.inner(**dict(zip(names, tensors, strict=True)))
            return outputs.last_hidden_state

    return EncoderWrapper(model).eval(), names, args, ["last_hidden_state"]


def _build_vision(spec: ModelSpec):
    """Vision encoder fed a deterministic synthetic image.

    A fixed pseudo-random image rather than a real photograph: the numbers must
    be reproducible by anyone without shipping image assets, and a vision
    transformer's latency does not depend on image content.
    """
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(spec.hf_id).eval()
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

    class VisionWrapper(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).last_hidden_state

    return VisionWrapper(model).eval(), ["pixel_values"], (pixel_values,), ["last_hidden_state"]


_BUILDERS = {"encoder": _build_encoder, "vision": _build_vision}


def export_onnx(
    spec: ModelSpec,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    opset: int = DEFAULT_OPSET,
    static_shapes: bool = True,
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
    directory = Path(artifact_root) / f"{spec.slug}__{artifact_key(spec, opset, static_shapes)}"
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

    builder = _BUILDERS.get(spec.exporter)
    if builder is None:
        raise ValueError(f"no exporter named {spec.exporter!r} for {spec.ref}")

    directory.mkdir(parents=True, exist_ok=True)
    wrapper, input_names, args, output_names = builder(spec)

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

    np.savez(
        inputs_path,
        **{name: tensor.numpy() for name, tensor in zip(input_names, args, strict=True)},
    )
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
