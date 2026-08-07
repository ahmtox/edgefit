"""Resolve the artifact a recipe describes.

The invariant this module exists to hold: **a recipe fully determines its
artifact.** The recipe names the model, the opset, whether shapes are static, and
the quantization scheme, so ``recipe_id -> artifact`` is a function. That is what
lets the sweep runner cache aggressively and resume after the laptop lid closes,
and it is why ``LoweringConfig`` lives on the recipe rather than being a CLI flag.

Artifacts are content-addressed and derived lazily:

    fp32 base export  ->  quantized variant  ->  measured
    (torch, once)         (ORT, seconds)         (per recipe)

torch is only imported on a base-export cache miss, so replaying an existing
artifact stays inside the thin runtime dependency set (PROJECT.md §8, Tier 3).
"""

from __future__ import annotations

import json
from pathlib import Path

from edgefit.backends.export_decoder import UnsupportedDecoderLowering, export_decoder
from edgefit.backends.export_onnx import (
    DEFAULT_ARTIFACT_ROOT,
    ExportedArtifact,
    artifact_key,
    export_onnx,
)
from edgefit.backends.quantize import (
    UnsupportedQuantizationError,
    copy_harness_inputs,
    quantize_artifact,
    variant_key,
)
from edgefit.models.registry import ModelSpec
from edgefit.schema.recipe import Recipe

__all__ = [
    "UnsupportedDecoderLowering",
    "UnsupportedQuantizationError",
    "resolve_artifact",
]


def recipe_applicability(spec: ModelSpec, recipe: Recipe) -> str | None:
    """Why this recipe cannot legally apply to this model, or None if it can.

    PROJECT.md §5.4 generates the recipe space from "legal recipes given task type +
    fingerprint + targets", so an illegal pair was never a candidate. Recording it as
    a *failure* would conflate "we tried and it broke" with "this was never a legal
    combination", and the corpus is for measurements.
    """
    if spec.exporter == "decoder":
        if recipe.lowering.static_shapes:
            return (
                "static shapes cannot express a KV-cached decoder: the cache grows one "
                "token per step, so the sequence axis is dynamic by construction"
            )
        if recipe.quantization is not None:
            return (
                "quantizing a KV-cached decoder is not implemented; the cache tensors "
                "need their own precision policy"
            )
    elif not recipe.lowering.static_shapes:
        return None
    return None


def resolve_artifact(
    spec: ModelSpec,
    recipe: Recipe,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    force: bool = False,
) -> ExportedArtifact:
    """Produce (or reuse) the artifact this recipe describes.

    Raises ``UnsupportedQuantizationError`` when the recipe asks for a scheme this
    backend cannot express — the caller records that as a ``lowering_failure``
    rather than measuring the wrong model under the right label.
    """
    root = Path(artifact_root)
    if spec.exporter == "decoder":
        # A KV-cached decoder is a different export shape, and quantizing one is not
        # implemented — the cache tensors would need their own scheme.
        if recipe.quantization is not None:
            raise UnsupportedQuantizationError(
                "quantizing a KV-cached decoder is not implemented: the cache tensors "
                "need their own precision policy, and applying weight-only "
                "quantization while leaving an fp32 cache would misreport both size "
                "and bandwidth"
            )
        return export_decoder(
            spec,
            artifact_root=root,
            opset=recipe.lowering.opset,
            static_shapes=recipe.lowering.static_shapes,
            force=force,
        )

    base = export_onnx(
        spec,
        artifact_root=root,
        opset=recipe.lowering.opset,
        static_shapes=recipe.lowering.static_shapes,
        frozen_tokens=recipe.lowering.frozen_token_inputs,
        force=force,
    )
    if recipe.quantization is None:
        return base

    base_key = artifact_key(
        spec,
        recipe.lowering.opset,
        recipe.lowering.static_shapes,
        recipe.lowering.frozen_token_inputs,
    )
    key = variant_key(base_key, recipe.quantization)
    directory = root / f"{spec.slug}__q{key}"
    model_path = directory / "model.onnx"
    meta_path = directory / "meta.json"

    if not force and model_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return ExportedArtifact(
            directory=directory,
            model_path=model_path,
            inputs_path=directory / "inputs.npz",
            reference_path=directory / "reference.npz",
            # Lowering cost is the *whole* pipeline: base export plus quantization
            # (hard rule #4 — users experience pipelines, not kernels).
            lowering_ms=float(meta["lowering_ms"]),
            was_cached=True,
        )

    quantize_ms = quantize_artifact(base.model_path, directory, recipe.quantization)
    copy_harness_inputs(base.directory, directory)

    # Carry the base export's architecture facts into the variant, so a quantized
    # artifact still fingerprints with an exact attention label.
    base_meta = json.loads((base.directory / "meta.json").read_text())
    architecture = {
        key: base_meta[key] for key in ("layers", "n_heads", "kv_heads") if key in base_meta
    }

    total_ms = quantize_ms + (0.0 if base.was_cached else base.lowering_ms)
    meta_path.write_text(
        json.dumps(
            {
                "ref": spec.ref,
                "base_key": base_key,
                "variant_key": key,
                "quantization": recipe.quantization.model_dump(mode="json"),
                "quantize_ms": quantize_ms,
                "lowering_ms": total_ms,
                **architecture,
            },
            indent=2,
        )
    )
    return ExportedArtifact(
        directory=directory,
        model_path=model_path,
        inputs_path=directory / "inputs.npz",
        reference_path=directory / "reference.npz",
        lowering_ms=total_ms,
        was_cached=False,
    )
