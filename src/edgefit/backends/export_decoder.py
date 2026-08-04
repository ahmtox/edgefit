"""Export a decoder-only LM to ONNX with KV cache I/O.

Generative inference has two phases with completely different cost profiles
(PROJECT.md §2.3): **prefill** consumes the whole prompt in one compute-bound pass
and produces the first token — that is TTFT — and **decode** then emits one token at
a time, bound by memory bandwidth as it re-reads the cache and the weights. A
harness that cannot tell them apart cannot report either honestly.

Getting the cache into the graph is the whole difficulty:

* Without KV cache I/O, each decode step would re-process the entire sequence, so
  decode cost would grow quadratically and the reported tok/s would be a fiction.
  Exporting the cache is not an optimisation, it is a correctness requirement.
* transformers 5 removed ``Cache.to_legacy_cache`` / ``from_legacy_cache``, so the
  tuple bridge every ONNX decoder export used to rely on is gone. The replacement
  is ``DynamicCache(ddp_cache_data=[(k, v), …])`` in and ``cache.layers[i].keys /
  .values`` out, which does trace.
* **One graph serves both phases.** Prefill is simply a decode step whose past has
  length zero, which avoids maintaining two graphs that can drift apart.
* ``position_ids`` must be an **explicit graph input**. Left implicit, the model
  derives rotary positions from tensor shapes, and the tracer bakes in whatever
  those were at export time — so every decode step gets prefill's positions. The
  model still runs and still emits plausible text; it just emits *different* text.
  Token agreement caught this at 25% where a float tolerance would have passed it.

Validated by token equality, not tolerance: greedy decoding through the exported
graph must produce the identical token sequence to PyTorch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgefit.backends.export_onnx import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_OPSET,
    HARNESS_SIDECARS,
    ExportedArtifact,
)
from edgefit.models.registry import ModelSpec
from edgefit.schema.common import content_hash

#: Fixed prompt so every measurement and the reference share one input.
PROMPT = "Explain in one sentence why on-device inference is hard."

#: Tokens greedily decoded for the reference sequence and the harness default.
REFERENCE_TOKENS = 16

#: v2 made position_ids an explicit graph input. Part of the artifact key: without
#: it, the fixed exporter happily reuses the broken cached graph, and two
#: measurements of two different graphs share one identity in the corpus.
DECODER_EXPORTER_VERSION = 2


class UnsupportedDecoderLowering(Exception):
    """The requested lowering cannot express a KV-cached decoder."""


@dataclass(frozen=True)
class DecoderShape:
    """Everything the harness needs to build past-key/value tensors."""

    layers: int
    kv_heads: int
    head_dim: int
    prompt_length: int

    def empty_past(self, batch: int = 1) -> dict[str, np.ndarray]:
        return {
            name: np.zeros((batch, self.kv_heads, 0, self.head_dim), dtype=np.float32)
            for name in self.past_names
        }

    @property
    def past_names(self) -> list[str]:
        return [f"past.{i}.{t}" for i in range(self.layers) for t in ("key", "value")]

    @property
    def present_names(self) -> list[str]:
        return [f"present.{i}.{t}" for i in range(self.layers) for t in ("key", "value")]


def decoder_shape(meta: dict) -> DecoderShape:
    return DecoderShape(
        layers=int(meta["layers"]),
        kv_heads=int(meta["kv_heads"]),
        head_dim=int(meta["head_dim"]),
        prompt_length=int(meta["prompt_length"]),
    )


def artifact_key(spec: ModelSpec, opset: int) -> str:
    return content_hash(
        {
            "hf_id": spec.hf_id,
            "exporter": "decoder",
            "exporter_version": DECODER_EXPORTER_VERSION,
            "opset": opset,
            "prompt": PROMPT,
            "reference_tokens": REFERENCE_TOKENS,
        }
    )


def export_decoder(
    spec: ModelSpec,
    artifact_root: Path | str = DEFAULT_ARTIFACT_ROOT,
    opset: int = DEFAULT_OPSET,
    static_shapes: bool = False,
    force: bool = False,
) -> ExportedArtifact:
    """Export ``spec`` with KV cache inputs and outputs.

    ``static_shapes=True`` is rejected rather than silently ignored. A cache that
    grows one token per step needs a dynamic sequence axis, and that tension is real
    rather than incidental: NPU delegates want static shapes, which is exactly why
    production mobile LLM stacks preallocate a fixed-size cache buffer and pass a
    position index instead. We do not implement that yet, so we say so.
    """
    if static_shapes:
        raise UnsupportedDecoderLowering(
            "a KV-cached decoder export needs a dynamic sequence axis, because the "
            "cache grows by one token per decode step. Fixed-size cache buffers with "
            "a position index — what mobile NPU stacks actually ship — are not "
            "implemented yet. Use static_shapes: false for generative recipes."
        )

    directory = Path(artifact_root) / f"{spec.slug}__dec{artifact_key(spec, opset)}"
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

    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=torch.float32).eval()
    config = model.config

    layers = int(config.num_hidden_layers)
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(
        getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    )

    encoded = tokenizer(PROMPT, return_tensors="pt")
    prompt_ids = encoded["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    shape = DecoderShape(layers, kv_heads, head_dim, prompt_length)

    class DecoderStep(torch.nn.Module):
        """One forward pass: prefill when past is empty, decode otherwise."""

        def __init__(self, inner, layer_count: int) -> None:
            super().__init__()
            self.inner = inner
            self.layer_count = layer_count

        def forward(self, input_ids, attention_mask, position_ids, *past):
            cache = DynamicCache(
                ddp_cache_data=[(past[2 * i], past[2 * i + 1]) for i in range(self.layer_count)]
            )
            outputs = self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            present: list[torch.Tensor] = []
            for layer in outputs.past_key_values.layers:
                present += [layer.keys, layer.values]
            return (outputs.logits, *present)

    wrapper = DecoderStep(model, layers).eval()
    empty = [torch.zeros(1, kv_heads, 0, head_dim) for _ in range(2 * layers)]
    prompt_positions = torch.arange(prompt_length, dtype=torch.long).unsqueeze(0)
    args = (
        prompt_ids,
        torch.ones(1, prompt_length, dtype=torch.long),
        prompt_positions,
        *empty,
    )

    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {1: "seq"},
        "attention_mask": {1: "total"},
        "position_ids": {1: "seq"},
        "logits": {1: "seq"},
    }
    for name in shape.past_names:
        dynamic_axes[name] = {2: "past_len"}
    for name in shape.present_names:
        dynamic_axes[name] = {2: "total"}

    started = time.perf_counter()
    torch.onnx.export(
        wrapper,
        args,
        str(model_path),
        input_names=["input_ids", "attention_mask", "position_ids", *shape.past_names],
        output_names=["logits", *shape.present_names],
        opset_version=opset,
        dynamo=False,
        dynamic_axes=dynamic_axes,
    )
    lowering_ms = (time.perf_counter() - started) * 1000.0

    # A model past the 2 GB protobuf limit spills weights into external files, and
    # torch writes one per tensor — ~150 files named after weights. Consolidate into
    # a single blob so the artifact is one graph plus one data file, which is what a
    # deployment pipeline expects to move around.
    _consolidate_external_data(model_path)

    np.savez(
        inputs_path,
        input_ids=prompt_ids.numpy().astype(np.int64),
        attention_mask=np.ones((1, prompt_length), dtype=np.int64),
        position_ids=prompt_positions.numpy().astype(np.int64),
    )

    # Reference is the *token sequence*, not logits. Token equality is a far
    # stronger known-answer test than a tolerance on floats: it either decodes the
    # same text or it does not.
    reference_tokens = _greedy_reference(model, prompt_ids, REFERENCE_TOKENS)
    np.savez(reference_path, tokens=np.asarray(reference_tokens, dtype=np.int64))

    meta_path.write_text(
        json.dumps(
            {
                "ref": spec.ref,
                "hf_id": spec.hf_id,
                "task": str(spec.task),
                "opset": opset,
                "static_shapes": False,
                "layers": layers,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "prompt": PROMPT,
                "prompt_length": prompt_length,
                "reference_tokens": reference_tokens,
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


def _consolidate_external_data(model_path: Path) -> None:
    """Rewrite a multi-file external-data export as ``model.onnx`` + one data blob."""
    import onnx  # noqa: PLC0415

    directory = model_path.parent
    before = {p.name for p in directory.iterdir() if p.is_file()}
    model = onnx.load(str(model_path))  # loads external data into memory
    for name in before - {model_path.name} - HARNESS_SIDECARS:
        (directory / name).unlink()
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{model_path.name}.data",
        size_threshold=1024,
    )


def _greedy_reference(model, prompt_ids, count: int) -> list[int]:
    """Greedy-decode ``count`` tokens in fp32 PyTorch."""
    import torch

    tokens: list[int] = []
    ids, cache = prompt_ids.clone(), None
    with torch.no_grad():
        for _ in range(count):
            outputs = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            nxt = int(outputs.logits[0, -1].argmax())
            tokens.append(nxt)
            ids = torch.tensor([[nxt]])
    return tokens
