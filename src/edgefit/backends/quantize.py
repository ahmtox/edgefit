"""Derive a quantized ONNX artifact from an fp32 base (PROJECT.md §6.1).

Quantization is where most of the deployment win — and most of the accuracy loss —
comes from, so it has to be a live axis rather than a footnote. The variants
implemented here are exactly the ones that cost nothing to explore:

* **int8/uint8 dynamic** — activations quantized at run time, so **no calibration
  data is required**. That is what makes a breadth sweep affordable today; static
  activation quantization needs a calibration set and waits for that machinery.
* **fp16** — the Apple Neural Engine is fp16-native, so this is not an exotic
  choice on this hardware; it is arguably the default a team should try first.

Anything not implemented raises ``UnsupportedQuantizationError``, which the runner
records as a ``lowering_failure`` rather than silently measuring the fp32 model and
labelling the row int4. A recipe that quietly did not apply is the worst possible
corpus entry: it looks like evidence and it is noise.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from edgefit.schema.common import ActivationQuant, Dtype, Granularity, content_hash
from edgefit.schema.recipe import QuantizationConfig


class UnsupportedQuantizationError(Exception):
    """This backend cannot express the requested quantization."""


def variant_key(base_key: str, quant: QuantizationConfig) -> str:
    """Key for a quantized variant.

    Derived from the base key, so bumping the base exporter's version invalidates
    every variant built from it — including their meta.json sidecars, which inherit
    the base export's architecture facts.
    """
    return content_hash({"base": base_key, "quant": quant.model_dump(mode="json")})


def _quantize_int(source: Path, destination: Path, quant: QuantizationConfig) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    weight_types = {Dtype.INT8: QuantType.QInt8, Dtype.UINT8: QuantType.QUInt8}
    weight_type = weight_types.get(quant.weight_dtype)
    if weight_type is None:
        raise UnsupportedQuantizationError(
            f"ORT dynamic quantization supports int8/uint8 weights, not {quant.weight_dtype}"
        )
    if quant.activation_quant is not ActivationQuant.DYNAMIC:
        raise UnsupportedQuantizationError(
            f"integer weights need dynamic activation quantization for this path, "
            f"got {quant.activation_quant}. Static activation quant requires calibration data."
        )
    if quant.weight_granularity is Granularity.BLOCKWISE:
        raise UnsupportedQuantizationError(
            "blockwise weight quantization is not implemented for the ORT backend yet"
        )

    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=weight_type,
        per_channel=quant.weight_granularity is Granularity.PER_CHANNEL,
        extra_options={"MatMulConstBOnly": True},
    )


def _quantize_fp16(source: Path, destination: Path) -> None:
    import onnx  # noqa: PLC0415
    from onnxconverter_common import float16  # noqa: PLC0415

    model = onnx.load(str(source))
    # keep_io_types: the harness feeds fp32 inputs and compares fp32 outputs, so
    # the boundary stays fp32 and only the interior converts. Changing the I/O
    # dtype would make the numerics comparison against the reference meaningless.
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, str(destination))


def quantize_artifact(
    source_model: Path,
    destination_dir: Path,
    quant: QuantizationConfig,
) -> float:
    """Write a quantized ``model.onnx`` into ``destination_dir``. Returns elapsed ms.

    Raises ``UnsupportedQuantizationError`` for anything this backend cannot
    actually express.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "model.onnx"

    started = time.perf_counter()
    if quant.weight_dtype is Dtype.FP16:
        if quant.activation_quant is not ActivationQuant.NONE:
            raise UnsupportedQuantizationError(
                "fp16 conversion does not take an activation quantization scheme"
            )
        _quantize_fp16(source_model, destination)
    elif quant.weight_dtype in (Dtype.INT8, Dtype.UINT8):
        _quantize_int(source_model, destination, quant)
    else:
        raise UnsupportedQuantizationError(
            f"no ORT quantization path for weight dtype {quant.weight_dtype}"
        )
    return (time.perf_counter() - started) * 1000.0


def copy_harness_inputs(base_dir: Path, destination_dir: Path) -> None:
    """Carry the pinned inputs and the fp32 reference into the variant.

    The reference stays the *fp32 PyTorch* output on purpose: numerics degradation
    is only meaningful against the unquantized truth, not against another
    quantized model.
    """
    for name in ("inputs.npz", "reference.npz"):
        shutil.copy2(base_dir / name, destination_dir / name)
