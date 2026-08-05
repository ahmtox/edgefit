"""Duplicate initializers — bytes an export ships twice (PROJECT.md §4 Stage 2.3).

Found by hand first. The fingerprint reported 1498.5M graph parameters for
Llama-3.2-1B against a card value near 1236M; the 262.5M difference was exactly one
embedding matrix (128256 x 2048), present twice because ``torch.onnx.export`` un-ties
tied embeddings. In fp32 that is ~1.05 GB of a 5.6 GB artifact — about 19% — on a
device where artifact size is a shipping constraint and an OTA budget.

Those bytes are genuinely in the file, so the fingerprint is right to count them.
What was missing is the *diagnosis*, and it generalises: "your export shipped 1 GB of
weights twice" is precisely the unasked-for warning §4 Stage 2.3 describes — the kind
a team optimizing by hand would not think to look for.

**Hashing alone does not find it.** Writing this detector as a content hash and
pointing it at the artifact that motivated it returned nothing, because the two copies
are ``(128256, 2048)`` and ``(2048, 128256)`` — the embedding and its transpose,
materialised for the LM head's MatMul. Same values, different bytes. So there are two
relations worth reporting, and the cheap one would have missed the whole finding:

* ``identical`` — byte-for-byte. Found by hashing; one copy is simply redundant.
* ``transpose`` — a 2-D weight and its transpose. Costs the same bytes, but the fix is
  different: the consumer must transpose at load, or the graph must use the other
  operand order.

Content is compared, never names. Two initializers are related because of their
values, not because one is called ``embed_tokens.weight`` and the other
``onnx::MatMul_5028``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto
from onnx.external_data_helper import ExternalDataInfo, uses_external_data

#: Bumped when the detector's *answer* could change for an unchanged model.
#: v2 adds transpose detection, without which the case that motivated the detector —
#: an un-tied embedding shipped alongside its transpose — reports clean.
DUPLICATE_DETECTOR_VERSION = 2

#: Groups below this are noise — a shared scalar or a repeated small bias costs
#: nothing to ship and reporting it would bury the finding that matters.
MIN_REPORTED_BYTES = 1024 * 1024

_CHUNK = 1024 * 1024

#: Rows of the transposed candidate compared per step, so a 1 GB pair is walked in
#: bounded memory rather than materialised twice.
_TRANSPOSE_ROWS = 256

#: Element probes before a transpose pair is verified in full. A mismatched pair almost
#: always fails within the first few, which keeps the expensive walk for real findings.
_TRANSPOSE_PROBES = 64


class DuplicateRelation(StrEnum):
    """How two initializers hold the same information."""

    IDENTICAL = "identical"
    TRANSPOSE = "transpose"


@dataclass(frozen=True)
class DuplicateGroup:
    """Initializers holding the same values, and how."""

    names: tuple[str, ...]
    dtype: str
    dims: tuple[int, ...]
    bytes_each: int
    relation: DuplicateRelation = DuplicateRelation.IDENTICAL

    @property
    def copies(self) -> int:
        return len(self.names)

    @property
    def wasted_bytes(self) -> int:
        """What could be dropped. One copy is legitimate; the rest are not."""
        return self.bytes_each * (self.copies - 1)


@dataclass(frozen=True)
class DuplicateWeightReport:
    detector_version: int
    initializer_bytes: int
    groups: tuple[DuplicateGroup, ...]

    @property
    def wasted_bytes(self) -> int:
        return sum(group.wasted_bytes for group in self.groups)

    @property
    def wasted_fraction(self) -> float | None:
        """Share of initializer bytes that is duplicated, or None if there are none.

        Deliberately a fraction of *initializer* bytes rather than of file size: the
        graph itself, and any harness sidecars, are not weights, and quoting a
        percentage of the wrong denominator is the mistake this project keeps
        catching elsewhere.
        """
        if self.initializer_bytes <= 0:
            return None
        return self.wasted_bytes / self.initializer_bytes

    @property
    def has_findings(self) -> bool:
        return bool(self.groups)


def _dims(tensor: TensorProto) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.dims)


def _external_path(tensor: TensorProto, base_dir: Path) -> tuple[Path, int, int] | None:
    """Where this tensor's bytes live on disk, if they live outside the graph."""
    if not uses_external_data(tensor):
        return None
    info = ExternalDataInfo(tensor)
    offset = int(info.offset or 0)
    path = base_dir / str(info.location)
    # No recorded length means "to the end of the file", per the ONNX spec.
    length = int(info.length) if info.length else max(path.stat().st_size - offset, 0)
    return path, offset, length


def _byte_length(tensor: TensorProto, base_dir: Path) -> int:
    located = _external_path(tensor, base_dir)
    if located is not None:
        return located[2]
    if tensor.HasField("raw_data"):
        return len(tensor.raw_data)
    # A typed field (float_data and friends). Materialise it — these are small by
    # construction, since anything large is written as raw or external data.
    return int(onnx.numpy_helper.to_array(tensor).nbytes)


def _content_digest(tensor: TensorProto, base_dir: Path) -> str:
    """Hash the tensor's bytes without holding the whole tensor in memory."""
    digest = hashlib.blake2b(digest_size=16)
    located = _external_path(tensor, base_dir)
    if located is not None:
        path, offset, length = located
        with path.open("rb") as handle:
            handle.seek(offset)
            remaining = length
            while remaining > 0:
                block = handle.read(min(_CHUNK, remaining))
                if not block:
                    break
                digest.update(block)
                remaining -= len(block)
        return digest.hexdigest()
    if tensor.HasField("raw_data"):
        digest.update(tensor.raw_data)
        return digest.hexdigest()
    digest.update(onnx.numpy_helper.to_array(tensor).tobytes())
    return digest.hexdigest()


def _as_array(tensor: TensorProto, base_dir: Path) -> np.ndarray | None:
    """View the tensor's values without copying them into memory.

    ``None`` means the bytes cannot be viewed as a plain buffer — a string tensor, or
    a dtype numpy has no equivalent for. Those are skipped rather than guessed at.
    """
    try:
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
    except (KeyError, TypeError, ValueError):
        return None
    if dtype.hasobject:
        return None
    dims = _dims(tensor)

    located = _external_path(tensor, base_dir)
    if located is not None:
        path, offset, length = located
        if length != int(np.prod(dims)) * dtype.itemsize:
            return None
        return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=dims)
    if tensor.HasField("raw_data"):
        return np.frombuffer(tensor.raw_data, dtype=dtype).reshape(dims)
    return onnx.numpy_helper.to_array(tensor)


def _is_transpose(left: np.ndarray, right: np.ndarray) -> bool:
    """Whether ``right`` holds ``left`` transposed.

    Probed before it is proved. A handful of element reads rejects unrelated weights
    of complementary shape immediately, so the full walk — which touches every byte of
    both tensors — only happens for pairs that are about to be reported.
    """
    rows, cols = left.shape
    rng = np.random.default_rng(0)  # fixed: the detector must answer the same twice
    probes = min(_TRANSPOSE_PROBES, rows * cols)
    for index in rng.choice(rows * cols, size=probes, replace=False):
        row, col = divmod(int(index), cols)
        if left[row, col] != right[col, row]:
            return False

    for start in range(0, rows, _TRANSPOSE_ROWS):
        stop = min(start + _TRANSPOSE_ROWS, rows)
        if not np.array_equal(left[start:stop, :], right[:, start:stop].T):
            return False
    return True


def _transpose_groups(
    buckets: dict[tuple[int, tuple[int, ...]], list[TensorProto]],
    base_dir: Path,
    min_bytes: int,
) -> list[DuplicateGroup]:
    """Pair 2-D buckets of complementary shape and report those that really transpose."""
    groups: list[DuplicateGroup] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for key, tensors in buckets.items():
        data_type, dims = key
        if len(dims) != 2 or dims[0] == dims[1] or key in seen:
            continue
        mirror = (data_type, (dims[1], dims[0]))
        if mirror not in buckets:
            continue
        seen.update({key, mirror})

        for left in tensors:
            bytes_each = _byte_length(left, base_dir)
            if bytes_each < min_bytes:
                continue
            left_array = _as_array(left, base_dir)
            if left_array is None:
                continue
            for right in buckets[mirror]:
                right_array = _as_array(right, base_dir)
                if right_array is None:
                    continue
                if _is_transpose(left_array, right_array):
                    groups.append(
                        DuplicateGroup(
                            names=tuple(sorted((left.name, right.name))),
                            dtype=TensorProto.DataType.Name(data_type).lower(),
                            dims=dims,
                            bytes_each=bytes_each,
                            relation=DuplicateRelation.TRANSPOSE,
                        )
                    )
    return groups


def find_duplicate_initializers(
    model_path: Path, *, min_bytes: int = MIN_REPORTED_BYTES
) -> DuplicateWeightReport:
    """Report weights shipped twice in one artifact, byte-identical or transposed.

    Only tensors that *could* be related are read. Candidates are bucketed by
    (dtype, shape), which is free and prunes almost everything: same-bucket tensors are
    hashed for identity, and buckets of complementary 2-D shape are checked for
    transposition. On a model with no findings this reads a few megabytes rather than
    the whole artifact.
    """
    base_dir = model_path.parent
    # External data stays on disk — the point is to hash it in slices, not load it.
    model = onnx.load(str(model_path), load_external_data=False)

    buckets: dict[tuple[int, tuple[int, ...]], list[TensorProto]] = {}
    initializer_bytes = 0
    for tensor in model.graph.initializer:
        initializer_bytes += _byte_length(tensor, base_dir)
        buckets.setdefault((tensor.data_type, _dims(tensor)), []).append(tensor)

    groups: list[DuplicateGroup] = []
    for (data_type, dims), tensors in buckets.items():
        if len(tensors) < 2:
            continue
        by_digest: dict[str, list[TensorProto]] = {}
        for tensor in tensors:
            by_digest.setdefault(_content_digest(tensor, base_dir), []).append(tensor)
        for matched in by_digest.values():
            if len(matched) < 2:
                continue
            bytes_each = _byte_length(matched[0], base_dir)
            if bytes_each * (len(matched) - 1) < min_bytes:
                continue
            groups.append(
                DuplicateGroup(
                    names=tuple(sorted(tensor.name for tensor in matched)),
                    dtype=TensorProto.DataType.Name(data_type).lower(),
                    dims=dims,
                    bytes_each=bytes_each,
                )
            )

    groups.extend(_transpose_groups(buckets, base_dir, min_bytes))
    groups.sort(key=lambda group: (-group.wasted_bytes, group.names))
    return DuplicateWeightReport(
        detector_version=DUPLICATE_DETECTOR_VERSION,
        initializer_bytes=initializer_bytes,
        groups=tuple(groups),
    )
