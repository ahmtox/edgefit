"""EdgeFit — deployment compiler for on-device AI.

Pass 1 is the measurement harness: ``(model, recipe, device) -> measurement``.
"""

# Read from installed metadata rather than hardcoded. The two drifted immediately:
# pyproject went to 0.1.0 for the first release and this string stayed at 0.0.1, so
# the published package told users the wrong version of itself. One source of truth
# removes the whole class of mistake.
from importlib import metadata as _md

try:  # pragma: no cover - trivial, and the fallback only fires when not installed
    __version__ = _md.version("edgefit")
except _md.PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"

# Bumped whenever anything that could change a measured number changes: timing
# policy, warmup counts, gate thresholds, subprocess protocol, metric definitions,
# **or the exporters** — the artifact is part of what gets measured. Learned by
# adding position_ids to the decoder graph without bumping this, which left rows
# measuring two different graphs sharing one identity.
# Measurements are immutable (PROJECT.md §14.3); a re-measure inserts a new row
# carrying the new harness_version rather than updating the old one.
# 0.3.4: the hosted fallback report counted `ALL` as a compute unit AI Hub reports,
# which it is not, so every unconstrained row recorded 100% fallback while running
# entirely on the NPU. The rows already written are wrong on a derived metric and
# cannot be edited, so they are superseded by re-measurement under this version.
# 0.3.5: EXPORTER_VERSION 3 → 4. The artifact is part of what gets measured, so an
# exporter change must bump this too — resume keys on (device, harness_version), and
# bumping only the exporter left 11 stale `lowering_failure` rows counting as "already
# done" against a defect that had just been fixed.
# 0.3.6: EXPORTER_VERSION 4 → 5 (frozen-token text export).
# 0.3.7: cold/warm load and first-inference are measured and recorded. New metrics
# mean a new corpus shape, and rows without them are not comparable to rows with.
# 0.3.8: 0.3.7 measured cold-load in the worker and dropped it crossing the process
# boundary, so its local rows carry null for a metric that was taken. They are not
# comparable to rows that have it, which is what a version bump is for.
# 0.3.9: 0.3.7/0.3.8 held two ORT sessions at once to time a warm reload, which
# inflated peak_rss_bytes by ~91% of a session — a new metric silently corrupting
# an existing one. Those rows overstate memory and are not comparable.
# 0.3.10: local warm_load is abandoned as unmeasurable in-process and recorded as
# null-with-reason. 0.3.9 rows carry a warm figure that is backwards on a quarter
# of them, so they are not comparable.
# 0.3.11: hosted recipes can quantize and compile before profiling, so the artifact
# measured may not be the ONNX we uploaded. That changes what a row means.
HARNESS_VERSION = "0.3.11"

__all__ = ["__version__", "HARNESS_VERSION"]
