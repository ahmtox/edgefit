"""EdgeFit — deployment compiler for on-device AI.

Pass 1 is the measurement harness: ``(model, recipe, device) -> measurement``.
"""

__version__ = "0.0.1"

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
HARNESS_VERSION = "0.3.5"

__all__ = ["__version__", "HARNESS_VERSION"]
