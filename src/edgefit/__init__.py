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
HARNESS_VERSION = "0.3.2"

__all__ = ["__version__", "HARNESS_VERSION"]
