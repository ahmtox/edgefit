"""EdgeFit — deployment compiler for on-device AI.

Pass 1 is the measurement harness: ``(model, recipe, device) -> measurement``.
"""

__version__ = "0.0.1"

# Bumped whenever anything that could change a measured number changes: timing
# policy, warmup counts, gate thresholds, subprocess protocol, metric definitions.
# Measurements are immutable (PROJECT.md §14.3); a re-measure inserts a new row
# carrying the new harness_version rather than updating the old one.
HARNESS_VERSION = "0.2.0"

__all__ = ["__version__", "HARNESS_VERSION"]
