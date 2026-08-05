"""Static and profile-based analysis of lowered graphs."""

from edgefit.backends.analysis.ep_placement import (
    KernelEvent,
    build_as_run_report,
    build_fallback_report,
    parse_profile,
)
from edgefit.backends.analysis.flops import FLOPS_ESTIMATOR_VERSION, FlopsTable, estimate_flops
from edgefit.backends.analysis.graph import fingerprint_onnx
from edgefit.backends.analysis.weights import (
    DUPLICATE_DETECTOR_VERSION,
    DuplicateGroup,
    DuplicateRelation,
    DuplicateWeightReport,
    find_duplicate_initializers,
)

__all__ = [
    "DUPLICATE_DETECTOR_VERSION",
    "FLOPS_ESTIMATOR_VERSION",
    "DuplicateGroup",
    "DuplicateRelation",
    "DuplicateWeightReport",
    "FlopsTable",
    "KernelEvent",
    "build_as_run_report",
    "build_fallback_report",
    "estimate_flops",
    "find_duplicate_initializers",
    "fingerprint_onnx",
    "parse_profile",
]
