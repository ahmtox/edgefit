"""Static and profile-based analysis of lowered graphs."""

from edgefit.backends.analysis.ep_placement import (
    KernelEvent,
    build_as_run_report,
    build_fallback_report,
    parse_profile,
)
from edgefit.backends.analysis.flops import FLOPS_ESTIMATOR_VERSION, FlopsTable, estimate_flops
from edgefit.backends.analysis.graph import fingerprint_onnx

__all__ = [
    "FLOPS_ESTIMATOR_VERSION",
    "FlopsTable",
    "KernelEvent",
    "build_as_run_report",
    "build_fallback_report",
    "estimate_flops",
    "fingerprint_onnx",
    "parse_profile",
]
