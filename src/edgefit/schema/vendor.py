"""Naming the vendor of a chipset, and nothing more.

This module exists because of a published mistake. Every hosted row in the corpus is
measured through Qualcomm AI Hub, and on Google Tensor and Samsung Exynos parts those
rows read **100% CPU**. We published that as a property of the device. It is not: that
toolchain has no code path to a rival's NPU, so running on the CPU there is expected
behaviour rather than a partitioning defect. The proof is that the same phone's GPU
*is* reachable through a different artifact format.

Blaming a rival's silicon for the reach of our own pipeline is precisely the
vendor-flavoured comparison PROJECT.md §12 says this business cannot make, so the fact
that decides the reading — whose toolchain measured whose silicon — is now recorded on
every row instead of being remembered.

Deliberately **not** a support matrix. Whether a particular Hexagon or ANE generation can
be targeted by a particular SDK is vendor documentation we would be guessing at, and a
guess there would surface as a published fallback percentage. This only answers "who makes
this part", from a string the device catalogue already gave us, and answers ``None`` when
it cannot tell.
"""

from __future__ import annotations

#: Public product-line prefixes, longest-lived first. Order matters only in that an
#: explicit vendor prefix should win over a bare product code.
_SOC_VENDOR_PREFIXES: tuple[tuple[str, str], ...] = (
    ("qualcomm", "qualcomm"),
    ("google", "google"),
    ("samsung", "samsung"),
    ("mediatek", "mediatek"),
    ("apple", "apple"),
    ("snapdragon", "qualcomm"),
    ("exynos", "samsung"),
    ("tensor", "google"),
    ("dimensity", "mediatek"),
    ("qcs", "qualcomm"),
    ("qcm", "qualcomm"),
    ("sdm", "qualcomm"),
    ("sa8", "qualcomm"),
    ("sm6", "qualcomm"),
    ("sm7", "qualcomm"),
    ("sm8", "qualcomm"),
    ("sc8", "qualcomm"),
)


def soc_vendor(soc: str | None) -> str | None:
    """Vendor of a chipset string, or ``None`` when we cannot say.

    ``None`` rather than a plausible default. This feeds
    :attr:`~edgefit.schema.measurement.FallbackReport.cross_vendor`, and a wrong vendor
    there silently converts "our toolchain cannot reach that accelerator" into "that
    accelerator refused the graph" — the two readings this whole module exists to keep
    apart.
    """
    if not soc:
        return None
    key = soc.strip().lower()
    for prefix, vendor in _SOC_VENDOR_PREFIXES:
        if key.startswith(prefix):
            return vendor
    return None


#: Who owns the accelerator an ONNX Runtime execution provider drives.
#:
#: CoreML on Apple silicon is the case that matters here: vendor is held constant, so
#: those fallback figures keep the strong reading — the partitioner really did decline
#: the ops, and three text models really were made slower by the accelerator.
_PROVIDER_VENDOR: dict[str, str | None] = {
    "CoreMLExecutionProvider": "apple",
    "CPUExecutionProvider": None,  # portable scalar code; no accelerator vendor to name
    "XnnpackExecutionProvider": None,
    "QNNExecutionProvider": "qualcomm",
    "NnapiExecutionProvider": "google",
}


def provider_vendor(intended_provider: str) -> str | None:
    """Vendor of the accelerator an EP targets, or ``None`` when unknown or N/A."""
    return _PROVIDER_VENDOR.get(intended_provider)
