"""Whose toolchain measured whose silicon (PROJECT.md §12 neutrality).

This file exists because of a published error rather than a hypothetical one. Every
hosted row in the corpus is measured through Qualcomm AI Hub. On Google Tensor and
Samsung Exynos parts those rows read **100% CPU**, and we published that as a finding
about the device. It is not one: that toolchain has no code path to a rival's NPU, so
CPU-only execution there is expected behaviour. The same Pixel 9's GPU turned out to be
reachable through a TFLite artifact, which proved the silicon was never the limit.

The corpus now records the fact that decides the reading, so a query cannot lose it.
"""

from __future__ import annotations

from edgefit.backends.analysis.ep_placement import provider_vendor
from edgefit.schema.measurement import FallbackReport
from edgefit.schema.vendor import soc_vendor


def _report(toolchain: str | None, device: str | None) -> FallbackReport:
    """A total-fallback report, which is the case where the distinction bites."""
    return FallbackReport(
        intended_provider="NPU",
        nodes_total=100,
        nodes_on_intended=0,
        fallback_node_pct=100.0,
        toolchain_vendor=toolchain,
        device_soc_vendor=device,
    )


class TestSocVendor:
    def test_names_the_vendors_actually_in_the_corpus(self) -> None:
        """Every chipset string the catalogue has handed us so far."""
        assert soc_vendor("google-tensor-g4") == "google"
        assert soc_vendor("google-tensor-g5") == "google"
        assert soc_vendor("qualcomm-snapdragon-8gen3") == "qualcomm"
        assert soc_vendor("qualcomm-snapdragon-x2-elite") == "qualcomm"
        assert soc_vendor("samsung-exynos-1280") == "samsung"
        assert soc_vendor("Apple M2") == "apple"
        # Bare product codes, which the catalogue also uses.
        assert soc_vendor("qcs6490") == "qualcomm"
        assert soc_vendor("qcs9075") == "qualcomm"
        assert soc_vendor("sa8775p") == "qualcomm"

    def test_unknown_is_none_and_not_a_plausible_default(self) -> None:
        """The whole point. A wrong vendor is worse than an absent one.

        Defaulting an unrecognised part to the toolchain's own vendor would mark a
        cross-vendor row as vendor-constant and hand it the strong reading — silently
        reintroducing the exact claim this module was written to retract.
        """
        assert soc_vendor("some-unannounced-part") is None
        assert soc_vendor("") is None
        assert soc_vendor(None) is None


class TestCrossVendor:
    def test_qualcomm_toolchain_on_tensor_is_not_diagnostic(self) -> None:
        """The published mistake, now mechanically flagged."""
        report = _report("qualcomm", "google")
        assert report.cross_vendor is True
        assert report.fallback_is_diagnostic is False
        # The measurement itself is still true and still recorded.
        assert report.fallback_node_pct == 100.0

    def test_qualcomm_toolchain_on_snapdragon_is_diagnostic(self) -> None:
        """Vendor held constant, so a fallback figure has nowhere else to come from.

        This is the comparison the post stands behind: mid-tier Snapdragons run every
        model on the CPU while 8 Gen 2 and newer accelerate every one, same compiler.
        """
        report = _report("qualcomm", "qualcomm")
        assert report.cross_vendor is False
        assert report.fallback_is_diagnostic is True

    def test_coreml_on_apple_silicon_keeps_the_strong_reading(self) -> None:
        """Guards against over-correcting. Apple's delegate on Apple's SoC is fair.

        The local CoreML findings — three text models made *slower* by the accelerator —
        must not be weakened by a fix aimed at the hosted rows.
        """
        report = _report(provider_vendor("CoreMLExecutionProvider"), soc_vendor("Apple M2"))
        assert report.toolchain_vendor == "apple"
        assert report.cross_vendor is False
        assert report.fallback_is_diagnostic is True

    def test_unknown_vendor_withholds_the_claim(self) -> None:
        """Absence of evidence is not evidence of neutrality.

        ``cross_vendor`` is None because we genuinely do not know, and
        ``fallback_is_diagnostic`` is False because the stronger reading has to be
        earned by evidence rather than by a gap in it.
        """
        for toolchain, device in (("qualcomm", None), (None, "google"), (None, None)):
            report = _report(toolchain, device)
            assert report.cross_vendor is None
            assert report.fallback_is_diagnostic is False

    def test_cpu_ep_names_no_accelerator_vendor(self) -> None:
        """The CPU EP drives no accelerator, so there is no vendor to attribute."""
        assert provider_vendor("CPUExecutionProvider") is None
        assert provider_vendor("SomeFutureExecutionProvider") is None
