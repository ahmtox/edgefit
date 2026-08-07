"""Peak-RSS unit regression test.

This exists because ``ru_maxrss`` is bytes on macOS and kilobytes on Linux, and a
1024x error in the memory column would look entirely believable in the atlas.
"""

from __future__ import annotations

import sys

from edgefit.harness.memory import maxrss_scale, peak_rss_bytes


def test_scale_matches_platform() -> None:
    assert maxrss_scale() == (1 if sys.platform == "darwin" else 1024)


def test_peak_rss_is_plausible_for_this_process() -> None:
    """A Python interpreter with numpy loaded lives comfortably inside this band."""
    peak = peak_rss_bytes()
    assert 5 * 1024**2 < peak < 4 * 1024**3, f"{peak} bytes is not a credible peak RSS"


def test_the_scale_is_actually_applied() -> None:
    """The conversion itself, with a known input. Deterministic by construction.

    This replaced an end-to-end allocation test, and the reason is worth keeping.
    That test allocated 200 MiB in a child, touched every page, and asserted the
    reported growth landed near 200 MiB. It passed on macOS for months. On a Linux
    CI runner it reported 130 MiB once and **0 bytes** the next run — the same code,
    the same allocation.

    Nothing was wrong with the scale factor. Linux simply does not guarantee that
    writing to a page keeps it resident long enough to move `ru_maxrss`, so the test
    was inferring a unit conversion from allocator behaviour it did not control. A
    flaky gate is worse than no gate: it trains you to re-run until green.

    The claim is that this platform's `ru_maxrss` unit is converted to bytes. So
    that is what is asserted, against a fixed value, with nothing left to the kernel.
    """
    import resource
    from unittest import mock

    fake = mock.Mock(ru_maxrss=200_000)
    with mock.patch.object(resource, "getrusage", return_value=fake):
        reported = peak_rss_bytes()

    expected = 200_000 * (1 if sys.platform == "darwin" else 1024)
    assert reported == expected, (
        f"ru_maxrss=200000 became {reported} bytes; this platform needs a "
        f"x{maxrss_scale()} scale and got x{reported // 200_000}"
    )


def test_a_wrong_scale_would_be_caught() -> None:
    """The 1024x error this file exists to prevent, shown to be detectable.

    macOS and Linux differ by exactly the factor that makes a wrong answer look
    plausible, so the guard is that the two platforms cannot silently share a scale.
    """
    assert maxrss_scale() in (1, 1024)
    assert (maxrss_scale() == 1) == (sys.platform == "darwin")
