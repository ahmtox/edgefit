"""Peak-RSS unit regression test.

This exists because ``ru_maxrss`` is bytes on macOS and kilobytes on Linux, and a
1024x error in the memory column would look entirely believable in the atlas.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from edgefit.harness.memory import maxrss_scale, peak_rss_bytes

_ALLOCATION_MIB = 200


def test_scale_matches_platform() -> None:
    assert maxrss_scale() == (1 if sys.platform == "darwin" else 1024)


def test_peak_rss_is_plausible_for_this_process() -> None:
    """A Python interpreter with numpy loaded lives comfortably inside this band."""
    peak = peak_rss_bytes()
    assert 5 * 1024**2 < peak < 4 * 1024**3, f"{peak} bytes is not a credible peak RSS"


def test_peak_rss_tracks_a_known_allocation() -> None:
    """Allocate a known amount in a child and check the reported figure lands near it.

    The real assertion is on the *units*: a wrong scale factor misses by 1024x,
    which no tolerance band could absorb.
    """
    program = textwrap.dedent(
        f"""
        from edgefit.harness.memory import peak_rss_bytes
        before = peak_rss_bytes()
        blob = bytearray({_ALLOCATION_MIB} * 1024 * 1024)
        blob[::4096] = b"\\x01" * len(blob[::4096])  # force the pages resident
        print(peak_rss_bytes() - before)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    growth = int(result.stdout.strip())

    expected = _ALLOCATION_MIB * 1024**2
    assert growth == pytest.approx(expected, rel=0.25), (
        f"peak RSS grew by {growth} bytes for a {expected}-byte allocation "
        "— check the ru_maxrss unit scale for this platform"
    )
