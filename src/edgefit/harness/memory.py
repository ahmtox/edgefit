"""Peak resident memory.

``getrusage`` reports ``ru_maxrss`` in **bytes on macOS/BSD** and **kilobytes on
Linux**. Nothing in the API says so and both produce plausible-looking numbers,
so getting it wrong yields a corpus that is silently wrong by 1024x on one
platform — precisely the Critical risk in PROJECT.md §13. Hence one helper, one
platform switch, and a regression test that allocates a known amount.
"""

from __future__ import annotations

import resource
import sys

# Verified empirically on macOS 15.2 / arm64: a 200 MiB child allocation reports
# ru_maxrss = 218529792, i.e. bytes. See tests/test_memory_units.py.
_MAXRSS_SCALE: int = 1 if sys.platform == "darwin" else 1024
"""Multiplier converting this platform's ru_maxrss into bytes."""


def maxrss_scale() -> int:
    """Bytes per ``ru_maxrss`` unit on this platform."""
    return _MAXRSS_SCALE


def peak_rss_bytes() -> int:
    """Peak resident set size of *this* process, in bytes.

    Deliberately self-scoped. The parent must not read ``RUSAGE_CHILDREN``: that
    is a running maximum over every child that has ever exited, so it would
    attribute a heavy config's memory to every lighter config measured after it.
    Each measurement subprocess reports its own figure instead.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _MAXRSS_SCALE
