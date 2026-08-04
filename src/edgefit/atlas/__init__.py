"""The atlas — the public benchmark site (PROJECT.md §4 Stage 1).

Distribution channel first, product second. The data is given away deliberately:
being the citation beats hoarding a corpus nobody has checked.
"""

from edgefit.atlas.build import DEFAULT_SITE_DIR, BuildReport, build

__all__ = ["DEFAULT_SITE_DIR", "BuildReport", "build"]
