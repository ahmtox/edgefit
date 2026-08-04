"""Inline SVG charts.

Hand-rolled rather than pulled from a library, because the atlas has to be
publishable as static files that fetch nothing. Two forms only, chosen by the job
the data does:

* **horizontal bar** — magnitude comparison across labelled recipes
* **scatter** — the latency/accuracy relationship, i.e. the Pareto view of
  PROJECT.md §4 Stage 2.3 in miniature

Marks follow the data-viz spec: bars capped at 24px with a 4px rounded data-end
square at the baseline, markers ≥8px with a 2px surface ring, hairline recessive
gridlines, and a legend whenever there are two series. Values are labelled at the
tip rather than on every mark. Series colour is the validated blue/orange
categorical pair, which clears every gate all-pairs in both modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

BAR_THICKNESS = 20  # spec caps at 24; the band keeps the rest as air
BAR_GAP = 12
LABEL_WIDTH = 190
VALUE_WIDTH = 92
CHART_PAD = 14


@dataclass(frozen=True)
class Bar:
    label: str
    value: float
    display: str
    series: int = 1
    """1 or 2 — the categorical slot, assigned by entity and never by rank."""
    note: str = ""


def _nice_step(span: float, count: int) -> float:
    """A round step covering ``span`` in about ``count`` intervals.

    Log-based rather than string-based. The original implementation derived the
    magnitude from ``len(str(int(raw)))``, which collapses to 1 for every value
    below 1 — so a cosine axis spanning 0.026 produced a single tick at zero and no
    readable scale at all.
    """
    if span <= 0:
        return 1.0
    raw = span / max(count, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        if raw <= magnitude * multiple:
            return magnitude * multiple
    return magnitude * 10


def _ticks(maximum: float, count: int = 4) -> list[float]:
    """Ticks from zero to ``maximum`` — for bars, which grow from a baseline."""
    if maximum <= 0:
        return [0.0]
    step = _nice_step(maximum, count)
    ticks, value = [], 0.0
    while value <= maximum * 1.0001 and len(ticks) < 24:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _domain_ticks(low: float, high: float, count: int = 3) -> list[float]:
    """Round ticks inside an arbitrary domain — for scatter axes that don't start at 0."""
    if high <= low:
        return [low]
    step = _nice_step(high - low, count)
    value = math.ceil(low / step) * step
    ticks = []
    while value <= high * (1 + 1e-9) + 1e-12 and len(ticks) < 24:
        ticks.append(round(value, 10))
        value += step
    return ticks or [low, high]


def horizontal_bars(
    bars: list[Bar],
    *,
    unit: str,
    series_names: tuple[str, str] | None = None,
    title: str = "",
) -> str:
    """Magnitude across labelled categories, biggest-to-smallest by caller order."""
    if not bars:
        return '<p class="dim">No measurements to plot.</p>'

    maximum = max(bar.value for bar in bars) or 1.0
    plot_width = 420
    height = CHART_PAD * 2 + 26 + len(bars) * (BAR_THICKNESS + BAR_GAP)
    width = LABEL_WIDTH + plot_width + VALUE_WIDTH
    axis_y = height - CHART_PAD - 8

    out: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{escape(title or "latency by recipe")}">'
    ]

    for tick in _ticks(maximum):
        x = LABEL_WIDTH + (tick / maximum) * plot_width
        out.append(
            f'<line x1="{x:.1f}" y1="{CHART_PAD + 18}" x2="{x:.1f}" y2="{axis_y}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{height - CHART_PAD + 2}" font-size="10" fill="var(--muted)" '
            f'text-anchor="middle">{tick:g}</text>'
        )
    out.append(
        f'<text x="{LABEL_WIDTH + plot_width / 2:.0f}" y="{CHART_PAD + 8}" font-size="10" '
        f'fill="var(--muted)" text-anchor="middle">{escape(unit)}</text>'
    )

    for index, bar in enumerate(bars):
        y = CHART_PAD + 26 + index * (BAR_THICKNESS + BAR_GAP)
        length = max((bar.value / maximum) * plot_width, 2.0)
        colour = f"var(--series-{bar.series})"
        # 4px rounded data-end, square at the baseline: a full round-rect shifted
        # left by its radius and clipped by the baseline edge would leak, so the
        # square end is restored with an overlapping rect.
        out.append(
            f'<rect x="{LABEL_WIDTH}" y="{y}" width="{length:.1f}" height="{BAR_THICKNESS}" '
            f'rx="4" fill="{colour}"/>'
            f'<rect x="{LABEL_WIDTH}" y="{y}" width="{min(4.0, length):.1f}" '
            f'height="{BAR_THICKNESS}" fill="{colour}"/>'
        )
        out.append(
            f'<text x="{LABEL_WIDTH - 8}" y="{y + BAR_THICKNESS / 2 + 3.5:.0f}" font-size="11" '
            f'fill="var(--ink-2)" text-anchor="end">{escape(bar.label)}</text>'
        )
        out.append(
            f'<text x="{LABEL_WIDTH + length + 7:.1f}" y="{y + BAR_THICKNESS / 2 + 3.5:.0f}" '
            f'font-size="11" fill="var(--ink)" font-weight="600">{escape(bar.display)}</text>'
        )
        if bar.note:
            out.append(
                f'<title>{escape(bar.label)}: {escape(bar.display)} — {escape(bar.note)}</title>'
            )

    out.append(
        f'<line x1="{LABEL_WIDTH}" y1="{CHART_PAD + 18}" x2="{LABEL_WIDTH}" y2="{axis_y}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
    )
    out.append("</svg>")

    chart = "".join(out)
    if series_names:
        chart = _legend(series_names) + chart
    return chart


def _legend(names: tuple[str, str]) -> str:
    return (
        '<div class="legend">'
        f'<span><i class="dot" style="background:var(--series-1)"></i>{escape(names[0])}</span>'
        f'<span><i class="dot" style="background:var(--series-2)"></i>{escape(names[1])}</span>'
        "</div>"
    )


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    radius: float
    label: str
    series: int = 1
    note: str = ""


def scatter(
    points: list[Point],
    *,
    x_label: str,
    y_label: str,
    series_names: tuple[str, str] | None = None,
    x_better_low: bool = True,
) -> str:
    """Two measures against each other — the Pareto view.

    Marks are ≥8px with a 2px surface ring so overlapping points stay legible,
    which is the documented mechanism rather than a stroke around the mark.
    """
    if not points:
        return '<p class="dim">No measurements to plot.</p>'

    width, height = 640, 320
    left, right, top, bottom = 56, 18, 22, 46
    plot_w, plot_h = width - left - right, height - top - bottom

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_pad = (x_hi - x_lo) * 0.12 or max(x_hi * 0.12, 1.0)
    y_pad = (y_hi - y_lo) * 0.18 or max(abs(y_hi) * 0.02, 0.001)
    x_lo, x_hi = max(0.0, x_lo - x_pad), x_hi + x_pad
    y_lo, y_hi = y_lo - y_pad, min(1.0, y_hi + y_pad) if y_hi <= 1 else y_hi + y_pad

    def sx(value: float) -> float:
        return left + (value - x_lo) / (x_hi - x_lo or 1) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_lo) / (y_hi - y_lo or 1) * plot_h

    out = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        f'aria-label="{escape(y_label)} against {escape(x_label)}">'
    ]

    for value in _domain_ticks(y_lo, y_hi, 4):
        y = sy(value)
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y + 3.5:.1f}" font-size="10" fill="var(--muted)" '
            f'text-anchor="end">{value:.6g}</text>'
        )
    for tick in _domain_ticks(x_lo, x_hi, 4):
        x = sx(tick)
        out.append(
            f'<text x="{x:.1f}" y="{height - bottom + 16}" font-size="10" fill="var(--muted)" '
            f'text-anchor="middle">{tick:g}</text>'
        )

    out.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" '
        f'stroke="var(--axis)" stroke-width="1"/>'
    )
    arrow = "lower is better" if x_better_low else "higher is better"
    out.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 8}" font-size="10" fill="var(--muted)" '
        f'text-anchor="middle">{escape(x_label)} · {arrow}</text>'
        f'<text x="14" y="{top + plot_h / 2:.0f}" font-size="10" fill="var(--muted)" '
        f'text-anchor="middle" transform="rotate(-90 14 {top + plot_h / 2:.0f})">'
        f"{escape(y_label)}</text>"
    )

    for point in points:
        cx, cy = sx(point.x), sy(point.y)
        radius = max(5.0, min(point.radius, 15.0))
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
            f'fill="var(--series-{point.series})" fill-opacity="0.85" '
            f'stroke="var(--surface-1)" stroke-width="2">'
            f"<title>{escape(point.label)}"
            f"{' — ' + escape(point.note) if point.note else ''}</title></circle>"
        )
    out.append("</svg>")

    chart = "".join(out)
    if series_names:
        chart = _legend(series_names) + chart
    return chart
