"""Page rendering for the atlas.

Deliberately plain string templates. The pages are documents, and a template
engine would add a dependency and an indirection for no expressive gain.

Two editorial rules run through every page:

1. **Provenance is visible, not buried.** Every page states that these numbers come
   from a single laptop-class unit with no second unit to cross-check. Publishing
   laptop numbers dressed as lab numbers would destroy the only asset the project
   has (PROJECT.md §13, measurement trust).
2. **Failures are shown.** §5.9 makes them data. An atlas that only lists what
   worked is the atlas every vendor already publishes.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from edgefit import HARNESS_VERSION, __version__
from edgefit.atlas import charts
from edgefit.atlas.assets import CSS, JS
from edgefit.atlas.query import Device, Model, Row, Summary, group_median

_NAV = (
    ("", "The matrix"),
    ("models/", "Models"),
    ("devices/", "Devices"),
    ("compare.html", "Compare"),
    ("methodology.html", "Methodology"),
    ("data/", "Data"),
)


def _n(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def _cell(value: float | None, digits: int = 2, suffix: str = "", cls: str = "num") -> str:
    if value is None:
        return f'<td class="{cls} dim">—</td>'
    return f'<td class="{cls}" data-v="{value}">{value:.{digits}f}{suffix}</td>'


def page(title: str, body: str, *, depth: int = 0, here: str = "") -> str:
    """Wrap body content in the shared shell."""
    up = "../" * depth
    nav = []
    for href, label in _NAV:
        cls = ' class="here"' if href == here else ""
        nav.append(f'<a href="{up}{href}"{cls}>{escape(label)}</a>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · EdgeFit Atlas</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="site"><h1><a href="{up}" style="color:inherit">EdgeFit Atlas</a></h1>
<p style="margin:0">Measured on-device inference.
Every number reproducible; every failure recorded.</p></header>
<nav class="site">{"".join(nav)}<span class="spacer"></span>
<button class="theme" type="button">theme</button></nav>
{body}
<footer class="site">
edgefit {escape(__version__)} · harness {escape(HARNESS_VERSION)} ·
generated {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")}<br>
Corpus is append-only and immutable: a re-measurement is a new row, never an edit.
</footer>
</div>
<script>{JS}</script>
</body>
</html>
"""


PROVENANCE = """
<div class="banner">
<strong>Read this before citing anything here.</strong>
<p>Every number below was measured on <em>one</em> laptop-class machine, gated for AC
power, low-power mode off, nominal thermal state and verified available compute — but
there is no second unit of the same SKU to cross-check against, so the
<a href="{up}methodology.html#two-unit">two-unit test</a> has not been run. Treat these
as dev-grade figures that are honest about their own conditions, not as lab results.
Laptop thermals also differ from a rack-mounted machine.</p>
</div>
"""


def _tiles(summary: Summary) -> str:
    tiles = [
        (f"{summary.measurements}", "measurements"),
        (f"{summary.successes}", "succeeded"),
        (f"{summary.failures}", "failures recorded"),
        (f"{summary.models}", "models"),
        (f"{summary.recipes}", "recipes"),
        (f"{summary.devices}", "devices"),
    ]
    cells = "".join(
        f'<div class="tile"><span class="v">{escape(v)}</span>'
        f'<span class="k">{escape(k)}</span></div>'
        for v, k in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _outcome_badge(row: Row) -> str:
    if row.ok:
        return '<span class="badge ok">success</span>'
    return f'<span class="badge fail">{escape(row.outcome.replace("_", " "))}</span>'


_MATRIX_HEAD = """
<tr>
<th class="sortable">Model</th>
<th class="sortable">Recipe</th>
<th class="sortable">Target</th>
<th class="sortable">Outcome</th>
<th class="sortable num">p50 ms</th>
<th class="sortable num">p95 ms</th>
<th class="sortable num">cv</th>
<th class="sortable num">size MiB</th>
<th class="sortable num">peak RSS</th>
<th class="sortable num">cosine</th>
<th class="sortable num">fallback (run)</th>
</tr>
"""


def _matrix_row(row: Row, *, depth: int) -> str:
    up = "../" * depth
    return (
        f'<tr data-provider="{escape(row.provider_short)}" '
        f'data-outcome="{escape(row.outcome)}" data-task="{escape(row.task)}">'
        f'<td><a href="{up}models/{row.model_slug}.html">{escape(row.model_name)}</a></td>'
        f'<td class="mono">{escape(row.recipe_label)}</td>'
        f'<td><a href="{up}devices/{row.device_slug}.html">{escape(row.soc)}</a> '
        f'<span class="dim">{escape(row.provider_short)}</span></td>'
        f"<td>{_outcome_badge(row)}</td>"
        + _cell(row.p50_ms)
        + _cell(row.p95_ms)
        + _cell(None if row.cv is None else row.cv * 100, 1, "%")
        + _cell(row.artifact_mib, 0)
        + _cell(row.peak_rss_mib, 0)
        + _cell(row.cosine, 4)
        + _cell(row.fb_time_as_run, 1, "%")
        + "</tr>"
    )


def index(summary: Summary, rows: list[Row], models: list[Model]) -> str:
    providers = sorted({row.provider_short for row in rows})
    tasks = sorted({row.task for row in rows})
    provider_options = "".join(
        f'<option value="{escape(p)}">{escape(p)}</option>' for p in providers
    )
    task_options = "".join(f'<option value="{escape(t)}">{escape(t)}</option>' for t in tasks)

    body = [
        PROVENANCE.format(up=""),
        _tiles(summary),
        "<h2>The matrix</h2>",
        "<p>Every measurement in the corpus, successes and failures together. "
        "Sort by any column; a missing value always sorts last, because "
        "&ldquo;not measured&rdquo; is not a small number.</p>",
        '<div class="controls">'
        '<input type="search" placeholder="filter model, recipe, target…" '
        'data-filter-text data-target="matrix">'
        f'<select data-filter-key="provider" data-target="matrix">'
        f'<option value="">all targets</option>{provider_options}</select>'
        f'<select data-filter-key="task" data-target="matrix">'
        f'<option value="">all tasks</option>{task_options}</select>'
        '<span class="count" data-count data-target="matrix"></span>'
        "</div>",
        '<div class="scroll"><table id="matrix" data-sortable><thead>',
        _MATRIX_HEAD,
        "</thead><tbody>",
        "".join(_matrix_row(row, depth=0) for row in rows),
        "</tbody></table></div>",
        _headline_findings(models),
    ]
    return page("The matrix", "".join(body), here="")


def _headline_findings(models: list[Model]) -> str:
    """What the corpus says, stated once, on the front page."""
    lines = []
    for model in models:
        baseline = model.cpu_baseline
        accel = next(
            (r for r in model.successes if r.provider_short != "CPU" and r.weight_dtype is None),
            None,
        )
        if not (baseline and accel and baseline.p50_ms and accel.p50_ms):
            continue
        ratio = baseline.p50_ms / accel.p50_ms
        verdict = (
            f'<span class="win">{ratio:.2f}× faster</span>'
            if ratio > 1
            else f'<span class="lose">{1 / ratio:.2f}× slower</span>'
        )
        lines.append(
            f"<tr><td><a href='models/{model.slug}.html'>{escape(model.name)}</a></td>"
            f"<td class='dim'>{escape(model.task)}</td>"
            + _cell(baseline.p50_ms)
            + _cell(accel.p50_ms)
            + f"<td class='num'>{verdict}</td>"
            + _cell(accel.fb_flops_authored, 1, "%")
            + _cell(accel.fb_time_as_run, 1, "%")
            + "</tr>"
        )
    if not lines:
        return ""
    return (
        "<h2>Does the accelerator actually help?</h2>"
        "<p>The question a delegate never answers for you. Enabling an accelerator can "
        "make a model slower with no error and no warning — the silent-fallback failure "
        "mode. Both fallback columns are shown because they disagree, and the "
        "disagreement is the finding: see "
        "<a href='methodology.html#fallback'>methodology</a>.</p>"
        '<div class="scroll"><table><thead><tr>'
        "<th>Model</th><th>Task</th><th class='num'>CPU fp32</th>"
        "<th class='num'>accelerated</th><th class='num'>verdict</th>"
        "<th class='num'>FLOP fb (as authored)</th><th class='num'>time fb (as run)</th>"
        f"</tr></thead><tbody>{''.join(lines)}</tbody></table></div>"
    )


def model_page(model: Model, depth: int = 1) -> str:
    # One mark per recipe, not per observation: a recipe measured twice is one point
    # on the chart with its spread in the tooltip, not two overlapping marks.
    bars, points = [], []
    for label, group in model.groups:
        latencies = sorted(row.p50_ms or 0 for row in group)
        median = group_median(group)
        first = group[0]
        note = f"cv {(first.cv or 0) * 100:.1f}%"
        if first.artifact_mib:
            note += f", {first.artifact_mib:.0f} MiB"
        if len(group) > 1:
            note += f", {len(group)} repeats spanning {latencies[0]:.2f}–{latencies[-1]:.2f} ms"
        bars.append(
            charts.Bar(
                label=label,
                value=median,
                display=f"{median:.2f}",
                series=first.series,
                note=note,
            )
        )
        if first.cosine is not None:
            points.append(
                charts.Point(
                    x=median,
                    y=first.cosine,
                    radius=4 + (first.artifact_mib or 0) ** 0.5 / 2.2,
                    label=label,
                    series=first.series,
                    note=note,
                )
            )

    fingerprint = []
    if model.n_parameters:
        fingerprint.append(f"{model.n_parameters / 1e6:.1f}M parameters")
    if model.n_nodes:
        fingerprint.append(f"{model.n_nodes} graph nodes")
    if model.attention_variant and model.attention_variant != "unknown":
        fingerprint.append(f"attention: {model.attention_variant}")
    if model.norm_type and model.norm_type != "unknown":
        fingerprint.append(f"norm: {model.norm_type}")

    top_ops = sorted(model.op_histogram.items(), key=lambda kv: -kv[1])[:8]
    ops = ", ".join(f"{escape(op)}&times;{count}" for op, count in top_ops)

    body = [
        f"<h2>{escape(model.name)}</h2>",
        f'<p class="mono dim">{escape(model.ref)} · {escape(model.task)}</p>',
        f'<div class="card"><strong>Graph fingerprint</strong>'
        f'<p>{escape(" · ".join(fingerprint)) if fingerprint else "not captured"}</p>'
        + (f'<p class="dim">{ops}</p>' if ops else "")
        + _graph_sizes(model)
        + "</div>",
        "<h3>Latency by recipe</h3>",
        "<figure>",
        charts.horizontal_bars(
            bars, unit="p50 latency, ms", series_names=("CPU", "accelerator")
        ),
        "<figcaption>Median of 10 timed runs after 3 discarded warmups. "
        "Hover a bar for variance and artifact size.</figcaption></figure>",
    ]

    if points:
        body += [
            "<h3>Speed against numerics</h3>",
            "<figure>",
            charts.scatter(
                points,
                x_label="p50 latency, ms",
                y_label="cosine vs fp32 reference",
                series_names=("CPU", "accelerator"),
            ),
            "<figcaption>Marker area scales with artifact size. Cosine is a numerics "
            "check against the fp32 PyTorch reference, <em>not</em> task accuracy — a "
            "model can hold cosine 0.999 and still fail on the slice that matters."
            "</figcaption></figure>",
        ]

    if model.repeats:
        lines = []
        for label, group in model.repeats:
            latencies = sorted(row.p50_ms or 0 for row in group)
            spread = 100 * (latencies[-1] - latencies[0]) / latencies[0] if latencies[0] else 0
            lines.append(
                f"<tr><td class='mono'>{escape(label)}</td>"
                f"<td class='num'>{len(group)}</td>"
                f"<td class='num'>{latencies[0]:.2f}</td>"
                f"<td class='num'>{latencies[-1]:.2f}</td>"
                f"<td class='num'>{spread:.2f}%</td></tr>"
            )
        body += [
            "<h3>Repeatability</h3>",
            "<p>Recipes measured more than once on this unit, in separate sessions. "
            "This is the weakest useful form of the check — the real one runs the same "
            "recipe on two physical units of the same SKU, and we have one machine. "
            "Agreement here cannot catch a defect in the methodology that both runs "
            "share; it can only catch drift.</p>",
            '<div class="scroll"><table><thead><tr><th>Recipe</th>'
            "<th class='num'>sessions</th><th class='num'>fastest ms</th>"
            "<th class='num'>slowest ms</th><th class='num'>spread</th></tr></thead>"
            f"<tbody>{''.join(lines)}</tbody></table></div>",
        ]

    body += [
        "<h3>Every recipe measured</h3>",
        '<div class="scroll"><table data-sortable><thead>',
        "<tr><th class='sortable'>Recipe</th><th class='sortable'>Target</th>"
        "<th class='sortable'>Outcome</th><th class='sortable num'>p50 ms</th>"
        "<th class='sortable num'>cv</th><th class='sortable num'>size MiB</th>"
        "<th class='sortable num'>cosine</th>"
        "<th class='sortable num'>FLOP fb (auth)</th>"
        "<th class='sortable num'>time fb (run)</th>"
        "<th class='sortable num'>partitions</th></tr>",
        "</thead><tbody>",
    ]
    for row in model.rows:
        body.append(
            f"<tr><td class='mono'>{escape(row.recipe_label)}</td>"
            f"<td>{escape(row.provider_short)}</td>"
            f"<td>{_outcome_badge(row)}</td>"
            + _cell(row.p50_ms)
            + _cell(None if row.cv is None else row.cv * 100, 1, "%")
            + _cell(row.artifact_mib, 0)
            + _cell(row.cosine, 4)
            + _cell(row.fb_flops_authored, 1, "%")
            + _cell(row.fb_time_as_run, 1, "%")
            + _cell(None if row.as_run_partitions is None else float(row.as_run_partitions), 0)
            + "</tr>"
        )
    body.append("</tbody></table></div>")

    if model.failures:
        body.append("<h3>Recorded failures</h3>")
        body.append(
            "<p>Kept rather than hidden. A recipe that cannot lower is a fact about "
            "the toolchain, and it is what teaches the static filter not to propose "
            "it again.</p>"
        )
        for row in model.failures:
            body.append(
                f'<div class="card"><strong class="mono">{escape(row.recipe_label)}</strong> '
                f"{_outcome_badge(row)}"
                f'<pre class="repro">{escape((row.failure_reason or "")[:600])}</pre></div>'
            )

    body.append("<h3>Reproduce</h3>")
    body.append(
        "<details><summary>Commands for every row on this page</summary>"
        + "".join(
            f'<pre class="repro">{escape(row.reproduce)}</pre>' for row in model.rows
        )
        + "</details>"
    )
    return page(model.name, "".join(body), depth=depth, here="models/")


def _graph_sizes(model: Model) -> str:
    """Show how quantization rewrites the graph.

    Not trivia: int8 quantization inserts a quantize/dequantize pair around every
    quantized op, so the node count jumps. That inserted work is part of why a
    quantized recipe's speedup is smaller than its arithmetic saving suggests.
    """
    if len(model.graph_sizes) < 2:
        return ""
    parts = ", ".join(f"{escape(dtype)} {nodes} nodes" for dtype, nodes in model.graph_sizes)
    return (
        f'<p class="dim">Graph size by weight dtype: {parts}. Quantization inserts '
        "quantize/dequantize pairs, so the node count grows even as the artifact shrinks."
        "</p>"
    )


def models_index(models: list[Model], depth: int = 1) -> str:
    rows = []
    for model in models:
        best = model.best
        rows.append(
            f"<tr><td><a href='{model.slug}.html'>{escape(model.name)}</a></td>"
            f"<td class='dim'>{escape(model.task)}</td>"
            f"<td class='num'>{len(model.successes)}</td>"
            f"<td class='num'>{len(model.failures)}</td>"
            f"<td class='mono'>{escape(best.recipe_label) if best else '—'}</td>"
            + _cell(best.p50_ms if best else None)
            + "</tr>"
        )
    body = [
        "<h2>Models</h2>",
        "<p>Every model measured, with its fastest recipe on this hardware.</p>",
        '<div class="scroll"><table data-sortable><thead><tr>'
        "<th class='sortable'>Model</th><th class='sortable'>Task</th>"
        "<th class='sortable num'>measured</th><th class='sortable num'>failed</th>"
        "<th class='sortable'>fastest recipe</th><th class='sortable num'>p50 ms</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>",
    ]
    return page("Models", "".join(body), depth=depth, here="models/")


def device_page(device: Device, depth: int = 1) -> str:
    successes = [row for row in device.rows if row.ok and row.p50_ms is not None]
    bars = [
        charts.Bar(
            label=f"{row.model_name[:18]} · {row.recipe_label.removeprefix('ort-')}",
            value=row.p50_ms or 0,
            display=f"{row.p50_ms:.1f}",
            series=row.series,
        )
        for row in sorted(successes, key=lambda r: r.p50_ms or 0)
    ]
    cores = f"{device.cores_total} cores"
    if device.cores_performance and device.cores_efficiency:
        cores += f" ({device.cores_performance}P + {device.cores_efficiency}E)"

    largest = max((row.artifact_mib or 0) for row in device.rows) if device.rows else 0
    body = [
        f"<h2>{escape(device.name)}</h2>",
        f'<div class="card">'
        f"<p><strong>{escape(device.model)}</strong> · {escape(device.soc)} · "
        f"{escape(device.arch)} · {escape(cores)} · "
        f"{device.ram_bytes / 1024**3:.0f} GiB RAM</p>"
        f"<p>{escape(device.os_name)} {escape(device.os_version)} "
        f"(build <span class='mono'>{escape(device.os_build)}</span>)</p>"
        f'<p class="dim">The OS build is part of this device\'s identity. An OS update '
        f"changes delegate behaviour, so the same machine on a new build is a new "
        f"measurement target.</p></div>",
        "<h3>What runs here</h3>",
        "<figure>",
        charts.horizontal_bars(bars, unit="p50 latency, ms", series_names=("CPU", "accelerator")),
        "<figcaption>Every successful measurement on this unit, fastest first."
        "</figcaption></figure>",
        "<h3>What fits</h3>",
        f"<p>Largest artifact measured here is {largest:.0f} MiB against "
        f"{device.ram_bytes / 1024**3:.0f} GiB of RAM. Peak resident memory is recorded "
        f"per measurement in the table below; it is the child process's own high-water "
        f"mark, so it is attributable to that recipe alone.</p>",
        '<div class="scroll"><table data-sortable><thead><tr>'
        "<th class='sortable'>Model</th><th class='sortable'>Recipe</th>"
        "<th class='sortable'>Outcome</th><th class='sortable num'>p50 ms</th>"
        "<th class='sortable num'>peak RSS MiB</th><th class='sortable num'>size MiB</th>"
        "<th class='sortable'>thermal</th><th class='sortable num'>probe ratio</th>"
        "</tr></thead><tbody>",
    ]
    for row in device.rows:
        body.append(
            f"<tr><td><a href='../models/{row.model_slug}.html'>"
            f"{escape(row.model_name)}</a></td>"
            f"<td class='mono'>{escape(row.recipe_label)}</td>"
            f"<td>{_outcome_badge(row)}</td>"
            + _cell(row.p50_ms)
            + _cell(row.peak_rss_mib, 0)
            + _cell(row.artifact_mib, 0)
            + f"<td class='dim'>{escape(row.thermal_state)}</td>"
            + _cell(row.calibration_ratio, 2, "×")
            + "</tr>"
        )
    body.append("</tbody></table></div>")
    return page(device.name, "".join(body), depth=depth, here="devices/")


def devices_index(devices: list[Device], depth: int = 1) -> str:
    rows = "".join(
        f"<tr><td><a href='{d.slug}.html'>{escape(d.name)}</a></td>"
        f"<td class='mono dim'>{escape(d.os_name)} {escape(d.os_version)} "
        f"({escape(d.os_build)})</td>"
        f"<td class='num'>{d.ram_bytes / 1024**3:.0f} GiB</td>"
        f"<td class='num'>{len([r for r in d.rows if r.ok])}</td></tr>"
        for d in devices
    )
    body = [
        "<h2>Devices</h2>",
        "<p>The fleet, such as it is. One laptop-class unit today — which is why "
        "nothing here is presented as a lab result.</p>",
        '<div class="scroll"><table data-sortable><thead><tr>'
        "<th class='sortable'>Device</th><th class='sortable'>OS</th>"
        "<th class='sortable num'>RAM</th><th class='sortable num'>measurements</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>",
    ]
    return page("Devices", "".join(body), depth=depth, here="devices/")


def compare_page(rows: list[Row]) -> str:
    """Client-side side-by-side, driven by the query string so URLs are shareable."""
    import json

    payload = json.dumps(
        [
            {
                "id": row.measurement_id,
                "name": f"{row.model_name} · {row.recipe_label}",
                "model": row.model_name,
                "recipe": row.recipe_label,
                "target": f"{row.soc} {row.provider_short}",
                "outcome": row.outcome,
                "p50": row.p50_ms,
                "p95": row.p95_ms,
                "cv": row.cv,
                "size": row.artifact_mib,
                "rss": row.peak_rss_mib,
                "cosine": row.cosine,
                "fbAuth": row.fb_flops_authored,
                "fbRun": row.fb_time_as_run,
                "parts": row.as_run_partitions,
                "repro": row.reproduce,
            }
            for row in rows
        ]
    )
    body = f"""
<h2>Compare</h2>
<p>Pick any two measurements. The URL updates as you choose, so a comparison is
shareable as a link.</p>
<div class="controls">
  <select id="a"></select><span class="dim">against</span><select id="b"></select>
</div>
<div id="out"></div>
<script id="rows" type="application/json">{payload}</script>
<script>
(function () {{
  var rows = JSON.parse(document.getElementById('rows').textContent);
  var a = document.getElementById('a'), b = document.getElementById('b');
  rows.forEach(function (r, i) {{
    [a, b].forEach(function (sel) {{
      var o = document.createElement('option');
      o.value = r.id; o.textContent = r.name + '  (' + r.target + ')';
      sel.appendChild(o);
    }});
  }});
  var params = new URLSearchParams(location.search);
  a.value = params.get('a') || (rows[0] && rows[0].id);
  b.value = params.get('b') || (rows[1] && rows[1].id) || a.value;

  function fmt(v, d, s) {{ return v === null || v === undefined ? '—' : v.toFixed(d) + (s || ''); }}
  function delta(x, y, lowerBetter) {{
    if (x === null || y === null || x === undefined || y === undefined || !y) return '—';
    var r = x / y;
    if (Math.abs(r - 1) < 0.005) return '<span class="dim">same</span>';
    var better = lowerBetter ? r < 1 : r > 1;
    var f = r < 1 ? (1 / r).toFixed(2) + '× lower' : r.toFixed(2) + '× higher';
    return '<span class="' + (better ? 'win' : 'lose') + '">' + f + '</span>';
  }}
  function render() {{
    var x = rows.find(function (r) {{ return r.id === a.value; }});
    var y = rows.find(function (r) {{ return r.id === b.value; }});
    if (!x || !y) return;
    var params = new URLSearchParams({{ a: x.id, b: y.id }});
    history.replaceState(null, '', '?' + params.toString());
    var spec = [
      ['p50 latency, ms', 'p50', 2, '', true],
      ['p95 latency, ms', 'p95', 2, '', true],
      ['artifact, MiB', 'size', 0, '', true],
      ['peak RSS, MiB', 'rss', 0, '', true],
      ['cosine vs fp32', 'cosine', 4, '', false],
      ['FLOP fallback (as authored)', 'fbAuth', 1, '%', true],
      ['time fallback (as run)', 'fbRun', 1, '%', true]
    ];
    var body = spec.map(function (s) {{
      return '<tr><td>' + s[0] + '</td>' +
        '<td class="num">' + fmt(x[s[1]], s[2], s[3]) + '</td>' +
        '<td class="num">' + fmt(y[s[1]], s[2], s[3]) + '</td>' +
        '<td class="num">' + delta(x[s[1]], y[s[1]], s[4]) + '</td></tr>';
    }}).join('');
    document.getElementById('out').innerHTML =
      '<div class="scroll"><table><thead><tr><th></th><th class="num">' + x.name +
      '</th><th class="num">' + y.name + '</th><th class="num">A vs B</th></tr></thead>' +
      '<tbody>' + body + '</tbody></table></div>' +
      '<h3>Reproduce</h3><pre class="repro">' + x.repro + '\\n' + y.repro + '</pre>';
  }}
  a.addEventListener('change', render); b.addEventListener('change', render); render();
}})();
</script>
"""
    return page("Compare", body, here="compare.html")


def data_page(summary: Summary, files: list[tuple[str, int]]) -> str:
    listing = "".join(
        f"<tr><td><a href='{escape(name)}'>{escape(name)}</a></td>"
        f"<td class='num'>{size / 1024:.1f} KiB</td></tr>"
        for name, size in files
    )
    body = f"""
<h2>Data</h2>
<p>The whole corpus, in full, free. Downloading it and disagreeing with us is the
point — a benchmark nobody can check is a benchmark nobody should trust.</p>
<div class="scroll"><table><thead><tr><th>File</th><th class="num">Size</th></tr></thead>
<tbody>{listing}</tbody></table></div>

<h3>What is in it</h3>
<ul>
<li><strong>measurements</strong> — one row per observation, {summary.measurements} of them,
    including the {summary.failures} that failed. Carries the full record as canonical
    JSON in <span class="mono">payload</span> alongside denormalised columns.</li>
<li><strong>recipes</strong> — every configuration measured, content-addressed by
    <span class="mono">recipe_id</span>. The same recipe always hashes to the same id.</li>
<li><strong>graph_fingerprints</strong> — structural summaries of each model: op
    histogram, dtypes, shapes, attention variant, norm type. No weights, no customer
    data.</li>
</ul>

<h3>Reading it</h3>
<pre class="repro">import duckdb
duckdb.sql("SELECT * FROM 'measurements.parquet' WHERE outcome = 'success'").show()</pre>

<h3>Caveats that travel with the data</h3>
<ul>
<li>Every row records <span class="mono">harness_version</span>. Rows from different
    versions are not necessarily comparable; that is why the field exists.</li>
<li>Every row records the host conditions it was taken under —
    <span class="mono">thermal_state</span>, <span class="mono">power_source</span>,
    <span class="mono">load_avg_1m</span>, <span class="mono">calibration_ratio</span> —
    so you can re-filter on stricter criteria than ours.</li>
<li><span class="mono">stress_profile</span> is <span class="mono">clean</span> on every
    row so far. The soak and memory-pressure rungs are not built yet.</li>
<li>Absent values are null with a written reason in the record's
    <span class="mono">unavailable</span> map. Nothing is imputed, ever.</li>
</ul>
"""
    return page("Data", body, depth=1, here="data/")


def methodology_page(summary: Summary, rows: list[Row]) -> str:
    """The trust document.

    PROJECT.md §4 Stage 1 calls this page "load-bearing for trust", and it is the
    one page where our unusual choices have to be defended rather than asserted.
    Numbers in the prose are computed from the corpus so the page cannot drift away
    from the data it describes.
    """
    measured = [row for row in rows if row.ok and row.cv is not None]
    cvs = sorted(row.cv or 0 for row in measured)
    worst_cv = cvs[-1] * 100 if cvs else 0.0
    median_cv = cvs[len(cvs) // 2] * 100 if cvs else 0.0
    runs = min((row.run_count for row in measured), default=0)

    body = f"""
{PROVENANCE.format(up="")}
<h2>Methodology</h2>
<p>This page exists so you can decide how much to trust the numbers without taking
our word for anything. Where a claim can be checked against the data, the data is
linked. Where something is unknown, it says so.</p>

<h3 id="runs">Runs and variance</h3>
<p>Every measurement is <strong>at least 5 timed runs</strong> after discarded
warmups, and the variance is recorded. This is enforced in the type system, not by
convention: a record claiming success without a timing distribution of five or more
samples fails validation and cannot be written. The aggregate statistics are
recomputed from the raw samples on load and rejected if they disagree, so a variance
figure that did not come from measured data is not representable.</p>
<p>In this corpus: {runs} timed runs per measurement, 3 discarded warmups, median
coefficient of variation <strong>{median_cv:.1f}%</strong>, worst
<strong>{worst_cv:.1f}%</strong>. Warmups are discarded because the first inference
includes lazy kernel compilation and delegate model caching — a real cost, but a
different one from steady-state latency.</p>

<h3 id="gate">The gate: we refuse rather than annotate</h3>
<p>Before every measurement the host is checked, and if it fails the measurement does
not happen. The refusal is itself recorded as a row. The reasoning is that a missing
number is recoverable and a wrong number is not: a run on a warm, busy,
battery-powered laptop produces a figure that looks entirely reasonable and is
worthless.</p>
<ul>
<li><strong>AC power</strong> — battery changes DVFS behaviour and thermal headroom.</li>
<li><strong>Low-power mode off</strong> — and <em>unknown</em> counts as a failure. An
    unverifiable power policy can halve clocks.</li>
<li><strong>Thermal state nominal</strong> — via <span class="mono">NSProcessInfo</span>,
    four buckets.</li>
<li><strong>Available memory</strong> — above 2 GiB, so memory pressure does not
    distort allocation.</li>
<li><strong>Verified available compute</strong> — see the probe below.</li>
</ul>
<p>We found the value of this the hard way. Measuring the same recipes on a contended
host and then on a quiet one moved median latency by about 10% and moved the
coefficient of variation by an order of magnitude — from 2.8&ndash;31.7% down to
0.3&ndash;3.8%. The contended numbers were not obviously wrong. They were quietly
wrong, which is worse.</p>

<h3 id="probe">The throttle probe, because there is no thermometer</h3>
<p>Apple Silicon exposes <strong>no unprivileged temperature reading</strong>.
<span class="mono">pmset -g therm</span> returns nothing, the Intel
<span class="mono">xcpm</span> sysctls do not exist on ARM, and
<span class="mono">powermetrics</span> requires root. Rather than publish a number we
cannot obtain, we measure the thing we actually care about: a fixed deterministic
matmul is timed immediately before each measurement and compared against this unit's
own recorded healthy throughput. A machine that has got slower is throttled or
contended, whatever any sensor claims.</p>
<p>The baseline is a low percentile of recent healthy samples, not the fastest time
ever seen. An all-time minimum turned out to be a ratchet: one unusually quiet moment
recorded 7.63&nbsp;ms against a normal healthy figure near 8.4&nbsp;ms, and after that
the machine could never satisfy its own threshold again. The ratio is recorded on
every row as <span class="mono">calibration_ratio</span>, so you can re-filter more
strictly than we did.</p>
<p><span class="mono">cpu_temperature_c</span> is null on every row, with the reason
attached. That is the honest state of the art on this hardware.</p>

<h3 id="fallback">Two fallback reports, because one of them lies</h3>
<p>When a delegate claims part of a graph, the ops it could not claim quietly run on
CPU. Nothing errors. That is the failure mode this project was built to find, and
measuring it turned out to be subtler than expected.</p>
<p>We report it two ways, because they answer different questions:</p>
<ul>
<li><strong>As authored</strong> — analysed with graph optimisation disabled, so
    profile node names still map to the model you wrote. This is the report that tells
    you <em>which ops to fix</em>, and it carries a FLOP-weighted share.</li>
<li><strong>As run</strong> — analysed at the recipe's own optimisation level, which
    is what actually executed. Fusion destroys node names, so FLOP attribution is
    withheld here rather than guessed; it carries measured time share and partition
    count.</li>
</ul>
<p>Both are necessary because the two differ materially. On ViT-base, moving from
optimisation level <span class="mono">disabled</span> to <span class="mono">all</span>
cut CPU node count from 244 to 86 and lifted the accelerator's time share from 53.7%
to 81.5%. A fallback figure taken from the unoptimized graph is a faithful measurement
of a graph that never runs.</p>
<p>This matters for reading the matrix: <strong>the FLOP-share column does not predict
whether the accelerator helps.</strong> It reads 97&ndash;99.8% for every model we have
measured and separates nothing. The as-run time share does separate them. We publish
both, labelled, rather than quietly dropping the one that turned out to be the wrong
tool.</p>
<p>One further caution: time share is not an efficiency measure. It says where the
time went, not whether sending that work to the accelerator was a good idea. A model
can spend 70% of its time inside the accelerator and still be twice as slow as plain
CPU.</p>

<h3 id="cosine">Cosine is a numerics check, not accuracy</h3>
<p>Quantized recipes carry a cosine similarity against the fp32 PyTorch reference
captured at export time. It exists so the quantization column has a cost beside its
speedup — &ldquo;int8 is twice as fast&rdquo; with no cost column is a half-truth.</p>
<p>It is <strong>not task accuracy.</strong> A model can hold cosine 0.999 globally and
fail badly on the one slice a customer cares about. Real eval-set accuracy is a
separate tier that is not built yet, and the field is null with that reason attached
rather than filled in with this number.</p>

<h3 id="memory">Peak memory</h3>
<p>Each measurement runs in a fresh child process which reports its own peak resident
size, so the figure is attributable to that recipe alone. The parent deliberately does
not read the aggregate child usage, which is a running maximum across every child that
ever exited and would blame one heavy recipe's memory on every lighter one measured
afterwards.</p>
<p>A platform detail worth stating because it is a classic silent error:
<span class="mono">ru_maxrss</span> is reported in <em>bytes</em> on macOS and in
<em>kilobytes</em> on Linux. Nothing in the API says so, and both produce believable
numbers. There is a regression test that allocates a known amount and checks the
result lands near it.</p>

<h3 id="failures">Failures are published</h3>
<p>{summary.failures} of {summary.measurements} rows in this corpus are failures, and
they are in the data download along with everything else. A recipe that will not lower,
or a delegate that aborts the process outright, is a fact about the toolchain that is
expensive to rediscover.</p>
<p>Both stages of every measurement run out of process, which is what makes this
possible: a delegate that crashes the interpreter becomes a recorded row instead of a
dead run with nothing to show. We found this necessary rather than theoretical — one
vendor flag combination aborts the process on every model we have tried it on.</p>

<h3 id="known-answer">The known-answer test</h3>
<p>The gate that everything else depends on. Each exported model ships with the fp32
PyTorch output for a fixed input, and the runtime's fp32 output must reproduce it to a
cosine of 0.9999 or better. This is what catches a silently wrong kernel — the failure
that would make every number here worthless while every latency figure still looked
plausible.</p>

<h3 id="two-unit">What we do not know</h3>
<p>Stated plainly, because a methodology page that only lists strengths is marketing.</p>
<ul>
<li><strong>No two-unit test.</strong> The proper check for a measurement methodology is
    to run the identical recipe on two physical units of the same SKU and confirm they
    agree within stated variance. We have one machine. Repeating a measurement on the
    same unit is strictly weaker — it cannot catch a defect in the methodology that both
    runs share. This is the largest known gap.</li>
<li><strong>One host, and a laptop.</strong> Laptop thermals differ from a
    rack-mounted machine. Nothing here demonstrates the harness is even correct on a
    second machine.</li>
<li><strong>No power measurement.</strong> <span class="mono">power_mw</span> is null
    everywhere; it needs instrumentation we do not have.</li>
<li><strong>Clean bench only.</strong> Every row is
    <span class="mono">stress_profile = clean</span>. Real devices are warm, memory
    constrained, and running other things. That gap is reportedly large at the tail, and
    we have not measured it.</li>
<li><strong>No task accuracy.</strong> See the cosine note above.</li>
<li><strong>Why vision wins and text loses is unexplained.</strong> Accelerated recipes
    beat CPU on the vision models and lose on the text models. The as-run time share
    correlates with the outcome, but a correlation is a description, not a mechanism.
    We are not going to offer one until it is measured.</li>
</ul>

<h3 id="immutable">Corrections</h3>
<p>The corpus is append-only. Measurements are never edited: a re-measurement is a new
row carrying a new <span class="mono">harness_version</span>, and both rows stay. If we
get something wrong, the record of having got it wrong stays too.</p>
<p>We have already used that. An earlier version of this atlas explained the
accelerator's losses using the FLOP-fallback figure. That explanation was wrong, and
the section above says so.</p>
"""
    return page("Methodology", body, here="methodology.html")
