"""Inline CSS and JS for the atlas.

Everything is inlined on purpose. The atlas has to be publishable as plain static
files anywhere, and a page that fetches nothing is a page that cannot break because
a CDN moved. Colours are the validated data-viz palette, declared as roles so the
light/dark pair swaps in one place.
"""

from __future__ import annotations

CSS = """
:root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --plane: #f9f9f7;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --good: #0ca30c;
  --warning: #fab219;
  --critical: #d03b3b;
  --good-text: #006300;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19; --plane: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --good-text: #0ca30c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19; --plane: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --good-text: #0ca30c;
}

* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 1.5rem 5rem; background: var(--plane); color: var(--ink);
  font: 15px/1.6 var(--sans); -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; }
a { color: var(--series-1); text-decoration: none; }
a:hover { text-decoration: underline; }
h1 { font-size: 1.75rem; margin: 0 0 .25rem; letter-spacing: -0.01em; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .75rem; letter-spacing: -0.005em; }
h3 { font-size: .95rem; margin: 1.75rem 0 .5rem; color: var(--ink-2); }
p { margin: .5rem 0; color: var(--ink-2); max-width: 68ch; }
code, .mono { font-family: var(--mono); font-size: .85em; }

header.site { padding: 2rem 0 0; }
nav.site { display: flex; gap: 1.25rem; align-items: center; flex-wrap: wrap;
  padding: .75rem 0 1.25rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
nav.site a { color: var(--ink-2); font-size: .9rem; }
nav.site a.here { color: var(--ink); font-weight: 600; }
nav.site .spacer { flex: 1; }
button.theme { background: none; border: 1px solid var(--border); color: var(--ink-2);
  border-radius: 6px; padding: .25rem .6rem; cursor: pointer; font: inherit; font-size: .8rem; }

.banner { border: 1px solid var(--border); border-left: 3px solid var(--warning);
  background: var(--surface-1); border-radius: 6px; padding: .85rem 1rem; margin: 1.25rem 0; }
.banner strong { color: var(--ink); }
.banner p { margin: .25rem 0 0; font-size: .9rem; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: .75rem; margin: 1.5rem 0; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
  padding: .85rem 1rem; }
.tile .v { font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; display: block; }
.tile .k { font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }

.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
  padding: 1.1rem 1.25rem; margin: 1rem 0; }

table { width: 100%; border-collapse: collapse; font-size: .875rem; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; }
th { text-align: left; font-weight: 600; font-size: .78rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: .04em; padding: .6rem .7rem;
  border-bottom: 1px solid var(--border); white-space: nowrap; position: sticky; top: 0;
  background: var(--surface-1); }
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { color: var(--ink); }
th[aria-sort]::after { content: " ↑"; }
th[aria-sort="descending"]::after { content: " ↓"; }
td { padding: .5rem .7rem; border-bottom: 1px solid var(--grid); vertical-align: top; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: var(--plane); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.dim { color: var(--muted); }
.win { color: var(--good-text); font-weight: 600; }
.lose { color: var(--critical); font-weight: 600; }

.badge { display: inline-flex; align-items: center; gap: .3rem; font-size: .75rem;
  padding: .1rem .45rem; border-radius: 4px; border: 1px solid var(--border);
  white-space: nowrap; }
.badge.ok { color: var(--good-text); }
.badge.fail { color: var(--critical); }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }

.controls { display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; margin: 1rem 0 .75rem; }
.controls input, .controls select { font: inherit; font-size: .875rem; padding: .35rem .55rem;
  border: 1px solid var(--border); border-radius: 6px; background: var(--surface-1); color: var(--ink); }
.controls input { min-width: 15rem; }
.count { color: var(--muted); font-size: .85rem; }

.legend { display: flex; gap: 1rem; flex-wrap: wrap; margin: .25rem 0 .75rem;
  font-size: .82rem; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: .4rem; }

figure { margin: 1rem 0 1.5rem; }
figcaption { font-size: .82rem; color: var(--muted); margin-top: .4rem; }
svg { display: block; max-width: 100%; height: auto; }
svg text { font-family: var(--sans); }

pre.repro { background: var(--plane); border: 1px solid var(--border); border-radius: 6px;
  padding: .6rem .7rem; overflow-x: auto; font-family: var(--mono); font-size: .78rem;
  color: var(--ink-2); margin: .35rem 0 0; }
details summary { cursor: pointer; color: var(--ink-2); font-size: .85rem; }
footer.site { margin-top: 4rem; padding-top: 1.25rem; border-top: 1px solid var(--border);
  color: var(--muted); font-size: .82rem; }
ul { color: var(--ink-2); max-width: 68ch; }
li { margin: .3rem 0; }
"""

JS = """
(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('edgefit-theme'); } catch (e) {}
  if (saved) root.setAttribute('data-theme', saved);
  var btn = document.querySelector('button.theme');
  if (btn) btn.addEventListener('click', function () {
    var dark = getComputedStyle(root).colorScheme === 'dark';
    var next = dark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('edgefit-theme', next); } catch (e) {}
  });

  // Table sort. Numeric columns are marked with class="num"; everything else
  // sorts as text. A missing value always sorts last regardless of direction —
  // "we did not measure this" is not a small number.
  document.querySelectorAll('table[data-sortable]').forEach(function (table) {
    var body = table.tBodies[0];
    table.querySelectorAll('th.sortable').forEach(function (th, index) {
      th.addEventListener('click', function () {
        var desc = th.getAttribute('aria-sort') !== 'descending';
        table.querySelectorAll('th').forEach(function (o) { o.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', desc ? 'descending' : 'ascending');
        var numeric = th.classList.contains('num');
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = a.cells[index], y = b.cells[index];
          var av = x ? (x.dataset.v !== undefined ? x.dataset.v : x.textContent.trim()) : '';
          var bv = y ? (y.dataset.v !== undefined ? y.dataset.v : y.textContent.trim()) : '';
          var an = av === '' || av === '—', bn = bv === '' || bv === '—';
          if (an && bn) return 0;
          if (an) return 1;
          if (bn) return -1;
          var r = numeric ? parseFloat(av) - parseFloat(bv) : av.localeCompare(bv);
          return desc ? -r : r;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    });
  });

  // Filters. Each control names the table it drives via data-target.
  function applyFilters(table) {
    var text = (document.querySelector('[data-filter-text][data-target="' + table.id + '"]') || {}).value || '';
    var needle = text.toLowerCase();
    var selects = document.querySelectorAll('[data-filter-key][data-target="' + table.id + '"]');
    var shown = 0;
    Array.prototype.forEach.call(table.tBodies[0].rows, function (row) {
      var ok = !needle || row.textContent.toLowerCase().indexOf(needle) !== -1;
      Array.prototype.forEach.call(selects, function (sel) {
        if (ok && sel.value) ok = row.dataset[sel.dataset.filterKey] === sel.value;
      });
      row.hidden = !ok;
      if (ok) shown++;
    });
    var out = document.querySelector('[data-count][data-target="' + table.id + '"]');
    if (out) out.textContent = shown + ' of ' + table.tBodies[0].rows.length + ' rows';
  }
  document.querySelectorAll('[data-target]').forEach(function (control) {
    if (!control.dataset.filterText && !control.dataset.filterKey) return;
    var table = document.getElementById(control.dataset.target);
    if (!table) return;
    var run = function () { applyFilters(table); };
    control.addEventListener('input', run);
    control.addEventListener('change', run);
  });
  document.querySelectorAll('table[data-sortable]').forEach(applyFilters);
})();
"""
