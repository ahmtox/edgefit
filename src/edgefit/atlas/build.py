"""Build the atlas from the corpus.

One command, no build server, no node toolchain: the output is a directory of
self-contained files that can be served from anywhere. PROJECT.md §7 names Next.js;
this deviates deliberately, because the atlas is a table and a few charts and a
framework buys nothing yet. Revisit when interactivity earns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edgefit.atlas import render
from edgefit.atlas.query import (
    load_devices,
    load_models,
    load_rows,
    recipe_paths,
    summarise,
)
from edgefit.corpus.export import export_csv, export_parquet
from edgefit.corpus.store import CorpusStore

DEFAULT_SITE_DIR = Path("site")


@dataclass(frozen=True)
class BuildReport:
    directory: Path
    pages: int
    rows: int
    models: int
    devices: int
    bytes_written: int

    @property
    def kib(self) -> float:
        return self.bytes_written / 1024


def build(store: CorpusStore, out_dir: Path | str = DEFAULT_SITE_DIR) -> BuildReport:
    """Render every page. Returns what was written."""
    site = Path(out_dir)
    for sub in ("", "models", "devices", "data"):
        (site / sub).mkdir(parents=True, exist_ok=True)

    rows = load_rows(store, recipe_paths())
    summary = summarise(store, rows)
    models = load_models(store, rows)
    devices = load_devices(store, rows)

    written: list[Path] = []

    def emit(path: Path, html: str) -> None:
        path.write_text(html, encoding="utf-8")
        written.append(path)

    emit(site / "index.html", render.index(summary, rows, models))
    emit(site / "methodology.html", render.methodology_page(summary, rows))
    emit(site / "compare.html", render.compare_page([r for r in rows if r.ok]))

    emit(site / "models" / "index.html", render.models_index(models))
    for model in models:
        emit(site / "models" / f"{model.slug}.html", render.model_page(model))

    emit(site / "devices" / "index.html", render.devices_index(devices))
    for device in devices:
        emit(site / "devices" / f"{device.slug}.html", render.device_page(device))

    # The raw corpus ships with the site — §4 Stage 1 gives the data away on
    # purpose, because a benchmark nobody can check is a benchmark nobody should
    # trust.
    data_dir = site / "data"
    exported = export_parquet(store, data_dir) | export_csv(store, data_dir)
    files = sorted((path.name, path.stat().st_size) for path in exported.values())
    emit(data_dir / "index.html", render.data_page(summary, files))

    return BuildReport(
        directory=site,
        pages=len(written),
        rows=len(rows),
        models=len(models),
        devices=len(devices),
        bytes_written=sum(path.stat().st_size for path in written),
    )
