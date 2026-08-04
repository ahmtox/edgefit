"""Parquet export — the `/data` page of the atlas (PROJECT.md §4 Stage 1).

> "Why we give the data away: it makes us the citation. Distribution and
> credibility beat data hoarding at this stage."

So export is a first-class operation, not an afterthought: whatever we can query
internally, anyone can download and re-derive.
"""

from __future__ import annotations

from pathlib import Path

from edgefit.corpus.store import CorpusStore

EXPORTABLE_TABLES = ("measurements", "configs", "graph_fingerprints")


def export_parquet(store: CorpusStore, out_dir: Path | str) -> dict[str, Path]:
    """Write every table to Parquet. Returns table name -> written path."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for table in EXPORTABLE_TABLES:
        path = destination / f"{table}.parquet"
        store.query(f"SELECT * FROM {table}").write_parquet(str(path))  # noqa: S608
        written[table] = path
    return written


def export_csv(store: CorpusStore, out_dir: Path | str) -> dict[str, Path]:
    """CSV alongside Parquet — the atlas offers both, and CSV is what people paste."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for table in EXPORTABLE_TABLES:
        path = destination / f"{table}.csv"
        store.query(f"SELECT * FROM {table}").write_csv(str(path))  # noqa: S608
        written[table] = path
    return written
