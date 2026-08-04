"""EdgeFit CLI.

The CLI is the real interface (PROJECT.md §4 Stage 2.2) — our users are ML
engineers who want this in CI. Output is written to be scanned in a terminal and
to make the uncomfortable numbers impossible to miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from edgefit import HARNESS_VERSION, __version__
from edgefit.cli.recipes import available_recipes, load_recipe
from edgefit.corpus.store import DEFAULT_CORPUS_PATH, CorpusStore
from edgefit.harness.gate import BaselineStore, GateThresholds, evaluate_gate, run_calibration_probe
from edgefit.harness.hostinfo import probe_device, probe_state
from edgefit.harness.runner import measure as run_measurement
from edgefit.harness.timing import MeasurementPolicy
from edgefit.models.registry import known_refs, resolve
from edgefit.schema.common import ThermalState
from edgefit.schema.measurement import Outcome

app = typer.Typer(
    help="EdgeFit — deployment compiler for on-device AI. Pass 1: the measurement harness.",
    no_args_is_help=True,
    add_completion=False,
)
corpus_app = typer.Typer(help="Inspect and export the measurement corpus.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")

console = Console()
_OK = "[green]ok[/green]"
_FAIL = "[red]refused[/red]"


def _mib(value: int | None) -> str:
    return f"{value / 1024**2:.0f} MiB" if value is not None else "—"


@app.command()
def version() -> None:
    """Print versions. The harness version is what measurements are stamped with."""
    console.print(f"edgefit {__version__}  ·  harness {HARNESS_VERSION}")


@app.command()
def models() -> None:
    """List the model subjects in the registry."""
    table = Table(title="Model registry", header_style="bold")
    table.add_column("ref")
    table.add_column("task")
    table.add_column("notes", overflow="fold")
    for ref in known_refs():
        spec = resolve(ref)
        table.add_row(ref, str(spec.task), spec.description)
    console.print(table)


@app.command()
def recipes() -> None:
    """List available recipe presets."""
    table = Table(title="Recipe library", header_style="bold")
    table.add_column("file")
    table.add_column("label")
    table.add_column("providers", overflow="fold")
    for path in available_recipes():
        recipe = load_recipe(path, known_refs()[0])
        table.add_row(
            str(path),
            recipe.label or "—",
            " > ".join(str(p) for p in recipe.runtime.providers),
        )
    console.print(table)


@app.command()
def doctor(
    calibrate: Annotated[
        bool, typer.Option(help="Run the throttle probe and record a baseline if the gate passes.")
    ] = True,
) -> None:
    """Report host identity and whether it is fit to measure on.

    A non-zero exit means the host is unfit. That is a feature: a measurement
    taken on a hot, busy, battery-powered machine is confident garbage, and hard
    rule #1 prefers a missing number to a wrong one.
    """
    device = probe_device()
    baselines = BaselineStore()

    identity = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    identity.add_row("device", f"{device.model} · {device.soc} · {device.arch}")
    cores = f"{device.cpu_cores_total} cores"
    if device.cpu_cores_performance and device.cpu_cores_efficiency:
        cores += f" ({device.cpu_cores_performance}P + {device.cpu_cores_efficiency}E)"
    identity.add_row("cpu", cores)
    identity.add_row("memory", f"{device.ram_bytes / 1024**3:.0f} GiB")
    identity.add_row("os", f"{device.os_name} {device.os_version} ({device.os_build})")
    identity.add_row("device_id", device.device_id)
    console.print(identity)
    console.print()

    probe = run_calibration_probe(baselines.get(device)) if calibrate else None
    report = evaluate_gate(probe_state(), calibration_probe=probe)

    checks = Table(header_style="bold")
    checks.add_column(" ", width=1)
    checks.add_column("check")
    checks.add_column("observed")
    checks.add_column("required")
    for check in report.checks:
        if check.advisory:
            mark = "[green]✓[/green]" if check.passed else "[yellow]•[/yellow]"
            style = "" if check.passed else "yellow"
        else:
            mark = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
            style = ""
        checks.add_row(mark, check.name, check.observed, check.required, style=style)
    console.print(checks)

    for field, reason in report.host_state.unavailable.items():
        console.print(f"[dim]{field}: unavailable — {reason}[/dim]")

    if report.passed:
        if probe is not None:
            best = baselines.record(device, probe.elapsed_ms)
            console.print(f"\ncalibration baseline: [bold]{best:.1f} ms[/bold]")
        console.print("\n[green]host is fit to measure[/green]")
        return

    console.print(f"\n[red]refusing to measure[/red] — {report.reason()}")
    raise typer.Exit(code=1)


@app.command()
def export(
    model: Annotated[str, typer.Option(help="Model ref, e.g. hf:google/vit-base-patch16-224")],
    dynamic: Annotated[
        bool, typer.Option(help="Export with dynamic batch/sequence instead of static shapes.")
    ] = False,
    force: Annotated[bool, typer.Option(help="Re-export even if cached.")] = False,
) -> None:
    """Export a model to ONNX with pinned inputs and an fp32 reference output."""
    from edgefit.backends.export_onnx import export_onnx  # noqa: PLC0415 - needs the export extra

    spec = resolve(model)
    with console.status(f"exporting {spec.hf_id}…"):
        artifact = export_onnx(spec, static_shapes=not dynamic, force=force)
    state = "cached" if artifact.was_cached else f"{artifact.lowering_ms:.0f} ms"
    console.print(
        f"{_OK}  {artifact.directory}  "
        f"[dim]{artifact.size_bytes / 1024**2:.1f} MiB · {state}[/dim]"
    )


@app.command()
def measure(
    model: Annotated[str, typer.Option(help="Model ref from the registry.")],
    recipe: Annotated[Path, typer.Option(help="Path to a recipe YAML.")],
    runs: Annotated[int, typer.Option(help="Timed runs. Minimum 5 (PROJECT.md §14.2).")] = 10,
    warmup: Annotated[int, typer.Option(help="Discarded warmup runs.")] = 3,
    corpus: Annotated[Path, typer.Option(help="Corpus database path.")] = DEFAULT_CORPUS_PATH,
    force_unfit: Annotated[
        bool, typer.Option("--force-unfit", help="Measure anyway on an unfit host. The row is "
                           "still recorded, and still not publishable.")
    ] = False,
) -> None:
    """Measure one (model, recipe) pair on this device and record the result."""
    from edgefit.backends.artifacts import (  # noqa: PLC0415
        UnsupportedQuantizationError,
        resolve_artifact,
    )

    spec = resolve(model)
    record_recipe = load_recipe(recipe, model)

    try:
        with console.status("resolving artifact…"):
            artifact = resolve_artifact(spec, record_recipe)
    except UnsupportedQuantizationError as exc:
        console.print(f"artifact           {_FAIL} [red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"artifact           {_OK} [dim]{artifact.size_bytes / 1024**2:.1f} MiB · "
        f"{'cached' if artifact.was_cached else f'{artifact.lowering_ms:.0f} ms'}[/dim]"
    )

    thresholds = None
    if force_unfit:
        thresholds = GateThresholds(
            require_ac_power=False,
            forbid_low_power_mode=False,
            max_thermal_state=ThermalState.CRITICAL,
            max_load_avg_1m=float("inf"),
            min_available_ram_bytes=0,
            max_calibration_ratio=float("inf"),
        )
        console.print("[yellow]gate overridden: this row is diagnostic only[/yellow]")

    with CorpusStore(corpus) as store, console.status("measuring…"):
        outcome = run_measurement(
            artifact.directory,
            record_recipe,
            store=store,
            policy=MeasurementPolicy(runs=runs, warmup=warmup),
            thresholds=thresholds,
        )
        total = store.count("measurements")

    record = outcome.record

    if record.outcome is Outcome.GATE_REFUSED:
        console.print(f"gate               {_FAIL}")
        for check in (outcome.gate.failures if outcome.gate else ()):
            console.print(f"  [red]✗[/red] {check.name}: {check.observed} (need {check.required})")
        console.print(f"\nrecorded as [yellow]{record.outcome}[/yellow] → {record.measurement_id}")
        raise typer.Exit(code=1)

    console.print(f"gate               {_OK}")

    if outcome.analysis and outcome.analysis.fallback:
        fallback = outcome.analysis.fallback
        console.print(
            f"partitioner        {fallback.nodes_on_intended}/{fallback.nodes_total} nodes on "
            f"{fallback.intended_provider.replace('ExecutionProvider', '')}"
            + (f" · {fallback.partition_count} partitions" if fallback.partition_count else "")
        )
        _print_fallback(fallback)

    if record.outcome is not Outcome.SUCCESS:
        console.print(f"\n[red]{record.outcome}[/red]: {record.failure_reason}")
        console.print(f"recorded → {record.measurement_id}  [dim](failures train tier 1)[/dim]")
        raise typer.Exit(code=1)

    metrics = record.metrics
    assert metrics is not None and metrics.latency_ms is not None
    stats = metrics.latency_ms
    console.print(f"runs               warmup {record.warmup_count} / timed {stats.n}")
    console.print()
    console.print(
        f"  latency_ms       p50 [bold]{stats.p50:.2f}[/bold]  p95 {stats.p95:.2f}  "
        f"sd {stats.stddev:.3f}  cv {stats.cv:.1%}"
    )
    console.print(f"  peak_rss         {_mib(metrics.peak_rss_bytes)}")
    console.print(f"  artifact         {_mib(metrics.artifact_bytes)}")
    if metrics.output_cosine_vs_reference is not None:
        cosine = metrics.output_cosine_vs_reference
        warn = "  [yellow](numerics degraded)[/yellow]" if cosine < 0.999 else ""
        console.print(f"  numerics         cosine {cosine:.5f} vs fp32 reference{warn}")
    if record.notes:
        console.print(f"\n[yellow]⚠ {record.notes}[/yellow]")

    console.print(
        f"\nwrote measurement [bold]{record.measurement_id}[/bold] → {corpus} "
        f"[dim](immutable · {total} rows)[/dim]"
    )


def _print_fallback(fallback) -> None:
    """Show all three proxies together, because their disagreement is the finding."""
    node = f"{fallback.fallback_node_pct:.1f}%"
    flops = (
        f"{fallback.fallback_flops_pct:.1f}%" if fallback.fallback_flops_pct is not None else "—"
    )
    time_pct = (
        f"{fallback.fallback_time_pct:.1f}%" if fallback.fallback_time_pct is not None else "—"
    )
    console.print(f"  fallback         nodes {node} · FLOPs [bold]{flops}[/bold] · time {time_pct}")

    if fallback.fallback_flops_pct is not None and fallback.fallback_flops_pct > 50:
        top = ", ".join(
            f"{op}×{count}" for op, count in list(fallback.unclaimed_op_types.items())[:5]
        )
        console.print(
            f"  [yellow]⚠ silent CPU fallback: {flops} of arithmetic left the intended "
            f"accelerator[/yellow]\n    [dim]unclaimed: {top}[/dim]"
        )


@app.command()
def sweep(
    model: Annotated[
        list[str] | None,
        typer.Option(help="Model refs. Repeatable. Omit for the whole registry."),
    ] = None,
    recipe: Annotated[
        list[Path] | None,
        typer.Option(help="Recipe YAMLs. Repeatable. Omit for the whole library."),
    ] = None,
    runs: Annotated[int, typer.Option(help="Timed runs per cell. Minimum 5.")] = 10,
    warmup: Annotated[int, typer.Option(help="Discarded warmup runs.")] = 3,
    corpus: Annotated[Path, typer.Option(help="Corpus database path.")] = DEFAULT_CORPUS_PATH,
    resume: Annotated[
        bool, typer.Option(help="Skip cells already measured on this device and harness version.")
    ] = True,
    wait: Annotated[
        float, typer.Option(help="Seconds to wait for a fit host before giving up.")
    ] = 600.0,
) -> None:
    """Measure the cross product of models and recipes.

    Waits for a fit host rather than refusing, resumes where it left off, and
    records every outcome including failures.
    """
    from edgefit.harness.sweep import expand, run_sweep  # noqa: PLC0415

    model_refs = model or known_refs()
    recipe_paths = recipe or available_recipes()
    cells = expand(model_refs, recipe_paths)

    console.print(
        f"[bold]{len(cells)} cells[/bold] — {len(model_refs)} models × "
        f"{len(recipe_paths)} recipes · {runs} runs each"
    )

    marks = {
        "measured": "[green]✓[/green]",
        "resumed": "[dim]·[/dim]",
        "failed": "[red]✗[/red]",
        "refused": "[yellow]![/yellow]",
        "waiting": "[yellow]⏸[/yellow]",
        "aborted": "[red]■[/red]",
    }
    width = max((len(c.label) for c in cells), default=10)

    def on_event(kind: str, cell, detail: str) -> None:
        if kind == "resumed":
            return  # already in the corpus; saying so for every cell is noise
        console.print(f"  {marks.get(kind, ' ')} {cell.label:<{width}}  [dim]{detail}[/dim]")

    with CorpusStore(corpus) as store:
        report = run_sweep(
            model_refs,
            recipe_paths,
            store=store,
            policy=MeasurementPolicy(runs=runs, warmup=warmup),
            resume=resume,
            wait_for_fit_s=wait,
            on_event=on_event,
        )
        total_rows = store.count("measurements")

    summary = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    summary.add_row("measured", f"[green]{report.measured}[/green]")
    if report.resumed:
        summary.add_row("already done", f"[dim]{report.resumed}[/dim]")
    if report.lowering_failures:
        summary.add_row("lowering failures", f"[red]{report.lowering_failures}[/red]")
    if report.runtime_failures:
        summary.add_row("runtime failures", f"[red]{report.runtime_failures}[/red]")
    if report.gate_refused:
        summary.add_row("gate refused", f"[yellow]{report.gate_refused}[/yellow]")
    summary.add_row("elapsed", f"{report.elapsed_s / 60:.1f} min")
    summary.add_row("corpus", f"{total_rows} rows")
    console.print()
    console.print(summary)

    if report.aborted_reason:
        console.print(f"\n[red]sweep aborted[/red] — {report.aborted_reason}")
        raise typer.Exit(code=1)


@corpus_app.command("list")
def corpus_list(
    corpus: Annotated[Path, typer.Option(help="Corpus database path.")] = DEFAULT_CORPUS_PATH,
    limit: Annotated[int, typer.Option(help="Rows to show.")] = 20,
) -> None:
    """Show recent measurements."""
    with CorpusStore(corpus) as store:
        rows = store.query(
            """
            SELECT m.created_at, m.model_ref, coalesce(c.label, c.intended_provider), m.outcome,
                   m.latency_p50_ms, m.latency_cv, m.fallback_flops_pct,
                   m.output_cosine_vs_reference
            FROM measurements m LEFT JOIN recipes c USING (recipe_id)
            ORDER BY m.created_at DESC LIMIT $limit
            """,
            {"limit": limit},
        ).fetchall()
        total = store.count("measurements")

    if not rows:
        console.print("[dim]corpus is empty[/dim]")
        return

    table = Table(title=f"Measurements ({total} total)", header_style="bold")
    columns = ("when", "model", "recipe", "outcome", "p50 ms", "cv", "fallback FLOPs", "cosine")
    for column in columns:
        table.add_column(column)
    for created, model_ref, provider, outcome, p50, cv, flops_pct, cosine in rows:
        table.add_row(
            created.strftime("%m-%d %H:%M"),
            model_ref.removeprefix("hf:").split("/")[-1],
            (provider or "—").replace("ExecutionProvider", ""),
            f"[green]{outcome}[/green]" if outcome == "success" else f"[red]{outcome}[/red]",
            f"{p50:.2f}" if p50 is not None else "—",
            f"{cv:.1%}" if cv is not None else "—",
            f"{flops_pct:.1f}%" if flops_pct is not None else "—",
            f"{cosine:.4f}" if cosine is not None else "—",
        )
    console.print(table)


@corpus_app.command("export")
def corpus_export(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("corpus/export"),
    corpus: Annotated[Path, typer.Option(help="Corpus database path.")] = DEFAULT_CORPUS_PATH,
    csv: Annotated[bool, typer.Option(help="Also write CSV alongside Parquet.")] = True,
) -> None:
    """Export the corpus to Parquet (and CSV) — the atlas /data download."""
    from edgefit.corpus.export import export_csv, export_parquet  # noqa: PLC0415

    with CorpusStore(corpus) as store:
        written = export_parquet(store, out)
        if csv:
            written |= export_csv(store, out)
    for _, path in sorted(written.items()):
        console.print(f"{_OK}  {path}  [dim]{path.stat().st_size / 1024:.1f} KiB[/dim]")


@app.command()
def verify(
    corpus: Annotated[Path, typer.Option(help="Corpus database path.")] = DEFAULT_CORPUS_PATH,
) -> None:
    """Run the golden fixtures — the gate for everything downstream (§9 step 4)."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    console.print("[bold]running golden fixtures[/bold] [dim](requires a fit host)[/dim]\n")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/golden", "-m", "device", "-v", "--no-header"],
        check=False,
        env={"EDGEFIT_CORPUS": str(corpus), **__import__("os").environ},
    )
    raise typer.Exit(code=completed.returncode)


if __name__ == "__main__":
    app()
