"""Corpus queries the atlas renders.

Kept apart from rendering so the SQL is reviewable on its own, and so the same
queries can back an API later without dragging HTML along.

One rule throughout: **failures are selected, not filtered out.** PROJECT.md §5.9
makes failures first-class data, and an atlas that only shows what worked is the
same atlas every vendor already publishes.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime

from edgefit.corpus.store import CorpusStore


def group_median(rows: list[Row]) -> float:
    """Median p50 across repeats of one recipe.

    A true median (averaging the two middle values for an even count), so the
    displayed number and the sort order come from the same statistic.
    """
    return statistics.median([row.primary_ms or 0.0 for row in rows])


def _slug(value: str) -> str:
    out = []
    for char in value.lower():
        out.append(char if char.isalnum() else "-")
    return "-".join(filter(None, "".join(out).split("-")))


@dataclass(frozen=True)
class Summary:
    measurements: int
    successes: int
    failures: int
    models: int
    recipes: int
    devices: int
    first_at: datetime | None
    last_at: datetime | None
    harness_versions: tuple[str, ...]

    @property
    def failure_rate(self) -> float:
        return 100.0 * self.failures / self.measurements if self.measurements else 0.0


@dataclass(frozen=True)
class Row:
    """One measurement, flattened for display."""

    measurement_id: str
    model_ref: str
    model_name: str
    model_slug: str
    task: str
    recipe_label: str
    recipe_id: str
    intended_provider: str
    provider_short: str
    weight_dtype: str | None
    granularity: str | None
    device_slug: str
    device_model: str
    soc: str
    os_version: str
    os_build: str
    outcome: str
    failure_reason: str | None
    run_count: int
    p50_ms: float | None
    p95_ms: float | None
    cv: float | None
    ttft_p50_ms: float | None
    ttft_cv: float | None
    decode_tok_s: float | None
    token_agreement: float | None
    peak_rss_mib: float | None
    artifact_mib: float | None
    lowering_ms: float | None
    cosine: float | None
    fb_flops_authored: float | None
    fb_node_authored: float | None
    fb_time_as_run: float | None
    as_run_partitions: int | None
    thermal_state: str
    power_source: str
    calibration_ratio: float | None
    harness_version: str
    created_at: datetime
    stress_profile: str
    measurement_source: str
    source_detail: str | None
    recipe_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == "success"

    @property
    def is_ours(self) -> bool:
        """Whether we took this measurement on a device we controlled.

        Load-bearing. "Third-party figures are recorded but never impersonate our
        measurements" is a recorded decision, and rendering a hosted row identically to
        one of ours is precisely how it would be broken — the corpus keeps them apart
        and then the published page puts them in the same cell.
        """
        return self.measurement_source != "third_party"

    @property
    def generative(self) -> bool:
        return self.ttft_p50_ms is not None

    @property
    def primary_cv(self) -> float | None:
        """The variance of whichever distribution this task actually has."""
        return self.ttft_cv if self.generative else self.cv

    @property
    def primary_ms(self) -> float | None:
        """Whichever timing this task actually has.

        Never averaged with the other kind: a generative recipe has TTFT and a decode
        rate, and no single latency at all.
        """
        return self.ttft_p50_ms if self.generative else self.p50_ms

    @property
    def series(self) -> int:
        """Categorical slot: 1 = CPU, 2 = an accelerator. Identity, not rank."""
        return 1 if self.provider_short == "CPU" else 2

    @property
    def reproduce(self) -> str:
        """The command that regenerates this row.

        Hosted rows take a different one. They were never produced by `measure`, and
        they have no recipe YAML in the library — so the generic path would have
        printed "recipe … is no longer in the library" for every third-party row and
        quietly broken the per-row reproducibility the methodology page promises.
        """
        if not self.is_ours:
            return (
                f"uv run edgefit measure-remote --model {self.model_ref} "
                f'--device "{self.device_model}" --compute-unit '
                f"{self.intended_provider.lower()}"
            )
        if not self.recipe_path:
            return f"# recipe {self.recipe_id} is no longer in the library"
        return (
            f"uv run edgefit measure --model {self.model_ref} --recipe {self.recipe_path}"
        )


@dataclass(frozen=True)
class Device:
    slug: str
    device_id: str
    model: str
    soc: str
    arch: str
    # Optional because a hosted farm does not expose them. Absent is rendered as
    # unknown, never as zero: "0 GiB RAM" is a synthesized measurement value, and
    # hard rule #1 forbids inventing one to keep a template simple.
    cores_total: int | None
    cores_performance: int | None
    cores_efficiency: int | None
    ram_bytes: int | None
    os_name: str
    os_version: str
    os_build: str
    rows: list[Row] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.model} · {self.soc}"


@dataclass(frozen=True)
class Model:
    ref: str
    name: str
    slug: str
    task: str
    n_nodes: int | None
    n_parameters: int | None
    attention_variant: str | None
    norm_type: str | None
    op_histogram: dict[str, int] = field(default_factory=dict)
    rows: list[Row] = field(default_factory=list)
    graph_sizes: tuple[tuple[str, int], tuple[str, int], ...] | tuple = ()
    """(quantization label, node count) — quantization rewrites the graph."""

    @property
    def successes(self) -> list[Row]:
        return [row for row in self.rows if row.ok]

    @property
    def failures(self) -> list[Row]:
        return [row for row in self.rows if not row.ok]

    @property
    def best(self) -> Row | None:
        ranked = [r for r in self.successes if r.primary_ms is not None]
        return min(ranked, key=lambda r: r.primary_ms or 0) if ranked else None

    @property
    def generative(self) -> bool:
        return any(row.generative for row in self.successes)

    @property
    def groups(self) -> list[tuple[str, list[Row]]]:
        """Successful rows grouped by recipe, fastest group first.

        Repeats matter: measuring the same recipe on the same unit twice is the only
        repeatability evidence available without a second unit, so the atlas shows the
        spread rather than rendering two mystery rows with slightly different numbers.
        """
        grouped: dict[str, list[Row]] = {}
        for row in self.successes:
            if row.primary_ms is not None:
                grouped.setdefault(row.recipe_label, []).append(row)
        # Ordered by the same statistic the chart displays. Sorting by the minimum
        # while labelling the median puts bars out of order for repeated recipes.
        return sorted(grouped.items(), key=lambda kv: group_median(kv[1]))

    @property
    def repeats(self) -> list[tuple[str, list[Row]]]:
        return [(label, rows) for label, rows in self.groups if len(rows) > 1]

    @property
    def cpu_baseline(self) -> Row | None:
        """The unaccelerated fp32 reference every other recipe is judged against."""
        for row in self.successes:
            if row.provider_short == "CPU" and row.weight_dtype is None:
                return row
        return None


_ROW_SQL = """
SELECT m.measurement_id, m.model_ref, r.task, coalesce(r.label, r.intended_provider) AS label,
       m.recipe_id, r.intended_provider, r.weight_dtype, r.weight_granularity,
       m.device_id, m.device_model, m.soc, m.os_version, m.os_build,
       m.outcome, m.failure_reason, m.run_count,
       m.latency_p50_ms, m.latency_p95_ms, m.latency_cv,
       m.ttft_p50_ms, m.ttft_cv, m.decode_tok_s_p50, m.token_agreement,
       m.peak_rss_bytes, m.artifact_bytes, m.lowering_ms, m.output_cosine_vs_reference,
       m.fallback_flops_pct, m.fallback_node_pct, m.as_run_time_pct, m.as_run_partitions,
       m.thermal_state, m.power_source, m.calibration_ratio,
       m.harness_version, m.created_at, m.stress_profile,
       m.measurement_source, m.source_detail
FROM measurements m LEFT JOIN recipes r USING (recipe_id)
ORDER BY m.model_ref, m.latency_p50_ms NULLS LAST
"""

_MIB = 1024**2


def _short_provider(provider: str | None) -> str:
    return (provider or "unknown").replace("ExecutionProvider", "") or "unknown"


def load_rows(store: CorpusStore, recipe_paths: dict[str, str] | None = None) -> list[Row]:
    """Every measurement, successes and failures alike."""
    paths = recipe_paths or {}
    rows: list[Row] = []
    for record in store.query(_ROW_SQL).fetchall():
        (
            mid, model_ref, task, label, recipe_id, provider, dtype, granularity,
            device_id, device_model, soc, os_version, os_build,
            outcome, reason, run_count, p50, p95, cv, ttft, ttft_cv, decode, tok_agree,
            rss, artifact, lowering, cosine,
            fb_flops, fb_node, as_run_time, as_run_parts,
            thermal, power, calib, harness, created, stress,
            source, source_detail,
        ) = record
        name = model_ref.removeprefix("hf:").split("/")[-1]
        rows.append(
            Row(
                measurement_id=mid,
                model_ref=model_ref,
                model_name=name,
                model_slug=_slug(name),
                task=task or "unknown",
                recipe_label=label or "unknown",
                recipe_id=recipe_id,
                intended_provider=provider or "unknown",
                provider_short=_short_provider(provider),
                weight_dtype=dtype,
                granularity=granularity,
                device_slug=_slug(f"{device_model}-{os_build}"),
                device_model=device_model,
                soc=soc,
                os_version=os_version,
                os_build=os_build,
                outcome=outcome,
                failure_reason=reason,
                run_count=run_count,
                p50_ms=p50,
                p95_ms=p95,
                cv=cv,
                ttft_p50_ms=ttft,
                ttft_cv=ttft_cv,
                decode_tok_s=decode,
                token_agreement=tok_agree,
                peak_rss_mib=rss / _MIB if rss else None,
                artifact_mib=artifact / _MIB if artifact else None,
                lowering_ms=lowering,
                cosine=cosine,
                fb_flops_authored=fb_flops,
                fb_node_authored=fb_node,
                fb_time_as_run=as_run_time,
                as_run_partitions=as_run_parts,
                thermal_state=thermal,
                power_source=power,
                calibration_ratio=calib,
                harness_version=harness,
                created_at=created,
                stress_profile=stress,
                measurement_source=source,
                source_detail=source_detail,
                recipe_path=paths.get(label or ""),
            )
        )
    return rows


def summarise(store: CorpusStore, rows: list[Row]) -> Summary:
    versions = sorted({row.harness_version for row in rows})
    stamps = sorted(row.created_at for row in rows)
    return Summary(
        measurements=len(rows),
        successes=sum(1 for row in rows if row.ok),
        failures=sum(1 for row in rows if not row.ok),
        models=len({row.model_ref for row in rows}),
        recipes=len({row.recipe_label for row in rows}),
        devices=len({row.device_slug for row in rows}),
        first_at=stamps[0] if stamps else None,
        last_at=stamps[-1] if stamps else None,
        harness_versions=tuple(versions),
    )


def load_models(store: CorpusStore, rows: list[Row]) -> list[Model]:
    """Group rows by model, attaching the graph fingerprint where one exists."""
    import json

    # A model has several fingerprints, because quantization rewrites the graph —
    # MiniLM is 339 nodes unquantized, 340 as fp16 and 474 as int8, the extra nodes
    # being inserted quantize/dequantize pairs. The header shows the *as-authored*
    # graph, chosen deterministically: an arbitrary pick made the published node
    # count flicker between builds, which is exactly the kind of instability that
    # costs a benchmark its credibility.
    fingerprints: dict[str, dict] = {}
    for model_ref, payload in store.query(
        """
        SELECT m.model_ref, g.payload
        FROM measurements m
        JOIN graph_fingerprints g ON g.fingerprint_id = m.graph_fingerprint_id
        JOIN recipes r USING (recipe_id)
        ORDER BY m.model_ref, r.weight_dtype IS NOT NULL, g.fingerprint_id
        """
    ).fetchall():
        fingerprints.setdefault(model_ref, json.loads(payload))

    graph_sizes: dict[str, list[tuple[str, int]]] = {}
    for model_ref, dtype, nodes in store.query(
        """
        SELECT DISTINCT m.model_ref, coalesce(r.weight_dtype, 'fp32'), g.n_nodes
        FROM measurements m
        JOIN graph_fingerprints g ON g.fingerprint_id = m.graph_fingerprint_id
        JOIN recipes r USING (recipe_id)
        ORDER BY 1, 3
        """
    ).fetchall():
        graph_sizes.setdefault(model_ref, []).append((dtype, nodes))

    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.model_ref, []).append(row)

    models = []
    for model_ref, model_rows in grouped.items():
        first = model_rows[0]
        fingerprint = fingerprints.get(model_ref, {})
        models.append(
            Model(
                ref=model_ref,
                name=first.model_name,
                slug=first.model_slug,
                task=first.task,
                n_nodes=fingerprint.get("n_nodes"),
                n_parameters=fingerprint.get("n_parameters"),
                attention_variant=fingerprint.get("attention_variant"),
                norm_type=fingerprint.get("norm_type"),
                op_histogram=fingerprint.get("op_histogram", {}),
                rows=sorted(model_rows, key=lambda r: (r.primary_ms is None, r.primary_ms or 0)),
                graph_sizes=tuple(graph_sizes.get(model_ref, ())),
            )
        )
    return sorted(models, key=lambda m: m.name)


def load_devices(store: CorpusStore, rows: list[Row]) -> list[Device]:
    """Group rows by device. Two OS builds on one machine are two devices."""
    import json

    facts: dict[str, dict] = {}
    for payload, in store.query("SELECT payload FROM measurements").fetchall():
        record = json.loads(payload)
        device = record["device"]
        facts.setdefault(_slug(f"{device['model']}-{device['os_build']}"), device)

    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.device_slug, []).append(row)

    devices = []
    for slug, device_rows in grouped.items():
        device = facts.get(slug, {})
        devices.append(
            Device(
                slug=slug,
                device_id=device.get("device_id", slug),
                model=device.get("model", device_rows[0].device_model),
                soc=device.get("soc", device_rows[0].soc),
                arch=device.get("arch", "unknown"),
                # `.get(key, default)` is not enough: a hosted fingerprint *has* these
                # keys, set to null. The default only fires when the key is missing.
                cores_total=device.get("cpu_cores_total") or None,
                cores_performance=device.get("cpu_cores_performance"),
                cores_efficiency=device.get("cpu_cores_efficiency"),
                ram_bytes=device.get("ram_bytes") or None,
                os_name=device.get("os_name", "unknown"),
                os_version=device.get("os_version", device_rows[0].os_version),
                os_build=device.get("os_build", device_rows[0].os_build),
                rows=sorted(device_rows, key=lambda r: (r.primary_ms is None, r.primary_ms or 0)),
            )
        )
    return sorted(devices, key=lambda d: d.name)


def recipe_paths() -> dict[str, str]:
    """Map a recipe label back to the file that produced it.

    Needed for the reproduction command PROJECT.md §4 Stage 1 promises on every
    row. The path is not stored on the record because it is not part of the
    recipe's meaning — two files with identical contents are the same recipe.
    """
    from edgefit.cli.recipes import available_recipes, load_recipe_payload

    mapping: dict[str, str] = {}
    for path in available_recipes():
        try:
            payload = load_recipe_payload(path)
        except (OSError, ValueError):
            continue
        label = payload.get("label")
        if label:
            mapping[label] = str(path)
    return mapping
