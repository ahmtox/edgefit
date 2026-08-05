"""Atlas generation.

The site is published output, so the properties worth testing are the ones that
would embarrass us in public: unstable numbers between builds, broken axes, hidden
failures, and missing provenance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from edgefit import HARNESS_VERSION
from edgefit.atlas import build
from edgefit.atlas.charts import (
    Bar,
    Point,
    _domain_ticks,
    _nice_step,
    _ticks,
    horizontal_bars,
    scatter,
)
from edgefit.atlas.query import group_median, load_models, load_rows, summarise
from edgefit.corpus import CorpusStore
from edgefit.schema import (
    Dtype,
    MeasurementRecord,
    Metrics,
    Outcome,
    QuantizationConfig,
    RunStats,
)

SAMPLES = [9.6, 9.7, 9.8, 9.65, 9.75, 9.68]


class TestTicks:
    def test_step_handles_sub_unit_spans(self) -> None:
        """The bug this exists for: a cosine axis spanning 0.026 produced one tick at 0.

        The original step calculation derived its magnitude from the string length of
        int(raw), which is 1 for every value below 1.
        """
        assert _nice_step(0.026, 3) == pytest.approx(0.01)
        assert _nice_step(20, 4) == pytest.approx(5.0)

    def test_bar_ticks_start_at_zero(self) -> None:
        """Bars grow from a baseline, so their axis must include it."""
        assert _ticks(20, 4)[0] == 0.0
        assert _ticks(0.9, 3) == [0.0, 0.5]

    def test_domain_ticks_land_inside_the_range(self) -> None:
        ticks = _domain_ticks(0.9746, 1.0, 4)
        assert ticks, "a narrow domain must still produce ticks"
        assert all(0.9746 <= tick <= 1.0 + 1e-9 for tick in ticks)
        assert len(ticks) >= 2

    def test_domain_ticks_are_round_numbers(self) -> None:
        assert _domain_ticks(4.0, 21.0, 4) == [5.0, 10.0, 15.0, 20.0]

    def test_ticks_terminate_on_degenerate_input(self) -> None:
        assert _ticks(0) == [0.0]
        assert _domain_ticks(5.0, 5.0) == [5.0]


class TestCharts:
    def test_bars_render_a_baseline_and_values(self) -> None:
        svg = horizontal_bars(
            [Bar(label="cpu", value=9.7, display="9.70"), Bar(label="ane", value=20.1,
                                                             display="20.10", series=2)],
            unit="p50 ms",
            series_names=("CPU", "accelerator"),
        )
        assert svg.count("<rect") >= 4  # two bars, each with its square baseline end
        assert "9.70" in svg and "20.10" in svg
        assert "var(--series-1)" in svg and "var(--series-2)" in svg

    def test_two_series_always_get_a_legend(self) -> None:
        """Identity must never rest on colour alone."""
        svg = horizontal_bars([Bar("a", 1.0, "1.0")], unit="ms", series_names=("CPU", "NPU"))
        assert 'class="legend"' in svg

    def test_empty_input_says_so_rather_than_drawing_nothing(self) -> None:
        assert "No measurements" in horizontal_bars([], unit="ms")
        assert "No measurements" in scatter([], x_label="x", y_label="y")

    def test_scatter_marks_carry_a_surface_ring(self) -> None:
        """The documented mechanism for overlapping marks, not a border."""
        svg = scatter(
            [Point(x=9.7, y=0.999, radius=6, label="a")], x_label="ms", y_label="cosine"
        )
        assert 'stroke="var(--surface-1)"' in svg
        assert "stroke-width=\"2\"" in svg


def _record(device, host_state, recipe, samples, **overrides) -> MeasurementRecord:
    base = {
        "harness_version": HARNESS_VERSION,
        "recipe_id": recipe.recipe_id,
        "model_ref": recipe.model.ref,
        "device": device,
        "host_state": host_state,
        "outcome": Outcome.SUCCESS,
        "run_count": len(samples),
        "warmup_count": 3,
        "metrics": Metrics(
            latency_ms=RunStats.from_samples(samples),
            artifact_bytes=90 * 1024**2,
            output_cosine_vs_reference=1.0,
        ),
    }
    return MeasurementRecord(**(base | overrides))


@pytest.fixture
def populated(tmp_path, device, host_state, cpu_recipe, coreml_recipe):
    """A corpus with a success, a repeat, and a recorded failure."""
    with CorpusStore(tmp_path / "corpus.duckdb") as store:
        for recipe in (cpu_recipe, coreml_recipe):
            store.insert_recipe(recipe)
        store.insert_measurement(_record(device, host_state, cpu_recipe, SAMPLES))
        store.insert_measurement(
            _record(
                device, host_state, cpu_recipe, [s + 0.05 for s in SAMPLES],
                created_at=datetime.now(UTC) + timedelta(minutes=5),
            )
        )
        store.insert_measurement(
            _record(
                device, host_state, coreml_recipe, SAMPLES,
                outcome=Outcome.LOWERING_FAILURE,
                failure_reason="the runtime aborted the process (SIGABRT)",
                metrics=None,
                run_count=0,
            )
        )
        yield store


def test_group_median_averages_the_middle_of_an_even_count(device, host_state, cpu_recipe):
    """Display statistic and sort statistic must be the same one.

    The bars previously sorted by the minimum across repeats while labelling the
    median, which put two near-identical recipes visibly out of order.
    """
    from edgefit.atlas.query import Row

    def row(p50: float) -> Row:
        return Row(
            measurement_id=f"m{p50}", model_ref="hf:x/y", model_name="y", model_slug="y",
            task="embed", recipe_label="r", recipe_id="rid",
            intended_provider="CPUExecutionProvider", provider_short="CPU",
            weight_dtype=None, granularity=None, device_slug="d", device_model="Mac",
            soc="M2", os_version="15.2", os_build="24C101", outcome="success",
            failure_reason=None, run_count=10, p50_ms=p50, p95_ms=p50, cv=0.01,
            ttft_p50_ms=None, ttft_cv=None, decode_tok_s=None, token_agreement=None,
            peak_rss_mib=1.0, artifact_mib=1.0, lowering_ms=1.0, cosine=1.0,
            fb_flops_authored=None, fb_node_authored=None, fb_time_as_run=None,
            as_run_partitions=None, thermal_state="nominal", power_source="ac",
            calibration_ratio=1.0, harness_version=HARNESS_VERSION,
            created_at=datetime.now(UTC), stress_profile="clean",
            measurement_source="edgefit", source_detail=None,
        )

    assert group_median([row(9.68), row(9.72)]) == pytest.approx(9.70)
    assert group_median([row(4.0)]) == pytest.approx(4.0)
    assert group_median([row(1.0), row(2.0), row(9.0)]) == pytest.approx(2.0)


class TestQuery:
    def test_loads_failures_alongside_successes(self, populated) -> None:
        """§5.9: an atlas that only shows what worked is every vendor's atlas."""
        rows = load_rows(populated)
        assert len(rows) == 3
        assert sum(1 for row in rows if not row.ok) == 1

    def test_summary_counts_failures_separately(self, populated) -> None:
        rows = load_rows(populated)
        summary = summarise(populated, rows)
        assert (summary.measurements, summary.successes, summary.failures) == (3, 2, 1)
        assert summary.devices == 1

    def test_repeats_are_grouped_not_duplicated(self, populated) -> None:
        model = load_models(populated, load_rows(populated))[0]
        labels = [label for label, _ in model.groups]
        assert len(labels) == len(set(labels)), "one group per recipe"
        assert model.repeats, "a recipe measured twice must show up as a repeat"
        label, group = model.repeats[0]
        assert len(group) == 2
        assert group_median(group) == pytest.approx(
            (group[0].p50_ms + group[1].p50_ms) / 2
        )

    def test_reproduction_command_is_present(self, populated) -> None:
        """PROJECT.md §4 Stage 1 promises one for every row."""
        from edgefit.atlas.query import recipe_paths

        rows = load_rows(populated, recipe_paths())
        commands = [row.reproduce for row in rows]
        assert all(cmd.startswith(("uv run edgefit measure", "#")) for cmd in commands)


class TestBuild:
    def test_writes_every_page(self, populated, tmp_path) -> None:
        report = build(populated, tmp_path / "site")
        site = tmp_path / "site"
        for relative in (
            "index.html", "methodology.html", "compare.html",
            "models/index.html", "devices/index.html", "data/index.html",
            "data/measurements.parquet", "data/measurements.csv",
        ):
            assert (site / relative).exists(), f"missing {relative}"
        assert report.pages >= 6

    def test_is_deterministic_across_builds(self, populated, tmp_path) -> None:
        """A published node count that flickers between builds costs credibility."""
        first = (tmp_path / "a", tmp_path / "b")
        build(populated, first[0])
        build(populated, first[1])
        page = "models/index.html"
        assert (first[0] / page).read_text() == (first[1] / page).read_text()

    def test_pages_carry_provenance(self, populated, tmp_path) -> None:
        """Publishing laptop numbers as lab numbers is the one unrecoverable mistake."""
        build(populated, tmp_path / "site")
        for name in ("index.html", "methodology.html"):
            text = (tmp_path / "site" / name).read_text()
            assert "two-unit test" in text
            assert "laptop-class" in text

    def test_failures_appear_in_the_output(self, populated, tmp_path) -> None:
        build(populated, tmp_path / "site")
        index = (tmp_path / "site" / "index.html").read_text()
        assert "lowering failure" in index

    def test_fetches_nothing_external(self, populated, tmp_path) -> None:
        """The atlas must survive being served from anywhere, forever."""
        build(populated, tmp_path / "site")
        for page in (tmp_path / "site").rglob("*.html"):
            text = page.read_text()
            for marker in ("src=\"http", "href=\"http", "@import", "fonts.googleapis"):
                assert marker not in text, f"{page.name} reaches out via {marker}"


class TestThirdPartyRows:
    """A hosted row must be visibly not ours, on the page as well as in the corpus.

    The atlas loaded no provenance at all until AI Hub rows existed, so it would have
    rendered a phone in someone else's rack identically to a gate-passed measurement on
    a machine we control — breaking a recorded decision on the published page rather
    than in the data.
    """

    @pytest.fixture
    def hosted(self, tmp_path, host_state):
        from edgefit.harness.remote import remote_host_state
        from edgefit.schema import (
            DeviceFingerprint,
            MeasurementSource,
            ModelRef,
            QaiHubComputeUnit,
            QaiHubRuntimeConfig,
            Recipe,
            StressProfile,
            TaskType,
        )

        recipe = Recipe(
            model=ModelRef(ref="hf:google/vit-base-patch16-224-in21k", task=TaskType.VISION),
            runtime=QaiHubRuntimeConfig(
                device_name="Samsung Galaxy S24 (Family)", compute_unit=QaiHubComputeUnit.NPU
            ),
        )
        record = MeasurementRecord(
            harness_version=HARNESS_VERSION,
            recipe_id=recipe.recipe_id,
            model_ref=recipe.model.ref,
            device=DeviceFingerprint(
                kind="hosted",
                model="Samsung Galaxy S24 (Family)",
                soc="sm8650",
                arch="aarch64",
                os_name="android",
                os_version="14",
                os_build="unknown",
            ),
            host_state=remote_host_state(),
            measurement_source=MeasurementSource.THIRD_PARTY,
            source_detail="Qualcomm AI Hub profile job jabc123; not end-to-end",
            stress_profile=StressProfile.UNKNOWN,
            outcome=Outcome.SUCCESS,
            run_count=len(SAMPLES),
            warmup_count=3,
            metrics=Metrics(latency_ms=RunStats.from_samples(SAMPLES)),
        )
        with CorpusStore(tmp_path / "hosted.duckdb") as store:
            store.insert_recipe(recipe)
            store.insert_measurement(record)
            yield store

    def test_the_row_knows_it_is_not_ours(self, hosted) -> None:
        row = load_rows(hosted)[0]
        assert not row.is_ours
        assert row.measurement_source == "third_party"

    def test_reproduction_uses_the_remote_command(self, hosted) -> None:
        """`measure --recipe <path>` cannot reproduce it: there is no recipe YAML."""
        command = load_rows(hosted)[0].reproduce
        assert command.startswith("uv run edgefit measure-remote")
        assert '--device "Samsung Galaxy S24 (Family)"' in command
        assert "no longer in the library" not in command

    def test_the_page_marks_it_and_names_the_job(self, hosted, tmp_path) -> None:
        build(hosted, tmp_path / "site")
        index = (tmp_path / "site" / "index.html").read_text()
        assert 'class="thirdparty"' in index
        assert "jabc123" in index, "the mark must name the job that produced the number"

    def test_methodology_scopes_its_claims(self, hosted, tmp_path) -> None:
        """Our gate and thermal probe say nothing about someone else's rack."""
        build(hosted, tmp_path / "site")
        page = (tmp_path / "site" / "methodology.html").read_text()
        assert "Rows we did not measure ourselves" in page
        assert "not end-to-end" in page


def test_quantized_graphs_are_reported_separately(
    tmp_path, device, host_state, cpu_recipe
) -> None:
    """Quantization rewrites the graph, so a model has several fingerprints.

    The header must show the as-authored one deterministically rather than whichever
    row the database happened to return first.
    """
    from edgefit.schema import GraphFingerprint

    quantized = cpu_recipe.derive(
        label="int8",
        quantization=QuantizationConfig(
            weight_dtype=Dtype.INT8,
            activation_quant="dynamic",
            activation_dtype=Dtype.INT8,
        ).model_dump(mode="json"),
    )
    base_fp = GraphFingerprint(n_nodes=339, n_parameters=22_600_000, n_initializers=100)
    quant_fp = GraphFingerprint(n_nodes=474, n_parameters=22_600_000, n_initializers=100)

    with CorpusStore(tmp_path / "corpus.duckdb") as store:
        for recipe, fingerprint in ((cpu_recipe, base_fp), (quantized, quant_fp)):
            store.insert_recipe(recipe)
            store.insert_fingerprint(fingerprint)
            store.insert_measurement(
                _record(
                    device, host_state, recipe, SAMPLES,
                    graph_fingerprint_id=fingerprint.fingerprint_id,
                )
            )
        model = load_models(store, load_rows(store))[0]

    assert model.n_nodes == 339, "header must show the as-authored graph"
    assert dict(model.graph_sizes) == {"fp32": 339, "int8": 474}
