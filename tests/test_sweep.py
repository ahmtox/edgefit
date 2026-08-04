"""Sweep runner behaviour (PROJECT.md §9 step 6).

The measurement itself is stubbed. What matters here is the orchestration: the
cross product, resumption, waiting rather than refusing, and that every outcome
becomes a row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edgefit import HARNESS_VERSION
from edgefit.corpus import CorpusStore
from edgefit.harness.sweep import expand, run_sweep
from edgefit.schema import Outcome

MINILM = "hf:sentence-transformers/all-MiniLM-L6-v2"
VIT = "hf:google/vit-base-patch16-224-in21k"


class TestExpand:
    def test_is_the_cross_product(self) -> None:
        cells = expand([MINILM, VIT], [Path("a.yaml"), Path("b.yaml")])
        assert len(cells) == 4

    def test_order_is_stable(self) -> None:
        """An interrupted sweep must resume in a predictable place."""
        first = expand([VIT, MINILM], [Path("b.yaml"), Path("a.yaml")])
        second = expand([MINILM, VIT], [Path("a.yaml"), Path("b.yaml")])
        assert first == second

    def test_label_is_readable(self) -> None:
        cell = expand([MINILM], [Path("recipes/ort_cpu_fp32.yaml")])[0]
        assert cell.label == "all-MiniLM-L6-v2 × ort_cpu_fp32"


@pytest.fixture
def store(tmp_path):
    with CorpusStore(tmp_path / "corpus.duckdb") as corpus:
        yield corpus


@pytest.fixture
def stubbed(monkeypatch, device, host_state):
    """Replace the gate, artifact resolution and measurement with fakes."""
    from edgefit.harness import sweep as sweep_module
    from edgefit.harness.gate import GateReport
    from edgefit.harness.runner import MeasurementOutcome
    from edgefit.schema import MeasurementRecord, Metrics, RunStats

    calls: list[str] = []
    gate = GateReport(checks=(), host_state=host_state, calibration_probe=None)

    monkeypatch.setattr(sweep_module, "probe_device", lambda: device)
    monkeypatch.setattr(sweep_module, "wait_until_fit", lambda *a, **k: gate)

    def fake_measure(artifact_dir, recipe, *, store, policy, thresholds):
        calls.append(recipe.label or "?")
        record = MeasurementRecord(
            harness_version=HARNESS_VERSION,
            recipe_id=recipe.recipe_id,
            model_ref=recipe.model.ref,
            device=device,
            host_state=host_state,
            outcome=Outcome.SUCCESS,
            run_count=5,
            warmup_count=3,
            metrics=Metrics(latency_ms=RunStats.from_samples([1.0, 1.1, 1.2, 1.05, 1.15])),
        )
        store.insert_recipe(recipe)
        store.insert_measurement(record)
        return MeasurementOutcome(record=record)

    monkeypatch.setattr(sweep_module, "measure", fake_measure)
    return calls


class TestRunSweep:
    def test_measures_every_cell(self, store, stubbed) -> None:
        report = run_sweep(
            [MINILM], [Path("recipes/ort_cpu_fp32.yaml"), Path("recipes/ort_cpu_4thread.yaml")],
            store=store,
        )
        assert report.measured == 2
        assert store.count("measurements") == 2

    def test_resumes_already_measured_cells(self, store, stubbed) -> None:
        recipes = [Path("recipes/ort_cpu_fp32.yaml")]
        run_sweep([MINILM], recipes, store=store)
        second = run_sweep([MINILM], recipes, store=store)
        assert (second.measured, second.resumed) == (0, 1)
        assert store.count("measurements") == 1, "resumption must not duplicate rows"

    def test_resume_can_be_disabled(self, store, stubbed) -> None:
        """--no-resume re-measures, which is how repeatability data is collected."""
        recipes = [Path("recipes/ort_cpu_fp32.yaml")]
        run_sweep([MINILM], recipes, store=store)
        second = run_sweep([MINILM], recipes, store=store, resume=False)
        assert second.measured == 1

    def test_aborts_when_the_host_stays_unfit(
        self, store, stubbed, monkeypatch, host_state
    ) -> None:
        """Forty identical refusal rows say nothing the first one did not."""
        from edgefit.harness import sweep as sweep_module
        from edgefit.harness.gate import GateCheck, GateReport

        unfit = GateReport(
            checks=(GateCheck(name="thermal state", passed=False, observed="serious",
                              required="<= nominal"),),
            host_state=host_state,
            calibration_probe=None,
        )
        monkeypatch.setattr(sweep_module, "wait_until_fit", lambda *a, **k: unfit)

        report = run_sweep(
            [MINILM], [Path("recipes/ort_cpu_fp32.yaml"), Path("recipes/ort_cpu_4thread.yaml")],
            store=store,
        )
        assert report.aborted_reason is not None
        assert "thermal state" in report.aborted_reason
        assert report.measured == 0

    def test_records_an_unsupported_recipe_as_a_lowering_failure(
        self, store, stubbed, monkeypatch
    ) -> None:
        """§5.9: 'ORT cannot do this' is exactly what tier 1 should learn once."""
        from edgefit.backends.artifacts import UnsupportedQuantizationError

        def refuse(*args, **kwargs):
            raise UnsupportedQuantizationError("blockwise int4 is not implemented")

        monkeypatch.setattr("edgefit.backends.artifacts.resolve_artifact", refuse)
        report = run_sweep([MINILM], [Path("recipes/ort_cpu_fp32.yaml")], store=store)

        assert report.lowering_failures == 1
        row = store.query(
            "SELECT outcome, failure_reason FROM measurements"
        ).fetchone()
        assert row == ("lowering_failure", "blockwise int4 is not implemented")
