"""Golden fixtures — PROJECT.md §9 step 4, the gate for everything after.

These tests touch real hardware and are marked ``device``, so they are excluded
from the fast suite. Run them with ``edgefit verify``.

They will refuse to run on an unfit host rather than produce a number that looks
fine and isn't — which is the entire point of the exercise.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from edgefit.corpus.store import CorpusStore
from edgefit.harness.gate import GateThresholds, current_gate
from edgefit.harness.hostinfo import probe_device
from edgefit.harness.runner import measure
from edgefit.harness.timing import MeasurementPolicy
from edgefit.models.registry import resolve
from edgefit.schema.common import ThermalState
from edgefit.schema.measurement import Outcome

pytestmark = pytest.mark.device

FIXTURES_PATH = Path(__file__).parent / "fixtures.yaml"
_MIB = 1024**2


def _load() -> dict:
    return yaml.safe_load(FIXTURES_PATH.read_text())


FIXTURES = _load()


#: Set to run the load-insensitive assertions on a busy machine.
#:
#: A normal desktop session (browser, editor, chat) holds load average well above
#: the gate's threshold, so on a developer laptop the strict fixtures never run at
#: all — which is why PROJECT.md §7 puts Phase-0 measurement on a dedicated Mac
#: mini rather than the founder's laptop.
#:
#: This does NOT soften the gate. It splits the fixtures by what load actually
#: affects: numerics correctness, partition attribution, provenance completeness
#: and variance recording are all valid on a busy host. Latency and memory bands
#: are not, and are skipped rather than checked against a meaningless number.
ALLOW_UNFIT = os.environ.get("EDGEFIT_GOLDEN_ALLOW_UNFIT") == "1"


@pytest.fixture(scope="session")
def host_gate():
    """One fitness decision for the whole session, via the canonical path.

    Evaluating without the calibration probe would apply the load-average rule the
    probe supersedes, and disagree with what `edgefit measure` and the sweep decide
    about the same machine.
    """
    return current_gate()


@pytest.fixture(scope="session")
def host_is_fit(host_gate) -> bool:
    return host_gate.passed


@pytest.fixture(scope="session")
def fit_host(host_is_fit: bool, host_gate) -> None:
    """Refuse to produce golden numbers on a host that cannot produce good ones."""
    if host_is_fit:
        return
    report = host_gate
    if not ALLOW_UNFIT:
        pytest.skip(
            f"host is not fit to measure — {report.reason()}. "
            "Set EDGEFIT_GOLDEN_ALLOW_UNFIT=1 to run the load-insensitive checks anyway."
        )


@pytest.fixture(scope="session")
def quiet_host(host_is_fit: bool) -> None:
    """Gate for assertions whose meaning depends on an idle machine."""
    if not host_is_fit:
        pytest.skip("timing bands require an idle host; this one is busy")


@pytest.fixture(scope="session")
def host_class_matches() -> None:
    expected = FIXTURES["host_class"]
    device = probe_device()
    if device.soc != expected["soc"] or device.os_name != expected["os_name"]:
        pytest.skip(
            f"fixtures were recorded on {expected['soc']}/{expected['os_name']}, "
            f"this host is {device.soc}/{device.os_name}. Record a host_class entry for it."
        )


@pytest.fixture(scope="session")
def artifacts() -> dict[str, Path]:
    """Export every model the fixtures reference, once."""
    from edgefit.backends.export_onnx import export_onnx

    directories: dict[str, Path] = {}
    for fixture in FIXTURES["fixtures"]:
        ref = fixture["model"]
        if ref not in directories:
            directories[ref] = export_onnx(resolve(ref)).directory
    return directories


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> CorpusStore:
    """Golden runs go to their own corpus unless one is named explicitly."""
    configured = os.environ.get("EDGEFIT_CORPUS")
    path = Path(configured) if configured else tmp_path_factory.mktemp("golden") / "corpus.duckdb"
    with CorpusStore(path) as store:
        yield store


@pytest.fixture(scope="session")
def results(fit_host, host_class_matches, artifacts, corpus, host_is_fit) -> dict[str, object]:
    """Run every fixture once and share the outcomes across assertions."""
    from edgefit.cli.recipes import load_recipe

    # When running on a busy host the gate would refuse every fixture, so the
    # thresholds are relaxed *for the run itself* while the timing assertions
    # stay skipped. Rows produced this way are diagnostic, never publishable.
    thresholds = (
        GateThresholds(
            require_ac_power=False,
            forbid_low_power_mode=False,
            max_thermal_state=ThermalState.CRITICAL,
            max_load_avg_1m=float("inf"),
            min_available_ram_bytes=0,
            max_calibration_ratio=float("inf"),
        )
        if ALLOW_UNFIT and not host_is_fit
        else None
    )

    outcomes = {}
    for fixture in FIXTURES["fixtures"]:
        recipe = load_recipe(fixture["recipe"], fixture["model"])
        outcomes[fixture["id"]] = measure(
            artifacts[fixture["model"]],
            recipe,
            store=corpus,
            policy=MeasurementPolicy(runs=10, warmup=3),
            thresholds=thresholds,
        )
    return outcomes


def _fixture(fixture_id: str) -> dict:
    return next(f for f in FIXTURES["fixtures"] if f["id"] == fixture_id)


def _ids() -> list[str]:
    return [f["id"] for f in FIXTURES["fixtures"]]


def _record(results, fixture_id: str):
    """The record for a fixture, or a skip if the host refused mid-session.

    A gate refusal is not a measurement, so asserting anything about the
    measurement's outcome would be asserting about a run that never happened.
    Fixtures execute sequentially, so a borderline host can pass the gate for the
    first fixture and refuse a later one — that is the gate working, not a
    regression, and it must not read as a fixture failure.
    """
    record = results[fixture_id].record
    expected = _fixture(fixture_id)["expect"]["outcome"]
    if record.outcome is Outcome.GATE_REFUSED and expected != str(Outcome.GATE_REFUSED):
        pytest.skip(f"host refused mid-session: {record.failure_reason}")
    return record


@pytest.mark.parametrize("fixture_id", _ids())
def test_outcome_matches_fixture(results, fixture_id: str) -> None:
    expect = _fixture(fixture_id)["expect"]
    record = _record(results, fixture_id)
    assert str(record.outcome) == expect["outcome"], (
        f"{fixture_id}: expected {expect['outcome']}, got {record.outcome} "
        f"({record.failure_reason})"
    )
    if "failure_reason_contains" in expect:
        assert expect["failure_reason_contains"] in (record.failure_reason or "")


# Bands whose meaning depends on the host being idle.
_LOAD_SENSITIVE = frozenset({"latency_p50_ms", "peak_rss_mib"})


def _observed(record) -> dict[str, float | None]:
    metrics = record.metrics
    return {
        "latency_p50_ms": metrics.latency_ms.p50 if metrics and metrics.latency_ms else None,
        "peak_rss_mib": (metrics.peak_rss_bytes or 0) / _MIB if metrics else None,
        "artifact_mib": (metrics.artifact_bytes or 0) / _MIB if metrics else None,
        "fallback_flops_pct": record.fallback.fallback_flops_pct if record.fallback else None,
        "fallback_node_pct": record.fallback.fallback_node_pct if record.fallback else None,
    }


def _check_bands(results, fixture_id: str, names: frozenset[str], keep: bool) -> None:
    expect = _fixture(fixture_id)["expect"]
    record = _record(results, fixture_id)
    if record.outcome is not Outcome.SUCCESS:
        pytest.skip("fixture expects a failure outcome; no metrics to band-check")

    observed = _observed(record)
    checked = 0
    for name, band in expect.items():
        if not isinstance(band, dict) or ((name in names) is not keep):
            continue
        value = observed.get(name)
        assert value is not None, f"{fixture_id}: {name} was not measured"
        assert band["min"] <= value <= band["max"], (
            f"{fixture_id}: {name}={value:.3f} outside recorded band "
            f"[{band['min']}, {band['max']}]"
        )
        checked += 1
    if not checked:
        pytest.skip(f"{fixture_id} declares no bands in this category")


@pytest.mark.parametrize("fixture_id", _ids())
def test_structural_metrics_within_bands(results, fixture_id: str) -> None:
    """Attribution and artifact size — valid regardless of how busy the host is.

    Wide bands on purpose: these catch order-of-magnitude regressions such as a
    delegate silently disabled or a 1024x memory unit error, not normal jitter.
    """
    _check_bands(results, fixture_id, _LOAD_SENSITIVE, keep=False)


@pytest.mark.parametrize("fixture_id", _ids())
def test_timing_metrics_within_bands(results, quiet_host, fixture_id: str) -> None:
    """Latency and memory bands. Meaningless on a contended machine, so gated."""
    _check_bands(results, fixture_id, _LOAD_SENSITIVE, keep=True)


@pytest.mark.parametrize("fixture_id", _ids())
def test_variance_is_recorded_and_sane(results, fixture_id: str) -> None:
    """Hard rule #2, checked on live data rather than only in the schema."""
    record = _record(results, fixture_id)
    if record.outcome is not Outcome.SUCCESS:
        pytest.skip("fixture expects a failure outcome")

    stats = record.metrics.latency_ms  # type: ignore[union-attr]
    assert stats is not None
    assert stats.n >= 5
    assert len(stats.samples) == stats.n
    assert stats.stddev > 0, "zero variance across 10 runs means the timer is not working"
    assert stats.minimum <= stats.p50 <= stats.maximum


@pytest.mark.parametrize("fixture_id", _ids())
def test_provenance_is_complete(results, fixture_id: str) -> None:
    """A measurement without provenance is not reproducible, so it is not valid."""
    record = _record(results, fixture_id)
    assert record.harness_version
    assert record.device.os_build and record.device.os_build != "unknown"
    assert record.device.soc and record.device.soc != "unknown"
    assert record.host_state.power_source
    assert record.calibration_probe is not None, "throttle proxy must be recorded"


def test_numerics_match_the_pytorch_reference(results, artifacts) -> None:
    """Known-answer test (PROJECT.md §9).

    ORT's fp32 output must reproduce the PyTorch fp32 reference captured at
    export time. This is the check that catches a silently wrong kernel — the
    failure mode that would make every number in the corpus worthless while
    every latency figure still looked entirely plausible.
    """
    from edgefit.backends.export_onnx import export_onnx

    fixture_id = "minilm-cpu-fp32"
    _record(results, fixture_id)  # skips if the host refused; there is nothing to compare
    outcome = results[fixture_id]
    assert outcome.outputs, "no outputs captured from the measurement run"

    artifact = export_onnx(resolve(_fixture(fixture_id)["model"]))
    reference = next(iter(artifact.load_reference().values())).ravel()
    measured = np.asarray(next(iter(outcome.outputs.values())), dtype=np.float64)
    reference = reference[: measured.size].astype(np.float64)

    cosine = float(
        np.dot(measured, reference) / (np.linalg.norm(measured) * np.linalg.norm(reference))
    )
    max_abs = float(np.max(np.abs(measured - reference)))

    thresholds = FIXTURES["numerics"]
    assert cosine >= thresholds["min_cosine_similarity"], (
        f"ORT fp32 output diverged from the PyTorch reference: cosine={cosine:.6f}"
    )
    assert max_abs <= thresholds["max_abs_difference"], (
        f"max absolute difference {max_abs:.6f} exceeds tolerance"
    )


def test_corpus_recorded_every_fixture(results, corpus) -> None:
    """Including the failures — §5.8, they train the tier-1 filter."""
    assert corpus.count("measurements") >= len(FIXTURES["fixtures"])
