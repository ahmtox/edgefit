"""Controlled stress — the second rung of the validation ladder (PROJECT.md §5.6).

The tests that matter here are the ones asserting the stressor *actually applies
load*, because both of its failure modes are silent: workers that never start, and
workers the scheduler quietly moves out of the way. Either would label an ordinary
measurement as a stressed one, which is worse than having no stress bench at all.
"""

from __future__ import annotations

from edgefit.harness.stress import Stressor, StressSpec
from edgefit.schema.common import StressProfile


class TestDescription:
    """A profile name alone is unfalsifiable; the parameters travel with the row."""

    def test_concurrent_load_names_the_worker_count(self) -> None:
        spec = StressSpec(StressProfile.CONCURRENT_LOAD, workers=4)
        assert "4" in spec.description and "worker" in spec.description

    def test_memory_pressure_names_the_size(self) -> None:
        spec = StressSpec(StressProfile.MEMORY_PRESSURE, balloon_mib=2048)
        assert "2048" in spec.description

    def test_clean_says_the_host_was_verified_idle(self) -> None:
        assert "idle" in StressSpec(StressProfile.CLEAN).description


class TestLifecycle:
    def test_workers_start_and_are_reaped(self) -> None:
        """A leaked burner would contend with every later measurement on this host.

        That is the worst possible bug for this module: the corpus would fill with
        rows labelled `clean` that were measured against a hidden load.
        """
        spec = StressSpec(StressProfile.CONCURRENT_LOAD, workers=2)
        with Stressor(spec) as stressor:
            assert len(stressor._workers) == 2
            assert all(p.poll() is None for p in stressor._workers), "a worker died at once"
        assert stressor._workers == []

    def test_cleanup_happens_even_if_the_block_raises(self) -> None:
        spec = StressSpec(StressProfile.CONCURRENT_LOAD, workers=2)
        stressor = Stressor(spec)
        try:
            with stressor:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert stressor._workers == []

    def test_the_memory_balloon_is_resident_not_virtual(self) -> None:
        """An untouched allocation applies no pressure, which would make it a lie."""
        with Stressor(StressSpec(StressProfile.MEMORY_PRESSURE, balloon_mib=64)) as s:
            assert s._balloon is not None
            assert len(s._balloon) == 64 * 1024 * 1024
        assert s._balloon is None


def test_the_burner_raises_its_qos() -> None:
    """Without this the stressor is nearly inert on Apple Silicon.

    Eight default-QoS burners saturated the machine — 611% CPU across 8 cores — and
    slowed the calibration probe by 4%, because darwin parks background work on
    efficiency cores and leaves the foreground measurement on the performance cores.
    With USER_INTERACTIVE requested, four workers slow the same probe by 1.74x.

    Asserted on the burner source because the alternative is a timing test, and a
    timing test for "is the scheduler cooperating" is exactly the kind of flaky gate
    this project has already had to delete once.
    """
    from edgefit.harness.stress import _BURNER

    assert "pthread_set_qos_class_self_np" in _BURNER
    assert "0x21" in _BURNER, "QOS_CLASS_USER_INTERACTIVE"
