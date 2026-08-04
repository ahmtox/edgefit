"""Device inventory and fleet resolution (PROJECT.md §4 Stage 2, input #2).

The property under test throughout is the separation of **known** from
**reachable**. Collapsing the two turns a coverage report into a sales document,
and we learned the distinction the hard way: Qualcomm AI Hub lists 78 devices for
our account and provisions none of them.
"""

from __future__ import annotations

import json

import pytest

from edgefit.devices import (
    FleetEntry,
    Inventory,
    InventoryDevice,
    normalise_soc,
    parse_fleet,
    qai_hub_inventory,
    resolve_fleet,
    suggest_aliases,
)


def _device(soc: str, *, reachable: bool = True, aliases: tuple[str, ...] = ()) -> InventoryDevice:
    return InventoryDevice(
        source="test", name=f"dev-{soc}", soc=soc, os_name="android", os_version="14",
        soc_aliases=aliases, reachable=reachable,
    )


@pytest.fixture
def inventory() -> Inventory:
    return Inventory(
        devices=(
            _device("Apple M2"),
            _device("sm8650", reachable=False, aliases=("qualcomm-snapdragon-8gen3",)),
            _device("sm8550", reachable=False),
            _device("google-tensor-g4", reachable=False),
        ),
        notes=("hosted devices are catalogued but not provisionable",),
    )


class TestNormalisation:
    @pytest.mark.parametrize(
        "written", ["SM8650", "sm8650", "SM-8650", "sm 8650", "Sm_8650"]
    )
    def test_spelling_variants_collapse(self, written: str) -> None:
        """Fleet exports spell SoC codes a dozen ways."""
        assert normalise_soc(written) == "SM8650"

    def test_distinct_socs_stay_distinct(self) -> None:
        assert normalise_soc("SM8650") != normalise_soc("SM8550")


class TestMatching:
    def test_matches_the_primary_code(self, inventory: Inventory) -> None:
        assert [d.soc for d in inventory.matching("SM8650")] == ["sm8650"]

    def test_matches_a_marketing_alias(self, inventory: Inventory) -> None:
        """AI Hub lists both sm8650 and qualcomm-snapdragon-8gen3 for one device."""
        assert inventory.matching("qualcomm-snapdragon-8gen3")

    def test_unknown_soc_matches_nothing(self, inventory: Inventory) -> None:
        assert inventory.matching("MT6989") == ()

    def test_reachable_is_a_subset(self, inventory: Inventory) -> None:
        assert len(inventory.reachable) == 1
        assert len(inventory.devices) == 4


class TestParseFleet:
    def test_reads_percentages(self) -> None:
        entries = parse_fleet("SM8650,22%\nSM8550,18%")
        assert [(e.soc, e.share) for e in entries] == [("SM8650", 22.0), ("SM8550", 18.0)]

    def test_skips_a_header(self) -> None:
        assert len(parse_fleet("soc,share\nSM8650,22%")) == 1

    def test_reads_fractions_as_percentages(self) -> None:
        """Analytics exports emit 0.22 as often as 22%."""
        assert parse_fleet("SM8650,0.22")[0].share == pytest.approx(22.0)

    def test_bare_integers_are_already_percentages(self) -> None:
        assert parse_fleet("SM8650,22")[0].share == pytest.approx(22.0)

    def test_ignores_extra_columns(self) -> None:
        entries = parse_fleet("SM8650,22%,Galaxy S24,someone's note")
        assert entries[0].soc == "SM8650" and entries[0].share == 22.0

    def test_an_unreadable_share_is_dropped_not_guessed(self) -> None:
        """Forgiving about format, strict about meaning."""
        assert parse_fleet("SM8650,lots\nSM8550,18%") == parse_fleet("SM8550,18%")

    def test_blank_and_short_lines_are_ignored(self) -> None:
        assert parse_fleet("\nSM8650\n\nSM8550,18%\n") == parse_fleet("SM8550,18%")


class TestCoverage:
    def test_separates_covered_from_reachable(self, inventory: Inventory) -> None:
        """The whole point. 66% recognised, 6% measurable is a very different promise."""
        coverage = resolve_fleet(
            parse_fleet("SM8650,22%\nApple M2,6%\nMT6989,20%"), inventory
        )
        assert coverage.covered_share == pytest.approx(28.0)
        assert coverage.reachable_share == pytest.approx(6.0)

    def test_reports_unknown_socs(self, inventory: Inventory) -> None:
        coverage = resolve_fleet(parse_fleet("MT6989,20%"), inventory)
        assert [t.entry.soc for t in coverage.unknown] == ["MT6989"]

    def test_reports_known_but_unreachable_separately(self, inventory: Inventory) -> None:
        coverage = resolve_fleet(parse_fleet("SM8650,22%"), inventory)
        assert coverage.unknown == ()
        assert [t.entry.soc for t in coverage.unreachable] == ["SM8650"]

    def test_status_labels_are_distinct(self, inventory: Inventory) -> None:
        coverage = resolve_fleet(
            parse_fleet("Apple M2,6%\nSM8650,22%\nMT6989,20%"), inventory
        )
        assert [t.status for t in coverage.targets] == [
            "reachable", "known but unreachable", "unknown",
        ]

    def test_total_share_is_reported_as_given(self, inventory: Inventory) -> None:
        """Shares that do not sum to 100 are surfaced, never silently normalised."""
        coverage = resolve_fleet(parse_fleet("SM8650,22%\nSM8550,18%"), inventory)
        assert coverage.total_share == pytest.approx(40.0)


class TestSuggestions:
    def test_suggests_near_misses(self, inventory: Inventory) -> None:
        """An unrecognised SoC is usually a naming mismatch, not an unknown chip."""
        hints = suggest_aliases(FleetEntry(soc="sm8651", share=1.0), inventory)
        assert "sm8650" in hints

    def test_offers_nothing_for_a_genuinely_foreign_soc(self, inventory: Inventory) -> None:
        assert suggest_aliases(FleetEntry(soc="MT6989", share=1.0), inventory) == []


class TestQaiHubCache:
    def test_missing_cache_is_a_note_not_a_crash(self, tmp_path) -> None:
        inventory = qai_hub_inventory(tmp_path / "absent.json")
        assert inventory.devices == ()
        assert any("refresh" in note for note in inventory.notes)

    def test_prefers_the_silicon_code_over_the_marketing_name(self, tmp_path) -> None:
        """A fleet export contains sm8650, not qualcomm-snapdragon-8gen3."""
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"devices": [{
            "name": "Samsung Galaxy S24", "os": "14",
            "attributes": ["chipset:qualcomm-snapdragon-8gen3", "chipset:sm8650",
                           "os:android", "format:phone", "hexagon:v75",
                           "htp-supports-fp16:true", "framework:tflite"],
        }]}))
        device = qai_hub_inventory(path).devices[0]
        assert device.soc == "sm8650"
        assert "qualcomm-snapdragon-8gen3" in device.soc_aliases
        assert device.accelerator == "v75"
        assert device.supports_fp16 is True

    def test_catalogue_entries_default_to_unreachable(self, tmp_path) -> None:
        """A catalogue entry is not capacity."""
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"devices": [
            {"name": "d", "os": "14", "attributes": ["chipset:sm8650"]}
        ]}))
        inventory = qai_hub_inventory(path, unreachable_reason="no entitlement")
        assert inventory.devices[0].reachable is False
        assert inventory.devices[0].unreachable_reason == "no entitlement"
