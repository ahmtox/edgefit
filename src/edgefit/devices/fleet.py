"""Fleet resolution — PROJECT.md §4 Stage 2, input #2.

> *fleet paste*: a device-distribution CSV from their analytics
> (`SM8650, 22%` …). We map SoC codes to inventory and optimize weighted by real
> distribution.

The point of asking for the real distribution instead of a device list is that it
changes the answer: optimising for the median device is not the same as optimising
for the device 22% of users actually hold.

Two numbers come out of this, and conflating them would be the whole failure mode:

* **covered** — the share of the fleet whose SoC we recognise at all
* **reachable** — the share we can actually measure on today

A coverage report that quotes only the first is a sales document. The gap between
them is the honest answer to "what can you do for me right now".
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from edgefit.devices.inventory import Inventory, InventoryDevice, normalise_soc


@dataclass(frozen=True)
class FleetEntry:
    """One line of a customer's device distribution."""

    soc: str
    share: float
    raw: str = ""


@dataclass(frozen=True)
class ResolvedTarget:
    entry: FleetEntry
    devices: tuple[InventoryDevice, ...]

    @property
    def known(self) -> bool:
        return bool(self.devices)

    @property
    def reachable(self) -> bool:
        return any(device.reachable for device in self.devices)

    @property
    def status(self) -> str:
        if self.reachable:
            return "reachable"
        return "known but unreachable" if self.known else "unknown"


@dataclass(frozen=True)
class FleetCoverage:
    targets: tuple[ResolvedTarget, ...]
    total_share: float

    @property
    def covered_share(self) -> float:
        return sum(t.entry.share for t in self.targets if t.known)

    @property
    def reachable_share(self) -> float:
        return sum(t.entry.share for t in self.targets if t.reachable)

    @property
    def unknown(self) -> tuple[ResolvedTarget, ...]:
        return tuple(t for t in self.targets if not t.known)

    @property
    def unreachable(self) -> tuple[ResolvedTarget, ...]:
        return tuple(t for t in self.targets if t.known and not t.reachable)

    def summary(self) -> str:
        return (
            f"{self.covered_share:.0f}% of the stated fleet is in inventory; "
            f"{self.reachable_share:.0f}% is measurable today"
        )


def parse_fleet(text: str) -> list[FleetEntry]:
    """Parse a device-distribution CSV.

    Deliberately forgiving about formatting, because this is pasted out of someone's
    analytics dashboard: an optional header, shares written as ``22%`` or ``0.22`` or
    ``22``, and extra columns ignored. Forgiving about *format*, strict about
    *meaning* — a row whose share cannot be read is reported, never guessed at.
    """
    entries: list[FleetEntry] = []
    reader = csv.reader(io.StringIO(text))
    for fields in reader:
        cells = [cell.strip() for cell in fields if cell.strip()]
        if len(cells) < 2:
            continue
        soc, share_text = cells[0], cells[1]
        if not soc or soc.lower() in {"soc", "chipset", "device", "model"}:
            continue  # header row
        cleaned = share_text.rstrip("%").replace(",", "")
        try:
            share = float(cleaned)
        except ValueError:
            continue
        # A bare fraction is a fraction; anything above 1 is already a percentage.
        # Ambiguous only at exactly 1, where "1%" and "100%" would both be defensible
        # readings — treated as 1% because fleet lines are conventionally percentages.
        if share <= 1 and "%" not in share_text and "." in cleaned:
            share *= 100
        entries.append(FleetEntry(soc=soc, share=share, raw=",".join(cells)))
    return entries


def resolve_fleet(entries: list[FleetEntry], inventory: Inventory) -> FleetCoverage:
    """Map each fleet entry onto inventory devices."""
    targets = tuple(
        ResolvedTarget(entry=entry, devices=inventory.matching(entry.soc)) for entry in entries
    )
    return FleetCoverage(targets=targets, total_share=sum(e.share for e in entries))


def load_fleet(path: Path | str, inventory: Inventory) -> FleetCoverage:
    return resolve_fleet(parse_fleet(Path(path).read_text()), inventory)


def suggest_aliases(entry: FleetEntry, inventory: Inventory, limit: int = 3) -> list[str]:
    """Near-miss inventory SoCs for an unmatched entry.

    An unrecognised SoC is usually a naming mismatch rather than an unknown chip, and
    saying "did you mean SM8650" is far more useful than "not found".
    """
    key = normalise_soc(entry.soc)
    if not key:
        return []
    scored: list[tuple[int, str]] = []
    for soc in inventory.socs():
        candidate = normalise_soc(soc)
        shared = len(set(key) & set(candidate))
        prefix = 0
        for a, b in zip(key, candidate, strict=False):
            if a != b:
                break
            prefix += 1
        if prefix >= 2 or shared >= max(3, len(key) - 2):
            scored.append((prefix * 10 + shared, soc))
    return [soc for _, soc in sorted(scored, reverse=True)[:limit]]
