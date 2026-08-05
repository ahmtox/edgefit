"""Device inventory and fleet resolution."""

from edgefit.devices.fleet import (
    FleetCoverage,
    FleetEntry,
    ResolvedTarget,
    load_fleet,
    parse_fleet,
    resolve_fleet,
    suggest_aliases,
)
from edgefit.devices.inventory import (
    QAI_HUB_LIMITS,
    Inventory,
    InventoryDevice,
    combined_inventory,
    local_inventory,
    normalise_soc,
    qai_hub_inventory,
    refresh_qai_hub_cache,
)

__all__ = [
    "QAI_HUB_LIMITS",
    "FleetCoverage",
    "FleetEntry",
    "Inventory",
    "InventoryDevice",
    "ResolvedTarget",
    "combined_inventory",
    "load_fleet",
    "local_inventory",
    "normalise_soc",
    "parse_fleet",
    "qai_hub_inventory",
    "refresh_qai_hub_cache",
    "resolve_fleet",
    "suggest_aliases",
]
