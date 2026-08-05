"""Device inventory — what we can target, and what we can actually reach.

PROJECT.md §5.1 requires that a submitted job's targets "exist in inventory", and
Stage 2 input #2 lets a customer hand over a device-distribution CSV instead of a
device list. Both need one answer to "what devices do we know about", spanning
hardware we own and hardware someone else hosts.

The distinction this module insists on is **known vs reachable**. A hosted farm can
list a device in its catalogue that our account cannot provision, and treating the
catalogue as capacity is how a fleet-coverage report ends up promising something we
cannot deliver. Every entry therefore carries ``reachable`` and, when false, the
reason.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

QAI_HUB_CACHE = Path(".edgefit/qai_hub_devices.json")

#: SoC codes are written a dozen ways across analytics exports and vendor docs.
#: Normalising to uppercase-alphanumeric makes "SM-8650", "sm8650" and "SM8650"
#: the same key without pretending to understand marketing names.
_NORMALISE = re.compile(r"[^A-Z0-9]")


def normalise_soc(value: str) -> str:
    return _NORMALISE.sub("", value.upper())


@dataclass(frozen=True)
class InventoryDevice:
    """One targetable device, owned or hosted."""

    source: str
    """``local`` for hardware we own, or the name of the farm hosting it."""

    name: str
    soc: str
    os_name: str
    os_version: str
    soc_aliases: tuple[str, ...] = ()
    form_factor: str | None = None
    frameworks: tuple[str, ...] = ()
    accelerator: str | None = None
    supports_fp16: bool | None = None
    reachable: bool = True
    unreachable_reason: str | None = None

    @property
    def keys(self) -> frozenset[str]:
        """Every normalised SoC string that should match this device."""
        return frozenset(normalise_soc(v) for v in (self.soc, *self.soc_aliases) if v)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.os_name} {self.os_version})"


@dataclass(frozen=True)
class Inventory:
    devices: tuple[InventoryDevice, ...] = ()
    notes: tuple[str, ...] = field(default=())
    """Caveats that must travel with the inventory, e.g. why nothing is reachable."""

    @property
    def reachable(self) -> tuple[InventoryDevice, ...]:
        return tuple(device for device in self.devices if device.reachable)

    def matching(self, soc: str) -> tuple[InventoryDevice, ...]:
        key = normalise_soc(soc)
        return tuple(device for device in self.devices if key in device.keys)

    def socs(self) -> dict[str, tuple[InventoryDevice, ...]]:
        grouped: dict[str, list[InventoryDevice]] = {}
        for device in self.devices:
            grouped.setdefault(device.soc, []).append(device)
        return {soc: tuple(items) for soc, items in sorted(grouped.items())}


def local_inventory() -> Inventory:
    """The hardware we own — currently whatever host this is running on."""
    from edgefit.harness.hostinfo import probe_device  # noqa: PLC0415

    device = probe_device()
    return Inventory(
        devices=(
            InventoryDevice(
                source="local",
                name=device.model,
                soc=device.soc,
                os_name=device.os_name,
                os_version=f"{device.os_version} ({device.os_build})",
                form_factor=device.kind,
                frameworks=("onnxruntime",),
                reachable=True,
            ),
        )
    )


def _attribute(attributes: list[str], prefix: str) -> str | None:
    for attribute in attributes:
        if attribute.startswith(prefix):
            return attribute[len(prefix) :]
    return None


def _attributes(attributes: list[str], prefix: str) -> tuple[str, ...]:
    return tuple(a[len(prefix) :] for a in attributes if a.startswith(prefix))


def qai_hub_inventory(
    cache_path: Path | str = QAI_HUB_CACHE,
    *,
    reachable: bool = False,
    unreachable_reason: str | None = None,
) -> Inventory:
    """Load the Qualcomm AI Hub catalogue from its local cache.

    Read from a cache rather than the network so the inventory works offline, is
    diffable in review, and does not make an atlas build depend on a vendor's
    uptime. ``edgefit devices refresh`` updates it.

    ``reachable`` defaults to False because a catalogue entry is not capacity: a farm
    can list a device our account cannot provision. ``combined_inventory`` passes True,
    having verified we measure on these. See :data:`QAI_HUB_LIMITS` for what that
    reachability does and does not include.
    """
    path = Path(cache_path)
    if not path.exists():
        return Inventory(notes=(f"no AI Hub cache at {path}; run `edgefit devices refresh`",))

    payload = json.loads(path.read_text())
    devices = []
    for record in payload.get("devices", []):
        attributes = list(record.get("attributes", []))
        chipsets = _attributes(attributes, "chipset:")
        # AI Hub lists both the silicon code (sm8650) and the marketing name
        # (qualcomm-snapdragon-8gen3). The silicon code is what a fleet export
        # contains, so it is the primary key and the rest are aliases.
        primary = next((c for c in chipsets if "snapdragon" not in c), None) or (
            chipsets[0] if chipsets else "unknown"
        )
        fp16 = _attribute(attributes, "htp-supports-fp16:")
        devices.append(
            InventoryDevice(
                source="qai_hub",
                name=record["name"],
                soc=primary,
                soc_aliases=tuple(c for c in chipsets if c != primary),
                os_name=_attribute(attributes, "os:") or "unknown",
                os_version=str(record.get("os", "unknown")),
                form_factor=_attribute(attributes, "format:"),
                frameworks=_attributes(attributes, "framework:"),
                accelerator=_attribute(attributes, "hexagon:"),
                supports_fp16={"true": True, "false": False}.get(fp16 or ""),
                reachable=reachable,
                unreachable_reason=None if reachable else unreachable_reason,
            )
        )
    notes = () if reachable else (unreachable_reason,) if unreachable_reason else ()
    return Inventory(devices=tuple(devices), notes=notes)


#: What AI Hub reachability includes. Stated once, carried everywhere.
#:
#: This constant has been wrong twice, in opposite directions. It first claimed a
#: Qualcomm entitlement blocked provisioning (a generalisation from the one broken job
#: type), then that our own backend was missing — which stayed after the backend
#: shipped, so the inventory reported "1 of 80 reachable" on the day three of those
#: devices wrote corpus rows. Reachability is a fact about the harness, and it has to
#: move when the harness does.
#: Reachable as of 2026-08-05, when the profile backend landed and wrote real rows on
#: three of these devices. Kept as a *note* rather than an unreachability reason,
#: because the limits are on what we can ask for, not on whether we can measure at all.
QAI_HUB_LIMITS = (
    "Qualcomm AI Hub devices are reachable: the profile backend measures on them and "
    "writes third-party rows. Two limits travel with that. Compile jobs are still "
    "rejected server-side, so recipes are fp32 ONNX only — no .tflite or QNN artifact, "
    "and therefore no Qualcomm-side quantization or delegate axis; the one live recipe "
    "axis is compute_unit. And profile jobs synthesize their own random inputs, so a "
    "model with integer index inputs cannot be profiled at all and is refused rather "
    "than failed against the device. Full diagnosis: docs/qai-hub-device-access.md"
)


def combined_inventory(cache_path: Path | str = QAI_HUB_CACHE) -> Inventory:
    """Everything we know about, owned and hosted, with reachability marked.

    Hosted devices were hardcoded unreachable with the reason "the AI Hub measurement
    backend is not implemented". It is now, and the inventory reported "1 of 80
    reachable" on the same day three of those devices produced corpus rows. Reachability
    is a fact about the harness, so it has to move when the harness does.
    """
    local = local_inventory()
    hosted = qai_hub_inventory(cache_path, reachable=True, unreachable_reason=None)
    return Inventory(
        devices=local.devices + hosted.devices,
        notes=local.notes + hosted.notes + (QAI_HUB_LIMITS,),
    )


def refresh_qai_hub_cache(cache_path: Path | str = QAI_HUB_CACHE) -> int:
    """Re-fetch the AI Hub catalogue. Requires the `qai-hub` package and a token."""
    import qai_hub as hub  # noqa: PLC0415

    devices = [
        {"name": device.name, "os": device.os, "attributes": sorted(device.attributes)}
        for device in hub.get_devices()
    ]
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_from": "qai_hub", "devices": devices}, indent=1))
    return len(devices)
