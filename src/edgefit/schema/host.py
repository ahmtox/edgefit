"""Device identity and host condition at measurement time (PROJECT.md §6.2).

Two separate things, deliberately:

* ``DeviceFingerprint`` — what the machine *is*. Stable across runs. The atlas
  device axis.
* ``HostState`` — what the machine *was doing* when we measured. Varies per run,
  and is the difference between a number you can publish and a number you can't.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from edgefit.schema.common import PowerSource, ThermalState, content_hash


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class DeviceFingerprint(_Frozen):
    """Identity of the physical unit. Every measurement carries one."""

    kind: str = Field(description="host | phone | tablet | sbc | virtual")
    model: str = Field(description="e.g. Mac14,2, SM-S928B")
    soc: str = Field(description="e.g. Apple M2, SM8650")
    arch: str
    cpu_cores_total: int
    cpu_cores_performance: int | None = None
    cpu_cores_efficiency: int | None = None
    ram_bytes: int
    os_name: str
    os_version: str
    os_build: str = Field(description="Exact build. OS updates silently change delegates.")
    # Distinguishes two physical units of the same SKU — the two-unit test
    # (PROJECT.md §9) is meaningless without it.
    unit_serial_hash: str | None = None

    @property
    def device_id(self) -> str:
        """Stable id for this unit in this OS state.

        Includes os_build on purpose: the same hardware on a new OS build is a
        different measurement target, which is the whole premise of Stage 3.
        """
        return content_hash(self.model_dump(mode="json"))

    @property
    def sku_id(self) -> str:
        """Identity of the SKU, ignoring OS and individual unit."""
        return content_hash({"model": self.model, "soc": self.soc, "ram_bytes": self.ram_bytes})


class HostState(_Frozen):
    """Conditions at the start of a measurement.

    ``unavailable`` maps field name -> reason. Hard rule #1: a value we cannot
    obtain is null with an explanation, never a plausible-looking placeholder.
    """

    power_source: PowerSource
    battery_percent: float | None = None
    low_power_mode: bool | None = None
    thermal_state: ThermalState = ThermalState.UNAVAILABLE
    load_avg_1m: float | None = None
    load_avg_5m: float | None = None
    available_ram_bytes: int | None = None
    cpu_temperature_c: float | None = Field(
        default=None,
        description="Almost always null: no unprivileged temperature on Apple Silicon.",
    )
    unavailable: dict[str, str] = Field(default_factory=dict)
