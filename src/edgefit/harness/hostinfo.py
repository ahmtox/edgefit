"""Probe what this machine is, and what it is doing right now.

Split deliberately (see ``edgefit.schema.host``): identity is stable and indexes
the atlas; condition varies per run and decides whether a number is publishable.

Everything here degrades honestly. When a value cannot be obtained we return
``None`` and record *why* in ``HostState.unavailable`` — never a placeholder
(PROJECT.md §14.1).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import psutil

from edgefit.schema.common import PowerSource, ThermalState, content_hash
from edgefit.schema.host import DeviceFingerprint, HostState

_TIMEOUT_S = 5.0


def _run(command: list[str]) -> str | None:
    """Best-effort shell-out. Returns None rather than raising: an unavailable
    probe is a fact to record, not a reason to abort a measurement."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _sysctl(name: str) -> str | None:
    return _run(["sysctl", "-n", name])


def _sysctl_int(name: str) -> int | None:
    raw = _sysctl(name)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Device identity
# --------------------------------------------------------------------------


def _macos_serial_hash() -> str | None:
    """Hashed, never raw.

    We need to tell two units of one SKU apart for the two-unit test; we do not
    need — and should not hold — the serial number itself.
    """
    output = _run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
    if not output:
        return None
    for line in output.splitlines():
        if "IOPlatformSerialNumber" in line and '"' in line:
            serial = line.rsplit('"', 2)[-2]
            return content_hash(serial, length=12) if serial else None
    return None


def _macos_device() -> DeviceFingerprint:
    os_version = platform.mac_ver()[0] or "unknown"
    os_build = _run(["sw_vers", "-buildVersion"]) or "unknown"
    total_cores = _sysctl_int("hw.ncpu") or os.cpu_count() or 1

    return DeviceFingerprint(
        kind="host",
        model=_sysctl("hw.model") or "unknown",
        soc=_sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        arch=platform.machine(),
        cpu_cores_total=total_cores,
        # perflevel0 is the performance cluster, perflevel1 the efficiency
        # cluster. Absent on Intel Macs, which is why these are optional.
        cpu_cores_performance=_sysctl_int("hw.perflevel0.logicalcpu"),
        cpu_cores_efficiency=_sysctl_int("hw.perflevel1.logicalcpu"),
        ram_bytes=_sysctl_int("hw.memsize") or psutil.virtual_memory().total,
        os_name="macOS",
        os_version=os_version,
        os_build=os_build,
        unit_serial_hash=_macos_serial_hash(),
    )


def _generic_device() -> DeviceFingerprint:
    """Fallback for Linux hosts (the Android device box, eventually).

    Intentionally coarse: better to say "unknown" than to guess a SoC name that
    then becomes a corpus key.
    """
    return DeviceFingerprint(
        kind="host",
        model=platform.node() or "unknown",
        soc=platform.processor() or "unknown",
        arch=platform.machine(),
        cpu_cores_total=os.cpu_count() or 1,
        ram_bytes=psutil.virtual_memory().total,
        os_name=platform.system() or "unknown",
        os_version=platform.release() or "unknown",
        os_build=platform.version() or "unknown",
    )


def probe_device() -> DeviceFingerprint:
    """Identify this machine."""
    return _macos_device() if sys.platform == "darwin" else _generic_device()


# --------------------------------------------------------------------------
# Host condition
# --------------------------------------------------------------------------


_THERMAL_BY_NS_VALUE = {
    0: ThermalState.NOMINAL,
    1: ThermalState.FAIR,
    2: ThermalState.SERIOUS,
    3: ThermalState.CRITICAL,
}


def _macos_process_info() -> tuple[ThermalState, bool | None, str | None]:
    """Thermal state and low-power mode via NSProcessInfo.

    The only unprivileged thermal signal Apple Silicon offers. Coarse — four
    buckets, no temperature — but real, which beats a number we made up.
    """
    try:
        from Foundation import NSProcessInfo  # noqa: PLC0415
    except ImportError:
        return ThermalState.UNAVAILABLE, None, "pyobjc-framework-Cocoa is not installed"

    try:
        info = NSProcessInfo.processInfo()
        thermal = _THERMAL_BY_NS_VALUE.get(int(info.thermalState()), ThermalState.UNAVAILABLE)
        return thermal, bool(info.isLowPowerModeEnabled()), None
    except Exception as exc:  # noqa: BLE001 - a probe must never take down a run
        return ThermalState.UNAVAILABLE, None, f"NSProcessInfo query failed: {exc}"


def _macos_power() -> tuple[PowerSource, float | None]:
    """Parse ``pmset -g ps``.

    Battery power means DVFS and thermal behaviour that will not match a
    plugged-in run, so this is a gating input, not a footnote.
    """
    output = _run(["pmset", "-g", "ps"])
    if not output:
        return PowerSource.UNKNOWN, None

    source = PowerSource.UNKNOWN
    if "'AC Power'" in output:
        source = PowerSource.AC
    elif "'Battery Power'" in output:
        source = PowerSource.BATTERY

    percent: float | None = None
    for token in output.replace(";", " ").split():
        if token.endswith("%"):
            try:
                percent = float(token.rstrip("%"))
            except ValueError:
                percent = None
            break
    return source, percent


def probe_state() -> HostState:
    """Capture the conditions a measurement is about to be taken under."""
    unavailable: dict[str, str] = {
        "cpu_temperature_c": (
            "no unprivileged temperature sensor on Apple Silicon "
            "(powermetrics requires root; pmset -g therm reports nothing)"
        )
        if sys.platform == "darwin"
        else "no temperature probe implemented for this platform"
    }

    try:
        load_1m, load_5m, _ = os.getloadavg()
    except OSError:
        load_1m = load_5m = None  # type: ignore[assignment]
        unavailable["load_avg_1m"] = "getloadavg unavailable on this platform"

    if sys.platform == "darwin":
        power_source, battery_percent = _macos_power()
        thermal_state, low_power_mode, thermal_error = _macos_process_info()
        if thermal_error:
            unavailable["thermal_state"] = thermal_error
            unavailable["low_power_mode"] = thermal_error
        if power_source is PowerSource.UNKNOWN:
            unavailable["power_source"] = "pmset -g ps could not be parsed"
    else:
        battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        if battery is None:
            power_source, battery_percent = PowerSource.UNKNOWN, None
            unavailable["power_source"] = "no battery/AC information exposed by this platform"
        else:
            power_source = PowerSource.AC if battery.power_plugged else PowerSource.BATTERY
            battery_percent = battery.percent
        thermal_state, low_power_mode = ThermalState.UNAVAILABLE, None
        unavailable["thermal_state"] = "no thermal API implemented for this platform"
        unavailable["low_power_mode"] = "no low-power-mode API implemented for this platform"

    return HostState(
        power_source=power_source,
        battery_percent=battery_percent,
        low_power_mode=low_power_mode,
        thermal_state=thermal_state,
        load_avg_1m=load_1m,
        load_avg_5m=load_5m,
        available_ram_bytes=psutil.virtual_memory().available,
        cpu_temperature_c=None,
        unavailable=unavailable,
    )
