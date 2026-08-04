"""Shared fixtures. Deliberately hand-built so tests don't depend on real hardware."""

from __future__ import annotations

import pytest

from edgefit.schema import (
    DeviceFingerprint,
    HostState,
    ModelRef,
    OrtProvider,
    OrtRuntimeConfig,
    PowerSource,
    Recipe,
    TaskType,
    ThermalState,
)

MINILM = "hf:sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def device() -> DeviceFingerprint:
    return DeviceFingerprint(
        kind="host",
        model="Mac14,2",
        soc="Apple M2",
        arch="arm64",
        cpu_cores_total=8,
        cpu_cores_performance=4,
        cpu_cores_efficiency=4,
        ram_bytes=16 * 1024**3,
        os_name="macOS",
        os_version="15.2",
        os_build="24C101",
    )


@pytest.fixture
def host_state() -> HostState:
    return HostState(
        power_source=PowerSource.AC,
        battery_percent=100.0,
        low_power_mode=False,
        thermal_state=ThermalState.NOMINAL,
        load_avg_1m=0.4,
        load_avg_5m=0.5,
        available_ram_bytes=9 * 1024**3,
        unavailable={"cpu_temperature_c": "no unprivileged temperature sensor on Apple Silicon"},
    )


@pytest.fixture
def cpu_recipe() -> Recipe:
    return Recipe(
        model=ModelRef(ref=MINILM, task=TaskType.EMBED),
        runtime=OrtRuntimeConfig(providers=(OrtProvider.CPU,)),
    )


@pytest.fixture
def coreml_recipe() -> Recipe:
    return Recipe(
        model=ModelRef(ref=MINILM, task=TaskType.EMBED),
        runtime=OrtRuntimeConfig(providers=(OrtProvider.COREML, OrtProvider.CPU)),
    )
