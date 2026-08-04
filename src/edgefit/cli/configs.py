"""Load a ConfigRecord from YAML.

Config files describe everything *except* the model, so one file can be applied
across every subject in the registry. That separation is what makes a sweep a
cross product rather than a pile of near-duplicate files.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from edgefit.models.registry import resolve
from edgefit.schema.config import ConfigRecord, ModelRef

CONFIG_DIR = Path("configs")


def load_config(path: Path | str, model_ref: str) -> ConfigRecord:
    """Build a ConfigRecord from a YAML file plus a model reference."""
    payload = yaml.safe_load(Path(path).read_text()) or {}
    spec = resolve(model_ref)
    payload["model"] = ModelRef(ref=spec.ref, task=spec.task).model_dump(mode="json")
    return ConfigRecord.model_validate(payload)


def available_configs(directory: Path | str = CONFIG_DIR) -> list[Path]:
    return sorted(Path(directory).glob("*.yaml"))
