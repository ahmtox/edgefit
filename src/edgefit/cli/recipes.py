"""Load a Recipe from YAML.

Recipe files describe everything *except* the model, so one file applies across
every subject in the registry. That separation is what makes a sweep a cross
product rather than a pile of near-duplicate files.

They also compose. A file may name a base with ``extends:``, and its own keys are
deep-merged over it — PROJECT.md §6.1 wants recipes to inherit from expert-vetted
defaults, and `recipes/` is that default library. Expressing "the baseline, but on
the Neural Engine" as a two-line file rather than a copy keeps the library honest
as it grows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from edgefit.models.registry import resolve
from edgefit.schema.recipe import ModelRef, Recipe

RECIPE_DIR = Path("recipes")

# Guards an extends cycle into a message rather than a stack overflow.
_MAX_INHERITANCE_DEPTH = 8


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """One level deep, matching the shape of a recipe: sections of flat knobs."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merged[key] | value
        else:
            merged[key] = value
    return merged


def load_recipe_payload(path: Path | str, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a recipe file, resolving any ``extends:`` chain."""
    resolved = Path(path).resolve()
    if resolved in _seen:
        chain = " -> ".join(p.name for p in (*_seen, resolved))
        raise ValueError(f"circular recipe inheritance: {chain}")
    if len(_seen) >= _MAX_INHERITANCE_DEPTH:
        raise ValueError(f"recipe inheritance deeper than {_MAX_INHERITANCE_DEPTH} levels")

    payload = yaml.safe_load(Path(path).read_text()) or {}
    parent = payload.pop("extends", None)
    if parent is None:
        return payload

    # Resolved relative to the child, so the library stays movable as a directory.
    base = load_recipe_payload((Path(path).parent / parent), (*_seen, resolved))
    return _merge(base, payload)


def load_recipe(path: Path | str, model_ref: str) -> Recipe:
    """Build a Recipe from a YAML file plus a model reference."""
    payload = load_recipe_payload(path)
    spec = resolve(model_ref)
    payload["model"] = ModelRef(ref=spec.ref, task=spec.task).model_dump(mode="json")
    return Recipe.model_validate(payload)


def available_recipes(directory: Path | str = RECIPE_DIR) -> list[Path]:
    return sorted(Path(directory).glob("*.yaml"))
