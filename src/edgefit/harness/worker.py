"""Measurement subprocess — runs one tier of the cascade in a fresh interpreter.

Both tiers run out-of-process, and the reason is not theoretical. Measuring
all-MiniLM-L6-v2 through the CoreML EP with ORT graph optimisation enabled trips
an MPSGraph assertion:

    'mps.matmul' op contracting dimensions differ 1 & 384
    failed assertion `original module failed verification'

which aborts the process outright (SIGABRT). In-process, that takes down the
whole sweep and leaves nothing behind. Out-of-process, the parent records a
``runtime_failure`` row and carries on — and per §5.8 that row is exactly what
trains the tier-1 static filter to stop proposing the recipe next time.

Isolation also keeps peak RSS attributable: each tier reports its own
``RUSAGE_SELF`` high-water mark, so analysis overhead never inflates the memory
figure attributed to inference.

Protocol: a JSON job on stdin, a JSON result on stdout. Deliberately dumb, so
the same worker can eventually run on a phone over adb.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from edgefit.schema.recipe import Recipe


def _analyze(job: dict) -> dict:
    from edgefit.backends import get_backend  # noqa: PLC0415

    recipe = Recipe.model_validate(job["recipe"])
    analysis = get_backend(recipe.runtime.kind).analyze(Path(job["artifact_dir"]), recipe)
    return {
        "lowered": analysis.lowered,
        "artifact_bytes": analysis.artifact_bytes,
        "failure_reason": analysis.failure_reason,
        "lowering_ms": analysis.lowering_ms,
        "fingerprint": (
            analysis.fingerprint.model_dump(mode="json") if analysis.fingerprint else None
        ),
        "fallback": analysis.fallback.model_dump(mode="json") if analysis.fallback else None,
        "fallback_as_run": (
            analysis.fallback_as_run.model_dump(mode="json")
            if analysis.fallback_as_run
            else None
        ),
    }


def _measure(job: dict) -> dict:
    from edgefit.backends import get_backend  # noqa: PLC0415

    recipe = Recipe.model_validate(job["recipe"])
    result = get_backend(recipe.runtime.kind).measure(
        Path(job["artifact_dir"]),
        recipe,
        runs=int(job["runs"]),
        warmup=int(job["warmup"]),
    )
    return asdict(result)


_HANDLERS = {"analyze": _analyze, "measure": _measure}


def run_job(job: dict) -> dict:
    """Execute one job and return a serialisable result."""
    mode = job.get("mode", "measure")
    handler = _HANDLERS.get(mode)
    if handler is None:
        raise ValueError(f"unknown job mode {mode!r}")
    return handler(job)


def main() -> int:
    try:
        job = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        json.dump({"failure_reason": f"malformed job: {exc}"}, sys.stdout)
        return 2

    try:
        result = run_job(job)
    except Exception as exc:  # noqa: BLE001 - a failed run is a record, not a traceback
        result = {"failure_reason": f"{type(exc).__name__}: {exc}"}

    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
