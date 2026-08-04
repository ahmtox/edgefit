# EdgeFit

A deployment compiler for on-device AI. Give it a model, target devices, and
constraints; it searches the space of deployment configurations, measures the
promising ones on real hardware, and returns the optimal configuration plus the
compiled artifact plus a proof that it hits the budget.

**Status: pre-product.** What exists today is pass 1 — the measurement harness.
Everything downstream depends on the numbers being trustworthy, so that came
first. See [docs/PROJECT.md](docs/PROJECT.md) for the product, and
[docs/STATUS.md](docs/STATUS.md) for what is actually built.

---

## What works today

`(model, recipe, device) → measurement`, on ONNX Runtime with CPU and CoreML
execution providers, with the result written immutably to a local corpus.

```bash
uv sync --extra export --group dev

uv run edgefit doctor        # is this host fit to measure on?
uv run edgefit models        # what subjects are registered
uv run edgefit measure --model hf:sentence-transformers/all-MiniLM-L6-v2 \
                       --recipe recipes/ort_coreml_fp32.yaml
uv run edgefit corpus list
uv run edgefit corpus export # Parquet + CSV
uv run edgefit verify        # golden fixtures — the gate for everything after
```

## The first finding

On Apple M2 with ONNX Runtime 1.28, the CoreML execution provider claims roughly
half the *nodes* of `all-MiniLM-L6-v2` and leaves **99.5% of the arithmetic** on
CPU — every MatMul, every LayerNormalization, fragmented across 37 partitions.
Nothing errors, nothing warns, and enabling CoreML makes the model 2.3× slower.

That is the failure mode EdgeFit exists to find, and it is why fallback is
recorded three independent ways (node share, FLOP share, measured time share).
A team reading only node share would conclude the delegate was working.

## Why the harness refuses to run

Measurement trust is the asset. If one hallucinated number enters the corpus,
every model trained on it is compromised, so the harness is built to fail loudly
rather than produce a plausible wrong answer:

- **A preflight gate** checks AC power, low-power mode, thermal state, load
  average and free memory, and refuses if any fails. On a laptop with a browser
  open, it refuses — correctly.
- **A measured throttle probe** times a fixed kernel against the host's own
  recorded best, because Apple Silicon exposes no unprivileged temperature and
  inventing one is worse than admitting it.
- **Variance is mandatory.** `RunStats` can only be built from raw samples and
  revalidates its own aggregates, so a fabricated standard deviation cannot be
  represented.
- **The corpus is insert-only.** No update, no delete, anywhere.
- **Unavailable values are null plus a written reason**, never a placeholder.
- **Both cascade tiers run out of process**, so a delegate that aborts the
  interpreter becomes a recorded failure instead of a dead sweep.

## Layout

```
src/edgefit/
  schema/     recipe, measurement, fingerprint, host records
  corpus/     insert-only DuckDB store + Parquet export
  harness/    host probes, preflight gate, run protocol, subprocess workers
  backends/   ONNX Runtime; graph/FLOP/EP-placement analysis
  models/     registered model subjects
  cli/        typer entry point
tests/
  golden/     known-answer fixtures (marked `device`)
```

```bash
uv run pytest -m "not device"   # fast suite, no hardware
uv run ruff check .
```
