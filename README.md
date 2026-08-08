# EdgeFit

Neutral, reproducible measurement of on-device AI inference — across silicon vendors,
including the failures.

Longer term this is a deployment compiler: give it a model, target devices and
constraints; it searches deployment recipes, measures the promising ones on real
hardware, and returns the best one plus the artifact plus a proof it meets budget. That
comes later and deliberately so. **Measurement first, because everything downstream is
worthless if the numbers are wrong.**

What is built, what is not, and what we know to be shaky is in
[the finding](docs/silent-fallback.md) and in *What is not true yet* below.
Product strategy and roadmap are not published.

---

## The finding

**[Your accelerator probably isn't running your model](docs/silent-fallback.md)**

Five models — three text, two vision — each exported once to fp32 ONNX and profiled on
**30 devices from four silicon vendors**: phones, Windows-on-ARM laptops, automotive
boards and embedded vision kits.

**Every device either runs the whole graph on its accelerator, or every node on the
CPU.** No middle. On ViT-base it is nine devices each way and the groups are **223×
apart** — no error, no warning, correct results throughout.

The line is not recency (Tensor G5 in the Pixel 10 falls back, like G2–G4 before it),
not vendor (six Qualcomm parts fall back too), and not form factor. Laptop against
laptop, same file: **Snapdragon X2 Elite 4.42 ms against an M2 at 71.54 ms through
CoreML** — 16× between directly competing products.

And the mirror image on Apple: ONNX Runtime's CoreML provider makes **four of six
models slower** than plain CPU, also silently.

## What works today

```bash
uv sync --extra export --group dev

uv run edgefit doctor                  # is this host fit to measure on?
uv run edgefit probe --model hf:...    # how would this model be measured?
uv run edgefit measure --model hf:sentence-transformers/all-MiniLM-L6-v2 \
                       --recipe recipes/ort_coreml_fp32.yaml
uv run edgefit sweep                   # models × recipes, locally
uv run edgefit sweep-remote            # models × hosted phones
uv run edgefit atlas build             # the corpus as a static site
uv run edgefit corpus export           # Parquet + CSV
uv run edgefit verify                  # golden fixtures — the gate for everything after
```

Corpus today: **339 measurements** over 7 models and 30 devices — four silicon vendors,
four form factors — of which **48 rows are recorded failures**.

| | |
|---|---|
| **Backends** | ONNX Runtime — CPU and CoreML providers, locally; Qualcomm AI Hub for hosted phones |
| **Models** | Any HuggingFace repo id. Specs are inferred from `config.json`; the registry holds overrides for what inference cannot get right |
| **Recipes** | fp32 · fp16 · int8 dynamic (per-tensor and per-channel) · static vs dynamic shapes · provider and vendor flags |
| **Workloads** | Encoders, classifiers, vision, and decoder-only generation with KV-cache I/O (TTFT and decode reported separately, never averaged) |
| **Analysis** | Per-node accelerator placement three ways · static FLOP estimation · graph fingerprint · duplicate-weight detection |
| **Output** | Insert-only DuckDB corpus, Parquet/CSV export, and a static atlas with a reproduction command on every row |

## Why the harness refuses to run

Measurement trust is the whole asset. One hallucinated number compromises every model
trained on the corpus, so this is built to fail loudly rather than produce a plausible
wrong answer:

- **A preflight gate** checks AC power, low-power mode, thermal state and free memory,
  and refuses if any fails. On a laptop with a browser open it refuses — correctly.
- **A measured throttle probe** times a fixed kernel against the host's own recorded
  healthy throughput, because Apple Silicon exposes no unprivileged temperature and
  inventing one is worse than admitting it.
- **Variance is mandatory and structural.** `RunStats` is constructible only from raw
  samples and revalidates its own aggregates, so a fabricated standard deviation cannot
  be represented.
- **The corpus is insert-only.** No update, no delete, anywhere. A re-measurement is a
  new row carrying a new harness version.
- **Unavailable values are null plus a written reason**, never a placeholder.
- **Both cascade tiers run out of process**, so a delegate that aborts the interpreter
  becomes a recorded failure instead of a dead sweep.
- **Third-party rows never impersonate ours.** Hosted measurements are marked
  throughout, with their thermal state recorded as unknown rather than assumed clean.
- **A model we cannot place is refused, not approximated.** The wrong input harness does
  not error — it returns a plausible number for a workload nobody asked about.

## What is not true yet

Stated here rather than buried, because the gaps are the reason to trust the rest:

- **No two-unit test.** Four physical devices of one SoC agree to 0.87%, which is the
  closest substitute, but they are different products — a disagreement could have been
  real rather than methodological.
- **Apple numbers are dev-grade.** One laptop-class machine, no second unit.
- **No quantized hosted rows yet.** Every hosted row is fp32, so the fleet result
  describes fp32 out of the box rather than what those accelerators can do with a model
  in their preferred format. Compile jobs work, so this is unfinished rather than
  blocked — "does int8 recover them?" is the next measurement, not an obstacle.
- **No power instrumentation, no thermal soak, no accuracy tier.** All null with
  recorded reasons rather than estimated.

## A note on the `§` references

The source cites `PROJECT.md §N` in about 85 places. That design document is not
published — it is product strategy — so those are pointers you cannot follow, and the
honest thing is to say so rather than let you hunt for a missing file. Where a reference
is load-bearing for understanding *why the code does something*, the reasoning is
restated inline next to it.

## Layout

```
src/edgefit/
  schema/     recipe, measurement, fingerprint, host records
  corpus/     insert-only DuckDB store + Parquet export
  harness/    host probes, preflight gate, run protocol, hosted measurement
  backends/   ONNX Runtime, export, quantization, graph/FLOP/placement analysis
  models/     spec inference + registry overrides
  atlas/      static site generator
  devices/    device inventory and fleet resolution
  cli/        typer entry point
tests/golden/ known-answer fixtures (marked `device`)
```

```bash
uv run pytest        # fast suite, no hardware
uv run ruff check .
```

## Hard rules

Enforced mechanically where possible, not by discipline:

1. Never estimate, extrapolate or synthesize a measurement value. A failed run is
   recorded as a failure; an unavailable field is null plus a reason.
2. Every measurement needs n≥5 runs and reported variance, or the record is invalid.
3. Measurements are immutable. Never `UPDATE`.
4. Measure end-to-end, including framework overhead and lowering time.
5. Open-source the harness. Every published number independently reproducible.
