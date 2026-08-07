# Your accelerator probably isn't running your model

*Measured on 16 devices across four silicon vendors. Nothing errored.*

---

## The short version

We took five models — three text encoders and two vision — exported each once to fp32
ONNX, and profiled those artifacts on up to eleven mobile SoCs from Qualcomm, Google and
Samsung.

**Three of them ran it on the NPU. Eight ran every single node on the CPU.**

No error. No warning. No log line. Correct results everywhere. The fastest accelerated
device runs it in **6.26 ms**; the slowest CPU-only device takes **820.37 ms**. That is
**131× end to end**, for the same file.

| device | SoC | vendor | p50 | on accelerator |
|---|---|---|---:|---:|
| Snapdragon 8 Elite QRD | sm8750 | Qualcomm | **6.26 ms** | 429/429 |
| Samsung Galaxy S24 | sm8650 | Qualcomm | **7.68 ms** | 429/429 |
| Samsung Galaxy S23 | sm8550 | Qualcomm | **10.94 ms** | 429/429 |
| Google Pixel 9 | google-tensor-g4 | Google | 303.76 ms | 0/401 |
| Google Pixel 7 | google-tensor-g2 | Google | 328.49 ms | 0/401 |
| Samsung Galaxy Note 20 | samsung-exynos-990 | Samsung | 351.44 ms | 0/401 |
| Samsung Galaxy S21 | sm8350 | Qualcomm | 380.07 ms | 0/401 |
| Google Pixel 8 | google-tensor-g3 | Google | 444.12 ms | 0/401 |
| Google Pixel 5 | sm7250 | Qualcomm | 750.98 ms | 0/401 |
| Samsung Galaxy A53 5G | samsung-exynos-1280 | Samsung | 780.79 ms | 0/401 |
| Google Pixel 3 | sdm845 | Qualcomm | 820.37 ms | 0/401 |

Model: `google/vit-base-patch16-224-in21k`, fp32 ONNX, batch 1, 224×224.

**The split is not about the model — it is entirely about the SoC.** Every device is
either fully accelerated on *every* model, or fully on the CPU for *every* model. There
is not one mixed case in 45 measurements:

| device | SoC | models accelerated |
|---|---|---|
| Snapdragon 8 Elite QRD | sm8750 | **6 of 6** |
| Samsung Galaxy S24 | sm8650 | **5 of 5** |
| Samsung Galaxy S23 | sm8550 | **5 of 5** |
| Google Pixel 7 / 9 | tensor-g2 / g4 | 0 of 5 |
| Samsung Galaxy S21 | sm8350 | 0 of 5 |
| Google Pixel 3 | sdm845 | 0 of 5 |
| Samsung Galaxy A53 5G | exynos-1280 | 0 of 5 |

Text models behave identically to vision ones. `all-MiniLM-L6-v2` puts 226 of 226 nodes
on a Galaxy S24's NPU and runs in **1.52 ms**; on a Pixel 9 it runs entirely on the CPU
at 23.34 ms. `toxic-comment` and `bart-base` reproduce the same split exactly.

## The number that should bother you

Look at two phones from the same year, at the same price point, in the same pocket:

| model | Galaxy S24 | Pixel 9 | gap |
|---|---:|---:|---:|
| vit-base-patch16-224 | 7.68 ms | 303.76 ms | **39.6×** |
| toxic-comment | 2.62 ms | 91.88 ms | **35.1×** |
| bart-base | 2.66 ms | 93.00 ms | **35.0×** |
| clip-vit-base-patch32 | 2.80 ms | 84.76 ms | **30.3×** |
| all-MiniLM-L6-v2 | 1.52 ms | 23.34 ms | **15.4×** |

Same files. Both phones from 2024, both flagships.

If your fleet analytics say those two devices are 20% of your users each, no single
benchmark number describes your application. You cannot average them. You cannot pick
one and call it representative. And you will not discover this from a datasheet,
because both SoCs have a capable NPU and neither vendor is wrong about that.

## Why this is the interesting failure

An op that a delegate declines does not fail. It falls back to CPU and returns the
right answer, slowly. The pipeline reports success. Your tests pass. Your accuracy
metrics are unchanged. The only symptom is a latency number that you have nothing to
compare against, because it is the first time you have run this model on this device.

This is the failure mode we built EdgeFit to find, and it turns out to be less like a
bug and more like weather: it is the normal condition of deploying a model to a fleet
you do not control.

## What this does *not* say

Being precise here matters more than the headline, so:

**This is not evidence that Tensor or Exynos NPUs are slow.** These are **fp32**
artifacts. NPUs generally want int8 or fp16, and several of these accelerators may
simply decline fp32 outright — which would be a completely reasonable thing for them to
do. What we measured is not silicon quality. It is *what happens to a model you hand to
a device without tuning it*, which is the situation every team starts in.

**We could not test the quantized path**, and we would like to. Qualcomm AI Hub's
compile jobs are rejected server-side on our account, so we cannot produce the
`.tflite` or QNN context binaries that would answer "does int8 recover these eight
devices?". That question is the obvious next one and we are blocked on it. If you can
answer it, please do — the harness is open.

**Three SoCs accelerating is not a Qualcomm endorsement.** The three that worked are
Snapdragon 8-series Gen 2 and newer. Three *other* Qualcomm parts in this table — sdm845,
sm7250, sm8350 — fell back completely, same as the Tensor and Exynos devices. The line
is generational, not by vendor.

## The other half: an accelerator that makes things slower

The fleet above is Android. On our own Apple hardware we can measure the other failure
mode, the one where the delegate *does* claim the graph and you still lose.

ONNX Runtime's CoreML execution provider on an M2, five models, fp32:

| model | CPU | CoreML | verdict |
|---|---:|---:|---|
| all-MiniLM-L6-v2 | 9.62 | 20.11 | **2.09× slower** |
| toxic-comment (DistilBERT) | 32.00 | 39.51 | **1.23× slower** |
| bart-base (encoder) | 32.05 | 38.97 | **1.22× slower** |
| distilbert-base-uncased | 32.12 | 36.66 | **1.14× slower** |
| clip-vit-base-patch32 | 28.35 | 18.03 | 1.57× faster |
| vit-base-patch16-224-in21k | 105.53 | 71.54 | 1.48× faster |

**Four of six models are made slower by enabling the accelerator.** Silently. And
restricting to the Neural Engine specifically was slower than `ALL` on every model we
tried — it never once helped.

So the two vendors fail in opposite directions, and both quietly: on Android the
delegate often declines the graph, on Apple it often accepts a graph it should have
declined.

## Three ways to measure fallback, and only one of them works

We record fallback three ways, because when we started we did not know which was
honest. It turns out they disagree badly, and the disagreement is the most useful thing
we learned.

| model | verdict | FLOP share (as authored) | **time share (as run)** |
|---|---|---:|---:|
| clip-vit-base-patch32 | **1.57× faster** | 97.2% | **8.2%** |
| vit-base-patch16-224-in21k | **1.48× faster** | 99.1% | **20.4%** |
| all-MiniLM-L6-v2 | 2.09× slower | 99.5% | **29.5%** |
| bart-base | 1.22× slower | 99.8% | **31.8%** |
| toxic-comment | 1.23× slower | — | **37.1%** |
| distilbert-base-uncased | 1.14× slower | 99.8% | **39.5%** |

Static FLOP share reads 97–99.8% for every model and separates nothing. Measured time
share on the graph *as actually executed* sorts them perfectly: the two winners sit at
8.2% and 20.4%, the four losers at 29.5% and above, and the gap between the groups has
no model in it.

Why the first number lies: **ONNX Runtime rewrites the graph before partitioning it.**
On ViT-base, moving optimisation from `disabled` to `all` cut CPU nodes from 244 to 86.
The as-authored analysis is a faithful measurement of a graph that never runs.

We got this wrong ourselves first, and published the corrected version rather than the
flattering one: an earlier draft of our own README explained CoreML's losses using that
99.5% FLOP figure. It is not the explanation. Measuring both partitions is what showed
it.

One caution the data also forces: time share is **not** an efficiency measure. It says
where the time went, not whether sending that work to the accelerator was a good idea.
MiniLM spends ~70% of its time inside CoreML and is still twice as slow as plain CPU.

## Why you should believe these numbers

We would rather you check than trust us, so:

**Every row is reproducible.** The atlas prints the exact command for each measurement.
The harness is open source. The corpus downloads as Parquet and CSV.

**Every failure is published**, not just the successes. Recipes that crash the runtime,
converters that emit invalid graphs, jobs a vendor rejected — they are rows in the same
table, with the vendor's own error message attached.

**Variance is mandatory and structural.** A measurement record cannot be constructed
without raw per-run samples; the aggregate statistics are recomputed on load and
rejected if they disagree. A fabricated standard deviation is not representable in our
schema.

**Four physical devices of the same SoC agree to 0.87%.** We ran the same recipe on a
Galaxy S24, S24+, S24 Ultra and the S24 family device: 7.68, 7.68, 7.73, 7.75 ms, with a
worst within-run coefficient of variation of 0.7%. Between-device disagreement is the
same size as within-device noise.

**The hosted numbers are not ours and are marked as such.** The Android rows were
measured by Qualcomm AI Hub on their hardware. We record them as third-party throughout,
with their thermal state unknown rather than assumed quiet, and we note that their
timing excludes model load and host-side framework overhead — so those rows read faster
than ours by an unknown margin. We did not correct for it, because we cannot measure it
from what they expose.

**Our own Apple numbers are dev-grade and labelled that way.** They come from one
laptop-class machine with no second unit to cross-check. The harness refuses to measure
when that machine is on battery, in low-power mode, thermally stressed, or simply busy —
it waits, and if it cannot get a clean window it records the refusal instead of a
number.

## The thing nobody can tell you

A silicon vendor can profile your model on their own hardware, free, better than we can.
What none of them will ever publish is the row where a competitor wins — or the row
where their own delegate quietly declines your graph and hands it back to the CPU.

That comparison only exists if someone measures across vendors and publishes the
failures. That is the whole reason this project exists.

---

## Reproduce it

```bash
git clone <repo> && cd edgefit
uv sync --extra export --extra qai-hub

# the fleet result above
uv run edgefit sweep-remote \
  --model hf:google/vit-base-patch16-224-in21k \
  --device "Google Pixel 9" --device "Samsung Galaxy S24"

# any single row
uv run edgefit measure-remote --model hf:google/vit-base-patch16-224-in21k \
  --device "Google Pixel 9" --compute-unit all

# the Apple half
uv run edgefit measure --model hf:sentence-transformers/all-MiniLM-L6-v2 \
  --recipe recipes/ort_coreml_fp32.yaml

uv run edgefit atlas build   # the whole corpus as a static site
```

Requires a free Qualcomm AI Hub account for the hosted devices. The Apple measurements
need a Mac and will refuse to run if the machine is not idle.

*Corpus at time of writing: 110 measurements, 7 models, 16 devices and 12 SoCs across
four silicon vendors — 15 hosted phones plus one Mac. 22 of those rows are recorded
failures. Every number above is a row in it.*
