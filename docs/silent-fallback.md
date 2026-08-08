# Your accelerator probably isn't running your model

*30 devices · four silicon vendors · phones, laptops, cars and embedded boards. Nothing errored.*

---

## The short version

We took five models — three text encoders and two vision — exported each once to fp32
ONNX, and profiled those artifacts across **30 devices from four silicon vendors**,
spanning phones, Windows-on-ARM laptops, automotive boards and embedded vision kits.

**Every device either runs the whole graph on its accelerator, or every node on the
CPU.** There is no middle. On ViT-base the split is exactly nine devices each way, and
the two groups are **223× apart**.

Nothing errors anywhere. Every device returns correct results.

| | device | SoC | p50 |
|---|---|---|---:|
| **NPU** | Snapdragon X2 Elite CRD | sc8480xp | **4.42 ms** |
| **NPU** | Snapdragon 8 Elite QRD | sm8750 | 6.26 ms |
| **NPU** | Samsung Galaxy S24 | sm8650 | 7.68 ms |
| **NPU** | Snapdragon X Plus 8-Core | sc8340xp | 10.70 ms |
| **NPU** | Samsung Galaxy S23 | sm8550 | 11.01 ms |
| **NPU** | Snapdragon X Elite CRD | sc8380xp | 11.51 ms |
| **NPU** | SA8775P ADP *(automotive)* | sa8775p | 13.75 ms |
| **NPU** | SA8295P ADP *(automotive)* | sa8295p | 16.72 ms |
| **NPU** | Dragonwing IQ-9075 EVK *(embedded)* | qcs9075 | 17.28 ms |
| CPU | Google Pixel 10 | tensor-g5 | 264.26 ms |
| CPU | Google Pixel 9 | tensor-g4 | 308.80 ms |
| CPU | Snapdragon 7 Gen 4 QRD | sm7750 | 354.83 ms |
| CPU | Samsung Galaxy A73 5G | sm7325 | 390.35 ms |
| CPU | Samsung Galaxy Tab S7 | sm8250 | 463.10 ms |
| CPU | Google Pixel 4 | sm8150 | 465.26 ms |
| CPU | Dragonwing RB3 Gen 2 *(embedded)* | qcs6490 | 708.96 ms |
| CPU | Xiaomi Redmi Note 10 5G | sm6150 | 894.80 ms |
| CPU | Samsung Galaxy A14 5G | exynos-1330 | 987.58 ms |

Model: `google/vit-base-patch16-224-in21k`, fp32 ONNX, batch 1, 224×224. The other four
models — two vision, three text, all of them transformers — reproduce the split
device-for-device.

### It is not purely a property of the device

We later put **MobileNetV2**, a convolutional network built for mobile, through the same
protocol. It breaks the pattern:

| model | Pixel 9 | placement |
|---|---:|---|
| ViT-base *(transformer)* | 291.10 ms | **CPU 544 of 544** |
| MobileNetV2 *(CNN)* | **5.01 ms** | **GPU 65 of 65** |

The same Pixel 9 that declined every node of every transformer accelerates a CNN
completely. So the honest statement is **device × architecture**, not device alone: these
devices reject *these transformer graphs*, and a convolutional model reaches the
accelerator on hardware that refused all five.

This corrects an earlier version of this post, which said the split was "entirely about
the SoC". It was — across the five models measured at the time, every one of which was a
transformer. Adding a single CNN falsified the generalisation, which is about what should
happen to a generalisation drawn from one architecture family.

## The line is not what you would guess

It is not recency. **Google's Tensor G5, shipping in the Pixel 10, runs every node on
the CPU** — as do G2, G3 and G4 before it. Five generations, same behaviour.

It is not vendor. Six *Qualcomm* parts in that table fall back completely: sm6150,
sm7325, sm7750, sm8150, sm8250 and the qcs6490 vision kit.

It is not form factor. Two automotive boards and one embedded kit accelerate; another
embedded kit from the same product family does not.

What actually predicts it, in this data, is **Snapdragon 8-series and X-series
silicon** — plus the automotive and embedded parts built on the same Hexagon
generation. Everything else, from any vendor, in any form factor, silently runs on the
CPU.

## Laptop against laptop

Until now the comparison was open to an easy objection: our Apple numbers came from a
laptop and our Qualcomm numbers from phones. The Snapdragon X series closes that. Same
model, same file, same class of machine:

| laptop-class chip | p50 | on accelerator |
|---|---:|---|
| **Snapdragon X2 Elite** | **4.42 ms** | 429/429 |
| Snapdragon X Plus 8-Core | 10.70 ms | 429/429 |
| Snapdragon X Elite | 11.51 ms | 429/429 |
| Apple M2 — CoreML | 71.54 ms | ~50% fell back |
| Apple M2 — CPU | 105.53 ms | — |

**16× between directly competing products in the same form factor**, and the X chips
take every node while the CoreML EP declines roughly half the graph on the identical
file.

## Two phones from the same year, in the same pocket

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

**We tested int8, and the answer inverts the usual advice.** This was the obvious
objection — *you only measured fp32, of course an NPU declined it* — so here is the
result. ViT-base, every artifact built by AI Hub's own compiler:

| device | fp32 | int8 **calibrated** | penalty |
|---|---:|---:|---:|
| Samsung Galaxy S24 — *accelerates fp32* | **6.73 ms** *(cv 0.6%)* | **56.17 ms** *(cv 0.4%)* | **8.4× worse** |
| Google Pixel 9 — *falls back on fp32* | 306.09 ms *(cv 9.0%)* | **146.70 ms** *(cv 9.6%)* | **2.1× better** |

**On the S24, calibrated int8 is 8.4× slower. On the Pixel 9, it is 2.0× faster.** Same
model, same compiler, same flags, opposite conclusions — decided entirely by whether the
device's accelerator claimed the fp32 graph in the first place.

"Quantize so it fits on the NPU" is the standard recommendation, and here quantizing
*forfeited* the NPU advantage. The S24 kept all 921 nodes on its NPU and still lost 8.4×.
The Pixel 9 never got off the CPU and still gained 2×.

Node counts suggest why: 544 at fp32, 570 uncalibrated, **921** calibrated. Quantization
inserts conversion boundaries and calibration inserts many more. Where an accelerator was
already running the graph, every boundary is pure cost; where the work was falling to the
CPU anyway, narrower integer arithmetic wins by more than the boundaries lose. We are
calling that a hypothesis rather than a conclusion — we have not measured node-level
attribution, and we would rather not assert a mechanism we cannot show.

**The practical consequence is that no single artifact is right for a mixed fleet:**

| if you ship | S24 | Pixel 9 | worst case |
|---|---:|---:|---:|
| fp32 everywhere | 6.72 *(best)* | 291.10 | **2.0× penalty** |
| calibrated int8 everywhere | 56.22 | 144.91 *(best)* | **8.4× penalty** |
| the right one per device | 6.72 | 144.91 | — |

If those two phones are each 20% of your users, there is no version of "just quantize it"
that does not cost one of those groups badly.

**And the damage is transformer-specific.** We ran the same calibrated-int8 protocol on
MobileNetV2, a CNN built for mobile:

| model | S24 fp32 | S24 int8 | penalty |
|---|---:|---:|---:|
| ViT-base | 6.73 ms *(cv 0.6%)* | 56.17 ms *(cv 0.4%)* | **8.4×** |
| MobileNetV2 | 0.355 ms *(cv 7.9%)* | 0.375 ms *(cv 13.1%)* | **not resolvable** |

On the CNN, int8 costs nothing like the 8.4× it costs the transformer — that contrast is
large and unambiguous. But we will not put a number on MobileNetV2's penalty: 0.355
against 0.375 ms at 8–13% coefficient of variation is **inside the noise**. A sub-
millisecond model on hosted hardware is at the floor of what this method resolves, and an
earlier version of this post claimed "1.03×" from those figures, which the variance does
not support. So 8.4× is not what quantization does in
general — it is what per-tensor int8 does to attention, where tensors with very different
dynamic ranges force conversion boundaries throughout. Node counts rise by a similar
*ratio* in both cases (65→104 and 544→921), but MobileNetV2's graph is small enough that
the added boundaries barely register.

**One caveat remains.** Our calibration set is thin — eight samples derived from one
input, where real calibration uses hundreds of representative examples — and a better set
would plausibly move the transformer numbers.

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

*Corpus at time of writing: **347 measurements** over 8 models and 30 devices across four
silicon vendors, including 48 recorded failures. Every number above is a row in it, and
every row carries the `edgefit` command that reproduces it.*

*The [atlas](https://ahmtox.github.io/edgefit) renders **214** of those — the current
generation. The other 125 are rows a later harness version re-measured on the same
cell, kept because measurements are immutable and an old row is a truthful record of
what that instrument reported, but not shown because publishing four generations side
by side would average figures we have since corrected. The full set is in the
[downloadable snapshot](https://github.com/ahmtox/edgefit/tree/main/data).*
