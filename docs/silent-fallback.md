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

### It is the device — and one apparent exception taught us more than the rule

An earlier version of this post claimed the split was **device × architecture**, on the
strength of one result: **MobileNetV2**, a CNN built for mobile, appeared to be accelerated
by the same Pixel 9 that declined every node of five transformers. Then a second CNN,
ResNet-50, fell back on that Pixel exactly as the transformers did, so architecture family
plainly was not the answer either. MobileNetV2 was left as a lone exception.

It was not an exception. It was a **confound of our own making**, and finding it is the
most useful thing in this post.

MobileNetV2's accelerated result came from an artifact **compiled to TFLite**. Every model
it was compared against was profiled as **raw ONNX**. Two different paths onto the device,
presented as one comparison. The control settles it:

| model | path | Pixel 9 p50 | placement |
|---|---|---:|---|
| MobileNetV2 | raw ONNX | 8.25 ms *(cv 9.5%)* | **CPU 55/55** |
| MobileNetV2 | **TFLite** | **4.71 ms** *(cv 29.2%)* | **GPU 65/65** |
| ViT-base | raw ONNX | 306.09 ms *(cv 9.0%)* | CPU 544/544 |
| ViT-base | TFLite | 306.09 ms *(cv 9.0%)* | CPU 544/544 |

So the device-level rule holds without exception after all. Across **29 devices and ten
models on the raw-ONNX path, not one device is mixed**: every device accelerates every
model it is given, or runs every model entirely on the CPU. Model identity predicts
nothing. The Pixel 9 declines all ten.

But the MobileNetV2 row is not noise to be discarded — it is the single most important
measurement here, because it proves **the Pixel 9's accelerator was reachable the whole
time.** The silicon was never the limit. The *path* was. Change the artifact format and the
same model on the same phone moves off the CPU and gets 1.75× faster.

That reframes the 100% figure sharply, and we would rather say it plainly than keep a
better headline: **what we measured on those devices is how far one toolchain reaches, not
what the hardware can do.** See [what this does not say](#what-this-does-not-say) — it
matters enough that it is not a footnote.

Two claims about this data have now been withdrawn: that architecture family explained the
split, and that MobileNetV2 was a specially-supported model. The second was our error
rather than a generalisation outrunning its evidence — the control belonged *before*
publication, not after. It was catchable only because every row records how it was
produced, which is worth more than any single number on this page.

Which is, uncomfortably, the entire reason a project like this has to exist.

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

## A third failure mode: not fallback, refusal

Falling back to CPU is the quiet failure. There is a louder one.

**EfficientNet-B0 does not run at all on Qualcomm hardware.** Not slowly — it fails, on
both a Galaxy S24 and a Snapdragon X Elite, while running fine on the Pixel 9's CPU. The
cause is a single node:

```
Node '/inner/efficientnet/pooler/AveragePool_token_366'
  OpType:AveragePool with domain:com.ms.internal.nhwc
```

One `AveragePool` in the pooling head, in an ONNX Runtime-internal NHWC domain, and the
whole model is unrunnable on the NPU that handles every other vision model we have
measured.

This is arguably the *good* outcome. It errors instead of silently running two orders of
magnitude slower, so you find out in CI rather than in production. But it is a third
distinct thing that can happen to a model you hand to a device — accelerated, silently
demoted to CPU, or refused outright — and none of the three is predictable from the model
card.

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

**This measures one toolchain's reach, not the hardware's ceiling — and on non-Qualcomm
silicon that distinction is the whole story.** Every measurement here was taken through
**Qualcomm AI Hub**, on Qualcomm's hosted devices, using Qualcomm's compiler. That stack
has no path to Google's or Samsung's NPU, so a Tensor or Exynos device running 100% on the
CPU is **expected behaviour, not a defect of that device.** We are not neutral observers of
those parts and we will not pretend otherwise: reporting that number as a property of the
Pixel would be exactly the vendor-flavoured comparison this project exists to avoid.

The MobileNetV2 result above is the proof, and it cost us a retraction to find: on the same
Pixel 9, the TFLite path reaches the GPU where the ONNX path does not. **The accelerator
was always there.**

What the Tensor and Exynos rows *do* honestly tell you is a deployment fact, and a useful
one: if you ship an fp32 ONNX artifact through this toolchain, those devices run it on the
CPU, and nothing warns you. That is true, actionable, and reproducible. It is a statement
about your pipeline, not about their silicon. **A fair test of a Tensor NPU needs LiteRT or
Google's own stack, which we have not run.**

The one comparison here that *is* neutral is the one inside Qualcomm's own fleet, where
vendor is held constant — see the generational cliff below.

**Nor is this evidence that fp32 is the natural input.** These are **fp32**
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

**And the effect of quantization is not predictable either.** Same Galaxy S24, same
TFLite target, same int8 settings, same calibration procedure — three models:

| model | fp32 | int8 | effect |
|---|---:|---:|---|
| **ResNet-50** | 1.461 ms *(cv 4.5%)* | **0.646 ms** *(cv 6.7%)* | **2.26× faster** |
| MobileNetV2 | 0.355 ms *(cv 7.9%)* | 0.375 ms *(cv 13.1%)* | inside the noise |
| **ViT-base** | 6.725 ms *(cv 0.6%)* | **56.169 ms** *(cv 0.4%)* | **8.35× slower** |

A **19× spread** in what quantization does to you, from a 2.26× win to an 8.35× loss,
decided by nothing but which model you happened to bring.

The node counts are suggestive: ResNet-50 goes 78 → 81 on quantization, MobileNetV2
65 → 104, ViT-base 544 → 921. The model that barely gains conversion boundaries gains
speed; the one that gains hundreds loses badly. We offer that as a hypothesis and not a
rule — we have not measured node-level attribution, and this page has already had two
tidy explanations fail.

**One caveat remains.** Our calibration set is thin — eight samples derived from one
input, where real calibration uses hundreds of representative examples — and a better set
would plausibly move the transformer numbers.

**The accelerating SoCs are not a Qualcomm endorsement — and this is the one clean
comparison on the page.** Hold the vendor and the toolchain constant and the cliff is still
there. Eight Qualcomm parts run every model entirely on the CPU:

| Qualcomm SoC, every model on CPU | Qualcomm SoC, every model accelerated |
|---|---|
| Snapdragon 678 · 765G · 845 · 855 | Snapdragon 8 Gen 2 · 8 Gen 3 · 8 Elite |
| Snapdragon 778G · 865+ · 7 Gen 4 | X Elite · X Plus · X2 Elite |
| QCS6490 *(embedded)* | SA8295P · SA8775P *(automotive)* · QCS9075 |

Same vendor, same compiler, same artifacts, same protocol — and a binary split with nothing
in between. **The line is generational and tier-based, not by vendor**, and it lands between
Snapdragon 865+/778G and 8 Gen 2. Because vendor and toolchain are held fixed on both sides,
this is the comparison we would stand behind without the caveat above.

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
