# Qualcomm AI Hub — device access

Originally written 2026-08-03 as a blocker report. **Rewritten 2026-08-04: the
blocker was misdiagnosed and does not exist.** Devices provision fine. A narrower,
real defect remains in *compile* jobs.

---

## Correction

The 2026-08-03 diagnosis concluded that the account could read the catalogue but
provision nothing, and attributed it to a `Community`-tier entitlement. That was
wrong. It generalised from `submit_compile_job` — the one job type that is
actually broken — to device access as a whole.

`submit_profile_job` and `submit_inference_job` both provision real hardware and
return real measurements on the same account, same token, same devices.

## What was measured, 2026-08-04

A single-Gemm ONNX model (`[1,64] → [1,32]`, 8 KB) submitted as a profile job to
six devices. All six reached `SUCCESS`.

| Device | SoC era | est. inference | peak mem | node placement |
|---|---|---:|---:|---|
| Google Pixel 3 | Snapdragon 845 | 3 µs | 50.4 MiB | `CPU: 1` |
| Google Pixel 5 | Snapdragon 765G | 5 µs | 86.6 MiB | `CPU: 1` |
| Samsung Galaxy S21 (Family) | Snapdragon 888 | 3 µs | 94.9 MiB | `CPU: 1` |
| Samsung Galaxy S23 Ultra | Snapdragon 8 Gen 2 | 58 µs | 108.2 MiB | `NPU: 3` |
| Samsung Galaxy S24 (Family) | Snapdragon 8 Gen 3 | 46 µs | 123.4 MiB | `NPU: 3` |
| Snapdragon 8 Elite QRD | 8 Elite | 40 µs | 135.0 MiB | `NPU: 3` |

Job states observed in sequence: `CREATED → PROVISIONING_DEVICE →
MEASURING_PERFORMANCE → SUCCESS`. `PROVISIONING_DEVICE` is the literal answer to
the question this document was opened to ask.

**These numbers characterise the harness path, not a workload.** One Gemm is too
small to say anything about the devices; the point is only that provisioning,
execution and per-node reporting all work.

## What the profile job returns

Two sections, and both matter to us:

- **`execution_summary`** — `estimated_inference_time`,
  `estimated_inference_peak_memory`, `first_load_time` / `warm_load_time` (cold
  vs warm, separately), `compile_time`, and memory increase/peak *ranges*.
- **`all_inference_times`** — **100 raw per-run samples**, not an aggregate. This
  satisfies hard rule #2 natively and feeds `RunStats`, which is constructible
  only from raw samples. A vendor reporting one number would have forced
  `reported_latency_ms`; this does not.
- **`execution_detail`** — per node: `name`, `type`, **`compute_unit`**
  (`NPU`/`GPU`/`CPU`), `execution_time`, `execution_cycles`.

`compute_unit` is a *measured* per-node placement on Qualcomm hardware — the same
question our CoreML fallback proxies answer on Apple, from a second vendor. That
is the cross-vendor breadth §12's neutrality claim needs.

Note the table above already shows the effect in miniature: the three older SoCs
ran the Gemm entirely on CPU while reporting success.

## The real defect: compile jobs

Every `submit_compile_job` fails. The client is qai-hub **0.54.0**, the latest on
PyPI, so this is not a stale client.

```
qai_hub.client.UserError: No devices match the given OS name, version, and attributes.
```

Raw server response, `POST /api/v1/jobs/` → **404**:

```json
{"detail": "Unable to find a target device matching the specified constraints."}
```

### What was ruled out

| Hypothesis | Test | Result |
|---|---|---|
| Device genuinely unselectable | `GET /api/v1/devices/?select=true` — the *same* server call `_get_device` makes before submission | **79/79 devices select**; server-side filtering confirmed real (a bogus name returns 0) |
| Wrong device-spec form | name+os, name only, chipset attribute, full catalogue attribute list, and each of the 13 attributes individually | all fail identically |
| Singular vs plural protobuf field | hand-built `CompileJob(device=…)` and `CompileJob(devices=DeviceList(…))` | both fail; the two error strings differ but neither succeeds |
| Source model type | ONNX and TorchScript | both fail |
| Target runtime | default, `tflite`, `qnn_context_binary`, `onnx`, `precompiled_qnn_onnx` | all fail |
| Endpoint | `workbench.aihub.qualcomm.com` and `app.aihub.qualcomm.com` | identical on both |
| Project scoping | no project, and explicit `s80bvcaxsw` | both fail |
| Job creation broken generally | `create_quantize_job` on the same model | reaches a *different* error (`No Dataset matches the given query`), so job creation and model validation both work |
| Client-side rejection | grepped the installed package for the message | absent — the string is server-generated |

The failure does not vary with any client-controlled input. It is specific to the
compile endpoint's target-device resolution.

### Reproduce

```bash
uv run --extra qai-hub python - <<'EOF'
import qai_hub as hub
m = hub.upload_model("tiny.onnx")                      # any ONNX model
d = hub.Device("Samsung Galaxy S23 Ultra")
print(hub.submit_profile_job(model=m, device=d).job_id)  # works
print(hub.submit_compile_job(model=m, device=d).job_id)  # UserError
EOF
```

## Consequence for EdgeFit

Measurement is unblocked; artifact production is not.

- **Available now:** latency, peak memory, cold/warm load, 100 raw samples, and
  per-node NPU/GPU/CPU placement across 79 devices / 65 chipsets — from ONNX,
  which is what our exporter already emits.
- **Not available:** compiled `.tflite` / QNN context binaries, so no
  Qualcomm-side quantized or delegate-varied *recipes*. The recipe axis stays
  Apple-only until compile works.

That split is worth stating plainly, because it maps onto §11a: we can produce
the *number* on Qualcomm hardware but not yet the *artifact*.

## Account facts

| Field | Value |
|---|---|
| Organization | `Community` |
| Project | `Alan's Personal Project` (`s80bvcaxsw`) |
| Catalogue | 79 devices, 65 distinct chipsets |
| OS split | android 73, windows 3, qc_linux 2, ubuntu 1 |
| Form factor | phone 65, iot 6, compute 3, auto 3, tablet 2 |
| Frameworks | onnx 79, tflite 76, qnn 46 |
| Server frameworks | QAIRT 2.45 / 2.47 / 2.48 |

**The API token still needs rotating** — it was pasted into a chat transcript.

## Note on Qualcomm Device Cloud

QDC remains a separate product (SSH to reserved devices, keypair auth). With AI
Hub profiling working, it is no longer a needed alternative — reconsider only if
compiled artifacts become necessary before Qualcomm fixes compile.
