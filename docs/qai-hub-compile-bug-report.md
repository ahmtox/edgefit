# `submit_compile_job` fails for every device while `submit_profile_job` succeeds

*A report prepared for Qualcomm AI Hub support. Everything below is reproducible from
the snippet in §3.*

---

## 1. Summary

On our account, **every** `submit_compile_job` call fails with a client-side
`UserError` originating from a server-side 404, while `submit_profile_job` and
`submit_inference_job` succeed on the same model, the same device object and the same
token.

```
qai_hub.client.UserError: No devices match the given OS name, version, and attributes.
```

Raw server response, `POST /api/v1/jobs/`:

```json
HTTP 404
{"detail": "Unable to find a target device matching the specified constraints."}
```

The failure does not vary with any input we control. It appears specific to the
compile endpoint's target-device resolution rather than to the device, the model, or
the account's entitlement.

## 2. Environment

| | |
|---|---|
| Client | `qai-hub` **0.54.0** (latest on PyPI at time of writing) |
| Python | 3.11 |
| Host | macOS 15.2, arm64 |
| Endpoints tried | `workbench.aihub.qualcomm.com`, `app.aihub.qualcomm.com` |
| Devices in catalogue | 79 |

## 3. Minimal reproduction

```python
import qai_hub as hub

model = hub.upload_model("model.onnx")        # any valid ONNX model
device = hub.Device("Samsung Galaxy S23 Ultra")

print(hub.submit_profile_job(model=model, device=device).job_id)   # succeeds
print(hub.submit_compile_job(model=model, device=device).job_id)   # UserError
```

The profile job provisions the device and returns real measurements — latency, peak
memory, ~100 raw per-run samples, per-node `compute_unit`. The compile job submitted
one line later, against the same `Device` object, fails.

## 4. What we ruled out

This is the part we hope is most useful: the failure survives every variation we
could think of, which is why we believe it is server-side and specific to compile.

| Hypothesis | How we tested it | Result |
|---|---|---|
| The device genuinely isn't selectable for our account | `GET /api/v1/devices/?select=true` — the same server call the client's `_get_device` makes before submission | **79 of 79 devices select.** A bogus device name returns 0, so the filter is functioning |
| We're passing the device spec in the wrong form | name + OS; name only; `chipset:` attribute; the full catalogue attribute list; each of the 13 attributes individually | All fail identically |
| Singular vs plural protobuf field | Hand-built `CompileJob(device=…)` and `CompileJob(devices=DeviceList(…))` | Both fail. The two error strings differ, but neither succeeds |
| Source model type | ONNX and TorchScript | Both fail |
| Target runtime | default, `tflite`, `qnn_context_binary`, `onnx`, `precompiled_qnn_onnx` | All fail |
| Endpoint | `workbench.aihub.qualcomm.com` and `app.aihub.qualcomm.com` | Identical on both |
| Project scoping | No project, and an explicit project id | Both fail |
| Job creation is broken generally | `create_quantize_job` on the same uploaded model | Reaches a **different** error (`No Dataset matches the given query`), so job creation and model validation both work |
| The client is rejecting it locally | Grepped the installed package for the error string | Absent — the message is server-generated |

## 5. Why this blocks us

We publish neutral, reproducible cross-vendor measurements of on-device inference
([atlas](https://ahmtox.github.io/edgefit), [harness](https://github.com/ahmtox/edgefit)).
Profile jobs work well and we are grateful for them — they are the reason we can
report Snapdragon numbers at all.

Because compile jobs fail, we can only submit **fp32 ONNX**. That produces a result
we would rather be able to qualify: across eleven mobile SoCs, three accelerate an
fp32 model onto the NPU and eight run every node on the CPU, silently. We are careful
to say this is *not* evidence about those accelerators' capability, because fp32 is
not what an NPU wants.

The obvious next question — **does int8 or fp16 recover those eight devices?** — needs
a compiled `.tflite` or QNN context binary, which is exactly what we cannot produce.
We would much rather publish the complete picture than the partial one.

## 6. What would help

Any of:

- A fix, or a workaround for the compile endpoint's device resolution
- Confirmation that this is an account-level restriction, if it is one — we will say
  so plainly rather than leaving it ambiguous
- A path to producing quantized artifacts for profiling by some other route

Happy to provide job ids, full request/response traces, or to test a patch.
