# Qualcomm AI Hub — device access diagnosis

Written 2026-08-03. Paste-ready for Qualcomm support if needed.

**Summary:** the API token authenticates and can read everything, but cannot
provision any device. Every diagnosis path that is on our side has been exhausted;
what remains is an account entitlement on Qualcomm's side.

---

## What works

| Capability | Result |
|---|---|
| Authentication | ✅ token accepted |
| Model upload | ✅ 5 models uploaded and listed |
| `GET /api/v1/devices/` | ✅ 78 devices, 31 distinct SoCs |
| `GET /api/v1/organizations/` | ✅ `Community` (slug `community`) |
| `GET /api/v1/projects/` | ✅ `Alan's Personal Project` (`s80bvcaxsw`) |
| `get_device_attributes()` | ✅ 10 attribute namespaces |

## What fails

Every `submit_compile_job` — which is the first step of any measurement — is
rejected with:

```
qai_hub.client.UserError: No devices match the given OS name, version, and attributes.
```

## What was ruled out

| Hypothesis | Test | Result |
|---|---|---|
| Wrong device-spec form | `Device(name)`, `Device(name, os)`, `Device(attributes="chipset:sm8650")`, `Device(attributes="chipset:qualcomm-snapdragon-8gen3")` | all 4 fail |
| Device object constructed wrongly | passed the exact objects `get_devices()` returns | fails |
| Only newer devices gated | 8 devices spanning Snapdragon 845 → 8 Elite (Pixel 3, 3a, 4, 5, Tab S7, Note 20, S21, Redmi Note 10) | **0 of 8** |
| Wrong API endpoint | `workbench.aihub.qualcomm.com` and `app.aihub.qualcomm.com`, same token | identical: 78 readable, 0 provisionable |
| Missing project scoping | explicit `project="s80bvcaxsw"`, project slug, and no project | all fail |
| Device marked unavailable in the catalogue | decoded the device protobuf | no availability field exists; fields are `name`, `os`, `attributes`, `soc_description` only |
| Client or credential problem | upload and all reads succeed against the same endpoint with the same token | ruled out |

The device record the server rejects is byte-identical to the one the server
returned:

```
name: "Samsung Galaxy S24"
os: "14"
attributes: "chipset:sm8650", "chipset:qualcomm-snapdragon-8gen3", "os:android",
            "hexagon:v75", "framework:tflite", "framework:onnx", "framework:qnn",
            "htp-supports-fp16:true", ...
soc_description: "Snapdragon® 8 Gen 3 | SM8650"
```

## Reproduce

```bash
pip install qai-hub
qai-hub configure --api_token <token>
python -c "
import qai_hub as hub
d = next(x for x in hub.get_devices() if 'chipset:sm8650' in x.attributes)
hub.submit_compile_job(model='model.onnx', device=d)"
```

## Most likely cause

The account's organization is **`Community`**, AI Hub's self-signup tier. The
pattern — full catalogue read access, working uploads, zero device provisioning —
is consistent with a tier or account-verification state that grants API access
without device minutes. Worth asking Qualcomm directly, since nothing in the web UI
surfaced a pending approval or unaccepted terms.

## Note on Qualcomm Device Cloud

QDC is a **separate product** from AI Hub: SSH access to reserved physical devices,
authenticated with an SSH keypair rather than an API token. An SSH key existing for
QDC has no bearing on AI Hub device access, and the `qai-hub` client cannot use it.

QDC is a viable alternative path, but a substantially larger integration — see
`docs/STATUS.md` for the cost estimate before committing to it.
