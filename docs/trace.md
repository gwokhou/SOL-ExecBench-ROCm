# Trace Schema

A Trace is an immutable record for one workload evaluation. It links a
Solution, Definition, concrete Workload, correctness result, performance
measurement, and ROCm environment snapshot.

## Top-Level Object

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `definition` | string | Yes | Name of the Definition used in this run. |
| `solution` | string or null | No | Name of the Solution tested. |
| `workload` | object | Yes | Concrete workload axes and input descriptors. |
| `evaluation` | object or null | No | Evaluation result. |

A workload-only trace has `solution == null` and `evaluation == null`.

## Workload

The embedded workload object follows [Workload](workload.md). It includes:

- `uuid`
- concrete `axes`
- `inputs`
- optional `tolerance`

## Evaluation

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `status` | string | Yes | Final evaluation status. |
| `log` | string | No | Captured stdout/stderr or diagnostic text. |
| `correctness` | object or null | Depends on status | Correctness metrics. |
| `performance` | object or null | Depends on status | Timing metrics. |
| `environment` | object | Yes | AMD/ROCm hardware and software snapshot. |
| `timestamp` | string | Yes | ISO-style timestamp. |

Status values:

- `PASSED`
- `INCORRECT_SHAPE`
- `INCORRECT_NUMERICAL`
- `INCORRECT_DTYPE`
- `RUNTIME_ERROR`
- `COMPILE_ERROR`
- `TIMEOUT`
- `REWARD_HACK`
- `INVALID_REFERENCE`

## Correctness

| Field | Type | Description |
| --- | --- | --- |
| `max_relative_error` | float | Maximum relative error. |
| `max_absolute_error` | float | Maximum absolute error. |
| `has_nan` | bool | True when solution or reference output contains NaN. |
| `has_inf` | bool | True when solution or reference output contains Inf but no NaN. |
| `extra` | object or null | Optional extra metrics. |

When outputs contain non-finite values, max error fields are set to `0.0` and
`has_nan` or `has_inf` explains the reason.

## Performance

| Field | Type | Description |
| --- | --- | --- |
| `latency_ms` | float | Solution latency in milliseconds. |
| `reference_latency_ms` | float | PyTorch reference latency on the same hardware. |
| `speedup_factor` | float | `reference_latency_ms / latency_ms`. |

Timing uses PyTorch's HIP-backed device event API. See [Analysis](analysis.md).

## Environment

| Field | Type | Description |
| --- | --- | --- |
| `hardware` | string | AMD GPU/device identifier, such as `AMD Radeon Graphics gfx1200`. |
| `libs` | object | Library/tool versions, such as `torch`, `hip`, `rocm`, or `triton`. |

PyTorch ROCm still exposes devices through `torch.cuda` compatibility APIs, but
trace environment data should identify AMD hardware and ROCm/HIP versions.

## Nullable Fields By Status

| Status | `correctness` | `performance` |
| --- | --- | --- |
| `PASSED` | Required | Required |
| `INCORRECT_NUMERICAL` | Required | `null` |
| `INCORRECT_SHAPE` | `null` | `null` |
| `INCORRECT_DTYPE` | `null` | `null` |
| `RUNTIME_ERROR` | `null` | `null` |
| `COMPILE_ERROR` | `null` | `null` |
| `TIMEOUT` | `null` | `null` |
| `REWARD_HACK` | `null` | `null` |
| `INVALID_REFERENCE` | `null` | `null` |

## Example

```json
{
  "definition": "rmsnorm",
  "solution": "rmsnorm_triton_rocm_v1",
  "workload": {
    "uuid": "6120f144-b973-4bd9-b884-77ecb132914e",
    "axes": {
      "batch_size": 32
    },
    "inputs": {
      "input": {
        "type": "safetensors",
        "path": "data/rmsnorm/b32_input.safetensors",
        "tensor_key": "input"
      },
      "weight": {
        "type": "safetensors",
        "path": "data/rmsnorm/rmsnorm_weight.safetensors",
        "tensor_key": "weight"
      }
    }
  },
  "evaluation": {
    "status": "PASSED",
    "log": "",
    "correctness": {
      "max_relative_error": 0.0000115,
      "max_absolute_error": 0.0000089,
      "has_nan": false,
      "has_inf": false,
      "extra": null
    },
    "performance": {
      "latency_ms": 0.008,
      "reference_latency_ms": 0.019,
      "speedup_factor": 2.375
    },
    "environment": {
      "hardware": "AMD Radeon Graphics gfx1200",
      "libs": {
        "torch": "2.10.0+rocm7.1",
        "hip": "7.1.25424",
        "triton": "3.6.0"
      }
    },
    "timestamp": "2026-05-21T12:45:00Z"
  }
}
```
