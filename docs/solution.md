# Solution Schema

A Solution is a concrete ROCm-compatible kernel implementation for a Definition.
It contains source files, build metadata, dependencies, and the entry point that
the evaluator calls.

This port accepts ROCm solution metadata only. CUDA/NVIDIA schema values are
rejected with migration guidance.

## Top-Level Schema

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | string | Yes | Unique solution name. |
| `definition` | string | Yes | Name of the Definition this solves. |
| `author` | string | Yes | Agent or human author identifier. |
| `description` | string | No | Brief implementation description. |
| `spec` | object | Yes | Build specification. |
| `sources` | array | Yes | Source code files reconstructed in staging. |

## Build Specification

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `languages` | array[string] | Yes | ROCm-supported language/category values. |
| `target_hardware` | array[string] | Yes | `["gfx1200"]`, `["LOCAL"]`, or both. |
| `entry_point` | string | Yes | `"filename::function_name"`. |
| `destination_passing_style` | bool | No | Outputs passed as final args when `true`; default is `true`. |
| `binding` | string or null | No | `torch` for HIP/C++ categories. Ignored for Python/Triton. |
| `dependencies` | array[string] | No | Required packages/libraries. |
| `compile_options` | object | No | HIP/C++ compiler and linker flags. |

## Supported Languages

| Value | Entry point | Notes |
| --- | --- | --- |
| `pytorch` | `.py::function` | PyTorch running on ROCm. |
| `triton` | `.py::function` | Triton ROCm kernels plus PyTorch. |
| `hip_cpp` | `.hip`, `.cpp`, `.cc`, `.cxx`, `.c`, `.h`, or `.hpp` | HIP/C++ extension compiled through PyTorch. |
| `hipblas` | Native C/C++ entry point | hipBLAS-oriented implementation category. |
| `miopen` | Native C/C++ entry point | MIOpen-oriented implementation category. |
| `ck` | Native C/C++ entry point | Composable Kernel-oriented implementation category. |
| `rocwmma` | Native C/C++ entry point | rocWMMA-oriented implementation category. |

Python languages (`pytorch`, `triton`) cannot be mixed with native C/C++
languages in the same solution.

## Unsupported Legacy Values

These values are not active ROCm schema values:

| Legacy value | Replacement direction |
| --- | --- |
| `cuda_cpp` | `hip_cpp` |
| `cutlass` | `ck` or `rocwmma`, or a HIP/Triton implementation |
| `cublas` | `hipblas` |
| `cudnn`, `cudnn_frontend` | `miopen` or a HIP/Triton implementation |
| `cute_dsl`, `cutile` | No direct runtime in this port; use HIP, Triton ROCm, CK, or rocWMMA where feasible |
| `cuda_cflags` | `hip_cflags` |
| `B200` | `gfx1200` or `LOCAL` |

## Destination Passing Style

Destination passing style is the default. The evaluator pre-allocates output
tensors and passes them after all inputs.

```python
def run(input, weight, eps, output):
    output[:] = normalize(input, weight, eps)
```

When `destination_passing_style` is `false`, the solution returns output
tensors instead:

```python
def run(input, weight, eps):
    return normalize(input, weight, eps)
```

Argument order is Definition inputs first, followed by Definition outputs.

## Compile Options

HIP/C++ solution categories accept:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `cflags` | array[string] | `[]` | Extra host compiler flags. |
| `hip_cflags` | array[string] | `["-O3"]` | Extra HIP compiler flags. |
| `ld_flags` | array[string] | `[]` | Extra linker flags. |

Example:

```json
"compile_options": {
  "hip_cflags": ["-O3", "-ffast-math", "--offload-arch=gfx1200"],
  "ld_flags": []
}
```

PyTorch's extension API still names the device-compiler keyword
`extra_cuda_cflags`; that is an implementation detail in the build template.
Solution JSON must use `hip_cflags`.

## Dependencies

Dependencies are validated against the current environment. Common values:

| Dependency | Notes |
| --- | --- |
| `torch` | PyTorch ROCm runtime. |
| `triton-rocm` or `triton` | Triton ROCm environment, depending on installed package metadata. |
| `hipblas` | Native hipBLAS implementation category. |
| `miopen` | Native MIOpen implementation category. |
| `ck` | Composable Kernel implementation category. |
| `rocwmma` | rocWMMA implementation category. |

## Sources

Each source entry has:

| Field | Required | Description |
| --- | --- | --- |
| `path` | Yes | Relative path without `..` or a leading slash. |
| `content` | Yes | Complete source text. |

The entry point file must be present in `sources`.

## PyTorch Example

```json
{
  "name": "swiglu_pytorch_rocm_v1",
  "definition": "swiglu_h4096",
  "author": "my-agent",
  "spec": {
    "languages": ["pytorch"],
    "target_hardware": ["LOCAL"],
    "entry_point": "kernel.py::run",
    "dependencies": ["torch"],
    "destination_passing_style": true
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import torch\n\ndef run(gate, up, output):\n    output[:] = torch.nn.functional.silu(gate) * up"
    }
  ]
}
```

## Triton ROCm Example

```json
{
  "name": "rmsnorm_triton_rocm_v1",
  "definition": "rmsnorm_h4096",
  "author": "my-agent",
  "spec": {
    "languages": ["triton"],
    "target_hardware": ["gfx1200"],
    "entry_point": "kernel.py::run",
    "dependencies": ["torch", "triton-rocm"],
    "destination_passing_style": true
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import triton\nimport triton.language as tl\n\n@triton.jit\ndef _kernel(...):\n    ...\n\ndef run(input, weight, eps, output):\n    _kernel[(input.shape[0],)](input, weight, output, input.shape[1], eps, BLOCK_SIZE=1024)"
    }
  ]
}
```

## HIP/C++ Example

```json
{
  "name": "rope_hip_v1",
  "definition": "rope_apply_rotation",
  "author": "my-agent",
  "spec": {
    "languages": ["hip_cpp"],
    "target_hardware": ["gfx1200"],
    "entry_point": "kernel.hip::run",
    "dependencies": ["torch"],
    "destination_passing_style": false,
    "binding": "torch",
    "compile_options": {
      "hip_cflags": ["-O3", "-ffast-math"]
    }
  },
  "sources": [
    {
      "path": "kernel.hip",
      "content": "#include <torch/extension.h>\n#include <hip/hip_runtime.h>\n\ntorch::Tensor run(torch::Tensor input) {\n    auto output = torch::empty_like(input);\n    // launch HIP kernel and write output\n    return output;\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.def(\"run\", &run);\n}\n"
    }
  ]
}
```
