# SOL ExecBench ROCm

**SOL ExecBench ROCm** is an anti-distortion GPU kernel evaluation and analytical Speed-of-Light (SOL) diagnostic benchmark suite for AMD ROCm GPUs, implementing the methodology from [SOL-ExecBench](https://arxiv.org/abs/2603.19173).

The **v0.1** release focuses on **same-card `AMD Instinct MI300X (gfx942)` vertical ROCm stack SOL diagnosis**—isolating PyTorch/Portable Triton, `vLLM` ROCm default dispatch, and explicit `ROCm/AITER` kernels against transparent dual-ceiling Roofline bounds (hardware Spec + live startup Empirical microbenchmarks) and verified `rocprofv3` hardware counters. Full results are published in the [v0.1 MI300X Vertical SOL Diagnosis Report](docs/v0.1_mi300x_sol_report.md).

## v0.1 Highlights — AMD MI300X (`gfx942`) Vertical 3-Tier SOL Diagnosis

Evaluated on `AMD Instinct MI300X VF` (`304` CUs, `191.69 GiB` HBM3, `256 MiB` Infinity Cache LLC) inside ROCm (`torch 2.13.0+rocm10.0.0`, `Triton 3.8.0`, `vllm 0.34.0`, `aiter`, `rocprofv3`) across **6 core inference primitive families, 8 `FlashInfer-Trace` Definitions, and 34 Full Workloads** (`portable` `34/34`, `vllm` `34/34`, `aiter` `30/34` across `7/8` Definitions):

| Primitive Family | Headline MI300X Finding (Same Card) | Key Numbers & `rocprofv3` Root Cause |
| :--- | :--- | :--- |
| **`skinny_gemm` (`FP8`)** | `vLLM` default `_scaled_mm` trails `AITER` pre-shuffled GEMM | `vLLM` `0.1025–0.1455 ms` (`6.9%–7.7%` Emp SOL) vs. `AITER` `0.0497–0.0585 ms` (`15.6%–17.4%`) — **2.05x–2.49x** dispatch gap (`Amp = 1.03x` on both, isolating tile scheduling rather than HBM traffic) |
| **`mla_paged_decode` (`BF16`)** | `AITER` ASM leads long-context decode (`S=32768`) | `aiter_mla_decode_fwd` hits **51.0% Emp SOL** (`0.3151 ms` at `B=16`), beating `vllm_triton_mla_decode` (`0.4213 ms`) by **1.34x–1.70x** (`S=4096` favors lightweight `portable` Triton `0.0726–0.0793 ms`) |
| **`gqa_paged_decode` (`BF16`)** | ROCm paged assembly/HIP kernels crush portable Triton | `vllm_paged_attention_rocm` (`0.3757 ms`, **76.1% Emp SOL**) & `aiter_pa_fwd_asm` (`0.3902 ms`, **73.2% Emp SOL**) beat `portable` (`1.2078 ms`) by **3.10x–3.21x** at `B=16, S=32768` |
| **`fused_moe` (`BF16`/`FP8`)** | Native FP8 routing & CK/ASM eliminate 2x weight traffic | `vLLM`/`AITER` beat `portable` by **1.33x–1.99x**; `rocprofv3` at `FP8 M=32` measures `Amp = 2.02x` (`portable` `813.7 MB`) vs. **1.02x / 1.01x** (`410.7 / 408.2 MB`) |
| **`fused_rmsnorm_rope` & `swiglu_fp8_quant`** | Silicon counters measure `1.00x` vs. `1.68x/1.80x` `Amp` | Single-pass `portable` Triton hits **`Amp = 1.00x`** vs. 2-op `vLLM`/`AITER` **`1.68x–1.80x`**; at small `M`, launch latency (`~33–47 µs`) dominates HBM bandwidth |

Read the full 34-workload table, analytical Roofline equations, and 15-case `rocprofv3` counter attribution in [`docs/v0.1_mi300x_sol_report.md`](docs/v0.1_mi300x_sol_report.md).

## Quickstart (`v0.1` MI300X 3-Tier Suite)

```bash
# 1. Install local development dependencies
uv sync --all-groups

# 2. Validate Python 3.12 compatibility, FlashInfer-Trace schemas, and CPU smoke suite
uv run python -m scripts.mvp_v1.cloud_preflight twin --target MI300X

# 3. Run the anti-distortion 3-tier vertical benchmark on a ROCm GPU (or --device cpu for smoke)
for tier in portable vllm aiter; do
  uv run python -m scripts.mvp_v1.cloud_runner \
    --mode full --device cuda:0 --target-override MI300X \
    --backend "${tier}" --no-collect-counters --lock-clocks \
    --output-dir "data/outputs/mi300x_v01/${tier}"
done

# 4. Generate the side-by-side vertical bottleneck table
uv run python -m scripts.mvp_v1.cloud_runner --compare \
  data/outputs/mi300x_v01/portable/summary.json \
  data/outputs/mi300x_v01/vllm/summary.json \
  data/outputs/mi300x_v01/aiter/summary.json

# 5. Verify locally generated or downloaded cloud result bundles under gitignored data/outputs/ (SHA-256 + Trace schema gate)
uv run python -m scripts.mvp_v1.cloud_preflight verify \
  data/outputs/mi300x_stage3b_final/vllm.tar.gz \
  --expected-sha256 be999975a263eb5d665f454816e6be116d9d55adf6e51de1b9d25be733222896
```

Anti-distortion timing guarantees ([`anti_distortion.py`](scripts/mvp_v1/anti_distortion.py), [`counter_capture.py`](scripts/mvp_v1/counter_capture.py), [`gfx942_v1.yaml`](src/sol_execbench/data/rocprofv3_counters/gfx942_v1.yaml)):
- Dynamic `RotatingBufferPool` replicas and `>= 2x` LLC (`512 MiB` on `MI300X`) `LLCRotatingFlusher` invalidation before every warmup and timed launch;
- Setup-time pre-packing (`w_t.t()`, weight `B-preshuffle`, block tables, and token CSR expansion) excluded from `torch.cuda.Event` timing windows;
- Full materialization, FP8 `isfinite`, and `FlashInfer-Trace` `ToleranceSpec` validation plus marker-scoped (`MVP_MEASURE`) `rocprofv3` hardware counters.

## Core Packages (`sol_execbench` & `solar`)

The repository maintains a strict boundary between its two sibling packages:

- `sol_execbench` owns problem/workload contracts (`FlashInfer-Trace`), input generation, anti-distortion correctness/timing, ROCm profiler manifests, GPU ownership locking, local scoring formulae, and the CLI.
- `solar` owns operator graph extraction, validated extended-einsum conversion, and SOL analysis as described in [SOL-ExecBench](https://arxiv.org/abs/2603.19173) §4.2.

```bash
# Corpus management and candidate evaluation
uv run sol-execbench dataset corpus validate
./scripts/run_docker.sh -- sol-execbench evaluate \
  problems/local/AMD_AKA/gfx1200/torch2hip/l1n1_square_matmul \
  --solution /sol-execbench/path/to/solution.json \
  --trace-output /outputs/matmul.trace.jsonl

# Formal SOLAR analysis
uv run sol-execbench solar analyze \
  problems/local/AMD_AKA/gfx1200/torch2hip/l1n36_rmsnorm \
  --workload aka-l1n36_rmsnorm-w0 \
  --orojenesis-home /path/to/pinned/orojenesis \
  --output out/solar/norm_forward_bf16
```

SOLAR exposes two fixed IR paths selected through `--backend`: `torchview_extended_einsum` (default) and `make_fx_aten`. Extraction and IR paths never mix independently, and a run never falls back from one path to the other. See [SOLAR boundary](docs/SOLAR-BOUNDARY.md), [scoring contract](docs/SCORING-V3.md), and [configuration](docs/user/CONFIGURATION.md#gpu-ownership-and-recovery) for `GPU_LOCK_DIR` / `GPU_LOCK_UUID` locking rules and normative contracts.

## Archived Milestone: Radeon RX 9060 XT (`gfx1200`)

The prior formal pipeline validation milestone on AMD Radeon RX 9060 XT (`gfx1200`, ROCm 7.2) is archived as completed (`gfx1200-official-score-v2`, `0.497818327` official SOL score over 43 problems and 163 workloads using the self-evaluated PyTorch reference baseline `rx9060xt-gfx1200-reference-v2`):

```bash
uv run sol-execbench score official RELEASE/release-bundle.json
```

See [`RELEASE/README.md`](RELEASE/README.md), [release scoring workflow](docs/user/RELEASE-SCORING.md), [GPU qualification](docs/user/GPU-QUALIFICATION.md), and [RDNA4 validation scope](docs/user/RDNA4-VALIDATION.md) for details on the frozen `gfx1200` artifact bundle. Hardware on `gfx1200` now serves as a free offline development/CI host while datacenter SOL captures target `MI300X (gfx942)` (see [HANDSOFF.md](HANDSOFF.md)).

## Development

Only Linux x86-64 with Python 3.14 is supported locally (`uv sync --all-groups`), while cloud ROCm containers match their pre-baked Python/PyTorch ABI via [`cloud_preflight.py`](scripts/mvp_v1/cloud_preflight.py).

```bash
uv run --with ruff ruff check .
uv run --with ruff ruff format .
uv run ty check
uv run pytest tests/
```

Build the optional local ROCm container with `./scripts/run_docker.sh --build`.

## License

Apache-2.0. Imported SOLAR files retain their original SPDX attribution.
