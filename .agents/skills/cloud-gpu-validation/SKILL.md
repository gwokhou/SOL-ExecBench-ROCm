---
name: cloud-gpu-validation
description: >-
  Use this skill when preparing, connecting to, or executing benchmarks and
  hardware calibration on remote or cloud GPU instances (AMD Developer Cloud
  MI300X, ModelScope MI308X, or cloud NVIDIA H20/H100/B200).
---

# Cloud GPU Validation Runbook

Follow these 5 gates in order before and during any remote GPU benchmark run.

## 1. Pre-Flight Resource & Billing Guardrails

- Verify GPU quota/inventory and note the exact image stack (`Python` minor
  version, `ROCm` or `CUDA` version, and `PyTorch` version) before launching.
- Never run `uv sync` inside a pre-baked cloud GPU container; reuse the image's
  `python3` (`torch` / `triton` match the container ABI).
- Keep GPU-only credits isolated: do not enable paid automated backups or
  extra network block volumes on credit-only GPU droplets.
- Always **Destroy** hourly GPU droplets immediately once artifacts are
  verified locally (`Power Off` retains GPU billing).

## 2. Local Python 3.12 Twin Gate (Before Booting Remote GPU)

Run the uncached Python 3.12 twin check locally to catch syntax (PEP 758 bare
`except`) and runtime annotation issues (PEP 649 self-references without
`from __future__ import annotations`):

```bash
uv run python -m scripts.mvp_v1.cloud_preflight twin --target MI300X
```

Do not provision or upload to a `py312` cloud host until this gate passes.

## 3. 3-Second Connectivity Probe & Channel Selection

Classify SSH vs HTTP reachability in one bounded pass before running `ssh`:

```bash
uv run python -m scripts.mvp_v1.cloud_preflight probe <IP> --proxy "${all_proxy:-}"
```

- Inspect `local_ssh_fingerprints` to confirm the matching private key exists
  locally.
- If `verdict == "ssh_banner_blocked_http_open"`, the proxy exit node blocks
  outbound port 22. **Do not retry port 22.**
- Use a browser Web Console only as a 10-second bootstrap to either open a high
  SSH port or start the single-shot bundle under `nohup`, never as a step-by-step
  interactive polling shell.

## 4. One-Shot Self-Extracting Runner

Generate a single self-contained script containing `src/`, `scripts/mvp_v1/`,
and the two-stage `smoke` -> `full` gate:

```bash
uv run python -m scripts.mvp_v1.cloud_preflight bundle \
  --output data/outputs/bundles/sol_execbench_oneshot_cloud.sh
```

Run it detached on the target host with `nohup` so console disconnects cannot
abort capture. If running internal `native_gpu_execution()` probes such as
`scripts/internal/cdna3/run_cdna3_diagnostic_calibration.py`, pass explicit
`GPU_LOCK_DIR` and full AMD-SMI `GPU_LOCK_UUID` first.

## 5. Hard Evidence Gate (Fail-Closed Reporting)

Never persist remote outputs under `data/outputs/` or report measured numbers
until:

1. Any browser subagent claim matches the system `BROWSER_HOST_FACTS` message;
2. Downloaded tarballs or directories pass schema + SHA256 verification:

```bash
uv run python -m scripts.mvp_v1.cloud_preflight verify <downloaded_tar_or_dir> \
  --expected-sha256 <sha256>
```

This rejects HTML login pages, `403 Forbidden` JSON payloads, truncated archives,
and invalid `Trace` records.
