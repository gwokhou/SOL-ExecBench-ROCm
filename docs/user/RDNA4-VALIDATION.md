# RDNA4 Validation Scope

The recorded RDNA4 engineering-validation scope is deliberately exact:

| Dimension | Accepted identity |
| --- | --- |
| GPU | AMD Radeon RX 9060 XT |
| ISA target | `gfx1200` |
| ROCm user space | `7.2.0` |
| PyTorch | `2.11.0+rocm7.2` |
| PyTorch HIP runtime | `7.2.26015` |
| Triton ROCm | `3.6.0` from the locked project environment |
| Host | Linux x86-64 with exactly one visible GPU |

This is not an RDNA4-family claim. In particular, `gfx1201`, a generic
`gfx12` target, another RDNA4 product, or a different ROCm/PyTorch stack must
produce new evidence before it can inherit the validation status. The
`requires_rdna4` marker therefore selects the exact `gfx1200` target in this
repository.

## Covered behavior

The local hardware gate exercises canonical device-event timing, stream-hiding
checks, a real `rocprofv3` kernel trace, and evaluator correctness for PyTorch,
Triton, HIP, hipBLAS, MIOpen, Composable Kernel, and rocWMMA paths. The native
fixtures compile for the observed `gfx1200` target and execute correctness
checks; a schema value or successful compilation alone is not treated as
hardware validation.

The packaged RX 9060 XT resource audit has a narrower authority. Its current
AMD-SMI telemetry reports `THROTTLED`, so the artifact is accepted only as
instruction/runtime corroboration. It is not unthrottled resource-peak
evidence. Canonical benchmark latency remains HIP device-event timing.
`rocprofv3` timing is derived diagnostic evidence, and its optional overhead
calibration is accepted only when the v2 artifact matches the GPU architecture,
current profiler binary SHA-256, and clock-lock state.

## Running the local gate

GitHub-hosted runners do not provide this GPU. The repository workflow is
manual-only and requires a separately administered self-hosted runner carrying
the labels `linux`, `x64`, `rocm`, and `gfx1200`. It is not a pull-request or
branch-protection gate.

Run the same content-addressed gate directly on the validated host:

```bash
uv run python scripts/internal/rdna4/run_rdna4_validation.py \
  --output-dir out/diagnostics/rdna4-local

uv run python scripts/internal/rdna4/run_rdna4_validation.py \
  --verify out/diagnostics/rdna4-local \
  --expected-source-revision "$(git rev-parse HEAD)"
```

The bundle contains the environment doctor payload, JUnit XML, bounded pytest
logs, and a manifest with hashes for every artifact. Verification rejects any
skip, test failure, source-revision mismatch, artifact change, target/toolchain
mismatch, or missing required artifact.

## Authority boundary

`sol_execbench.rdna4_validation.v2` is local engineering evidence. Both direct
runs and manual GitHub Actions runs on a self-hosted machine have
`release_eligible=false` and `trusted_execution=false`. The GitHub variant
records workflow/run provenance, but that metadata is not a trusted execution
attestation. A content checksum proves internal consistency, not who controlled
the runner.

After a successful manual workflow run, the GitHub job emits a separate
`sol_execbench.rdna4_validation_receipt.v1` receipt. The receipt binds the
workflow run ID and attempt, exact commit SHA, `gfx1200` target, and evidence
manifest digest. Diagnostic and score release workflows independently download
and re-verify the receipt and complete evidence tree; they stop unless the
receipt SHA exactly matches the release tag commit. The local manifest remains
non-authoritative and cannot manufacture this workflow binding.

The verifier therefore cannot promote this schema into a publisher score bundle,
including when a caller edits the booleans and recomputes the manifest
checksum. Official scoring separately requires a clean immutable source
revision and the baseline, candidate-execution, and SOLAR evidence described in
[Release and Official Score Workflow](RELEASE-SCORING.md).
