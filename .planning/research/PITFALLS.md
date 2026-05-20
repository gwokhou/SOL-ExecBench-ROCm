# Pitfalls Research

**Domain:** ROCm port of GPU kernel benchmark framework
**Researched:** 2026-05-21
**Confidence:** HIGH for common porting pitfalls, MEDIUM for project-specific profiler replacement details

## Critical Pitfalls

### Pitfall 1: Assuming HIPIFY Completes the Port

**What goes wrong:** CUDA code compiles after automatic conversion but benchmark
semantics drift or architecture-specific bugs remain.

**Why it happens:** HIPIFY handles API translation, not every semantic issue.
AMD's HIP porting guide specifically calls out architecture differences such as
warp size and lane masks.

**How to avoid:** Use HIPIFY `--examine` to inventory CUDA surface area, then
manually port and test each subsystem.

**Warning signs:** `warpSize`, lane masks, inline PTX, hard-coded SM/gencode
flags, CUDA stream/event assumptions, and NVIDIA-only headers remain.

**Phase to address:** Initial audit and native compile phases.

---

### Pitfall 2: Replacing CUPTI Timing With Weak Timing

**What goes wrong:** The benchmark reports timings that miss non-default stream
work or profiler-relevant GPU activity.

**Why it happens:** Existing timing relies on CUPTI concepts and has tests for
stream-hiding attacks. A simple HIP event replacement may not preserve those
defenses.

**How to avoid:** Prototype rocprofiler-sdk/rocprofv3 timing and validate it
against existing reward-hack and stream tests before migrating examples.

**Warning signs:** Timing tests are skipped, removed, or rewritten only to match
implementation convenience.

**Phase to address:** Timing/profile subsystem phase.

---

### Pitfall 3: Porting Examples Before Runtime Contracts

**What goes wrong:** Example kernels are ported repeatedly because language
enums, build flags, loader behavior, and output normalization keep changing.

**Why it happens:** Examples exercise all backend assumptions.

**How to avoid:** Stabilize schemas, HIP build, eval import, hardware detection,
and timing before broad example migration.

**Warning signs:** Example PRs contain Docker, build, timing, schema, and kernel
changes all at once.

**Phase to address:** Roadmap should separate environment/build/runtime from
example migration.

---

### Pitfall 4: Treating RDNA 4 and CDNA 3 as Equivalent

**What goes wrong:** Tests pass on one AMD architecture but fail or perform
incorrectly on the other.

**Why it happens:** Consumer RDNA and datacenter CDNA differ in supported
features, library tuning, matrix instructions, and performance counters.

**How to avoid:** Add architecture detection early and keep per-architecture
test gates explicit.

**Warning signs:** Test names say "ROCm" but no run records show both RDNA 4 and
CDNA 3.

**Phase to address:** Test migration and final validation.

---

### Pitfall 5: License Drift During Dependency Replacement

**What goes wrong:** The port accidentally introduces dependencies or copied
code that conflict with the repository license or third-party notices.

**Why it happens:** CUDA-to-ROCm replacement often involves borrowing examples,
headers, or snippets from multiple upstream projects.

**How to avoid:** Track dependency licenses during each replacement and update
notices/docs as part of the same phase.

**Warning signs:** New vendored code, copied kernels, or dependency additions
without license review.

**Phase to address:** Every phase; final compliance pass before completion.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Leaving CUDA names in ROCm-only code | Smaller diff | Confusing APIs and tests | Only as temporary aliases with deprecation notes. |
| Skipping hardware-specific tests | Faster CI | False confidence | Only for local development, never completion gate. |
| Disabling reward-hack tests | Unblocks timing port | Breaks benchmark trust | Never acceptable for final v1. |
| Hard-coding one gfx target | Simplifies compile | Fails RDNA/CDNA requirement | Only for a narrow prototype branch. |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Docker | Keeping NVIDIA runtime flags | Use ROCm device mounts, groups, and official ROCm base image. |
| PyTorch | Installing CUDA wheels | Use ROCm wheels or ROCm PyTorch image. |
| Native build | Reusing `-gencode` flags | Map to AMD gfx targets and HIP compiler options. |
| Profiling | Shelling out to profiler without JSON discipline | Preserve stdout JSON contract and route profiler noise to logs/stderr. |
| SMI tools | Porting `nvidia-smi` call names mechanically | Rebuild discovery/clock logic around AMD SMI/ROCm SMI/rocminfo. |

## "Looks Done But Isn't" Checklist

- [ ] **Docker:** Container builds and imports torch, but HIP examples do not compile.
- [ ] **PyTorch:** `torch.cuda.is_available()` works, but architecture-specific kernels fail.
- [ ] **Timing:** Latency is nonzero, but stream-hiding tests are not equivalent.
- [ ] **Examples:** One HIP sample passes, but CUTLASS/cuDNN/CuTe/cuTile replacements are unplanned.
- [ ] **Tests:** Unit tests pass on CPU, but RDNA 4 and CDNA 3 hardware runs are missing.
- [ ] **Licensing:** Code compiles, but third-party notices are stale.

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| HIPIFY over-trust | Audit/build phase | HIPIFY report plus manual removal of CUDA-only patterns. |
| Weak timing | Timing/profile phase | Stream/reward-hack tests pass under ROCm. |
| Example churn | Runtime-before-examples ordering | Examples migrate after schemas/build/eval are stable. |
| RDNA/CDNA mismatch | Validation phase | Test logs from both architecture families. |
| License drift | Every dependency phase | Updated notices and dependency review. |

## Sources

- HIP porting guide — CUDA-to-HIP migration and semantic traps.
- rocprofv3 docs — profiler and tracing replacement direction.
- ROCm compatibility matrix — ROCm 7.x components and hardware/system support.
- Current codebase map — high-risk eval driver, timing, and native compilation areas.

---
*Pitfalls research for: ROCm port of SOL ExecBench*
*Researched: 2026-05-21*
