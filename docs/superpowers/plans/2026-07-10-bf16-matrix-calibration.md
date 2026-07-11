# BF16 Matrix Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shipped HIP calibration backend measure exact BF16 matrix paths (WMMA on RDNA4 and MFMA on CDNA3/CDNA4) so validated models are attainable only after real BF16 evidence is collected.

**Architecture:** Architecture adapters declare only their exact BF16 instruction path. `HipCommandBackend` selects a source generator from the full key, compiles it for the adapter ISA, and uses in-program deterministic CPU-reference validation before reporting samples. State classification has a strict boundary: an explicit device-capability negative is `unavailable`; any failed or incomplete attempt to determine capability is `unknown`.

**Tech Stack:** Python 3.12, HIP/hipcc, AMD GPU matrix intrinsics, Pytest, Ruff.

## Global Constraints

- `gfx12*` declares `compute.matrix.bf16.bf16.wmma`; `gfx94*` and `gfx95*` declare `compute.matrix.bf16.bf16.mfma`.
- Matrix samples are BF16-input fused multiply-add throughput in TFLOP/s and count two operations per FMA.
- A candidate is `measured` only after successful compile, execution, numerical check, stability check, and conservative sampling.
- `unavailable` is returned only by an explicit architecture-aware capability negative; compilation, runtime, output, numerical, and stability failures are `unknown`.
- The adapter's complete declared matrix remains required for `validation_status="validated"`; unavailable or unknown BF16 evidence remains diagnostic and cannot build a model.
- Do not add rocWMMA as a runtime dependency, change packaged models, or weaken official-score authority gates.
- Live GPU tests use existing architecture markers and do not make claims for hardware absent from the host.

---

### Task 1: Declare exact architecture matrix paths and strict capability states

**Files:**
- Modify: `src/sol_execbench/core/scoring/hardware_calibration/environment.py`
- Modify: `src/sol_execbench/core/scoring/hardware_calibration/hip_probe.py`
- Test: `tests/sol_execbench/core/scoring/hardware_calibration/test_environment.py`
- Test: `tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py`

**Interfaces:**
- Produces `ArchitectureAdapter.candidates` containing `...bf16.bf16.wmma` for `gfx12*` and `...bf16.bf16.mfma` for `gfx94*`/`gfx95*`.
- Produces `HipCommandBackend.compile(key) -> "passed" | "unsupported" | "missing" | "failed"`, where only an explicit architecture/key mismatch produces `"unsupported"`.

- [ ] **Step 1: Write failing adapter-path tests**

```python
@pytest.mark.parametrize(
    ("architecture", "matrix_path"),
    [("gfx1200", "wmma"), ("gfx942", "mfma"), ("gfx950", "mfma")],
)
def test_adapter_declares_its_exact_bf16_matrix_path(architecture, matrix_path):
    keys = {key.value for key in adapter_for(architecture).candidates}
    assert f"compute.matrix.bf16.bf16.{matrix_path}" in keys
    assert f"compute.matrix.bf16.bf16.{'mfma' if matrix_path == 'wmma' else 'wmma'}" not in keys
```

- [ ] **Step 2: Run the adapter test to verify it fails**

Run: `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_environment.py::test_adapter_declares_its_exact_bf16_matrix_path -n 0 -v`

Expected: FAIL for `gfx1200`, because the adapter currently declares MFMA for every family.

- [ ] **Step 3: Write failing state-separation tests**

```python
def test_matrix_path_mismatch_is_explicitly_unavailable(tmp_path):
    backend = HipCommandBackend(workspace=tmp_path, hipcc="hipcc")
    key = CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma")
    assert backend.compile(key) == "unsupported"

def test_missing_compiler_is_unknown_not_unavailable(tmp_path):
    probe = default_hip_probe(HipCommandBackend(workspace=tmp_path, hipcc=None))
    key = CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "mfma")
    candidate = probe.measure(key)
    assert candidate.state == "unknown"
    assert candidate.reason_code == "hip_probe_compile_missing"
```

- [ ] **Step 4: Run the state tests to verify current behavior is insufficient**

Run: `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py -k 'matrix_path_mismatch or missing_compiler_is_unknown' -n 0 -v`

Expected: the mismatch test fails until `HipCommandBackend` validates the path against the selected architecture; the missing-compiler test passes or confirms the existing `unknown` contract.

- [ ] **Step 5: Implement the minimal declarations and state router**

```python
def _matrix_key(family: str) -> CalibrationProfileKey:
    path = "wmma" if family == "gfx12" else "mfma"
    return CalibrationProfileKey("compute", "matrix", "bf16", "bf16", path)

def _matrix_path_is_supported(key: CalibrationProfileKey, architecture: str) -> bool:
    return (
        (architecture.startswith("gfx12") and key.path == "wmma")
        or (architecture.startswith(("gfx94", "gfx95")) and key.path == "mfma")
    )
```

Pass the adapter architecture to `HipCommandBackend`; return `"unsupported"` only when `_matrix_path_is_supported()` is false. Preserve `"missing"` and `"failed"` for absent/failed `hipcc` and let `HipProbe.measure()` map them to `unknown`.

- [ ] **Step 6: Run focused tests to verify green**

Run: `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_environment.py tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py -n 0 -v`

Expected: PASS.

- [ ] **Step 7: Commit the independently testable state/declaration change**

```bash
git add src/sol_execbench/core/scoring/hardware_calibration/environment.py src/sol_execbench/core/scoring/hardware_calibration/hip_probe.py tests/sol_execbench/core/scoring/hardware_calibration/test_environment.py tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py
git commit -s -m "Declare exact BF16 calibration paths"
```

### Task 2: Implement self-validating BF16 WMMA and MFMA probe sources

**Files:**
- Modify: `src/sol_execbench/core/scoring/hardware_calibration/hip_probe.py`
- Test: `tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py`

**Interfaces:**
- Produces `_hip_source(key) -> str` for existing FP32 candidates and `_matrix_hip_source(key) -> str` for the exact BF16 WMMA/MFMA key.
- `HipCommandBackend.compile()` invokes `hipcc -O3 --offload-arch=<architecture> <source> -o <executable>` for a declared matrix candidate.

- [ ] **Step 1: Write failing source-routing and compiler-command tests**

```python
def test_wmma_source_uses_bf16_wmma_intrinsic():
    source = _hip_source(CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma"))
    assert "__hip_bfloat16" in source
    assert "wmma" in source.lower()
    assert "RESULT" in source

def test_mfma_source_uses_bf16_mfma_intrinsic():
    source = _hip_source(CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "mfma"))
    assert "__hip_bfloat16" in source
    assert "mfma" in source.lower()
    assert "RESULT" in source

def test_matrix_compile_targets_selected_architecture(tmp_path):
    commands = []
    backend = HipCommandBackend(
        workspace=tmp_path, hipcc="hipcc", architecture="gfx1200",
        run=lambda command, **_: commands.append(command) or Result(0),
    )
    assert backend.compile(CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma")) == "passed"
    assert "--offload-arch=gfx1200" in commands[0]
```

- [ ] **Step 2: Run tests to verify red**

Run: `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py -k 'wmma_source or mfma_source or matrix_compile_targets' -n 0 -v`

Expected: FAIL because `_hip_source()` only emits FP32 vector/copy source and compile does not target an adapter ISA.

- [ ] **Step 3: Implement the two matrix source generators**

Implement a WMMA generator for the `wmma` key and an MFMA generator for the `mfma` key using HIP's architecture-supported BF16 matrix intrinsic exposed by the repository's minimum ROCm toolchain. Each generated program must:

```cpp
// required structure in both generated programs
// 1. initialize fixed BF16 A and B tiles;
// 2. invoke the exact WMMA or MFMA instruction in a timed repetition loop;
// 3. copy a deterministic accumulator result to host;
// 4. compare against an FP32 CPU reference with a named tolerance;
// 5. print seven "RESULT <positive-tflops>" lines only after validation passes.
```

The program returns nonzero before printing results for a failed numerical check. Use `2 * M * N * K * repetitions / elapsed_seconds / 1e12` as the throughput calculation, with tile dimensions matching the selected intrinsic.

- [ ] **Step 4: Run source-routing and existing HIP probe tests to verify green**

Run: `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py -n 0 -v`

Expected: PASS.

- [ ] **Step 5: Commit the self-validating matrix probes**

```bash
git add src/sol_execbench/core/scoring/hardware_calibration/hip_probe.py tests/sol_execbench/core/scoring/hardware_calibration/test_hip_probe.py
git commit -s -m "Measure BF16 matrix calibration paths"
```

### Task 3: Prove validated calibration depends on real BF16 evidence

**Files:**
- Modify: `tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py`
- Modify: `tests/sol_execbench/core/scoring/hardware_calibration/test_live_calibration.py`

**Interfaces:**
- Verifies `run_calibration(CalibrationRequest(...)) -> HardwareCalibrationArtifact` retains strict all-declared-profile validation after the exact matrix-path split.

- [ ] **Step 1: Write failing builder tests**

```python
def test_rdna4_validates_when_real_wmma_candidate_is_measured():
    artifact = run_calibration(CalibrationRequest(
        environment=GpuEnvironment(0, "gfx1200"),
        hip_probe=_probe_with_states({"compute.matrix.bf16.bf16.wmma": "measured"}),
        clock_controller=_locked_clock(),
    ))
    assert artifact.validation_status == "validated"

def test_rdna4_remains_provisional_when_wmma_is_unknown():
    artifact = run_calibration(CalibrationRequest(
        environment=GpuEnvironment(0, "gfx1200"),
        hip_probe=_probe_with_states({"compute.matrix.bf16.bf16.wmma": "unknown"}),
        clock_controller=_locked_clock(),
    ))
    assert artifact.validation_status == "provisional"
```

- [ ] **Step 2: Run builder tests to verify red**

Run: `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py -k 'real_wmma or wmma_is_unknown' -n 0 -v`

Expected: FAIL until test helpers model the exact split path and production adapters declare WMMA.

- [ ] **Step 3: Add marker-gated live tests for actual BF16 matrix execution**

```python
@pytest.mark.requires_rdna4
def test_live_rdna4_wmma_calibration_is_measured():
    candidate = default_hip_probe(HipCommandBackend(architecture=discover_gpu(0).architecture)).measure(
        CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "wmma")
    )
    assert candidate.state == "measured"

@pytest.mark.requires_cdna3
def test_live_cdna3_mfma_calibration_is_measured():
    candidate = default_hip_probe(HipCommandBackend(architecture=discover_gpu(0).architecture)).measure(
        CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "mfma")
    )
    assert candidate.state == "measured"
```

Add the analogous `requires_cdna4` MFMA test. Each test is collected only on its matching host and uses the production backend, not an injected result.

- [ ] **Step 4: Run unit tests to verify green**

Run: `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py tests/sol_execbench/core/scoring/hardware_calibration/test_live_calibration.py -n 0 -v`

Expected: unit tests PASS; marker-gated tests skip when their required GPU family is unavailable.

- [ ] **Step 5: Run available live architecture checks**

Run: `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration/test_live_calibration.py -m 'requires_rdna4 or requires_cdna3 or requires_cdna4' -n 0 -v`

Expected: each test passes on its matching available hardware; absent hardware is skipped by repository marker policy.

- [ ] **Step 6: Commit calibration-authority regression coverage**

```bash
git add tests/sol_execbench/core/scoring/hardware_calibration/test_builder.py tests/sol_execbench/core/scoring/hardware_calibration/test_live_calibration.py
git commit -s -m "Verify BF16 evidence gates calibration authority"
```

### Task 4: Run the complete calibration and CLI verification suite

**Files:**
- Modify: `docs/TESTING.md`

**Interfaces:**
- Documents the focused matrix-probe command and the meaning of `unavailable` versus `unknown` in live calibration reports.

- [ ] **Step 1: Write the documentation assertions as required text**

Add a calibration-testing subsection containing these exact operational rules:

```text
unavailable means an explicit architecture-aware capability check proved that the exact path is unsupported.
unknown means support could not be determined reliably; it includes compiler, runtime, output, correctness, and stability failures.
```

- [ ] **Step 2: Verify the documentation change is present**

Run: `rg -n 'explicit architecture-aware capability check|support could not be determined reliably' docs/TESTING.md`

Expected: two matching lines.

- [ ] **Step 3: Run final tests and lint**

Run: `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration tests/sol_execbench/cli/commands/test_hardware_model_cli.py -n 0 -v && uv run --with ruff ruff check .`

Expected: PASS with zero lint violations.

- [ ] **Step 4: Commit documentation and final verification scope**

```bash
git add docs/TESTING.md docs/superpowers/specs/2026-07-10-bf16-matrix-calibration-design.md docs/superpowers/plans/2026-07-10-bf16-matrix-calibration.md
git commit -s -m "Document BF16 calibration evidence semantics"
```
