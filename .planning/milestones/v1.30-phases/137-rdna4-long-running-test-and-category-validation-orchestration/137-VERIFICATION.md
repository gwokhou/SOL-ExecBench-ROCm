# Phase 137 Verification

## Hardware Preflight

```bash
rocminfo
rocm-smi
lspci
```

Result: host tools observed RDNA4 `gfx1200` / Navi 44 Radeon RX 9060 XT.
`rocm-smi` reported a low-power warning while idle.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from pathlib import Path; import torch; print('/dev/kfd', Path('/dev/kfd').exists()); print('/dev/dri', Path('/dev/dri').exists()); print('hip', torch.version.hip); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
```

Result: `/dev/kfd False`, `/dev/dri False`, HIP `7.1.25424`,
`cuda_available False`, `device_count 0`. Classified as an
execution-environment device-passthrough boundary.

## Commands

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m requires_rdna4 -q -rs --junitxml .planning/phases/137-rdna4-long-running-test-and-category-validation-orchestration/137-rdna4-pytest.xml
```

Result: 3 skipped. Artifact: `137-rdna4-pytest.xml`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/examples/test_examples.py -k "hipblas or miopen or ck or rocwmma or triton or hip_cpp" -q -rs --junitxml .planning/phases/137-rdna4-long-running-test-and-category-validation-orchestration/137-category-examples.xml
```

Result: 9 passed, 9 skipped. Artifact: `137-category-examples.xml`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_library_examples.py tests/sol_execbench/test_rocm_library_readiness_docs.py tests/sol_execbench/test_rocm_diagnostics_reporting.py -q --junitxml .planning/phases/137-rdna4-long-running-test-and-category-validation-orchestration/137-category-guardrails.xml
```

Result: 38 passed after restoring README guardrail wording. Artifact:
`137-category-guardrails.xml`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py::test_phase_137_rdna4_category_evidence_stays_bounded tests/sol_execbench/test_rocm_library_readiness_docs.py::test_readme_links_library_readiness_and_names_supported_libraries -q
```

Result: 2 passed.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check tests/sol_execbench/test_public_contract_guardrails.py
```

Result: passed.

```bash
git diff --check
```

Result: passed.

## Residual Risk

The current command environment cannot access ROCm device nodes through `uv`.
The next RDNA4 execution attempt should run in an environment where the same
pytest process can see `/dev/kfd` and `/dev/dri`.
