# v1.15 Release Closure

v1.15 packages the ROCm port as a small, research-grade benchmark preview. The
release is meant to help GPU kernel researchers, compiler/backend researchers,
and agent researchers reproduce bounded ROCm evidence without treating that
evidence as paper-level or leaderboard parity.

## Release Scope

The release includes:

- Claim boundaries in `docs/CLAIMS.md`.
- A deterministic curated ROCm slice in `docs/curated_rocm_slice.md`.
- Researcher-facing workflows in `docs/RESEARCHER-GUIDE.md`.
- Cookbook recipes in `docs/COOKBOOK.md`.
- Existing evaluator, trace, environment, profiler, and AMD-native score paths.

The release does not include:

- Full 235-problem paper validation.
- Original 124-model extraction or curation parity.
- NVIDIA B200, Blackwell, or official leaderboard parity.
- Upstream NVlabs/SOLAR equivalence.
- Unvalidated CDNA 3, CDNA 4, NVFP4, or MXFP4 claims.

## Reproducibility Checklist

Use this checklist when reproducing v1.15 evidence.

1. Install dependencies.

   ```bash
   uv sync --all-groups
   ```

2. Confirm the GPU runtime and Python environment.

   ```bash
   uv run sol-execbench environment --json
   ```

3. Run at least one smoke problem through the public evaluator path.

   ```bash
   uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
     --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
     --output out/v1.15-rmsnorm.trace.jsonl
   ```

4. Run the curated ROCm slice entries listed in
   `docs/curated_rocm_slice.md` through `sol-execbench` or
   `scripts/run_dataset.py`.

5. Record optional profiler evidence when `rocprofv3` is available.

   ```bash
   uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
     --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
     --profile rocprofv3 \
     --output out/v1.15-rmsnorm-profile.trace.jsonl
   ```

6. Generate AMD-native score evidence only when the required local timing and
   bound artifacts are present.

   ```bash
   uv run sol-execbench-score amd --help
   ```

## Artifact Families

| Artifact | Purpose | Claim Level |
| --- | --- | --- |
| Trace JSONL | Canonical pass/fail/skip/runtime result records. | ROCm-port evidence |
| Environment JSON | ROCm, PyTorch, device, and process evidence. | Runtime evidence |
| Profile sidecars | Optional diagnostic kernel timing evidence from `rocprofv3`. | Profiling evidence |
| AMD score reports | Local roofline-derived score evidence for AMD hardware. | AMD-native-derived evidence |
| Readiness and closure docs | Scope, exclusions, and reproducibility status. | Research-preview evidence |

Profiling and AMD score artifacts supplement the trace. They do not replace
correctness results and do not independently establish benchmark parity.

## Result Semantics

| State | Meaning |
| --- | --- |
| pass | The solution passed correctness and timing checks for the selected workload. |
| fail | The solution executed but violated correctness, timing, schema, or runtime expectations. |
| skip | The test or example intentionally did not run because a declared prerequisite was missing. |
| unavailable | Required hardware, software, dataset, or profiler support was not present. |
| unscored | Execution evidence may exist, but score evidence was not produced or was not valid for the claim. |

Reports must keep these states visible. Skipped, unavailable, or unscored
entries are not successful results and must remain in denominator accounting.

## Known Gaps

| Gap | Future Requirement |
| --- | --- |
| Full paper-scale 235-problem validation is not reproduced. | PAPER-02 |
| Original 124-model extraction and curation are not reproduced. | PAPER-01 |
| Local AMD SOLAR derivation is not compared against upstream SOLAR at paper scale. | PAPER-03 |
| MI300X/CDNA 3 and CDNA 4 full-suite evidence is not archived. | HW-01, HW-02 |
| Static kernel evidence is not collected with RGA or code-object analysis. | STATIC-01 |
| Hosted leaderboard and submission service are not available. | SERV-01 |

## Release Decision

v1.15 is suitable as a bounded ROCm research preview when the checklist above is
completed and the claim boundaries in `docs/CLAIMS.md` are followed. It is not a
paper-parity, official leaderboard, or vendor-comparison release.

The next likely milestone should focus on static GPU kernel evidence:
RGA/code-object extraction, GPUOpen ISA classification, and compiler-facing
artifact reports that deepen kernel analysis without requiring full paper-scale
dataset work first.
