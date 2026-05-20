---
last_mapped: 2026-05-20
last_mapped_commit: unknown
focus: concerns
---

# Concerns

## GPU Environment Sensitivity

Core behavior depends on CUDA, CUPTI, PyTorch CUDA wheels, NVIDIA drivers,
Docker runtime flags, and sometimes Blackwell `sm_100+` hardware. Failures can
come from environment drift rather than code regressions. Tests that require
specific hardware should keep using explicit markers and clear skip reasons.

## Eval Driver Complexity

`src/sol_execbench/driver/templates/eval_driver.py` is a large self-contained
script with many responsibilities: import isolation, input generation,
correctness, timing, reward-hack checks, and trace emission. Changes here have
high blast radius and should be backed by subprocess tests in
`tests/sol_execbench/driver/test_eval_driver.py`.

## Reward-Hack Surface

The project explicitly evaluates untrusted or adversarial submission code.
Existing defenses cover function monkey patching, lazy outputs, thread
injection, blocked inline extension compilation, and CUPTI-based stream timing.
Any new optimization path or output normalization path should be reviewed for
ways a submission could avoid measurement or fake correctness.

## Native Compilation Boundaries

C++/CUDA compilation is staged through `build_ext.py`. Build flags are derived
from solution specs and auto-injected architecture flags in
`ProblemPackager._inject_gencode_flags`. Changes here can break CUDA, CUTLASS,
cuDNN, or local hardware targets.

## Schema Compatibility

Definitions, workloads, solutions, and traces are public JSON contracts
documented in `docs/`. Tightening validators in `src/sol_execbench/core/data/`
can break existing benchmark datasets or examples. Schema changes should update
docs and sample fixtures together.

## Dataset And Output Volume

`scripts/run_dataset.py` can write large outputs under `out/` and depends on
downloaded data under `data/`. These paths should remain out of source control.
Batch runs should support resume/skip behavior to avoid wasting GPU time.

## Minor Code Quality Notes

There are a few typo-level issues in comments/docstrings, such as
`explcitly synchronize` in `src/sol_execbench/core/bench/timing.py`. These are
low risk but can be cleaned opportunistically.

## Planning Artifact State

This codebase map was created before full GSD project initialization. After
requirements and roadmap exist, refresh this map if project scope changes
architecture, schemas, or evaluation isolation.
