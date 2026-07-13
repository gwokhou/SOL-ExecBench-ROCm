---
phase: 117
phase_name: "First-Run User Path"
created_at: "2026-06-01"
autonomous: true
requirements: [FIRST-01, FIRST-02, FIRST-03, FIRST-04]
---

# Phase 117 Context

## Goal

Make the first-run path self-contained for a new ROCm user: install
dependencies, run a minimal example, write trace JSONL, interpret correctness
and timing fields, and diagnose common failures without confusing PyTorch ROCm
compatibility names with NVIDIA/CUDA support.

## Inputs

- `docs/user/GETTING-STARTED.md` already contains install commands, ROCm visibility
  checks, example commands, Docker path, dataset setup, and common setup
  issues.
- `docs/user/trace.md` defines Trace JSONL fields and states that PyTorch ROCm uses
  historical `torch.cuda` compatibility APIs.
- `docs/user/CONFIGURATION.md` documents no-trace diagnostic sidecars.
- Existing docs tests use direct Markdown substring assertions in
  `tests/sol_execbench/test_research_release_docs.py`.

## Locked Decisions

- Keep the first-run path on existing examples and CLI flags.
- Do not add new commands or runtime behavior in this phase.
- The documented minimal command should produce canonical trace JSONL using
  `--output`.
- `torch.cuda` references are allowed only as PyTorch ROCm compatibility names,
  not as NVIDIA runtime support claims.
- No live ROCm, Docker, dataset, or network checks should be required by the
  new verification tests.

## Risks

- A new user may run a command without `--output` and miss where trace JSONL is
  written.
- `torch.cuda` wording can look like CUDA/NVIDIA support unless paired with
  explicit ROCm compatibility language.
- No-trace diagnostics may be hard to find from the first-run page if only
  configuration docs mention them.

## Deferred

- Running examples on live hardware.
- Adding an interactive tutorial or notebook.
- Changing CLI output, trace schema, or no-trace sidecar behavior.
