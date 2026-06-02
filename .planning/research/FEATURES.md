# Project Research: Features

## Milestone

v1.27 Copyright Provenance Cleanup

## Question

How should copyright and provenance cleanup work for this ROCm fork?

## Table Stakes

- Classify active files by provenance:
  - upstream retained;
  - derivative modified;
  - independent ROCm work;
  - generated, planning, or documentation material.
- Update file headers according to classification:
  - upstream retained: keep NVIDIA notice;
  - derivative modified: keep NVIDIA notice and add project attribution;
  - independent ROCm work: use project attribution, not NVIDIA-only
    attribution;
  - generated/planning/docs: omit source-style headers or use project
    attribution unless copied upstream expression is present.
- Preserve Apache-2.0 license identity.
- Document that the project is a ROCm port/adaptation of NVIDIA SOL-ExecBench,
  not an NVIDIA- or AMD-endorsed release.
- Keep paper attribution separate from source copyright. The paper is the
  benchmark/method citation, not proof that every independent implementation
  file should carry NVIDIA copyright.
- Add automated checks that prevent future blanket NVIDIA headers in
  independent ROCm files.

## Differentiators

- A machine-readable provenance allowlist or manifest that the audit tests can
  consume.
- Release-readiness integration so public prerelease bundles fail when
  provenance metadata is inconsistent.
- Directory-level policy that reduces per-file churn for future contributors.

## Anti-Features

- Removing NVIDIA notices from files that still retain upstream expression.
- Treating all files as NVIDIA-owned because the fork originated upstream.
- Treating all files as project-owned because the current work is extensive.
- Rewriting history for ordinary metadata correction.
- Expanding into full legal review or dependency relicensing.

## Sources

- Upstream repository: https://github.com/NVIDIA/SOL-ExecBench
- Original paper: https://arxiv.org/abs/2603.19173
- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
