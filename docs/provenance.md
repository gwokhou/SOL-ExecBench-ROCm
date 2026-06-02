# Copyright and Provenance Policy

This repository is an Apache-2.0 ROCm port of NVIDIA SOL-ExecBench. The port
preserves benchmark semantics where practical while replacing CUDA/NVIDIA
runtime, build, timing, scoring evidence, documentation, and release workflows
with ROCm-oriented equivalents.

This document is engineering provenance guidance, not legal advice. It defines
how this project classifies active files before SPDX/copyright cleanup.

## Source References

- Upstream project: <https://github.com/NVIDIA/SOL-ExecBench>
- Upstream reference used for this policy: `upstream/main`
- License: Apache-2.0, see `LICENSE`
- Machine-readable classification artifact: `provenance.toml`

The SOL-ExecBench paper is the benchmark and methodology citation. It does not
decide file-level source copyright ownership for independent implementation
work in this ROCm port.

## Classification

### Upstream Retained

Files copied verbatim or nearly verbatim from upstream SOL-ExecBench.

Header policy:

- keep applicable NVIDIA `SPDX-FileCopyrightText`;
- keep `SPDX-License-Identifier: Apache-2.0`;
- do not replace NVIDIA attribution with project-only attribution.

### Derivative Modified

Files that retain substantial upstream expression after ROCm adaptation.

Header policy:

- keep applicable NVIDIA `SPDX-FileCopyrightText`;
- add this project's attribution where appropriate:
  `SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port`;
- keep `SPDX-License-Identifier: Apache-2.0`.

### Independent ROCm Work

Files created for this ROCm port that do not retain substantial upstream
expression. Examples include AMD/ROCm evidence models, release readiness tools,
prerelease artifact tooling, ROCm-only tests, and newly written documentation.

Header policy:

- use this project's attribution;
- keep `SPDX-License-Identifier: Apache-2.0`;
- do not use NVIDIA-only file copyright attribution.

### Generated Or Planning Material

Planning artifacts, generated evidence examples, and release draft materials
may not need source-style file headers. Their provenance should be documented
at the artifact, directory, or policy level unless copied source expression
requires a file-level notice.

## NVIDIA Notice Allowlist

`provenance.toml` lists files currently allowed to retain NVIDIA notices under
`nvidia_notice.allowed`. In Phase 123 these are active files that currently
carry the NVIDIA SPDX line and also exist at the same path in `upstream/main`.
They are treated as upstream-retained or derivative-modified candidates.

`provenance.toml` also lists `nvidia_notice.cleanup_candidates`: active files
that currently carry a NVIDIA-only SPDX line but do not exist at the same
upstream path. Phase 124 will remove or replace those headers unless direct
review finds copied upstream expression that requires retaining the notice.

## History Policy

Prior blanket headers are corrected through ordinary commits. Git history is
not rewritten for this metadata cleanup unless a separate legal review requires
it.

## Public Wording Policy

Public docs should describe this repository as a ROCm port or adaptation of
NVIDIA SOL-ExecBench. They must not imply NVIDIA or AMD endorsement unless
explicit approval exists.
