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

## Dataset License And Redistribution Boundaries

`provenance.toml` also contains a machine-readable
`sol_execbench.dataset_provenance_policy.v1` policy for dataset sources and
locally generated migration outputs. That policy is authoritative for automated
guardrails.

### NVIDIA SOL-ExecBench Evaluation Dataset

The NVIDIA SOL-ExecBench evaluation dataset is governed by the NVIDIA
Evaluation Dataset License. This project does not redistribute original NVIDIA
dataset rows, definitions, workloads, traces, solutions, blobs, or ROCm-migrated
derivatives of that dataset. Users must obtain NVIDIA/SOL-ExecBench content
from upstream and run local migration tooling only when they have applicable
rights under the NVIDIA Evaluation Dataset License.

The policy classifies NVIDIA original and derivative dataset payloads as
`excluded` and `release_bundle_blocked`. These files may appear in a local,
ignored `data/` or `out/` workflow only; they must not be committed, published
as fixtures, or copied into prerelease or release bundles.

### FlashInfer Trace

FlashInfer Trace is tracked separately from NVIDIA SOL-ExecBench. The
machine-readable policy records `flashinfer-ai/flashinfer-trace` as Apache-2.0
content. Apache-2.0 FlashInfer Trace material may be redistributed only when
the required license and attribution notices are preserved. Migration manifests
must keep FlashInfer Trace source identity separate from NVIDIA/SOL-ExecBench
source identity.

### Generated Local Migration Artifacts

Generated migration artifacts inherit the redistribution boundary of their
source dataset. Local manifests may record generated refs, checksums, source
dataset IDs, source revisions, and license-boundary metadata. Generated payloads
derived from NVIDIA SOL-ExecBench remain local-only and release-bundle-blocked.

### Project-Owned ROCm Code

Project-owned ROCm code, tests, scripts, examples, and documentation remain
Apache-2.0 project material unless a file explicitly records retained upstream
expression. These files are publishable with this repository's normal Apache-2.0
attribution.

### Guardrails

`scripts/check_dataset_redistribution.py` enforces the dataset policy for staged
repository paths and release-bundle directories. The prerelease readiness check
also scans bundle contents and blocks restricted NVIDIA dataset paths before
publication.

## NVIDIA Notice Allowlist

`provenance.toml` lists files currently allowed to retain NVIDIA notices under
`nvidia_notice.allowed`. In Phase 123 these are active files that currently
carry the NVIDIA SPDX line and also exist at the same path in `upstream/main`.
They are treated as upstream-retained or derivative-modified candidates.

`provenance.toml` also lists `nvidia_notice.cleanup_candidates`: active files
identified in Phase 123 as carrying a NVIDIA-only SPDX line without existing at
the same upstream path. Phase 124 replaces those headers with project
attribution unless direct review finds copied upstream expression that requires
retaining the notice.

## History Policy

Prior blanket headers are corrected through ordinary commits. Git history is
not rewritten for this metadata cleanup unless a separate legal review requires
it.

## Public Wording Policy

Public docs should describe this repository as a ROCm port or adaptation of
NVIDIA SOL-ExecBench. They must not imply NVIDIA or AMD endorsement unless
explicit approval exists.
