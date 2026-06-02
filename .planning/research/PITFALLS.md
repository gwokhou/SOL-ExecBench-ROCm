# Project Research: Pitfalls

## Milestone

v1.27 Copyright Provenance Cleanup

## Question

What mistakes are likely when adding copyright/provenance cleanup to this
existing ROCm port?

## Pitfalls

- Blanket replacement in either direction.
  Replacing every NVIDIA notice with project attribution is as risky as keeping
  NVIDIA attribution on every independent file. The decision has to follow
  provenance.

- Using paper citation as source copyright evidence.
  The paper supports methodology and benchmark attribution; it does not decide
  source file ownership for independent implementation work.

- Removing notices from derivative files.
  Apache-2.0 requires retaining applicable upstream notices in derivative
  distributions.

- Making the release look officially endorsed.
  Public wording should avoid implying NVIDIA or AMD endorsement unless there
  is explicit approval.

- Over-scoping into legal audit.
  The milestone should improve release hygiene and provenance evidence, while
  explicitly avoiding legal opinions, relicensing, and full dependency audits.

- Breaking existing residue tests.
  The current residue audit treats NVIDIA SPDX as an acceptable retained
  upstream notice. It must be converted carefully so it catches incorrect
  headers without flagging legitimate upstream-derived files.

- Losing reviewability.
  A generated bulk edit without a manifest will be hard to audit. A small,
  explicit provenance policy makes the cleanup defensible.

## Prevention Strategy

- Require a classification artifact before bulk header edits.
- Keep NVIDIA notices in upstream-retained and derivative files.
- Add project attribution for modified derivative files.
- Use project attribution only for independent files.
- Add tests before or alongside bulk edits so future additions do not regress.
- Document the limitations: not legal advice, not complete dependency audit,
  not history rewrite.
