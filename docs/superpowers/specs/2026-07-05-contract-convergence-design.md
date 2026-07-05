# Contract Convergence Design

Date: 2026-07-05

## Context

The evaluator contract and diagnostic sidecars have moved to the current v2
shape:

- `build_evaluator_contract()` emits `sol_execbench.evaluator_contract.v2`.
- `capabilities` is a mapping of capability key to requirement level.
- Agent feedback sidecars emit `sol_execbench.agent_feedback.v2`.
- Profile summary sidecars emit `sol_execbench.profile_summary.v2`.
- Feedback identity aliases such as `sol_contract_version`, `candidate_hash`,
  and `source_hash` have been removed from the current sidecar schema.
- Sidecar authority is now the enum value `authority: "diagnostic"`.

The repository currently still has some active documentation and inactive code
that can make the current contract look ambiguous. There are no real users yet,
so this work should converge the active repository on the current contract
rather than preserve v1 compatibility.

## Scope

This work updates only the active contract surface:

- Current source code under `src/`.
- Current user-facing documentation under `docs/`.
- Current tests and fixtures under `tests/`.

Historical records are out of scope:

- `.planning/` milestone, phase, audit, and quick-task records remain historical.
- `docs/releases/` entries remain historical release notes.
- Historical references to v1 behavior in those files do not need rewriting.

## Source Of Truth

`build_evaluator_contract()` in `src/sol_execbench/core/data/contract.py` is the
source of truth for the current evaluator contract payload. Active docs and tests
must match its emitted JSON shape.

The active documentation should distinguish these names:

- Contract capability keys: keys in the `capabilities` mapping, such as
  `runtime.evidence`, `profiling.evidence`, `static_kernel.evidence`,
  `agent_feedback.sidecar`, and `profile_summary.sidecar`.
- Requirement levels: values in the `capabilities` mapping, such as `always`,
  `optional`, and `profile:diagnostic`.
- Artifact schema versions: concrete payload schema identifiers, such as
  `sol_execbench.agent_feedback.v2`, `sol_execbench.profile_summary.v2`, and
  `sol_execbench.static_kernel_evidence.v1`.

Current docs must not describe old `.v1` strings as current evaluator contract
capability keys when the builder emits unversioned capability keys.

## Code Cleanup Boundary

Delete only inactive code that conflicts with the current sidecar schema:

- Remove `AgentFeedbackAuthority`.
- Remove `ProfileSummaryAuthority`.

Keep governance guardrail models unchanged:

- `AgentFeedbackGovernanceGuardrail`.
- `ProfileSummaryGovernanceGuardrail`.

Those guardrail models are not sidecar payload schemas. They are claim-upgrade
boundaries that explicitly keep all `*_authority` fields false, so preserving
their boolean shape makes the guardrail tests and intent clearer.

## Documentation Changes

Update active contract documentation so it reflects the builder payload:

- `docs/EVALUATOR-CONTRACT.md` should list current capability keys and
  requirement levels.
- It should document `agent_feedback.sidecar` and `profile_summary.sidecar` as
  contract capability keys whose concrete artifact schemas are v2.
- It should not list `runtime.evidence.v1`, `profiling.evidence.v1`,
  `toolchain.routing.v1`, `static_kernel_evidence.v1`,
  `agent_feedback.sidecar.v1`, or `profile_summary.sidecar.v1` as current
  evaluator contract capability keys.
- `docs/trace.md` should refer to the current `runtime.evidence` capability key
  when discussing evaluator contract discovery.

Do not edit release notes only to modernize old terms.

## Testing

Add or update tests so the current contract cannot drift from the current docs:

- Keep `tests/sol_execbench/test_contract.py` as the direct payload guard:
  `schema_version == "sol_execbench.evaluator_contract.v2"`, `capabilities` is
  a mapping, current keys have expected levels, and `source_boundary_claims` is
  absent.
- Add a focused current-documentation guard that checks
  `docs/EVALUATOR-CONTRACT.md` contains the current capability keys and levels.
- The same guard should reject old evaluator capability key spellings in the
  active contract doc.
- Keep sidecar tests focused on v2 identity, authority, freshness, and fixture
  shapes.

The expected verification command for the implementation is:

```bash
uv run pytest tests/sol_execbench/test_contract.py tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_profile_summary.py tests/sol_execbench/test_cli_environment_snapshot.py
```

If broader historical-doc tests fail for pre-existing planning or release-note
assertions, they are not part of this convergence acceptance unless the failure
touches current contract docs or current contract payload behavior.
