# Contract Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converge the active evaluator contract code, docs, and tests on the current v2 payload shape without preserving v1 compatibility.

**Architecture:** Treat `build_evaluator_contract()` as the source of truth for current capability keys and requirement levels. Active documentation must distinguish contract capability keys from concrete artifact schema versions, and inactive v1 sidecar authority models should be removed while governance guardrail models remain unchanged.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Click CLI tests, Markdown documentation.

---

## File Structure

- Modify: `tests/sol_execbench/test_contract.py`
  - Adds a current-documentation guard that compares `docs/EVALUATOR-CONTRACT.md` against the builder payload shape.
- Modify: `docs/EVALUATOR-CONTRACT.md`
  - Replaces old optional capability token wording with the current capability mapping and artifact schema wording.
- Modify: `docs/trace.md`
  - Updates the runtime evidence capability reference from `runtime.evidence.v1` to `runtime.evidence`.
- Modify: `src/sol_execbench/core/bench/agent_feedback.py`
  - Removes the unused obsolete agent-feedback sidecar authority model.
- Modify: `src/sol_execbench/core/bench/profile_summary.py`
  - Removes the unused obsolete profile-summary sidecar authority model.

---

### Task 1: Add Current Contract Documentation Guard

**Files:**
- Modify: `tests/sol_execbench/test_contract.py`

- [ ] **Step 1: Add `Path` import and doc guard constants**

Change the import block near the top of `tests/sol_execbench/test_contract.py` from:

```python
from __future__ import annotations

import json
```

to:

```python
from __future__ import annotations

import json
from pathlib import Path
```

Then add these constants after `OPTIONAL_CAPABILITIES`:

```python
CURRENT_CONTRACT_DOC = Path("docs/EVALUATOR-CONTRACT.md")
OLD_EVALUATOR_CAPABILITY_TOKENS = {
    "runtime.evidence.v1",
    "profiling.evidence.v1",
    "toolchain.routing.v1",
    "static_kernel_evidence.v1",
    "agent_feedback.sidecar.v1",
    "profile_summary.sidecar.v1",
}
```

- [ ] **Step 2: Add the failing documentation convergence test**

Append this test after `test_evaluator_contract_advertises_optional_evidence_without_bump`:

```python
def test_current_contract_doc_matches_builder_capabilities():
    text = CURRENT_CONTRACT_DOC.read_text()
    payload = build_evaluator_contract().model_dump(mode="json")
    capabilities = payload["capabilities"]

    assert isinstance(capabilities, dict)
    for old_token in OLD_EVALUATOR_CAPABILITY_TOKENS:
        assert f"`{old_token}`" not in text

    for capability, level in capabilities.items():
        assert f"`{capability}`" in text
        assert f"`{level}`" in text

    assert "`sol_execbench.agent_feedback.v2`" in text
    assert "`sol_execbench.profile_summary.v2`" in text
```

- [ ] **Step 3: Run the new test and verify it fails**

Run:

```bash
uv run pytest tests/sol_execbench/test_contract.py::test_current_contract_doc_matches_builder_capabilities -q
```

Expected result: `FAILED`, because `docs/EVALUATOR-CONTRACT.md` still contains old current-capability wording such as `` `runtime.evidence.v1` ``.

---

### Task 2: Update Active Contract Documentation

**Files:**
- Modify: `docs/EVALUATOR-CONTRACT.md`
- Modify: `docs/trace.md`
- Test: `tests/sol_execbench/test_contract.py`

- [ ] **Step 1: Replace the optional capabilities section**

In `docs/EVALUATOR-CONTRACT.md`, replace the whole section from `## Optional Capabilities` through the paragraph ending with `` `sol_version`, `candidate_id`, and `source_sha256`.`` with:

```markdown
## Optional Capabilities

The `capabilities` field is a mapping of current evaluator contract capability
keys to requirement levels. Required trace and baseline semantics use `always`;
optional diagnostics use `optional` or a narrower diagnostic profile.

| Capability key | Level | Meaning |
| --- | --- | --- |
| `trace.correctness` | `always` | Canonical correctness status and correctness metrics. |
| `trace.timing` | `always` | Canonical timing fields and timing interpretation. |
| `trace.scoring` | `always` | Canonical scoring fields and score provenance. |
| `baseline.measured_export` | `always` | Measured baseline registry export fields. |
| `baseline.scoring_artifact` | `always` | Scoring baseline artifact fields. |
| `compatibility.metadata` | `always` | Metadata consumers can persist for compatibility diagnostics. |
| `failure_categories` | `always` | Stable consumer-facing failure buckets. |
| `runtime.evidence` | `optional` | Optional runtime environment evidence beside canonical trace rows. |
| `profiling.evidence` | `optional` | Optional profiler evidence and metadata beside benchmark output. |
| `toolchain.routing` | `optional` | Optional toolchain availability and provenance diagnostics. |
| `static_kernel.evidence` | `optional` | Optional static kernel evidence sidecar diagnostics. |
| `agent_feedback.sidecar` | `profile:diagnostic` | Optional bounded next-experiment feedback diagnostics. |
| `profile_summary.sidecar` | `profile:diagnostic` | Optional normalized profiler summary diagnostics. |

Concrete artifact schema versions are separate from contract capability keys.
`agent_feedback.sidecar` currently emits `sol_execbench.agent_feedback.v2` as
`<trace>.agent-feedback.json`. `profile_summary.sidecar` currently emits
`sol_execbench.profile_summary.v2` as `<trace>.profile-summary.json` when a
trace output path is available. Static kernel evidence remains the concrete
`sol_execbench.static_kernel_evidence.v1` sidecar schema behind the
`static_kernel.evidence` capability key. Current ROCm profiling metadata is
still emitted separately as the trace-adjacent `<trace>.profile.json`
rocprofv3 sidecar.

These capabilities are intentionally optional unless their level is `always`.
A compatible consumer must keep working when a SOL version only provides
canonical trace/profile surfaces and does not produce feedback sidecars. For
HIP freshness checks, SOL sidecars emit canonical identity fields only:
`sol_version`, `candidate_id`, and `source_sha256`.
```

- [ ] **Step 2: Update feedback sidecar wording in the contract doc**

In `docs/EVALUATOR-CONTRACT.md`, replace this sentence:

```markdown
`agent_feedback.sidecar.v2`, `profile_summary.sidecar.v2`, and the current
`<trace>.profile.json` profiler metadata are trace-adjacent diagnostic surfaces.
```

with:

```markdown
`agent_feedback.sidecar`, `profile_summary.sidecar`, and the current
`<trace>.profile.json` profiler metadata are trace-adjacent diagnostic surfaces.
Their concrete feedback and profile-summary artifact schemas are
`sol_execbench.agent_feedback.v2` and `sol_execbench.profile_summary.v2`.
```

- [ ] **Step 3: Update runtime evidence wording in trace docs**

In `docs/trace.md`, replace:

```markdown
v1.13 adds an optional runtime environment evidence contract for richer
diagnostics. That evidence is not a canonical trace JSONL field in v1.13:
`evaluation.environment` remains limited to the stable `hardware` and `libs`
shape described here, while consumers can detect optional support through the
evaluator contract capability `runtime.evidence.v1`.
```

with:

```markdown
Runtime environment evidence is optional diagnostic metadata for richer
environment reporting. It is not a canonical trace JSONL field:
`evaluation.environment` remains limited to the stable `hardware` and `libs`
shape described here, while consumers can detect optional support through the
current evaluator contract capability `runtime.evidence`.
```

- [ ] **Step 4: Run the documentation guard and verify it passes**

Run:

```bash
uv run pytest tests/sol_execbench/test_contract.py::test_current_contract_doc_matches_builder_capabilities -q
```

Expected result: `1 passed`.

- [ ] **Step 5: Run the full contract test file**

Run:

```bash
uv run pytest tests/sol_execbench/test_contract.py -q
```

Expected result: all tests in `tests/sol_execbench/test_contract.py` pass.

- [ ] **Step 6: Commit the documentation convergence change**

Run:

```bash
git add tests/sol_execbench/test_contract.py docs/EVALUATOR-CONTRACT.md docs/trace.md
git commit -s -m "docs: converge evaluator contract capabilities"
```

Expected result: commit succeeds with DCO sign-off.

---

### Task 3: Remove Inactive Sidecar Authority Models

**Files:**
- Modify: `src/sol_execbench/core/bench/agent_feedback.py`
- Modify: `src/sol_execbench/core/bench/profile_summary.py`
- Test: `tests/sol_execbench/test_agent_feedback.py`
- Test: `tests/sol_execbench/test_profile_summary.py`

- [ ] **Step 1: Confirm the inactive classes are only locally defined**

Run:

```bash
rg -n "<inactive sidecar authority class-name pattern>" src tests docs
```

Expected result: only the class definitions appear in `agent_feedback.py` and `profile_summary.py`.

- [ ] **Step 2: Remove the obsolete agent-feedback sidecar authority model**

In `src/sol_execbench/core/bench/agent_feedback.py`, delete the whole obsolete
agent-feedback sidecar authority model with the docstring `Authority boundary for
agent feedback sidecars.` and the diagnostic-only `*_authority` literal fields.

Do not modify `AgentFeedbackGovernanceGuardrail`.

- [ ] **Step 3: Remove the obsolete profile-summary sidecar authority model**

In `src/sol_execbench/core/bench/profile_summary.py`, delete the whole obsolete
profile-summary sidecar authority model with the docstring `Authority boundary
for profile summary sidecars.` and the diagnostic-only `*_authority` literal
fields.

Do not modify `ProfileSummaryGovernanceGuardrail`.

- [ ] **Step 4: Verify the inactive class names are gone**

Run:

```bash
rg -n "<inactive sidecar authority class-name pattern>" src tests docs
```

Expected result: no matches.

- [ ] **Step 5: Run sidecar and governance tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_profile_summary.py -q
```

Expected result: all tests in both files pass.

- [ ] **Step 6: Commit the cleanup**

Run:

```bash
git add src/sol_execbench/core/bench/agent_feedback.py src/sol_execbench/core/bench/profile_summary.py
git commit -s -m "refactor: remove inactive sidecar authority models"
```

Expected result: commit succeeds with DCO sign-off.

---

### Task 4: Final Verification

**Files:**
- Verify: `tests/sol_execbench/test_contract.py`
- Verify: `tests/sol_execbench/test_agent_feedback.py`
- Verify: `tests/sol_execbench/test_profile_summary.py`
- Verify: `tests/sol_execbench/test_cli_environment_snapshot.py`

- [ ] **Step 1: Run the accepted convergence test set**

Run:

```bash
uv run pytest tests/sol_execbench/test_contract.py tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_profile_summary.py tests/sol_execbench/test_cli_environment_snapshot.py
```

Expected result: all selected tests pass.

- [ ] **Step 2: Inspect current active contract documentation references**

Run:

```bash
rg -n '`(runtime\.evidence\.v1|profiling\.evidence\.v1|toolchain\.routing\.v1|static_kernel_evidence\.v1|agent_feedback\.sidecar\.v1|profile_summary\.sidecar\.v1)`' docs/EVALUATOR-CONTRACT.md docs/trace.md
```

Expected result: no matches. This command checks only backtick-quoted old
evaluator capability tokens, so it does not reject the valid concrete artifact
schema `sol_execbench.static_kernel_evidence.v1`.

- [ ] **Step 3: Inspect removed authority class references**

Run:

```bash
rg -n "<inactive sidecar authority class-name pattern>" src tests docs
```

Expected result: no matches.

- [ ] **Step 4: Confirm worktree status**

Run:

```bash
git status --short
```

Expected result: no unstaged or uncommitted implementation changes remain.
