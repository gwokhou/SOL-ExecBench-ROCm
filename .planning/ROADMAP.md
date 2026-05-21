# Roadmap: SOL ExecBench ROCm Port

**Created:** 2026-05-21
**Milestone:** v1.1 CDNA 3 Support and Migration Closure
**Mode:** standard
**Granularity:** standard

## Overview

This roadmap closes the remaining ROCm migration preparation gaps after v1.0.
It adds CDNA 3 support at the schema, build, marker, example, and documentation
layers while explicitly leaving real `gfx94*` hardware validation for the next
milestone.

The phase numbering continues from the shipped v1.0 roadmap. v1.0 completed
Phases 1 through 6, so this milestone starts at Phase 7.

## Phase Status

- [ ] **Phase 7: CDNA 3 Schema and Build Surface**
- [ ] **Phase 8: Migration Residue and Example Closure**
- [ ] **Phase 9: Support Documentation and Validation Handoff**

## Phases

### Phase 7: CDNA 3 Schema and Build Surface

**Goal:** Make CDNA 3 a first-class supported target in solution metadata,
native packaging, local hardware detection, and test marker semantics without
claiming real CDNA 3 validation evidence.

**Requirements:** CDNA-01, CDNA-02, CDNA-03, CDNA-04

**Success Criteria:**
1. `solution.json` accepts explicit CDNA 3 targets such as `gfx942` while still rejecting unsupported NVIDIA hardware values.
2. HIP/C++ packaging injects `--offload-arch=<gfx94*>` for explicit CDNA 3 targets and preserves existing `LOCAL` detection behavior.
3. Test marker and local architecture helpers consistently classify `gfx94*` as CDNA 3.
4. Code, tests, and docs distinguish CDNA 3 code/schema support from unrecorded hardware-validation evidence.

### Phase 8: Migration Residue and Example Closure

**Goal:** Turn the remaining active CUDA/NVIDIA/library residue into a
maintained audit surface and resolve ambiguous former NVIDIA example categories.

**Requirements:** AUDIT-01, AUDIT-02, AUDIT-03, EXMP-01, EXMP-02, EXMP-03

**Success Criteria:**
1. A source-level audit covers active source, tests, examples, and docs while excluding archived planning history.
2. Remaining CUDA/NVIDIA/CUPTI/library terms are removed or allowlisted with explicit compatibility or legacy-context reasons.
3. PyTorch `torch.cuda` and similar upstream compatibility names are documented as ROCm-compatible API namespaces, not NVIDIA runtime support.
4. Former CUTLASS/cuDNN/CuTe/cuTile example metadata and tests clearly label ROCm-native implementations, compatibility fallbacks, or intentional non-goals.
5. Portable examples include CDNA 3 target metadata where appropriate, without implying real hardware validation.

### Phase 9: Support Documentation and Validation Handoff

**Goal:** Publish accurate v1.1 support documentation and define the next
milestone's CDNA 3 hardware validation gate.

**Requirements:** DOC-01, DOC-02, DOC-03

**Success Criteria:**
1. README/setup/schema/compliance docs list CDNA 3 as code/schema-supported but not yet hardware-validated.
2. Known gaps and support matrices identify the exact missing evidence: a full adapted suite run on `gfx94*`.
3. The next milestone requirements or handoff notes define commands, expected artifacts, and acceptance criteria for CDNA 3 validation.
4. No docs claim CDNA 3 hardware validation before the real suite pass is recorded.

## Requirement Coverage

| Requirement | Phase |
|-------------|-------|
| CDNA-01 | Phase 7 |
| CDNA-02 | Phase 7 |
| CDNA-03 | Phase 7 |
| CDNA-04 | Phase 7 |
| AUDIT-01 | Phase 8 |
| AUDIT-02 | Phase 8 |
| AUDIT-03 | Phase 8 |
| EXMP-01 | Phase 8 |
| EXMP-02 | Phase 8 |
| EXMP-03 | Phase 8 |
| DOC-01 | Phase 9 |
| DOC-02 | Phase 9 |
| DOC-03 | Phase 9 |

**Coverage:** 13/13 v1.1 requirements mapped.

---
*Roadmap created: 2026-05-21*
