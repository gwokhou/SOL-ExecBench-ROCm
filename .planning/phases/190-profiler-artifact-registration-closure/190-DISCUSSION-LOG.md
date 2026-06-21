# Phase 190: Profiler Artifact Registration Closure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-21
**Phase:** 190-Profiler Artifact Registration Closure
**Areas discussed:** Discovery scope, Artifact formats, Status semantics, Citation cost

---

## Discovery Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Recursive scan with filtering | Scan `output_directory` recursively, but register only files matching the requested prefix, known profiler names, or known profiler output structures. | ✓ |
| Top-level plus known subdirectories | Scan top level and a small fixed set of subdirectories. | |
| Only fix current known failed layout | Add a narrow special case for the observed failure. | |

**User's choice:** Recursive scan with filtering.
**Notes:** The user selected broad nested-layout coverage while preserving filtering against unrelated files.

---

## Artifact Formats

| Option | Description | Selected |
|--------|-------------|----------|
| Classify common formats, unknown as other | Explicitly classify `rocpd`, CSV, JSON, Perfetto/PFTrace, and OTF2; register other matching files as `other`. | ✓ |
| Only rocpd/CSV/JSON | Limit explicit coverage to the currently common formats. | |
| Cover and parse all formats | Discover and parse every format into structured metadata. | |

**User's choice:** Classify common formats, unknown as other.
**Notes:** Phase 190 should register/cite artifacts, not parse every profiler format.

---

## Status Semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Keep success with coverage/warnings | Return-code-zero plus at least one artifact remains `success`; coverage/warnings express incompleteness. | ✓ |
| Add partial status | Mark return-code-zero incomplete artifact sets as `partial`. | |
| Add success_with_warnings status | Add a new top-level warning success state. | |

**User's choice:** Keep success with coverage/warnings.
**Notes:** This preserves existing `result.succeeded == status == "success"` semantics.

---

## Citation Cost

| Option | Description | Selected |
|--------|-------------|----------|
| Size-limited SHA256 | Hash small files and mark large files as skipped by size. | |
| Always compute SHA256 | Compute SHA256 for every registered profiler artifact. | ✓ |
| No default SHA256 | Record path, size, kind, and status only. | |

**User's choice:** Always compute SHA256.
**Notes:** The user prioritized evidence verifiability over citation hashing cost.

---

## the agent's Discretion

- Exact helper names, reason-code enum/module shape, and fixture layout are left to the agent.

## Deferred Ideas

- Rich profiler counter parsing and bottleneck inference belong to Phase 191.
- A future size limit for profiler DB hashing may be considered if always-hash is too expensive in real runs.
