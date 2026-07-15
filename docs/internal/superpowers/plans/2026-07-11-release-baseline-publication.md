# Release Baseline Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a complete-suite, provenance-complete release baseline and versioned score evidence bundle, then independently rerun and gate it without promoting derived or blocked workloads to official authority.

**Architecture:** Add a `release_baseline` scoring package that turns a canonical suite manifest plus an optimized-solution trace into the existing compact `scoring_baseline.v1` and a new auditable `release_baseline_bundle.v1`.  A separate verifier consumes that immutable bundle and an independently produced trace, classifies each workload as `official`, `derived`, or `blocked`, and writes a deterministic verification report.  Existing prerelease packaging copies and requires the final bundle and verification report, while preserving its engineering-prerelease claim boundary.

**Tech Stack:** Python 3.12, dataclasses, JSON/SHA-256, Click, argparse, Pytest, existing SOL trace and official-score evidence models.

## Global Constraints

- Historical planning note: the current `sol_execbench.scoring_baseline.v1` reader
  requires the exact current serialization and has no backward-compatibility mode.
- The selected suite manifest is the denominator: every `(definition, workload_uuid)` appears exactly once in the release bundle.
- Classification is exactly one of `official`, `derived`, or `blocked`; `derived`/`blocked` must remain visible in summaries.
- Only `official` rows may be described as official score authority; a nonzero derived or blocked count forbids full-suite-official wording.
- Reject non-positive and non-finite latency from `scoring_baseline.v1`; represent affected suite rows as `blocked` in the bundle.
- Baseline and rerun must be separate paths/runs and must match solution hash, suite checksum, environment fingerprint, timing policy, compiler/build identity, and bound/model checksums.
- Record a positive relative latency tolerance in the release bundle; equality at the tolerance boundary passes.
- Do not calibrate hardware models, alter SOL arithmetic, claim NVIDIA parity, or require ROCm hardware in unit tests.
- Keep generated release evidence under `out/` or supplied output paths; do not commit benchmark data or generated artifacts.

---

## File Structure

- Create: `src/sol_execbench/core/scoring/release_baseline/models.py` — frozen release bundle, workload, provenance, and verification dataclasses with deterministic serialization.
- Create: `src/sol_execbench/core/scoring/release_baseline/builder.py` — trace/manifest ingestion, complete-suite baseline construction, status classification, checksums, and atomic JSON output.
- Create: `src/sol_execbench/core/scoring/release_baseline/verifier.py` — independent rerun matching, latency tolerance comparison, final classification, and verification report generation.
- Create: `src/sol_execbench/core/scoring/release_baseline/__init__.py` — stable public APIs and schema constants.
- Modify: `src/sol_execbench/core/scoring/__init__.py` — lazy export the release-baseline public surface.
- Modify: `src/sol_execbench/cli/commands/baseline.py` — add `baseline release build` and `baseline release verify` subcommands.
- Modify: `scripts/internal/release/build_prerelease_artifact_bundle.py` — accept, validate, copy, checksum, and require finalized baseline release evidence.
- Modify: `scripts/internal/release/check_prerelease_readiness.py` — verify the required bundle/report pair and reject inconsistent classification or full-suite-official overclaim.
- Modify: `docs/internal/prerelease_artifact_bundle.md`, `docs/internal/prerelease_readiness.md`, `docs/user/EVALUATOR-CONTRACT.md`, `docs/internal/sol_score_gap_and_amd_reuse_report.md` — document commands, artifact authority, and closure of 7.3/7.7 without changing non-goals.
- Create: `tests/sol_execbench/core/scoring/test_release_baseline.py` — builder/model/verifier unit coverage.
- Modify: `tests/sol_execbench/cli/commands/test_baseline.py` — CLI invocation and options coverage.
- Modify: `tests/sol_execbench/core/dataset/test_prerelease_artifact_bundle.py` — packaging integration coverage.
- Modify: `tests/sol_execbench/core/dataset/test_prerelease_readiness.py` — final release-baseline readiness coverage.

### Task 1: Define the deterministic release-baseline contracts

**Files:**
- Create: `src/sol_execbench/core/scoring/release_baseline/models.py`
- Create: `src/sol_execbench/core/scoring/release_baseline/__init__.py`
- Modify: `src/sol_execbench/core/scoring/__init__.py`
- Test: `tests/sol_execbench/core/scoring/test_release_baseline.py`

**Consumes:** canonical workload objects containing `definition` and `workload_uuid`; existing `ScoringBaselineArtifact`.

**Produces:** `RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION`, `RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION`, `ReleaseBaselineBundle`, `ReleaseBaselineWorkload`, `ReleaseProvenance`, `ReleaseBaselineVerification`, and `sha256_file(path: Path) -> str`.

- [ ] **Step 1: Write failing contract tests for serialization, denominator, and enum validation.**

```python
from sol_execbench.core.scoring.release_baseline import (
    ReleaseBaselineBundle,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
)

def test_release_bundle_serializes_full_denominator_in_key_order():
    bundle = ReleaseBaselineBundle(
        release="v2.14-gfx1200-rocm7.1",
        suite_manifest_ref="suite.json",
        suite_manifest_sha256="a" * 64,
        baseline_artifact_ref="scoring-baseline.json",
        baseline_artifact_sha256="b" * 64,
        provenance=ReleaseProvenance(solution="hipblaslt", solution_sha256="c" * 64),
        workloads=(
            ReleaseBaselineWorkload("z", "2", "blocked", None, ("missing_baseline",)),
            ReleaseBaselineWorkload("a", "1", "derived", 1.0, ("model_not_validated",)),
        ),
        latency_tolerance_rel=0.05,
    )
    assert [row["definition"] for row in bundle.to_dict()["workloads"]] == ["a", "z"]
    assert bundle.summary == {"total": 2, "official": 0, "derived": 1, "blocked": 1}

def test_release_workload_rejects_unknown_classification():
    with pytest.raises(ValueError, match="classification"):
        ReleaseBaselineWorkload("gemm", "w1", "provisional", 1.0, ())
```

- [ ] **Step 2: Run the new tests to verify they fail.**

Run: `uv run pytest tests/sol_execbench/core/scoring/test_release_baseline.py -q`

Expected: FAIL during import because `sol_execbench.core.scoring.release_baseline` does not exist.

- [ ] **Step 3: Implement frozen models and deterministic JSON surface.**

```python
RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION = "sol_execbench.release_baseline_bundle.v1"
RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION = "sol_execbench.release_baseline_verification.v1"
CLASSIFICATIONS = ("official", "derived", "blocked")

@dataclass(frozen=True)
class ReleaseBaselineWorkload:
    definition: str
    workload_uuid: str
    classification: str
    latency_ms: float | None
    blocker_reason_codes: tuple[str, ...]
    trace_ref: str | None = None
    trace_sha256: str | None = None
    bound_ref: str | None = None
    bound_sha256: str | None = None
    hardware_model_ref: str | None = None
    hardware_model_sha256: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.definition, self.workload_uuid)
```

Implement `__post_init__` checks for classification, positive finite optional latency, non-empty identity, 64-character lowercase SHA-256 digests, unique workload keys, positive finite `latency_tolerance_rel`, and `total == official + derived + blocked`.  Ensure `to_dict()` sorts workload rows by `.key`, converts tuples to JSON lists, and includes `summary`, `schema_version`, artifact references, provenance, and immutable identity fields.

- [ ] **Step 4: Add parsing and round-trip tests.**

```python
def test_release_bundle_rejects_duplicate_workload_keys():
    row = ReleaseBaselineWorkload("gemm", "w1", "blocked", None, ("missing_baseline",))
    with pytest.raises(ValueError, match="duplicate workload"):
        ReleaseBaselineBundle(
            release="v2.14", suite_manifest_ref="suite.json", suite_manifest_sha256="a" * 64,
            baseline_artifact_ref="baseline.json", baseline_artifact_sha256="b" * 64,
            provenance=ReleaseProvenance(solution="hipblaslt", solution_sha256="c" * 64),
            workloads=(row, row), latency_tolerance_rel=0.05,
        )

def test_release_bundle_json_round_trip_is_stable(tmp_path):
    bundle = _bundle_fixture()
    path = tmp_path / "bundle.json"
    write_release_baseline_bundle(bundle, path)
    assert load_release_baseline_bundle(path).to_dict() == bundle.to_dict()
```

- [ ] **Step 5: Run focused tests and format.**

Run: `uv run pytest tests/sol_execbench/core/scoring/test_release_baseline.py -q && uv run --with ruff ruff check src/sol_execbench/core/scoring/release_baseline tests/sol_execbench/core/scoring/test_release_baseline.py && uv run --with ruff ruff format --check src/sol_execbench/core/scoring/release_baseline tests/sol_execbench/core/scoring/test_release_baseline.py`

Expected: PASS.

- [ ] **Step 6: Commit the contract.**

```bash
git add src/sol_execbench/core/scoring/release_baseline src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/core/scoring/test_release_baseline.py
git commit -s -m "feat: define release baseline evidence contract"
```

### Task 2: Build complete-suite release baselines from trace evidence

**Files:**
- Create: `src/sol_execbench/core/scoring/release_baseline/builder.py`
- Modify: `src/sol_execbench/core/scoring/release_baseline/__init__.py`
- Test: `tests/sol_execbench/core/scoring/test_release_baseline.py`

**Consumes:** `suite_manifest: list[dict[str, str]]`, trace JSONL, release/provenance metadata, and per-workload authority inputs.

**Produces:** `build_release_baseline_bundle(suite_workloads, trace_path, release, provenance, authority_by_key, latency_tolerance_rel) -> tuple[ScoringBaselineArtifact, ReleaseBaselineBundle]` and `write_release_baseline_outputs(baseline, bundle, baseline_path, bundle_path) -> ReleaseBaselineBundle`.

- [ ] **Step 1: Write tests for valid entries and blocked trace cases.**

```python
def test_builder_writes_compact_entries_and_keeps_missing_suite_rows_blocked(tmp_path):
    baseline, bundle = build_release_baseline_bundle(
        suite_workloads=[{"definition": "gemm", "workload_uuid": "w1"}, {"definition": "gemm", "workload_uuid": "w2"}],
        trace_path=_write_trace(tmp_path, _passed_trace("gemm", "w1", 1.25)),
        release="v2.14", provenance=_provenance(tmp_path), authority_by_key={}, latency_tolerance_rel=0.05,
    )
    assert [(entry.definition, entry.workload_uuid) for entry in baseline.entries] == [("gemm", "w1")]
    assert {row.workload_uuid: row.classification for row in bundle.workloads} == {"w1": "derived", "w2": "blocked"}
```

- [ ] **Step 2: Run focused builder tests to verify failure.**

Run: `uv run pytest tests/sol_execbench/core/scoring/test_release_baseline.py -q`

Expected: FAIL because `build_release_baseline_bundle` is not exported.

- [ ] **Step 3: Implement trace normalization and classification precedence.**

```python
def build_release_baseline_bundle(
    *,
    suite_workloads: Sequence[Mapping[str, str]],
    trace_path: Path,
    release: str,
    provenance: ReleaseProvenance,
    authority_by_key: Mapping[tuple[str, str], AuthorityInput],
    latency_tolerance_rel: float,
) -> tuple[ScoringBaselineArtifact, ReleaseBaselineBundle]:
    """Build complete-suite release evidence; never drop an expected workload."""
```

For each manifest workload, use only a `PASSED` trace record with a finite, positive `evaluation.performance.latency_ms`.  Duplicate matching trace records, absent records, failed status, invalid latency, unrecognized trace workload, or mismatched definition/UUID produce a `blocked` row with stable blocker codes.  Valid rows become `official` only if `AuthorityInput.official_blockers` is empty; otherwise become `derived` and preserve its blocker codes.  Create `ScoringBaselineEntry(definition=definition, workload_uuid=workload_uuid, latency_ms=latency_ms, source="release_baseline_bundle")` only for valid measured latencies.  Fail before output if the manifest itself has duplicate/malformed identities or the trace contains identities outside the suite.

- [ ] **Step 4: Add tests for duplicate, non-finite, failed, and authority-degraded inputs.**

```python
@pytest.mark.parametrize("latency", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_latency_becomes_blocked_and_is_excluded_from_compact_artifact(tmp_path, latency):
    baseline, bundle = _build_with_trace(tmp_path, _passed_trace("gemm", "w1", latency))
    assert baseline.entries == ()
    assert bundle.workloads[0].classification == "blocked"
    assert "invalid_baseline_latency" in bundle.workloads[0].blocker_reason_codes
```

- [ ] **Step 5: Implement atomic output and checksum linkage.**

```python
def write_release_baseline_outputs(
    *, baseline: ScoringBaselineArtifact, bundle: ReleaseBaselineBundle,
    baseline_path: Path, bundle_path: Path,
) -> ReleaseBaselineBundle:
    """Write deterministic baseline first, then a bundle referencing its digest."""
```

Write baseline JSON via temporary sibling then `Path.replace`; derive its SHA-256 after replacement; return/write a bundle with the exact `baseline_artifact_ref` and digest.  Never write a bundle whose referenced baseline file or digest differs from its content.

- [ ] **Step 6: Run tests and commit.**

Run: `uv run pytest tests/sol_execbench/core/scoring/test_release_baseline.py -q`

Expected: PASS.

```bash
git add src/sol_execbench/core/scoring/release_baseline tests/sol_execbench/core/scoring/test_release_baseline.py
git commit -s -m "feat: build complete release baseline evidence"
```

### Task 3: Verify an independent rerun without mutating baseline evidence

**Files:**
- Create: `src/sol_execbench/core/scoring/release_baseline/verifier.py`
- Modify: `src/sol_execbench/core/scoring/release_baseline/__init__.py`
- Test: `tests/sol_execbench/core/scoring/test_release_baseline.py`

**Consumes:** `ReleaseBaselineBundle`, independent rerun trace, and immutable rerun provenance.

**Produces:** `verify_release_baseline_rerun(bundle, rerun_trace_path, rerun_provenance) -> ReleaseBaselineVerification` and `write_release_baseline_verification(verification, output_path) -> None`.

- [ ] **Step 1: Write failing verifier tests for pass, boundary tolerance, and immutable mismatch.**

```python
def test_verifier_accepts_latency_at_relative_tolerance_boundary(tmp_path):
    bundle = _official_bundle(latency=10.0, tolerance=0.05)
    report = verify_release_baseline_rerun(
        bundle=bundle, rerun_trace_path=_write_trace(tmp_path, _passed_trace("gemm", "w1", 10.5)),
        rerun_provenance=bundle.provenance,
    )
    assert report.workloads[0].classification == "official"
    assert report.workloads[0].latency_delta_rel == pytest.approx(0.05)

def test_verifier_blocks_solution_hash_mismatch_even_when_latency_matches(tmp_path):
    report = verify_release_baseline_rerun(bundle=_official_bundle(), rerun_trace_path=_matching_trace(tmp_path), rerun_provenance=_provenance(tmp_path, solution_sha256="f" * 64))
    assert "solution_hash_mismatch" in report.workloads[0].blocker_reason_codes
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `uv run pytest tests/sol_execbench/core/scoring/test_release_baseline.py -q`

Expected: FAIL because `verify_release_baseline_rerun` does not exist.

- [ ] **Step 3: Implement immutable comparison and final classification.**

```python
def verify_release_baseline_rerun(
    *, bundle: ReleaseBaselineBundle, rerun_trace_path: Path,
    rerun_provenance: ReleaseProvenance,
) -> ReleaseBaselineVerification:
    """Verify a new run; preserve bundle latencies and downgrade only as required."""
```

Require a distinct resolved trace path from each baseline trace reference.  Compare solution hash, environment fingerprint, clock policy, compiler/build ID, timing policy, suite checksum, and each workload's bound/model checksums before comparing latency.  Calculate `abs(rerun - baseline) / baseline`; `<= latency_tolerance_rel` passes.  Preserve `derived` rows as derived if all rerun evidence matches.  Convert any `official` or `derived` row with missing rerun, invalid latency, immutable mismatch, extra/duplicate trace identity, or tolerance breach to `blocked` with a stable reason.  The report includes original/final classification, baseline/rerun latency, delta, passed flag, per-row reasons, and summary denominator.

- [ ] **Step 4: Add failure matrix tests and deterministic output test.**

```python
@pytest.mark.parametrize("reason", ["environment_fingerprint_mismatch", "timing_policy_mismatch", "bound_checksum_mismatch", "latency_outside_tolerance"])
def test_verifier_reports_stable_blocker_codes(tmp_path, reason):
    report = _verify_with_single_mismatch(tmp_path, reason)
    assert report.workloads[0].classification == "blocked"
    assert reason in report.workloads[0].blocker_reason_codes
```

- [ ] **Step 5: Run focused tests and commit.**

Run: `uv run pytest tests/sol_execbench/core/scoring/test_release_baseline.py -q`

Expected: PASS.

```bash
git add src/sol_execbench/core/scoring/release_baseline tests/sol_execbench/core/scoring/test_release_baseline.py
git commit -s -m "feat: verify independent release baseline reruns"
```

### Task 4: Expose release building and verification through the baseline CLI

**Files:**
- Modify: `src/sol_execbench/cli/commands/baseline.py`
- Test: `tests/sol_execbench/cli/commands/test_baseline.py`

**Consumes:** `baseline release build` options for suite manifest, trace, solution/provenance, authority JSON, tolerance, and output paths; `baseline release verify` options for bundle, rerun trace/provenance, and output.

**Produces:** two files from `release build` (`scoring_baseline.v1` and `release_baseline_bundle.v1`) and one `release_baseline_verification.v1` file from `release verify`.

- [ ] **Step 1: Write Click tests for the new commands.**

```python
def test_release_build_writes_compact_baseline_and_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_baseline, "build_release_baseline_bundle", _fake_build)
    result = CliRunner().invoke(cli, ["baseline", "release", "build", "--suite-manifest", str(tmp_path / "suite.json"), "--trace", str(tmp_path / "trace.jsonl"), "--release", "v2.14", "--baseline-output", str(tmp_path / "baseline.json"), "--bundle-output", str(tmp_path / "bundle.json"), "--solution", "hipblaslt", "--solution-sha256", "a" * 64, "--environment-fingerprint", "gfx1200-rocm7.1", "--timing-policy", "median-100", "--compiler-build-id", "rocm-7.1", "--latency-tolerance-rel", "0.05"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "baseline.json").exists()
    assert (tmp_path / "bundle.json").exists()
```

- [ ] **Step 2: Run CLI tests to verify failure.**

Run: `uv run pytest tests/sol_execbench/cli/commands/test_baseline.py -q`

Expected: FAIL before the nested `release build` command is implemented.

- [ ] **Step 3: Add commands with explicit required provenance.**

```python
@release_cli.command("build")
@click.option("--suite-manifest", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--trace", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--release", required=True)
@click.option("--baseline-output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--bundle-output", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--solution", required=True)
@click.option("--solution-sha256", required=True)
@click.option("--environment-fingerprint", required=True)
@click.option("--timing-policy", required=True)
@click.option("--compiler-build-id", required=True)
@click.option("--latency-tolerance-rel", required=True, type=click.FloatRange(min=0.0, min_open=True))
def _release_build_cli(
    suite_manifest_path: Path, trace_path: Path, release: str,
    baseline_output_path: Path, bundle_output_path: Path, solution: str,
    solution_sha256: str, environment_fingerprint: str, timing_policy: str,
    compiler_build_id: str, latency_tolerance_rel: float,
) -> None:
    """Build compact and complete release-baseline evidence from one trace."""
```

Add equivalent required `release verify` options for `--bundle`, `--rerun-trace`, `--output`, `--solution-sha256`, `--environment-fingerprint`, `--timing-policy`, and `--compiler-build-id`. Parse suite and authority JSON before invoking core code; show output locations and summaries. Do not add a command path that infers identity from host state.

- [ ] **Step 4: Add option validation and verify-command tests.**

```python
def test_release_build_requires_positive_tolerance(tmp_path):
    result = CliRunner().invoke(cli, ["baseline", "release", "build", "--suite-manifest", str(tmp_path / "suite.json"), "--trace", str(tmp_path / "trace.jsonl"), "--release", "v2.14", "--baseline-output", str(tmp_path / "baseline.json"), "--bundle-output", str(tmp_path / "bundle.json"), "--solution", "hipblaslt", "--solution-sha256", "a" * 64, "--environment-fingerprint", "gfx1200-rocm7.1", "--timing-policy", "median-100", "--compiler-build-id", "rocm-7.1", "--latency-tolerance-rel", "0"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output

def test_release_verify_writes_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_baseline, "verify_release_baseline_rerun", _fake_verify)
    result = CliRunner().invoke(cli, ["baseline", "release", "verify", "--bundle", str(tmp_path / "bundle.json"), "--rerun-trace", str(tmp_path / "rerun.jsonl"), "--output", str(tmp_path / "verification.json"), "--solution-sha256", "a" * 64, "--environment-fingerprint", "gfx1200-rocm7.1", "--timing-policy", "median-100", "--compiler-build-id", "rocm-7.1"])
    assert result.exit_code == 0, result.output
```

- [ ] **Step 5: Run CLI tests, lint, and commit.**

Run: `uv run pytest tests/sol_execbench/cli/commands/test_baseline.py -q && uv run --with ruff ruff check src/sol_execbench/cli/commands/baseline.py tests/sol_execbench/cli/commands/test_baseline.py`

Expected: PASS.

```bash
git add src/sol_execbench/cli/commands/baseline.py tests/sol_execbench/cli/commands/test_baseline.py
git commit -s -m "feat: add release baseline CLI workflow"
```

### Task 5: Include finalized release evidence in the prerelease artifact bundle

**Files:**
- Modify: `scripts/internal/release/build_prerelease_artifact_bundle.py`
- Modify: `tests/sol_execbench/core/dataset/test_prerelease_artifact_bundle.py`

**Consumes:** `--release-baseline-bundle PATH` and `--release-baseline-verification PATH` passed by a release maintainer.

**Produces:** copied `release_baseline/` input files, checksum-backed required artifacts, and a manifest summary that reports official/derived/blocked counts.

- [ ] **Step 1: Write failing package tests for accepted evidence and malformed reports.**

```python
def test_bundle_copies_final_release_baseline_evidence_and_records_summary(tmp_path, monkeypatch):
    baseline_bundle, verification = _write_final_release_evidence(tmp_path)
    assert build_prerelease_artifact_bundle.main(["--version", "v2.14-rc1", "--output-dir", str(tmp_path / "out"), "--release-baseline-bundle", str(baseline_bundle), "--release-baseline-verification", str(verification)]) == 0
    manifest = _load_manifest(tmp_path / "out")
    artifacts = {item["id"]: item for item in manifest["artifacts"]}
    assert artifacts["release_baseline_bundle"]["required"] is True
    assert manifest["release_baseline_summary"] == {"total": 2, "official": 1, "derived": 1, "blocked": 0}
```

- [ ] **Step 2: Run package tests to verify failure.**

Run: `uv run pytest tests/sol_execbench/core/dataset/test_prerelease_artifact_bundle.py -q`

Expected: FAIL because the builder does not accept release-baseline options.

- [ ] **Step 3: Implement validated copy and manifest entry generation.**

```python
parser.add_argument("--release-baseline-bundle", type=Path, default=None)
parser.add_argument("--release-baseline-verification", type=Path, default=None)

def _release_baseline_artifacts(
    bundle_path: Path, verification_path: Path, output_dir: Path,
    checksum_cache: dict[Path, str],
) -> tuple[list[BundleArtifact], dict[str, int]]:
    """Copy only a schema-valid, internally linked final release evidence pair."""
```

Require both options together.  Load them using core release-baseline parsers; reject a verification that does not name the supplied bundle digest or whose final summary differs from its rows.  Copy them to `output_dir / "release_baseline"` with deterministic filenames, calculate SHA-256 after copying, add both as required artifacts, and record exact counts in `release_baseline_summary`.  Mark the artifacts `provisional` unless all rows are official; never change global claim boundary to leaderboard or paper authority.

- [ ] **Step 4: Test failure paths.**

```python
@pytest.mark.parametrize("argument", ["--release-baseline-bundle", "--release-baseline-verification"])
def test_bundle_requires_release_evidence_pair(tmp_path, argument):
    with pytest.raises(SystemExit, match="must be supplied together"):
        build_prerelease_artifact_bundle.main(["--version", "v2.14", argument, str(tmp_path / "input.json")])

def test_bundle_rejects_verification_for_different_bundle(tmp_path):
    with pytest.raises(ValueError, match="bundle checksum"):
        _run_with_mismatched_release_evidence(tmp_path)
```

- [ ] **Step 5: Run focused tests and commit.**

Run: `uv run pytest tests/sol_execbench/core/dataset/test_prerelease_artifact_bundle.py -q`

Expected: PASS.

```bash
git add scripts/internal/release/build_prerelease_artifact_bundle.py tests/sol_execbench/core/dataset/test_prerelease_artifact_bundle.py
git commit -s -m "feat: package release baseline evidence"
```

### Task 6: Make readiness reject inconsistent release-baseline publication claims

**Files:**
- Modify: `scripts/internal/release/check_prerelease_readiness.py`
- Modify: `tests/sol_execbench/core/dataset/test_prerelease_readiness.py`

**Consumes:** prerelease artifact manifest entries and copied `release_baseline_bundle.v1` / verification report.

**Produces:** blocking findings for absent, corrupted, mismatched, non-final, or overclaimed release-baseline evidence.

- [ ] **Step 1: Write failing readiness tests.**

```python
def test_readiness_accepts_complete_release_evidence_with_derived_rows(tmp_path):
    bundle_dir = _write_prerelease_bundle_with_release_evidence(tmp_path, summary={"total": 2, "official": 1, "derived": 1, "blocked": 0})
    assert check_prerelease_readiness.main(["--bundle-dir", str(bundle_dir), "--output-dir", str(tmp_path / "report"), "--skip-doc-claim-checks"]) == 0

def test_readiness_blocks_full_suite_official_claim_when_derived_rows_exist(tmp_path):
    bundle_dir = _write_prerelease_bundle_with_release_evidence(tmp_path, full_suite_official=True, summary={"total": 2, "official": 1, "derived": 1, "blocked": 0})
    assert _finding_ids(_run_readiness(bundle_dir)) >= {"release_baseline_full_suite_authority_overclaim"}
```

- [ ] **Step 2: Run readiness tests to verify failure.**

Run: `uv run pytest tests/sol_execbench/core/dataset/test_prerelease_readiness.py -q`

Expected: FAIL because release-baseline evidence is not checked.

- [ ] **Step 3: Implement release-baseline-specific readiness checks.**

```python
def _check_release_baseline_evidence(
    manifest: dict[str, object], bundle_dir: Path, checksum_cache: dict[Path, str]
) -> list[Finding]:
    """Validate final evidence linkage, denominator conservation, and authority wording."""
```

Call it from `_check_manifest`.  When either release-baseline artifact is present, require both and load them.  Verify schema versions, artifact checksums, `verification.bundle_sha256 == sha256(bundle file)`, equal suite denominator, one final record per original workload, and `official + derived + blocked == total`.  Emit blocking findings for drift and invalid summaries.  If `release_baseline_summary["derived"] + ["blocked"] > 0`, require `claim_boundary["release_baseline_full_suite_official"] is False`; when it is true emit `release_baseline_full_suite_authority_overclaim`.  Preserve all existing generic artifact and claim-boundary checks.

- [ ] **Step 4: Add checksum and denominator regression tests.**

```python
def test_readiness_blocks_release_verification_checksum_drift(tmp_path):
    bundle_dir = _write_prerelease_bundle_with_release_evidence(tmp_path)
    _mutate_release_verification(bundle_dir)
    assert "release_baseline_verification_checksum_mismatch" in _finding_ids(_run_readiness(bundle_dir))

def test_readiness_blocks_nonconserving_release_summary(tmp_path):
    bundle_dir = _write_prerelease_bundle_with_release_evidence(tmp_path, summary={"total": 2, "official": 2, "derived": 1, "blocked": 0})
    assert "release_baseline_summary_mismatch" in _finding_ids(_run_readiness(bundle_dir))
```

- [ ] **Step 5: Run tests and commit.**

Run: `uv run pytest tests/sol_execbench/core/dataset/test_prerelease_readiness.py -q`

Expected: PASS.

```bash
git add scripts/internal/release/check_prerelease_readiness.py tests/sol_execbench/core/dataset/test_prerelease_readiness.py
git commit -s -m "feat: gate release baseline publication evidence"
```

### Task 7: Document the release workflow and execute the verification matrix

**Files:**
- Modify: `docs/internal/prerelease_artifact_bundle.md`
- Modify: `docs/internal/prerelease_readiness.md`
- Modify: `docs/user/EVALUATOR-CONTRACT.md`
- Modify: `docs/internal/sol_score_gap_and_amd_reuse_report.md`
- Test: `tests/sol_execbench/test_contract.py`

**Consumes:** implemented CLI commands, schemas, and prerelease integration.

**Produces:** operator instructions and precise authority wording that marks report items 7.3 and 7.7 as implemented in code while retaining any hardware/bound authority gaps.

- [ ] **Step 1: Add documentation assertions before editing prose.**

```python
def test_release_baseline_docs_preserve_complete_suite_authority_boundary():
    report = (REPO_ROOT / "docs/internal/sol_score_gap_and_amd_reuse_report.md").read_text()
    prerelease = (REPO_ROOT / "docs/internal/prerelease_artifact_bundle.md").read_text()
    assert "release_baseline_bundle.v1" in prerelease
    assert "derived" in prerelease and "blocked" in prerelease
    assert "7.3" in report and "7.7" in report
    assert "does not make a full-suite official claim" in prerelease
```

- [ ] **Step 2: Run the assertion to verify it fails.**

Run: `uv run pytest tests/sol_execbench/test_contract.py::test_release_baseline_docs_preserve_complete_suite_authority_boundary -q`

Expected: FAIL because the release-baseline publication wording has not been added.

- [ ] **Step 3: Document exact build, verify, package, and readiness commands.**

```bash
uv run sol-execbench baseline release build \
  --suite-manifest out/release/suite.json --trace out/release/baseline-trace.jsonl \
  --release v2.14-gfx1200-rocm7.1 --baseline-output out/release/scoring-baseline.json \
  --bundle-output out/release/release-baseline.json --solution hipblaslt \
  --solution-sha256 <sha256> --environment-fingerprint <fingerprint> \
  --timing-policy median-100 --compiler-build-id <build-id> --latency-tolerance-rel 0.05
uv run sol-execbench baseline release verify \
  --bundle out/release/release-baseline.json --rerun-trace out/rerun/baseline-trace.jsonl \
  --output out/release/release-baseline-verification.json --solution-sha256 <sha256> \
  --environment-fingerprint <fingerprint> --timing-policy median-100 --compiler-build-id <build-id>
```

State that every suite workload is represented; compact baseline entries are only positive measurements; `official`, `derived`, and `blocked` are different authority statuses; and nonzero derived/blocked counts prohibit a full-suite official claim.  Update report section 7 with dated implementation status for 7.3 and 7.7 while leaving 7.1, 7.2, and all model/bound caveats open unless independently evidenced.

- [ ] **Step 4: Run all impacted tests and static checks.**

Run: `uv run pytest tests/sol_execbench/core/scoring/test_release_baseline.py tests/sol_execbench/cli/commands/test_baseline.py tests/sol_execbench/core/dataset/test_prerelease_artifact_bundle.py tests/sol_execbench/core/dataset/test_prerelease_readiness.py tests/sol_execbench/test_contract.py -q && uv run --with ruff ruff check src/sol_execbench/core/scoring/release_baseline src/sol_execbench/cli/commands/baseline.py scripts/internal/release tests/sol_execbench/core/scoring/test_release_baseline.py tests/sol_execbench/cli/commands/test_baseline.py tests/sol_execbench/core/dataset/test_prerelease_artifact_bundle.py tests/sol_execbench/core/dataset/test_prerelease_readiness.py`

Expected: PASS.

- [ ] **Step 5: Run an offline synthetic end-to-end release rehearsal.**

Run: `uv run sol-execbench baseline release build --suite-manifest tests/fixtures/release_baseline/suite.json --trace tests/fixtures/release_baseline/baseline.jsonl --release v2.14-test --baseline-output out/release-baseline-rehearsal/scoring-baseline.json --bundle-output out/release-baseline-rehearsal/release-baseline.json --solution fixture-baseline --solution-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef --environment-fingerprint fixture-gfx1200-rocm7.1 --timing-policy median-100 --compiler-build-id fixture-build --latency-tolerance-rel 0.05 && uv run sol-execbench baseline release verify --bundle out/release-baseline-rehearsal/release-baseline.json --rerun-trace tests/fixtures/release_baseline/rerun.jsonl --output out/release-baseline-rehearsal/release-baseline-verification.json --solution-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef --environment-fingerprint fixture-gfx1200-rocm7.1 --timing-policy median-100 --compiler-build-id fixture-build && uv run scripts/internal/release/build_prerelease_artifact_bundle.py --version v2.14-test --output-dir out/release-baseline-package --release-baseline-bundle out/release-baseline-rehearsal/release-baseline.json --release-baseline-verification out/release-baseline-rehearsal/release-baseline-verification.json && uv run scripts/internal/release/check_prerelease_readiness.py --bundle-dir out/release-baseline-package --output-dir out/release-baseline-readiness --skip-doc-claim-checks`

Expected: exit 0; release summary counts add up to suite total; generated artifacts remain untracked under `out/`.

- [ ] **Step 6: Commit documentation and final verification.**

```bash
git add docs/internal/prerelease_artifact_bundle.md docs/internal/prerelease_readiness.md docs/user/EVALUATOR-CONTRACT.md docs/internal/sol_score_gap_and_amd_reuse_report.md tests/sol_execbench/test_contract.py
git commit -s -m "docs: close release baseline publication workflow"
git status --short
```

Expected: no unintended tracked changes; generated `out/` artifacts are not staged.

## Plan Self-Review

- **Spec coverage:** Task 1 establishes the versioned bundle and status model; Task 2 implements report item 7.3; Tasks 3 and 6 implement report item 7.7; Task 4 exposes the workflow; Task 5 publishes it; Task 7 documents and rehearses it.  Complete-suite coverage, classification, immutable provenance, tolerance, checksums, and full-suite claim restrictions each have explicit tests.
- **Placeholder scan:** No `TBD`, `TODO`, ellipsis, or deferred implementation instruction remains.
- **Type consistency:** Build API returns `(ScoringBaselineArtifact, ReleaseBaselineBundle)`; verification consumes `ReleaseBaselineBundle` and returns `ReleaseBaselineVerification`; CLI and release scripts only use these public interfaces and their serialized files.
