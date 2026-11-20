"""Frozen C2 target-view and C3 smoke campaign controller and CPU auditor."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from sol_execbench.core.bench.clock_lock import verify_clocks
from sol_execbench.core.bench.gpu_execution import (
    native_gpu_execution,
    resolve_gpu_device,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_json_value,
)
from sol_execbench.core.dataset.corpus import (
    TARGET_VIEW_MANIFEST_FILENAME,
    generate_corpus,
    load_target_descriptor,
    validate_corpus,
)
from sol_execbench.core.dataset.corpus_models import CorpusProfile
from sol_execbench.core.dataset.target_view import MaterializedTargetView
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.integrity.source_tree import source_tree_manifest
from sol_execbench.core.platform.memory_quota import (
    GPUMemoryQuotaEvidence,
    collect_gpu_memory_quota_isolated,
)
from sol_execbench.core.process import exclusive_file_lock
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.scoring.release_environment import (
    CONTAINER_IMAGE_ID_ENV,
    verify_release_source_state,
)
from sol_execbench.core.solar_bridge import target_coverage
from sol_execbench.core.solar_bridge.corpus_readiness import (
    audit_corpus_stage_readiness,
)
from sol_execbench.core.solar_bridge.coverage_models import (
    COVERAGE_SOURCE_SHA256,
    COVERAGE_TARGET_IDENTITY_SHA256,
    COVERAGE_TARGET_SHA256,
    CoverageOptions,
    CoverageRow,
    CoverageStatus,
    TargetCoverageReport,
)
from sol_execbench.core.solar_bridge.coverage_plan import (
    coverage_rows,
    load_coverage_target,
    new_coverage_report,
)
from sol_execbench.core.solar_bridge.coverage_resume import (
    authenticate_resumed_attempts,
)
from sol_execbench.core.solar_bridge.formal_device import require_formal_device
from sol_execbench.core.solar_bridge.resource_policy import (
    formal_mapper_thread_count,
)
from solar.analysis.orojenesis.runner import OrojenesisRunner
from solar.ir.contracts import IRPath
from solar.verification.execution import semantic_runtime_identity

_CACHE_ENV_KEYS = (
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_CUSTOM_CACHE_DIR",
)
_MANIFEST_REL = "problems/LLM_CORE/releases/LLM_CORE_V2/manifest.yaml"
_DESCRIPTOR_REL = (
    "problems/LLM_CORE/targets/configurations/rx9060xt/standard-16gb.yaml"
)
_SMOKE_DEFER_REASON = "smoke checkpoint reached; expansion pending review"
_ADMIT_TIMEOUT_SECONDS = 30.0
_CLI_PREFLIGHT_TIMEOUT_SECONDS = 60.0
_MIB = 1024**2
_GIB = 1024**3


class CampaignPhase(StrEnum):
    """Closed vocabulary for C2/C3 container phases."""

    READINESS = "readiness"
    ADMIT_GPU = "admit-gpu"
    REPRESENTATIVES = "b2-representatives"
    C2 = "c2"
    C3_SMOKE = "c3-smoke"
    C3_RESUME = "c3-resume"
    AUDIT = "audit"


@dataclass(frozen=True, slots=True, kw_only=True)
class CampaignPaths:
    """Canonical container paths inside repository and outputs bundle."""

    repo: Path
    outputs: Path
    freeze: Path

    @property
    def manifest(self) -> Path:
        """Frozen LLM_CORE_V2 corpus manifest."""
        return self.repo / _MANIFEST_REL

    @property
    def descriptor(self) -> Path:
        """Frozen RX 9060 XT standard-16gb target descriptor."""
        return self.repo / _DESCRIPTOR_REL

    @property
    def llm_core(self) -> Path:
        """Line C evidence root under /outputs."""
        return self.outputs / "llm-core"

    @property
    def target_view_dir(self) -> Path:
        """Sealed C2 materialized target-view directory."""
        return self.llm_core / "target-view"

    @property
    def target_view_manifest(self) -> Path:
        """Sealed C2 target-view manifest path."""
        return self.target_view_dir / TARGET_VIEW_MANIFEST_FILENAME

    @property
    def capacity_path(self) -> Path:
        """Measured C2 GPU memory-quota evidence."""
        return self.llm_core / "capacity-evidence.json"

    @property
    def c2_summary_path(self) -> Path:
        """Verified C2 summary and digest ledger."""
        return self.llm_core / "c2-summary.json"

    @property
    def coverage_dir(self) -> Path:
        """C3 target-coverage output directory."""
        return self.llm_core / "coverage"

    @property
    def coverage_report_path(self) -> Path:
        """Serialized TargetCoverageReport path."""
        return self.coverage_dir / target_coverage._REPORT_FILENAME


def clamped_coverage_options(
    orojenesis_home: Path,
    device: str = "cuda:0",
) -> CoverageOptions:
    """Match ``_audit_validated_target`` host-memory clamping for identical identity."""
    base = CoverageOptions(device=device, orojenesis_home=orojenesis_home)
    half_phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    return base.model_copy(
        update={"memory_bytes": min(base.memory_bytes, half_phys)}
    )


def verify_environment(
    paths: CampaignPaths,
    expected_freeze_sha256: str,
    *,
    online: bool,
) -> dict[str, Any]:
    """Authenticate freeze, checkout, image, semantic runtime, and mapper."""
    if sha256_file(paths.freeze) != expected_freeze_sha256:
        raise RuntimeError("C2/C3 freeze SHA-256 mismatch")
    spec: dict[str, Any] = load_json_value(paths.freeze)
    state = verify_release_source_state(
        paths.repo, expected_revision=spec["execution_commit"]
    )
    src_sha = stable_json_checksum(source_tree_manifest(paths.repo / "src"))
    if src_sha != spec["source_manifest_sha256"]:
        raise RuntimeError("source tree manifest SHA-256 mismatch")
    if semantic_runtime_identity() != spec["container_semantic_runtime"]:
        raise RuntimeError("semantic runtime identity mismatch")
    if os.environ.get(CONTAINER_IMAGE_ID_ENV) != spec["image_id"]:
        raise RuntimeError("container image identity mismatch")
    if (
        sha256_file(paths.manifest) != COVERAGE_SOURCE_SHA256
        or sha256_file(paths.descriptor) != COVERAGE_TARGET_SHA256
    ):
        raise RuntimeError("C0 frozen manifest or target descriptor drifted")
    mapper_home = Path(spec["mapper_home"])
    toolchain = OrojenesisRunner(mapper_home).toolchain_identity
    if (
        toolchain["artifact"]["sha256"] != spec["mapper_binary_sha256"]
        or toolchain["provenance_sha256"] != spec["mapper_provenance_sha256"]
    ):
        raise RuntimeError("Orojenesis mapper identity mismatch")
    if (
        sorted(os.sched_getaffinity(0))
        != spec["budgets"]["required_cpu_affinity"]
    ):
        raise RuntimeError("CPU affinity differs from frozen budget")
    if online:
        _verify_online_runtime(spec)
    return {**spec, "verified_source_revision": state.revision}


def _verify_online_runtime(spec: dict[str, Any]) -> None:
    if formal_mapper_thread_count() != spec["budgets"]["mapper_threads"]:
        raise RuntimeError("mapper thread count differs from frozen budget")
    cgroup_mem = Path("/sys/fs/cgroup/memory.max")
    if cgroup_mem.is_file():
        limit = int(cgroup_mem.read_text(encoding="utf-8").strip())
        if limit != spec["effective_container_memory_bytes"]:
            raise RuntimeError(
                f"cgroup memory limit {limit} != "
                f"{spec['effective_container_memory_bytes']}"
            )
    expected_caches = spec["cache_environment"]
    for key in _CACHE_ENV_KEYS:
        if os.environ.get(key) != expected_caches[key]:
            raise RuntimeError(f"cache environment differs: {key}")
        Path(expected_caches[key]).mkdir(parents=True, exist_ok=True)


def _write_state(
    outputs: Path, phase_name: str, started: float, **details: object
) -> None:
    payload = {
        "phase": phase_name,
        "elapsed_seconds": time.monotonic() - started,
        **details,
    }
    atomic_write_json_value(outputs / "state.json", payload)
    print(json.dumps(payload), flush=True)


def _admit_gpu_in_process(spec: dict[str, Any]) -> dict[str, Any]:
    with native_gpu_execution("cuda:0"):
        physical = resolve_gpu_device()
        expected = spec["device"]
        if physical.uuid != expected["uuid"] or physical.bdf != expected["bdf"]:
            raise RuntimeError("resolved GPU identity differs from freeze")
        if not verify_clocks(physical.uuid):
            raise RuntimeError("target clocks are not locked at STABLE_PEAK")
        require_formal_device("cuda:0")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one visible GPU device")
        props = torch.cuda.get_device_properties(0)
        if props.gcnArchName.split(":", 1)[0] != "gfx1200":
            raise RuntimeError("device architecture is not gfx1200")
        if (
            props.multi_processor_count != expected["runtime_wgp_count"]
            or props.total_memory != expected["visible_memory_bytes"]
        ):
            raise RuntimeError("device WGP count or VRAM size drifted")
        if int(torch.ones(4, device="cuda:0").sum().item()) != 4:
            raise RuntimeError("GPU tensor sanity sum failed")
        torch.cuda.synchronize()
        return {
            "uuid": physical.uuid,
            "bdf": physical.bdf,
            "name": props.name,
            "gfx": props.gcnArchName,
            "physical_compute_units": expected["physical_compute_units"],
            "runtime_wgp_count": props.multi_processor_count,
            "visible_memory_bytes": props.total_memory,
            "clock_policy": "STABLE_PEAK",
            "owner_token": os.environ.get("SOL_EXECBENCH_CLOCK_OWNER"),
        }


def _admit_gpu_isolated(
    paths: CampaignPaths, expected_freeze_sha256: str
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "scripts.internal.solar.c2_c3_campaign",
        CampaignPhase.ADMIT_GPU.value,
        "--repo",
        str(paths.repo),
        "--outputs",
        str(paths.outputs),
        "--freeze",
        str(paths.freeze),
        "--expected-freeze-sha",
        expected_freeze_sha256,
    ]
    result = run_in_process_group_bounded(cmd, timeout=_ADMIT_TIMEOUT_SECONDS)
    if result.returncode:
        raise RuntimeError(
            f"isolated GPU admission failed ({result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError("parent process unexpectedly initialized GPU")
    return json.loads(result.stdout)


def run_readiness(
    paths: CampaignPaths, expected_freeze_sha256: str
) -> dict[str, Any]:
    """Run device-free container preflight and verify corpus manifest."""
    if torch.cuda.is_initialized():
        raise RuntimeError("readiness initialized GPU")
    spec = verify_environment(paths, expected_freeze_sha256, online=False)
    corpus = validate_corpus(paths.manifest)
    receipt = {
        "verified": True,
        "performance_authority": False,
        "freeze_sha256": expected_freeze_sha256,
        "source_revision": spec["verified_source_revision"],
        "source_manifest_sha256": spec["source_manifest_sha256"],
        "semantic_runtime": spec["container_semantic_runtime"],
        "mapper_binary_sha256": spec["mapper_binary_sha256"],
        "mapper_provenance_sha256": spec["mapper_provenance_sha256"],
        "gpu_initialized": torch.cuda.is_initialized(),
        "corpus_validation": corpus,
    }
    atomic_write_json_value(
        paths.outputs / "preflight/container-readiness.json", receipt
    )
    return receipt


def _evaluate_backend_representatives(
    paths: CampaignPaths,
    spec: dict[str, Any],
    backend: IRPath,
) -> list[dict[str, Any]]:
    selected = spec["representatives"][backend.value]
    result = audit_corpus_stage_readiness(
        paths.repo / "problems/AMD_AKA/manifest.yaml",
        paths.outputs / "representatives" / backend.value,
        device="cuda:0",
        timeout_seconds=spec["budgets"]["readiness_seconds"],
        ir_path=backend,
        selected_item_ids=frozenset(selected),
    )
    records = [
        json.loads(line)
        for line in result.matrix_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(records) != len(selected):
        raise RuntimeError(
            f"{backend.value}: expected {len(selected)} records, got {len(records)}"
        )
    observed: set[str] = set()
    entries: list[dict[str, Any]] = []
    exact_map = spec["exact_rejections"].get(backend.value, {})
    for record in records:
        item = f"{record['problem_path']}/{record['workload_uuid']}"
        if item in observed:
            raise RuntimeError(f"duplicate representative item: {item}")
        observed.add(item)
        expectation = selected[item]
        passed = (
            record["status"] == "ready"
            if expectation == "ready"
            else record["status"] != "ready"
            and record["reason_code"] == expectation
        )
        exact = exact_map.get(item)
        if exact is not None:
            passed = passed and all(
                record.get(key) == value for key, value in exact.items()
            )
        entries.append(
            {
                "ir_path": backend.value,
                "item": item,
                "expected": expectation,
                "passed": passed,
                "record": record,
            }
        )
    if observed != set(selected):
        raise RuntimeError(f"representative mismatch on {backend.value}")
    return entries


def run_representatives(
    paths: CampaignPaths, expected_freeze_sha256: str
) -> dict[str, Any]:
    """Replay the 8 representative slots on the clean C2 execution checkout."""
    started = time.monotonic()
    spec = verify_environment(paths, expected_freeze_sha256, online=True)
    for rel, digest in spec["frozen_inputs_sha256"].items():
        if sha256_file(paths.repo / rel) != digest:
            raise RuntimeError(f"frozen input drifted: {rel}")
    admission = _admit_gpu_isolated(paths, expected_freeze_sha256)
    all_records: list[dict[str, Any]] = []
    for backend in IRPath:
        _write_state(
            paths.outputs, "b2-representatives", started, ir_path=backend.value
        )
        all_records.extend(
            _evaluate_backend_representatives(paths, spec, backend)
        )
    passed = len(all_records) == 8 and all(
        item["passed"] for item in all_records
    )
    verdict = {
        "passed": passed,
        "performance_authority": False,
        "source_revision": spec["verified_source_revision"],
        "gpu_admission": admission,
        "records": all_records,
        "elapsed_seconds": time.monotonic() - started,
    }
    verdict_path = paths.outputs / "preflight/representatives-verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json_value(verdict_path, verdict)
    if not passed:
        raise RuntimeError("representative regression differs from expectation")
    return verdict


def run_c2(paths: CampaignPaths, expected_freeze_sha256: str) -> dict[str, Any]:
    """Measure GPU memory quota and seal strict LLM_CORE_V2 core target view."""
    started = time.monotonic()
    verify_environment(paths, expected_freeze_sha256, online=True)
    verdict_path = paths.outputs / "preflight/representatives-verdict.json"
    if not verdict_path.is_file():
        raise RuntimeError("b2-representatives must pass before C2")
    if load_json_value(verdict_path).get("passed") is not True:
        raise RuntimeError("b2-representatives verdict is not passed")
    if paths.target_view_dir.exists() or paths.c2_summary_path.exists():
        raise FileExistsError("C2 outputs already exist; refusing overwrite")
    paths.llm_core.mkdir(parents=True, exist_ok=True)
    _write_state(paths.outputs, "c2-admission", started)
    admission = _admit_gpu_isolated(paths, expected_freeze_sha256)
    _write_state(paths.outputs, "c2-capacity-probe", started)
    with native_gpu_execution("cuda:0"):
        capacity: GPUMemoryQuotaEvidence = collect_gpu_memory_quota_isolated(
            "cuda:0", timeout_seconds=_ADMIT_TIMEOUT_SECONDS
        )
    if torch.cuda.is_initialized():
        raise RuntimeError("C2 parent unexpectedly initialized GPU")
    atomic_write_json_value(
        paths.capacity_path, capacity.model_dump(mode="json")
    )
    _write_state(
        paths.outputs,
        "c2-generate-corpus",
        started,
        usable_budget_bytes=capacity.usable_budget_bytes,
    )
    target_desc = load_target_descriptor(paths.descriptor)
    generate_corpus(
        paths.manifest,
        paths.target_view_dir,
        target=target_desc,
        profiles=(CorpusProfile.CORE,),
        capacity_evidence=capacity,
        require_complete_profile=True,
    )
    view = load_coverage_target(
        paths.manifest, paths.target_view_manifest, paths.descriptor
    )
    summary = _build_c2_summary(
        paths, view, capacity, admission, time.monotonic() - started
    )
    atomic_write_json_value(paths.c2_summary_path, summary)
    _write_state(paths.outputs, "c2-complete", started)
    return summary


def _build_c2_summary(
    paths: CampaignPaths,
    view: MaterializedTargetView,
    capacity: GPUMemoryQuotaEvidence,
    admission: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    rows = coverage_rows(view, "cuda:0")
    decisions = [d.model_dump(mode="json") for d in view.view.decisions]
    status_counts = dict(
        sorted(Counter(d["status"] for d in decisions).items())
    )
    family_counts = dict(
        sorted(Counter(r.family for r in rows if r.slot == "smoke").items())
    )
    if (
        len(decisions) != 36
        or len(view.view.problems) != 20
        or len(view.view.workloads) != 180
        or len(rows) != 360
    ):
        raise ValueError("C2 generated inventory violated C0 denominator")
    return {
        "status": "passed",
        "performance_authority": False,
        "gpu_admission": admission,
        "target_view_manifest_sha256": sha256_file(paths.target_view_manifest),
        "capacity_evidence_sha256": sha256_file(paths.capacity_path),
        "workload_view_digest": view.view.workload_view_digest,
        "generation_cohort_id": view.view.generation_cohort_id,
        "distribution_id": view.view.distribution_id,
        "hardware_configuration_id": view.view.hardware_configuration_id,
        "capacity_class_id": view.view.capacity_class_id,
        "capacity_class_bytes": view.view.capacity_class_bytes,
        "capacity_probe_digest": capacity.capacity_probe_digest,
        "runtime_total_bytes": capacity.runtime_total_bytes,
        "runtime_free_bytes": capacity.runtime_free_bytes,
        "stable_allocatable_bytes": capacity.stable_allocatable_bytes,
        "harness_reserve_bytes": capacity.harness_reserve_bytes,
        "safety_percent": capacity.safety_percent,
        "usable_budget_bytes": capacity.usable_budget_bytes,
        "coverage": view.view.coverage,
        "decisions_count": len(decisions),
        "decision_status_counts": status_counts,
        "problems_count": len(view.view.problems),
        "workloads_count": len(view.view.workloads),
        "planned_coverage_rows": len(rows),
        "smoke_definition_rows_by_family": family_counts,
        "decisions": decisions,
        "elapsed_seconds": elapsed_seconds,
    }


def run_c3_smoke(
    paths: CampaignPaths,
    expected_freeze_sha256: str,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute the 40 smoke rows while preserving exact 360-row resume identity."""
    started = time.monotonic()
    spec = verify_environment(paths, expected_freeze_sha256, online=True)
    c2_summary = _require_verified_c2(paths)
    admission = _admit_gpu_isolated(paths, expected_freeze_sha256)
    target = load_coverage_target(
        paths.manifest, paths.target_view_manifest, paths.descriptor
    )
    options = clamped_coverage_options(Path(spec["mapper_home"]))
    report = execute_smoke_slice(
        target,
        paths.coverage_dir,
        options,
        outputs=paths.outputs,
        started=started,
        resume=resume,
    )
    summary = build_smoke_summary(paths, report, options)
    atomic_write_json_value(paths.llm_core / "smoke-summary.json", summary)
    atomic_write_json_value(
        paths.llm_core / "exposure-ledger.json",
        _build_exposure_ledger(
            c2_summary=c2_summary,
            report=report,
            admission=admission,
            freeze_sha256=expected_freeze_sha256,
        ),
    )
    _write_state(paths.outputs, "c3-smoke-complete", started)
    return summary


def execute_smoke_slice(
    target: MaterializedTargetView,
    coverage_dir: Path,
    options: CoverageOptions,
    *,
    outputs: Path | None = None,
    started: float | None = None,
    resume: bool = False,
) -> TargetCoverageReport:
    """Admit only the frozen C0 source and run ``slot == 'smoke'`` rows."""
    if (
        target.view.source_manifest_sha256 != COVERAGE_SOURCE_SHA256
        or target.view.target_descriptor_sha256
        != COVERAGE_TARGET_IDENTITY_SHA256
    ):
        raise ValueError("M0-C0 source identity is not frozen")
    return _execute_smoke_validated(
        target,
        coverage_dir,
        options,
        outputs=outputs,
        started=started,
        resume=resume,
    )


def _execute_smoke_validated(
    target: MaterializedTargetView,
    coverage_dir: Path,
    options: CoverageOptions,
    *,
    outputs: Path | None = None,
    started: float | None = None,
    resume: bool = False,
) -> TargetCoverageReport:
    """Execute only ``slot == 'smoke'`` rows using identical C0 resume keys."""
    if torch.cuda.is_initialized():
        raise RuntimeError("controller must not hold an active GPU context")
    mapper = OrojenesisRunner(options.orojenesis_home)
    expected = new_coverage_report(target, options, mapper.toolchain_identity)
    coverage_dir = coverage_dir.resolve()
    coverage_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(coverage_dir / ".coverage.lock"):
        report = target_coverage._resume_or_create(
            coverage_dir, expected, resume
        )
        if resume:
            authenticate_resumed_attempts(target, report, coverage_dir, options)
        _run_smoke_rows(
            target,
            coverage_dir,
            options,
            report,
            outputs,
            started or time.monotonic(),
        )
        report.matrix_complete = all(
            row.attempts and row.status not in target_coverage._STOP_STATUSES
            for row in report.rows
        )
        report.costs_observed = target_coverage._costs_observed(report)
        report.complete = report.matrix_complete and report.costs_observed
        target_coverage._save(coverage_dir, report)
        # Prove standard resume authentication immediately on the written report.
        verified = target_coverage._resume_or_create(
            coverage_dir, expected, resume=True
        )
        authenticate_resumed_attempts(target, verified, coverage_dir, options)
    return report


def _run_smoke_rows(
    target: MaterializedTargetView,
    coverage_dir: Path,
    options: CoverageOptions,
    report: TargetCoverageReport,
    outputs: Path | None,
    started: float,
) -> None:
    for index, row in enumerate(report.rows):
        if row.slot != "smoke":
            if not row.attempts:
                row.status = CoverageStatus.NOT_REACHED
                row.reason = _SMOKE_DEFER_REASON
            continue
        if row.attempts:
            continue
        stop = target_coverage._stop_reason(report, coverage_dir, options, row)
        if stop:
            row.reason = stop
            continue
        if outputs is not None:
            _write_state(
                outputs,
                "c3-smoke-row",
                started,
                row_index=index,
                semantic_id=row.semantic_id,
                ir_path=row.ir_path.value,
                smoke_elapsed_seconds=report.smoke_elapsed_seconds,
            )
        target_coverage._execute_row(
            target, coverage_dir, options, report, row, index
        )


def _require_verified_c2(paths: CampaignPaths) -> dict[str, Any]:
    c2_summary: dict[str, Any] = load_json_value(paths.c2_summary_path)
    if c2_summary.get("status") != "passed":
        raise RuntimeError("C2 summary is not in passed state")
    if (
        sha256_file(paths.target_view_manifest)
        != c2_summary["target_view_manifest_sha256"]
    ):
        raise RuntimeError("sealed C2 target-view manifest changed before C3")
    if (
        sha256_file(paths.capacity_path)
        != c2_summary["capacity_evidence_sha256"]
    ):
        raise RuntimeError("sealed C2 capacity evidence changed before C3")
    return c2_summary


def _build_exposure_ledger(
    *,
    c2_summary: dict[str, Any],
    report: TargetCoverageReport,
    admission: dict[str, Any],
    freeze_sha256: str,
) -> dict[str, Any]:
    smoke_rows = [row for row in report.rows if row.slot == "smoke"]
    deferred_rows = [row for row in report.rows if row.slot != "smoke"]
    return {
        "schema_role": "llm_core_exposure_ledger",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "freeze_sha256": freeze_sha256,
        "report_identity_sha256": report.identity_sha256,
        "target_view_manifest_sha256": c2_summary[
            "target_view_manifest_sha256"
        ],
        "workload_view_digest": c2_summary["workload_view_digest"],
        "gpu_owner_token": admission.get("owner_token"),
        "exposed_smoke_workload_uuids": sorted(
            {row.workload_uuid for row in smoke_rows if row.attempts}
        ),
        "exposed_smoke_rows": sum(bool(row.attempts) for row in smoke_rows),
        "unexposed_development_holdout_rows": sum(
            not row.attempts for row in deferred_rows
        ),
        "policy": (
            "C2 materialized target view is evaluation-side only; smoke "
            "workloads inspected here cannot count as independent holdout. "
            "All 320 development/holdout rows have 0 worker attempts."
        ),
    }


def run_c3_resume(
    paths: CampaignPaths, expected_freeze_sha256: str
) -> TargetCoverageReport:
    """Resume full 320-row expansion via untouched production orchestrator."""
    spec = verify_environment(paths, expected_freeze_sha256, online=True)
    _require_verified_c2(paths)
    _admit_gpu_isolated(paths, expected_freeze_sha256)
    target = load_coverage_target(
        paths.manifest, paths.target_view_manifest, paths.descriptor
    )
    return target_coverage.audit_target_coverage(
        target,
        paths.coverage_dir,
        CoverageOptions(
            device="cuda:0", orojenesis_home=Path(spec["mapper_home"])
        ),
        resume=True,
    )


def run_audit(
    paths: CampaignPaths,
    expected_freeze_sha256: str,
    *,
    audit_out: Path,
) -> dict[str, Any]:
    """Authenticate C2 + C3 smoke artifacts inside a device-free container."""
    if torch.cuda.is_initialized():
        raise RuntimeError("device-free audit unexpectedly initialized GPU")
    spec = verify_environment(paths, expected_freeze_sha256, online=False)
    c2_summary = _require_verified_c2(paths)
    target = load_coverage_target(
        paths.manifest, paths.target_view_manifest, paths.descriptor
    )
    cli_output = _run_readonly_cli_preflight(paths)
    options = clamped_coverage_options(Path(spec["mapper_home"]))
    mapper = OrojenesisRunner(options.orojenesis_home)
    expected = new_coverage_report(target, options, mapper.toolchain_identity)
    report = load_json_file(TargetCoverageReport, paths.coverage_report_path)
    resumed = target_coverage._resume_or_create(
        paths.coverage_dir, expected, resume=True
    )
    authenticate_resumed_attempts(target, resumed, paths.coverage_dir, options)
    _verify_deferred_rows_untouched(resumed)
    smoke_summary = build_smoke_summary(paths, resumed, options)
    payload = {
        "audit_passed": True,
        "performance_authority": False,
        "gpu_initialized": torch.cuda.is_initialized(),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "freeze_sha256": expected_freeze_sha256,
        "c2_target_view_manifest_sha256": c2_summary[
            "target_view_manifest_sha256"
        ],
        "c2_workload_view_digest": c2_summary["workload_view_digest"],
        "cli_preflight_output": cli_output,
        "report_schema_version": report.schema_version,
        "report_identity_sha256": report.identity_sha256,
        "expected_identity_sha256": expected.identity_sha256,
        "resume_identity_matches_full_audit": (
            report.identity == expected.identity
            and report.identity_sha256 == expected.identity_sha256
        ),
        "matrix_complete": report.matrix_complete,
        "complete": report.complete,
        "smoke_summary": smoke_summary,
    }
    audit_out.mkdir(parents=True, exist_ok=True)
    atomic_write_json_value(audit_out / "smoke-summary.json", smoke_summary)
    atomic_write_json_value(audit_out / "cpu-audit.json", payload)
    return payload


def _run_readonly_cli_preflight(paths: CampaignPaths) -> str:
    cli_bin = Path(sys.executable).with_name("sol-execbench")
    cmd = [
        str(cli_bin),
        "solar",
        "audit-target-view",
        "--source-manifest",
        str(paths.manifest),
        "--target-view",
        str(paths.target_view_manifest),
        "--descriptor",
        str(paths.descriptor),
        "--device",
        "cuda:0",
    ]
    result = run_in_process_group_bounded(
        cmd,
        timeout=_CLI_PREFLIGHT_TIMEOUT_SECONDS,
        env=dict(os.environ),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"CLI metadata preflight failed ({result.returncode}): {result.stderr}"
        )
    text = result.stdout.strip()
    if "validated metadata: 360 rows; no worker executed" not in text:
        raise RuntimeError(f"unexpected CLI preflight output: {text!r}")
    return text


def _verify_deferred_rows_untouched(report: TargetCoverageReport) -> None:
    smoke = [r for r in report.rows if r.slot == "smoke"]
    deferred = [r for r in report.rows if r.slot != "smoke"]
    if len(smoke) != 40 or len(deferred) != 320:
        raise ValueError("unexpected smoke/deferred denominator split")
    if any(
        r.attempts or r.status is not CoverageStatus.NOT_REACHED
        for r in deferred
    ):
        raise ValueError("non-smoke row was unexpectedly executed")


def build_smoke_summary(
    paths: CampaignPaths,
    report: TargetCoverageReport,
    options: CoverageOptions,
) -> dict[str, Any]:
    """Summarize the 40-row smoke matrix, costs, two-path diffs, and projection."""
    smoke_rows = [row for row in report.rows if row.slot == "smoke"]
    row_details = [_summarize_row(row) for row in smoke_rows]
    evidence_bytes = _tree_bytes(paths.coverage_dir)
    stage_costs = _stage_cost_summary(row_details)
    two_path_diffs = _two_path_comparison(row_details)
    projection = _budget_projection(
        smoke_rows=smoke_rows,
        report=report,
        options=options,
        smoke_evidence_bytes=evidence_bytes,
    )
    return {
        "spec_id": report.spec_id.value,
        "performance_authority": False,
        "identity_sha256": report.identity_sha256,
        "total_rows": len(report.rows),
        "smoke_rows_count": len(smoke_rows),
        "deferred_rows_count": len(report.rows) - len(smoke_rows),
        "matrix_complete": report.matrix_complete,
        "complete": report.complete,
        "elapsed_seconds": report.elapsed_seconds,
        "smoke_elapsed_seconds": report.smoke_elapsed_seconds,
        "smoke_evidence_bytes": evidence_bytes,
        "overall_status_counts": dict(
            sorted(Counter(r.status.value for r in report.rows).items())
        ),
        "smoke_status_counts": dict(
            sorted(Counter(r["status"] for r in row_details).items())
        ),
        "by_ir_path": _grouped_status_counts(row_details, "ir_path"),
        "by_family": _grouped_status_counts(row_details, "family"),
        "by_family_and_ir_path": _family_path_breakdown(row_details),
        "reason_counts": _reason_breakdown(row_details),
        "readiness_stage_counts": _readiness_stage_counts(row_details),
        "formal_metrics_counts": _formal_metrics_counts(row_details),
        "stage_cost_summary": stage_costs,
        "two_path_comparison": two_path_diffs,
        "projection_remaining_320_rows": projection,
        "make_fx_real_retry_observability": {
            "observable_in_current_artifacts": False,
            "reason": (
                "make_fx_backend falls back on DynamicOutputShapeException/"
                "DataDependentOutputException without tagging operator_graph.yaml"
            ),
        },
        "smoke_rows": row_details,
    }


def _summarize_row(row: CoverageRow) -> dict[str, Any]:
    if not row.attempts:
        return {
            "key": row.key,
            "semantic_id": row.semantic_id,
            "workload_uuid": row.workload_uuid,
            "family": row.family,
            "ir_path": row.ir_path.value,
            "status": row.status.value,
            "reason": row.reason,
            "wall_seconds": 0.0,
            "readiness": None,
            "formal": None,
            "verification": None,
            "external_reads": None,
            "views": None,
            "total": None,
            "orojenesis": None,
            "stage_costs": [],
        }
    attempt = row.attempts[-1]
    meta = attempt.observations.get("metadata")
    metadata = meta if isinstance(meta, dict) else {}
    stage_records = _collect_attempt_records(attempt.observations)
    return {
        "key": row.key,
        "semantic_id": row.semantic_id,
        "workload_uuid": row.workload_uuid,
        "family": row.family,
        "ir_path": row.ir_path.value,
        "status": attempt.status.value,
        "reason": attempt.reason,
        "wall_seconds": attempt.wall_seconds,
        "artifact_count": len(attempt.artifacts),
        "readiness": attempt.readiness,
        "formal": attempt.formal,
        "verification": attempt.observations.get("verification"),
        "external_reads": metadata.get("external_reads"),
        "views": metadata.get("views"),
        "total": attempt.observations.get("total"),
        "orojenesis": metadata.get("orojenesis"),
        "stage_costs": stage_records,
    }


def _collect_attempt_records(
    observations: dict[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for phase_name in ("readiness", "formal"):
        obs = observations.get(phase_name)
        if not isinstance(obs, dict):
            continue
        raw_records = obs.get("records")
        if not isinstance(raw_records, list):
            continue
        for rec in raw_records:
            if isinstance(rec, dict):
                records.append(
                    {
                        "worker_phase": phase_name,
                        **{str(k): v for k, v in rec.items()},
                    }
                )
    return records


def _grouped_status_counts(
    rows: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter[str]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), Counter())[row["status"]] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(groups.items())}


def _family_path_breakdown(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    groups: dict[str, dict[str, Counter[str]]] = {}
    for row in rows:
        fam = groups.setdefault(row["family"], {})
        fam.setdefault(row["ir_path"], Counter())[row["status"]] += 1
    return {
        fam: {
            path: dict(sorted(c.items())) for path, c in sorted(paths.items())
        }
        for fam, paths in sorted(groups.items())
    }


def _reason_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str | None, str | None], list[str]] = {}
    for row in rows:
        if row["status"] == CoverageStatus.PASSED.value:
            continue
        readiness = row["readiness"] or {}
        formal = row["formal"] or {}
        stage = (
            formal.get("stage")
            if row["formal"] is not None
            else readiness.get("failure_stage")
        )
        key = (
            row["ir_path"],
            row["status"],
            str(stage) if stage is not None else None,
            row["reason"],
        )
        buckets.setdefault(key, []).append(row["semantic_id"])
    return [
        {
            "ir_path": ir_path,
            "status": status,
            "stage": stage,
            "reason": reason,
            "count": len(defs),
            "definitions": sorted(defs),
        }
        for (ir_path, status, stage, reason), defs in sorted(buckets.items())
    ]


def _readiness_stage_counts(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, Counter[str]]] = {}
    for row in rows:
        readiness = row["readiness"] or {}
        stages = readiness.get("stages") or []
        path_counts = result.setdefault(row["ir_path"], {})
        for entry in stages:
            if isinstance(entry, dict):
                st_name = str(entry.get("stage", "unknown"))
                st_status = str(entry.get("status", "unknown"))
                path_counts.setdefault(st_name, Counter())[st_status] += 1
    return {
        path: {st: dict(sorted(c.items())) for st, c in sorted(stages.items())}
        for path, stages in sorted(result.items())
    }


def _formal_metrics_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "formal_attempted": 0,
        "formal_passed": 0,
        "external_reads_quality": dict(Counter()),
        "view_availability": {},
    }
    ext_counter: Counter[str] = Counter()
    view_counters: dict[str, Counter[str]] = {}
    for row in rows:
        if row["formal"] is not None:
            counts["formal_attempted"] += 1
        if row["status"] == CoverageStatus.PASSED.value:
            counts["formal_passed"] += 1
        ext = row["external_reads"]
        if isinstance(ext, dict):
            ext_counter[str(ext.get("quality"))] += 1
        views = row["views"]
        if isinstance(views, dict):
            for v_name, v_data in views.items():
                if isinstance(v_data, dict):
                    label = f"{v_data.get('status')}:{v_data.get('reason')}"
                    view_counters.setdefault(v_name, Counter())[label] += 1
    counts["external_reads_quality"] = dict(sorted(ext_counter.items()))
    counts["view_availability"] = {
        k: dict(sorted(v.items())) for k, v in sorted(view_counters.items())
    }
    return counts


def _stage_cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wall_by_stage: dict[tuple[str, str], list[float]] = {}
    rss_by_stage: dict[tuple[str, str], list[int]] = {}
    dev_by_stage: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        for rec in row["stage_costs"]:
            key = (str(rec.get("worker_phase")), str(rec.get("stage")))
            wall = rec.get("wall_seconds")
            if isinstance(wall, (int, float)):
                wall_by_stage.setdefault(key, []).append(float(wall))
            rss = rec.get("host_process_high_water_bytes")
            if isinstance(rss, int) and not isinstance(rss, bool):
                rss_by_stage.setdefault(key, []).append(rss)
            dev = rec.get("device_peak_bytes")
            if isinstance(dev, int) and not isinstance(dev, bool):
                dev_by_stage.setdefault(key, []).append(dev)
    summary: dict[str, Any] = {}
    for key in sorted(wall_by_stage):
        label = f"{key[0]}:{key[1]}"
        walls = sorted(wall_by_stage[key])
        rss_vals = rss_by_stage.get(key, [])
        dev_vals = dev_by_stage.get(key, [])
        summary[label] = {
            "count": len(walls),
            "wall_seconds_total": sum(walls),
            "wall_seconds_median": statistics.median(walls),
            "wall_seconds_max": max(walls),
            "host_rss_peak_max_bytes": max(rss_vals) if rss_vals else None,
            "device_peak_max_bytes": max(dev_vals) if dev_vals else None,
        }
    return summary


def _two_path_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_def: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_def.setdefault(row["semantic_id"], {})[row["ir_path"]] = row
    pairs: list[dict[str, Any]] = []
    status_matrix: Counter[str] = Counter()
    for semantic_id, paths in sorted(by_def.items()):
        mfx = paths[IRPath.MAKE_FX_ATEN.value]
        tv = paths[IRPath.TORCHVIEW_EXTENDED_EINSUM.value]
        status_matrix[f"{mfx['status']}__vs__{tv['status']}"] += 1
        if (
            mfx["status"] != CoverageStatus.PASSED.value
            or tv["status"] != CoverageStatus.PASSED.value
        ):
            continue
        pairs.append(_compare_passed_pair(semantic_id, mfx, tv))
    return {
        "common_passed_count": len(pairs),
        "status_pair_counts": dict(sorted(status_matrix.items())),
        "common_passed_diffs": pairs,
    }


def _compare_passed_pair(
    semantic_id: str, mfx: dict[str, Any], tv: dict[str, Any]
) -> dict[str, Any]:
    mfx_tot, tv_tot = mfx["total"] or {}, tv["total"] or {}
    mfx_ext = mfx["external_reads"] or {}
    tv_ext = tv["external_reads"] or {}
    mfx_views = mfx["views"] or {}
    tv_views = tv["views"] or {}
    return {
        "semantic_id": semantic_id,
        "make_fx_flops": mfx_tot.get("flops"),
        "torchview_flops": tv_tot.get("flops"),
        "flops_equal": mfx_tot.get("flops") == tv_tot.get("flops"),
        "resource_work_equal": (
            mfx_tot.get("resource_work") == tv_tot.get("resource_work")
        ),
        "external_read_bytes_equal": (
            mfx_ext.get("bytes") == tv_ext.get("bytes")
        ),
        "make_fx_external_read_bytes": mfx_ext.get("bytes"),
        "torchview_external_read_bytes": tv_ext.get("bytes"),
        "make_fx_lower_bound_seconds": mfx_tot.get("lower_bound_seconds"),
        "torchview_lower_bound_seconds": tv_tot.get("lower_bound_seconds"),
        "views_status": {
            view: {
                "make_fx": (mfx_views.get(view) or {}).get("status"),
                "torchview": (tv_views.get(view) or {}).get("status"),
            }
            for view in ("unfused", "ideal_fused", "cache_aware")
        },
    }


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _budget_projection(
    *,
    smoke_rows: list[CoverageRow],
    report: TargetCoverageReport,
    options: CoverageOptions,
    smoke_evidence_bytes: int,
) -> dict[str, Any]:
    executed = [row for row in smoke_rows if row.attempts]
    walls = [row.attempts[-1].wall_seconds for row in executed]
    mean_wall = statistics.fmean(walls) if walls else None
    remaining_rows = len(report.rows) - len(smoke_rows)
    remaining_activity = max(
        0.0, options.activity_seconds - report.elapsed_seconds
    )
    mean_bytes = (smoke_evidence_bytes / len(executed)) if executed else None
    projected_wall = (
        mean_wall * remaining_rows if mean_wall is not None else None
    )
    projected_bytes = (
        int(mean_bytes * remaining_rows) + smoke_evidence_bytes
        if mean_bytes is not None
        else None
    )
    return {
        "executed_smoke_rows": len(executed),
        "remaining_rows": remaining_rows,
        "smoke_elapsed_seconds": report.smoke_elapsed_seconds,
        "smoke_budget_seconds": options.smoke_seconds,
        "remaining_activity_budget_seconds": remaining_activity,
        "mean_wall_seconds_per_smoke_row": mean_wall,
        "max_wall_seconds_smoke_row": max(walls) if walls else None,
        "linear_projected_remaining_seconds": projected_wall,
        "fits_remaining_activity_budget_linearly": (
            projected_wall is not None and projected_wall <= remaining_activity
        ),
        "smoke_evidence_mib": smoke_evidence_bytes / _MIB,
        "linear_projected_total_evidence_gib": (
            projected_bytes / _GIB if projected_bytes is not None else None
        ),
        "evidence_budget_gib": options.evidence_bytes / _GIB,
        "note": (
            "Linear smoke extrapolation is a lower-order reference only; "
            "development/holdout capacity-high shapes are larger and may cost "
            "more per row in mapper/verification replay."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point inside the pinned C2/C3 execution container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=tuple(item.value for item in CampaignPhase)
    )
    parser.add_argument("--repo", type=Path, default=Path("/sol-execbench"))
    parser.add_argument("--outputs", type=Path, default=Path("/outputs"))
    parser.add_argument(
        "--freeze", type=Path, default=Path("/outputs/freeze.json")
    )
    parser.add_argument("--expected-freeze-sha", required=True)
    parser.add_argument("--audit-out", type=Path, default=Path("/audit-out"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    paths = CampaignPaths(
        repo=args.repo, outputs=args.outputs, freeze=args.freeze
    )
    phase = CampaignPhase(args.phase)
    if phase is CampaignPhase.READINESS:
        print(json.dumps(run_readiness(paths, args.expected_freeze_sha)))
    elif phase is CampaignPhase.ADMIT_GPU:
        spec = verify_environment(paths, args.expected_freeze_sha, online=True)
        print(json.dumps(_admit_gpu_in_process(spec)))
    elif phase is CampaignPhase.REPRESENTATIVES:
        verdict = run_representatives(paths, args.expected_freeze_sha)
        print(
            json.dumps(
                {
                    "passed": verdict["passed"],
                    "slots": len(verdict["records"]),
                }
            )
        )
    elif phase is CampaignPhase.C2:
        print(json.dumps(run_c2(paths, args.expected_freeze_sha)))
    elif phase is CampaignPhase.C3_SMOKE:
        summary = run_c3_smoke(
            paths, args.expected_freeze_sha, resume=args.resume
        )
        print(
            json.dumps({"smoke_status_counts": summary["smoke_status_counts"]})
        )
    elif phase is CampaignPhase.C3_RESUME:
        report = run_c3_resume(paths, args.expected_freeze_sha)
        print(
            json.dumps({"rows": len(report.rows), "complete": report.complete})
        )
    else:
        payload = run_audit(
            paths, args.expected_freeze_sha, audit_out=args.audit_out
        )
        print(json.dumps({"audit_passed": payload["audit_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
