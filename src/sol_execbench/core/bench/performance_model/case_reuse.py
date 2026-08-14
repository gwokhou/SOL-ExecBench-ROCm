# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Case-granular held-out exposure and raw-evidence reuse governance."""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.performance_model.diagnostic_schema_versions import (
    DiagnosticAcceptanceArtifactKind,
    DiagnosticArtifactSchema,
    DiagnosticCaseReuseArtifactKind,
)
from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    acceptance_exposures_dir,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
)
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    NonEmptyString,
    StrictArtifactModel,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import (
    SHA256Digest,
    sha256_file,
    stable_json_checksum,
)

CASE_REUSE_MANIFEST_NAME = "case-reuse-manifest.json"
EXPOSURE_RECEIPT_NAME = "acceptance-exposure.json"
SOURCE_CORPUS_NAME = "source-held-out.json"
REPLACEMENT_FRAGMENT_NAME = "replacement-fragment.json"

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
_CASES_PER_FAMILY = 20
_TOTAL_CASES = 220
_SUPPORTED_FAMILIES = tuple(
    family for family in WorkloadKind if family is not WorkloadKind.UNSUPPORTED
)


class ReuseDisposition(StrEnum):
    """Whether one final held-out slot is inherited or freshly collected."""

    REUSE = "reuse"
    REPLACE = "replace"


class AcceptancePreconditionError(ValueError):
    """Acceptance stopped before a verdict or metric release was possible."""

    def __init__(
        self,
        *,
        case_id: str,
        workload_kind: WorkloadKind,
        reason_codes: tuple[str, ...],
    ) -> None:
        """Capture the first unavailable prediction without metric release."""
        detail = (
            f"{case_id} validation case lacks an available HW prediction: "
            f"{list(reason_codes)}"
        )
        super().__init__(detail)
        self.case_id = case_id
        self.workload_kind = workload_kind
        self.reason_codes = reason_codes
        self.evaluated_case_ids_before_failure: tuple[str, ...] = ()
        self.exposure_receipt_sha256: str | None = None

    def bind_exposure_receipt(self, digest: str) -> None:
        """Attach the durable receipt identity to the surfaced failure."""
        self.exposure_receipt_sha256 = digest
        self.args = (f"{self.args[0]}; exposure_receipt_sha256={digest}",)


class DiagnosticAcceptanceExposureReceipt(CurrentSchemaModel):
    """Exact information released by a pre-verdict acceptance failure."""

    model_config = _CONFIG
    current_schema_version = DiagnosticArtifactSchema.DIAGNOSTIC_ACCEPTANCE
    current_artifact_kind = DiagnosticAcceptanceArtifactKind.EXPOSURE

    schema_version: Literal[DiagnosticArtifactSchema.DIAGNOSTIC_ACCEPTANCE] = (
        DiagnosticArtifactSchema.DIAGNOSTIC_ACCEPTANCE
    )
    artifact_kind: Literal[DiagnosticAcceptanceArtifactKind.EXPOSURE] = (
        DiagnosticAcceptanceArtifactKind.EXPOSURE
    )
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    run_id: SHA256Digest
    held_out_corpus_sha256: SHA256Digest
    source_revision: NonEmptyString
    outcome: Literal["precondition_failed"] = "precondition_failed"
    evaluated_case_ids_before_failure: tuple[NonEmptyString, ...] = ()
    released_case_id: NonEmptyString
    released_workload_kind: WorkloadKind
    released_reason_codes: tuple[NonEmptyString, ...] = Field(min_length=1)
    released_metric_fields: tuple[NonEmptyString, ...] = ()
    acceptance_result_written: Literal[False] = False
    created_at: NonEmptyString

    @model_validator(mode="after")
    def _release_is_pre_verdict(self) -> DiagnosticAcceptanceExposureReceipt:
        if (
            self.purpose is DiagnosticEvidencePurpose.PRODUCTION
            and re.fullmatch(r"[0-9a-f]{40}", self.source_revision) is None
        ):
            raise ValueError("production exposure requires an exact revision")
        if self.released_metric_fields:
            raise ValueError(
                "precondition exposure cannot release acceptance metrics"
            )
        if self.released_case_id in self.evaluated_case_ids_before_failure:
            raise ValueError("failing case cannot also precede the failure")
        if len(set(self.evaluated_case_ids_before_failure)) != len(
            self.evaluated_case_ids_before_failure
        ):
            raise ValueError("exposure receipt repeats evaluated case IDs")
        return self


class DiagnosticHeldOutCorpusFragment(CurrentSchemaModel):
    """Fresh cases collected only for impact-classified held-out strata."""

    model_config = _CONFIG
    current_schema_version = DiagnosticArtifactSchema.DIAGNOSTIC_CASE_REUSE
    current_artifact_kind = DiagnosticCaseReuseArtifactKind.HELD_OUT_FRAGMENT

    schema_version: Literal[DiagnosticArtifactSchema.DIAGNOSTIC_CASE_REUSE] = (
        DiagnosticArtifactSchema.DIAGNOSTIC_CASE_REUSE
    )
    artifact_kind: Literal[
        DiagnosticCaseReuseArtifactKind.HELD_OUT_FRAGMENT
    ] = DiagnosticCaseReuseArtifactKind.HELD_OUT_FRAGMENT
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    role: Literal["held_out"] = "held_out"
    design_sha256: SHA256Digest
    configuration_frozen_before_collection: Literal[True] = True
    cases: list[DiagnosticValidationCase] = Field(min_length=1, max_length=220)

    @model_validator(mode="after")
    def _cases_are_unique(self) -> DiagnosticHeldOutCorpusFragment:
        for field in ("case_id", "pair_id"):
            values = [getattr(case, field) for case in self.cases]
            if len(values) != len(set(values)):
                raise ValueError(f"held-out fragment repeats {field}")
        if any(
            case.workload_kind is WorkloadKind.UNSUPPORTED
            for case in self.cases
        ):
            raise ValueError("held-out fragment contains unsupported family")
        return self


class SourceChangeImpact(StrictArtifactModel):
    """Reviewed effect of one changed path on raw collection evidence."""

    model_config = _CONFIG

    path: NonEmptyString
    previous_path: NonEmptyString | None = None
    change: Literal["added", "modified", "deleted", "renamed"]
    affects_raw_collection: bool
    affects_derived_diagnostics: bool
    affected_families: tuple[WorkloadKind, ...] = ()
    rationale: NonEmptyString

    @model_validator(mode="after")
    def _rename_fields_match_change(self) -> SourceChangeImpact:
        if (self.change == "renamed") != (self.previous_path is not None):
            raise ValueError("only renamed paths carry previous_path")
        has_effect = (
            self.affects_raw_collection or self.affects_derived_diagnostics
        )
        if has_effect != bool(self.affected_families):
            raise ValueError(
                "behavior-changing paths require exact affected families"
            )
        if len(set(self.affected_families)) != len(self.affected_families):
            raise ValueError("source change repeats affected families")
        if WorkloadKind.UNSUPPORTED in self.affected_families:
            raise ValueError("unsupported family cannot be impact-classified")
        return self


class DiagnosticCaseReuseDecision(StrictArtifactModel):
    """One content-bound decision in the final 220-case corpus."""

    model_config = _CONFIG

    case_id: NonEmptyString
    workload_kind: WorkloadKind
    disposition: ReuseDisposition
    pair_id: SHA256Digest
    evidence_manifest_sha256: SHA256Digest
    solar_manifest_sha256: SHA256Digest
    evidence_identity_sha256: SHA256Digest
    reason: NonEmptyString


class DiagnosticCaseReuseManifest(CurrentSchemaModel):
    """Reviewed impact proof for composing old and freshly collected cases."""

    model_config = _CONFIG
    current_schema_version = DiagnosticArtifactSchema.DIAGNOSTIC_CASE_REUSE
    current_artifact_kind = DiagnosticCaseReuseArtifactKind.MANIFEST

    schema_version: Literal[DiagnosticArtifactSchema.DIAGNOSTIC_CASE_REUSE] = (
        DiagnosticArtifactSchema.DIAGNOSTIC_CASE_REUSE
    )
    artifact_kind: Literal[DiagnosticCaseReuseArtifactKind.MANIFEST] = (
        DiagnosticCaseReuseArtifactKind.MANIFEST
    )
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    source_corpus_sha256: SHA256Digest
    replacement_fragment_sha256: SHA256Digest
    replacement_design_sha256: SHA256Digest
    exposure_receipt_sha256: SHA256Digest
    final_corpus_sha256: SHA256Digest
    base_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_changes: tuple[SourceChangeImpact, ...]
    tainted_families: tuple[WorkloadKind, ...] = Field(min_length=1)
    policy_frozen_before_composition: Literal[True] = True
    decisions: tuple[DiagnosticCaseReuseDecision, ...] = Field(
        min_length=_TOTAL_CASES,
        max_length=_TOTAL_CASES,
    )
    created_at: NonEmptyString

    @model_validator(mode="after")
    def _decisions_are_complete(self) -> DiagnosticCaseReuseManifest:
        _require_unique_decisions(self.decisions)
        _require_exact_family_counts(self.decisions)
        tainted = set(self.tainted_families)
        if len(tainted) != len(self.tainted_families):
            raise ValueError("reuse manifest repeats tainted families")
        if WorkloadKind.UNSUPPORTED in tainted:
            raise ValueError("unsupported family cannot be impact-classified")
        paths = tuple(change.path for change in self.source_changes)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("source change paths must be sorted and unique")
        if not self.source_changes and (
            self.base_source_revision != self.target_source_revision
        ):
            raise ValueError("revision change requires a reviewed source diff")
        for decision in self.decisions:
            expected = (
                ReuseDisposition.REPLACE
                if decision.workload_kind in tainted
                else ReuseDisposition.REUSE
            )
            if decision.disposition is not expected:
                raise ValueError("reuse disposition disagrees with taint scope")
        impacted_families = {
            family
            for change in self.source_changes
            for family in change.affected_families
        }
        if not impacted_families.issubset(tainted):
            raise ValueError("source-diff-affected families must be replaced")
        return self


def _require_unique_decisions(
    decisions: tuple[DiagnosticCaseReuseDecision, ...],
) -> None:
    for field in ("case_id", "pair_id", "evidence_manifest_sha256"):
        values = [getattr(item, field) for item in decisions]
        if len(values) != len(set(values)):
            raise ValueError(f"reuse manifest repeats {field}")


def _require_exact_family_counts(
    decisions: tuple[DiagnosticCaseReuseDecision, ...],
) -> None:
    counts = Counter(item.workload_kind for item in decisions)
    expected = set(_SUPPORTED_FAMILIES)
    if set(counts) != expected or set(counts.values()) != {_CASES_PER_FAMILY}:
        raise ValueError("reuse manifest requires exactly 20 cases per family")


def load_and_verify_case_reuse_bundle(
    final_corpus_path: Path,
) -> DiagnosticCaseReuseManifest | None:
    """Verify an optional sibling reuse bundle against every cited digest."""
    root = final_corpus_path.parent
    manifest_path = root / CASE_REUSE_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    manifest = load_json_file(DiagnosticCaseReuseManifest, manifest_path)
    source = root / SOURCE_CORPUS_NAME
    fragment = root / REPLACEMENT_FRAGMENT_NAME
    exposure = root / EXPOSURE_RECEIPT_NAME
    expected = (
        (source, manifest.source_corpus_sha256),
        (fragment, manifest.replacement_fragment_sha256),
        (exposure, manifest.exposure_receipt_sha256),
        (final_corpus_path, manifest.final_corpus_sha256),
    )
    for path, digest in expected:
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"case reuse bundle identity drift: {path}")
    _verify_bundle_semantics(root, manifest)
    return manifest


def _verify_bundle_semantics(
    root: Path,
    manifest: DiagnosticCaseReuseManifest,
) -> None:
    source = load_json_file(
        DiagnosticValidationCorpus, root / SOURCE_CORPUS_NAME
    )
    fragment = load_json_file(
        DiagnosticHeldOutCorpusFragment, root / REPLACEMENT_FRAGMENT_NAME
    )
    exposure = load_json_file(
        DiagnosticAcceptanceExposureReceipt, root / EXPOSURE_RECEIPT_NAME
    )
    final = load_json_file(DiagnosticValidationCorpus, root / "held_out.json")
    if exposure.held_out_corpus_sha256 != manifest.source_corpus_sha256:
        raise ValueError("exposure receipt does not cite the source corpus")
    if exposure.source_revision != manifest.base_source_revision:
        raise ValueError("reuse diff does not start at exposure revision")
    if fragment.design_sha256 != manifest.replacement_design_sha256:
        raise ValueError("replacement fragment design identity drift")
    _verify_exposure_scope(source, exposure, manifest.tainted_families)
    expected = _compose_cases(source, fragment, set(manifest.tainted_families))
    if final.cases != expected:
        raise ValueError("final corpus differs from reviewed case composition")
    if (
        tuple(
            _decision(case, source, manifest.tainted_families)
            for case in expected
        )
        != manifest.decisions
    ):
        raise ValueError("case reuse decisions differ from final corpus")


def _verify_exposure_scope(
    source: DiagnosticValidationCorpus,
    exposure: DiagnosticAcceptanceExposureReceipt,
    tainted_families: tuple[WorkloadKind, ...],
) -> None:
    """Require the replacement scope to cover the exact evaluated prefix."""
    source_by_id = {case.case_id: case for case in source.cases}
    released = source_by_id.get(exposure.released_case_id)
    if (
        released is None
        or released.workload_kind is not exposure.released_workload_kind
    ):
        raise ValueError("exposure case identity differs from source corpus")
    released_index = source.cases.index(released)
    expected_prefix = tuple(
        case.case_id for case in source.cases[:released_index]
    )
    if exposure.evaluated_case_ids_before_failure != expected_prefix:
        raise ValueError("exposure evaluated prefix differs from source corpus")
    exposed_families = {
        source_by_id[case_id].workload_kind
        for case_id in (*expected_prefix, exposure.released_case_id)
    }
    if not exposed_families.issubset(set(tainted_families)):
        raise ValueError("reuse policy omits an exposed workload family")


def _compose_cases(
    source: DiagnosticValidationCorpus,
    fragment: DiagnosticHeldOutCorpusFragment,
    tainted: set[WorkloadKind],
) -> list[DiagnosticValidationCase]:
    if source.role != "held_out" or len(source.cases) != _TOTAL_CASES:
        raise ValueError(
            "reuse source must be an exact 220-case held-out corpus"
        )
    replacements = {case.case_id: case for case in fragment.cases}
    expected_ids = {
        case.case_id for case in source.cases if case.workload_kind in tainted
    }
    if set(replacements) != expected_ids:
        raise ValueError(
            "replacement fragment does not exactly cover taint scope"
        )
    source_pairs = {case.pair_id for case in source.cases}
    if any(case.pair_id in source_pairs for case in fragment.cases):
        raise ValueError("replacement fragment reuses an exposed pair")
    return [replacements.get(case.case_id, case) for case in source.cases]


def _decision(
    case: DiagnosticValidationCase,
    source: DiagnosticValidationCorpus,
    tainted_families: tuple[WorkloadKind, ...],
) -> DiagnosticCaseReuseDecision:
    tainted = set(tainted_families)
    disposition = (
        ReuseDisposition.REPLACE
        if case.workload_kind in tainted
        else ReuseDisposition.REUSE
    )
    source_by_id = {item.case_id: item for item in source.cases}
    original = source_by_id[case.case_id]
    evidence_identity = (
        original.evidence_manifest.sha256
        if disposition is ReuseDisposition.REUSE
        else case.evidence_manifest.sha256
    )
    return DiagnosticCaseReuseDecision(
        case_id=case.case_id,
        workload_kind=case.workload_kind,
        disposition=disposition,
        pair_id=case.pair_id,
        evidence_manifest_sha256=case.evidence_manifest.sha256,
        solar_manifest_sha256=case.solar_manifest.sha256,
        evidence_identity_sha256=stable_json_checksum(
            {
                "evidence_manifest_sha256": evidence_identity,
                "pair_id": case.pair_id,
                "solar_manifest_sha256": case.solar_manifest.sha256,
            }
        ),
        reason=(
            "unexposed case with unchanged raw collection dependencies"
            if disposition is ReuseDisposition.REUSE
            else "fresh replacement for exposure-tainted family"
        ),
    )


def compose_case_reuse_corpus(
    source: DiagnosticValidationCorpus,
    fragment: DiagnosticHeldOutCorpusFragment,
    tainted_families: tuple[WorkloadKind, ...],
) -> DiagnosticValidationCorpus:
    """Compose one exact corpus while replacing every tainted-family slot."""
    if source.purpose is not fragment.purpose:
        raise ValueError("reuse inputs cross evidence purpose domains")
    cases = _compose_cases(source, fragment, set(tainted_families))
    return DiagnosticValidationCorpus(
        purpose=source.purpose,
        role="held_out",
        cases=cases,
    )


def build_case_reuse_decisions(
    final: DiagnosticValidationCorpus,
    source: DiagnosticValidationCorpus,
    tainted_families: tuple[WorkloadKind, ...],
) -> tuple[DiagnosticCaseReuseDecision, ...]:
    """Derive the only admitted decision ledger for a composed corpus."""
    return tuple(
        _decision(case, source, tainted_families) for case in final.cases
    )


def persist_acceptance_exposure(
    receipt: DiagnosticAcceptanceExposureReceipt,
    receipt_path: Path,
    store_root: Path,
) -> str:
    """Import one exposure into CAS and its immutable process registry."""
    store = BlobStore(store_root)
    digest = store.put_file(receipt_path)
    registry_path = (
        acceptance_exposures_dir(store_root) / receipt.run_id / f"{digest}.json"
    )
    if registry_path.is_file():
        existing = load_json_file(
            DiagnosticAcceptanceExposureReceipt, registry_path
        )
        if existing != receipt:
            raise ValueError(f"immutable exposure differs: {registry_path}")
    else:
        atomic_write_json_value(registry_path, receipt.model_dump(mode="json"))
    store.put_file(registry_path, expected_sha256=digest)
    return digest


__all__ = [
    "CASE_REUSE_MANIFEST_NAME",
    "EXPOSURE_RECEIPT_NAME",
    "REPLACEMENT_FRAGMENT_NAME",
    "SOURCE_CORPUS_NAME",
    "AcceptancePreconditionError",
    "DiagnosticAcceptanceExposureReceipt",
    "DiagnosticCaseReuseDecision",
    "DiagnosticCaseReuseManifest",
    "DiagnosticHeldOutCorpusFragment",
    "ReuseDisposition",
    "SourceChangeImpact",
    "build_case_reuse_decisions",
    "compose_case_reuse_corpus",
    "load_and_verify_case_reuse_bundle",
    "persist_acceptance_exposure",
]
