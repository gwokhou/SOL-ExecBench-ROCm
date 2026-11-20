"""Verify M0-M2 receipts and emit the M3/D0 decision evidence index bundle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file
from solar.contracts import FORMAL_BOUND_KIND

DEFAULT_RUN_ID = "20261120-d0"
PLANNING_ANCHOR_COMMIT = "8641b9dc78d00250ba1cf75d5e9434f9839e5ef4"
MAX_ARTIFACT_EXPECTATIONS = 128
MAX_REPORT_EXPECTATIONS = 64
MAX_FOLLOWUP_ITEMS = 3
MAX_CLAIM_RECORDS = 64
MAX_DEFERRED_ITEMS = 32
MAX_REFERENCED_FILE_BYTES = 64 * 1024 * 1024


class LineDomain(StrEnum):
    """Research line domains combined by the M3/D0 synthesis."""

    BASELINE = "baseline"
    JUDGE = "judge"
    LLM_CORE = "llm_core"


class ClaimVerdict(StrEnum):
    """Strict claim-to-evidence disposition classes for D0."""

    PASSED = "passed"
    CRITERION_NOT_MET = "criterion_not_met"
    BOUNDED_INCOMPLETE = "bounded_incomplete"
    UNPROVED_RETAINED = "unproved_retained"
    INVARIANT_PRESERVED = "invariant_preserved"


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactExpectation:
    """Expected path and optional exact SHA-256 for a referenced artifact."""

    domain: LineDomain
    relative_path: str
    expected_sha256: str | None = None
    role: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class VerifiedArtifact:
    """On-disk verification record emitted into domain index manifests."""

    domain: LineDomain
    relative_path: str
    role: str
    sha256: str
    size_bytes: int
    expected_sha256: str | None
    digest_matched: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimRecord:
    """One claim-to-evidence row in the M3/D0 claims matrix."""

    claim_id: str
    domain: LineDomain
    verdict: ClaimVerdict
    statement: str
    scope_boundary: str
    evidence_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class FollowupTask:
    """One bounded next-iteration work item selected by D0."""

    task_id: str
    execution_order: int
    priority: str
    title: str
    consumer: str
    owning_paths: tuple[str, ...]
    problem_statement: str
    acceptance_criteria: tuple[str, ...]
    exposure_constraint: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferredItem:
    """Explicitly deferred workstream with its non-trigger justification."""

    item_id: str
    title: str
    disposition: str
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class D0Inputs:
    """Bounded input specification for D0 bundle generation."""

    sealed_receipts: tuple[ArtifactExpectation, ...]
    tracked_reports: tuple[ArtifactExpectation, ...]
    claims: tuple[ClaimRecord, ...]
    followups: tuple[FollowupTask, ...]
    deferred_items: tuple[DeferredItem, ...]
    l3d_frontend_candidates_count: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class D0SynthesisResult:
    """Return contract summarizing the emitted D0 evidence index bundle."""

    run_id: str
    output_dir: Path
    all_referenced_artifacts_verified: bool
    sealed_receipts_count: int
    tracked_reports_count: int
    claims_count: int
    selected_count: int
    execution_order: tuple[str, ...]
    emitted_file_digests: dict[str, str]


DEFAULT_SEALED_RECEIPTS: tuple[ArtifactExpectation, ...] = (
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-b1-freeze-20261108/freeze-manifest.json",
        expected_sha256=(
            "1ceb04b24674daf7ed19090044761e3be29eb6ce879cfd679eef79d5d2f8280c"
        ),
        role="B1 original execution freeze manifest (160b9988)",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-b1-freeze-20261108/committed-source-identity.json",
        expected_sha256=(
            "69dacf8610359c87a25811b5aba884177f75b9fdc441b92bb7e3d927f39c0d81"
        ),
        role="B1 source tree identity receipt",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path=(
            "solar-b1-freeze-20261108/mapper-and-resource-identity.json"
        ),
        expected_sha256=(
            "62a3e8b132e6f012337b9c4b59881a56ee848fffa974311a5add33d719b002ec"
        ),
        role="Pinned Orojenesis mapper and resource identity",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-e1-c1-cpu-20261108/execution-manifest.json",
        expected_sha256=(
            "1adf01f7b02d9e6402fcd539379a87cd82a9e2d0fb2823bb9c22e44c68088056"
        ),
        role="E1/C1 CPU acceptance execution manifest",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-b2-20261108/freeze-r6.json",
        expected_sha256=(
            "b43c845bf141da84a9384bcc7bef61a09166de345a64df633c149cc8460aa495"
        ),
        role="B2 revised freeze at 043a6820",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-b2-20261108/representatives-verdict.json",
        expected_sha256=(
            "440143ed8c5e9c346429bcf0d90e28b85e7144a287c33d08025699be71c0136c"
        ),
        role="B2 8/8 representative regression verdict",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-b2-20261108/qualification/static/gate.json",
        expected_sha256=(
            "adc0c16911cab910ad87ae3e50ad1441c0c514fdcedcf7239158208f1524a725"
        ),
        role="AKA static gate (163 items)",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-b2-20261108/qualification/canary/gate.json",
        expected_sha256=(
            "d205a35ed4cf5f65351eb00e42835593580c1efa853db601ba3591a4f27c6b93"
        ),
        role="AKA canary gate (112/112 ready)",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-b2-20261108/qualification/full/gate.json",
        expected_sha256=(
            "329eac6a88553223e37e3d0c60feed5aa144509fb1259b9de58d173c3b201695"
        ),
        role="AKA full gate (163/163 ready)",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-b2-20261108/preflight/final-chain-cpu-replay.json",
        expected_sha256=(
            "f00a789c9187fc84b411994eb3b9750945e7bfafbe9aa85368db9f34a933c2f4"
        ),
        role="B2 independent GPU-free receipt replay verification",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-b2-20261108/final-manifest.json",
        expected_sha256=(
            "0c670dcdc8f3891845704f398c7ad1bd093e13a4f107d785de831c9dbe071389"
        ),
        role="B2 final integrity manifest (1,717 files)",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="solar-c2-c3-20261119-r2/preflight/source-diff-vs-prior.json",
        expected_sha256=(
            "6dc370555298d1c951752b7d98da912f563bb448983d8896ac228718d185a36f"
        ),
        role="RDNA4 WGP/VRAM hardware.py single-file diff vs 622c6ba3",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path=(
            "solar-c2-c3-20261119-r2/preflight/representatives-verdict.json"
        ),
        expected_sha256=(
            "b5348acaff2583f2f4ffe8d2fbe5d0a9612ae8e95279aaa391e8676a0fe5d4a2"
        ),
        role="B2 8/8 representative check at b45e6f9a",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="solar-e2-20261108-r2/freeze-e2.json",
        expected_sha256=(
            "081d27f127cb70eb91d92e354193c906e748bc6dfd74ce9419a448900d1d4062"
        ),
        role="Original E2 freeze manifest (622c6ba3, 2,764 files)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-successor-worker-reap-20261118/"
            "stopped-activity-manifest.json"
        ),
        expected_sha256=(
            "f78675ff01816b3347a79c89004b1ed9af071a3d3546ded820061788aef6e2aa"
        ),
        role="First E2 successor stopped activity manifest (500/1296 PID stop)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="solar-e2-successor-reaping-frozen-20261118/freeze.json",
        expected_sha256=(
            "5077e043805976ed817361650de8c09f5214db8fd1d101ed95e32ad86d427c94"
        ),
        role="PID-reaping E2 successor freeze manifest",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-successor-reaping-frozen-20261118/"
            "successor-confirmation/independent-evidence-audit.json"
        ),
        expected_sha256=(
            "f9a14b602dd0c7cbd65140cd8a9ac91a8f16617566b52796165f9f01308af810"
        ),
        role="Full E2 independent evidence audit (36,477 inputs)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-successor-reaping-frozen-20261118/"
            "successor-confirmation/gpu-closeout.json"
        ),
        expected_sha256=(
            "c78ec61955a4551b7cdb592a549ce26451ce3329dc99b82b8365bba527620791"
        ),
        role="Full E2 GPU handback receipt",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-successor-reaping-frozen-20261118/"
            "successor-confirmation/report/correctness.json"
        ),
        expected_sha256=(
            "543c616808498c2b0af1e4dd0d32ff35f570f9c704f15a8fdcbe80a184656ea6"
        ),
        role="Full E2 correctness report (432 units / 1,296 sessions)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-successor-reaping-frozen-20261118/"
            "successor-confirmation/report/cost.json"
        ),
        expected_sha256=(
            "3330f27e1f16f44664647503a3aa1b99f7a52a4c67355b856497083b91d870fb"
        ),
        role="Full E2 cost report (72 pairs / 144 sessions / 72 replays)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-successor-closeout-20261118/terminal-verification.json"
        ),
        expected_sha256=(
            "5d2b067e14ebf2f41a0c1cdb86b76a669704244b7aeb5afed7e93f8d9a831305"
        ),
        role="E2 successor terminal verification verdict",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-successor-closeout-20261118/"
            "counterexample-and-witness-index.json"
        ),
        expected_sha256=(
            "77705be64d20dd0d859cb9cb78f320775a8fe1905784907604792f435a6165bb"
        ),
        role="E2 counterexample and witness index (15 inconsistent units)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-layernorm-reference-cpu-20261119/"
            "retained-call-comparison.json"
        ),
        expected_sha256=(
            "b90bbc41be4a2a8be39d719e9b11087766c0b981bf927922e94d4e56870690b9"
        ),
        role="LayerNorm centered_fp64_v1 CPU comparison (3,669 retained calls)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="solar-e2-layernorm-gpu-dev-20261119/freeze.json",
        expected_sha256=(
            "142b2502b7c5693da11d16b3b6bef417e28706e8e74f11c25d6b5bad301a9efd"
        ),
        role="LayerNorm dev GPU confirmation freeze",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="solar-e2-layernorm-gpu-dev-20261119/audit.json",
        expected_sha256=(
            "9b1a010a2a31b867f1f3b9de11ebc5056a58ac88c88eae1ffa9c8fd0f63d5e45"
        ),
        role="LayerNorm dev GPU audit (156/156 sessions)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="solar-e2-layernorm-gpu-holdout-20261119/freeze.json",
        expected_sha256=(
            "de52da67655a2cf9e6d091f1df79fc5037cd5ab1d12999de096319511b049a4d"
        ),
        role="LayerNorm h0 holdout GPU confirmation freeze",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-layernorm-gpu-holdout-20261119/"
            "holdout-confirmation/holdout-commitment.json"
        ),
        expected_sha256=(
            "36385b8b851e63b2febaa9051714ff728341bc25a418bfb57e222f9c07fa3ab2"
        ),
        role="LayerNorm h0 pre-materialization commitment",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-layernorm-gpu-holdout-20261119/"
            "holdout-confirmation/online-summary.json"
        ),
        expected_sha256=(
            "47afe8af395183cd277ebe68dd3dab66a6514c502c874ac307a3b249e5ec1a56"
        ),
        role="LayerNorm h0 online execution summary (186/186 sessions)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path=(
            "solar-e2-layernorm-gpu-holdout-20261119/"
            "holdout-confirmation/gpu-closeout.json"
        ),
        expected_sha256=(
            "3e90f02f30805eb6cec1e66b1c3390570ed96962ba961c27747ce83749452372"
        ),
        role="LayerNorm h0 GPU handback receipt",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="solar-e2-layernorm-gpu-holdout-20261119/audit.json",
        expected_sha256=(
            "cdfbd3efb03f3ddda9255aeb8e7762da032bc376664e9f9655309a32b26ffdaf"
        ),
        role="LayerNorm h0 device-free independent audit",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119/freeze.json",
        expected_sha256=(
            "4dce5050341210d8e074db80ccee5acda75236a35f7e81c16e49ff3efbc43d03"
        ),
        role="C2/C3 r1 smoke campaign freeze (b45e6f9a)",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119/llm-core/capacity-evidence.json",
        expected_sha256=(
            "cb47591cdaf8a101bc37897d79b861c1590a6b483a85c60f8ffe19ee78e5d6bb"
        ),
        role="r1 measured RX 9060 XT memory quota evidence (mem-12gib)",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119/llm-core/c2-summary.json",
        expected_sha256=(
            "dbf06a42a505a1592c4b19972e9b758723331c2ec850ba64241b8e230d645293"
        ),
        role="r1 C2 target-view summary (20 core problems / 360 rows)",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119/llm-core/smoke-summary.json",
        expected_sha256=(
            "c43d5b9c0753cda33044ebca6f5bc5aea0fa2835a8bfc131523d23f7fd1dc7d9"
        ),
        role="r1 40-row C3 smoke summary",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119/llm-core/exposure-ledger.json",
        expected_sha256=(
            "ea350f98b9640fd40f1d2566038ab1e36e508dc3a71771a5f1df6d8fa601ad35"
        ),
        role="r1 smoke exposure ledger (20 common UUIDs)",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119/llm-core/cpu-audit.json",
        expected_sha256=(
            "851acbfec7ee1219ceb59a0b5788639712add40b1d20e557c8ef6845ec45cbd3"
        ),
        role="r1 device-free smoke audit",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/freeze.json",
        expected_sha256=(
            "ec6c3cea7e1135e5b54013069db2f0312b407ebd4894d5e5d18bed3b9c991e16"
        ),
        role="C2/C3 r2 clean rebuild freeze (b45e6f9a)",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/llm-core/capacity-evidence.json",
        expected_sha256=(
            "c05a4fd7775c203dc965cbc82f65b0e68cd59b94264f8570e5e3c91eeb9c7e93"
        ),
        role="r2 measured RX 9060 XT memory quota evidence (mem-12gib)",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/llm-core/c2-summary.json",
        expected_sha256=(
            "0ad8eadd9537d15c4e536ffb3e24c65697ab223abf3f328d23287c84a588fd44"
        ),
        role="r2 C2 target-view summary (identical workload_view_digest)",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/llm-core/smoke-summary.json",
        expected_sha256=(
            "746d043448363d7dfbc44e90c2c13f9a777aee524e2efba2056eedd9356af50f"
        ),
        role="r2 preserved 40-row smoke summary",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/llm-core/exposure-ledger.json",
        expected_sha256=(
            "e1cbd9a3e2c61d5a5675326b512d36b16c22826ae48ca10ee819ad9599a25bc7"
        ),
        role="r2 preserved smoke exposure ledger",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/llm-core/cpu-audit.json",
        expected_sha256=(
            "b412b5414d3b323d25b5bc53f4855fb2485f333e4a858122d99b90c41808d524"
        ),
        role="r2 preserved smoke CPU audit",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path=(
            "solar-c2-c3-20261119-r2/llm-core/coverage/target-coverage.json"
        ),
        expected_sha256=(
            "86dd98e5c1b2380afec56b618095df53604e82e5775725e9820a66a6730c917a"
        ),
        role="r2 terminal target-coverage ledger (168/360 executed)",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/host-run/c3-resume-closeout.json",
        expected_sha256=(
            "9ec4b4fe4b94ca112fdf947b458c555e3e00ca3ac8e76ae50945606679bff91d"
        ),
        role="r2 C3 GPU handback closeout receipt",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/llm-core/c3-summary.json",
        expected_sha256=(
            "ba77a820ae99e1e370d16d3e7e21ef92e59fa64b31515fb0b8ca02b810485b67"
        ),
        role="r2 post-C3 gap inventory and 2-path differential summary",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/llm-core/c3-exposure-ledger.json",
        expected_sha256=(
            "e6969c96cfaf74bb7008fae3cfcd9b5db5ed034f0558c8fd2686459eeb548588"
        ),
        role="r2 final exposure ledger (84 exposed / 96 unexposed UUIDs)",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="solar-c2-c3-20261119-r2/llm-core/c3-final-audit.json",
        expected_sha256=(
            "91fc0ebd9974f796581257fcee2e4bafea0244ccf15797c694247bae2a5b26ed"
        ),
        role="r2 device-free final audit and nested-PGID drift verification",
    ),
)

DEFAULT_TRACKED_REPORTS: tuple[ArtifactExpectation, ...] = (
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="docs/internal/solar-m0-baseline-evidence-2026-11-07.md",
        expected_sha256=(
            "84422daf4a6202ed03a7bda3c8d1309e31450a4104c9ab5c386291e258408077"
        ),
        role="M0/B0 read-only baseline inventory",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="docs/internal/solar-m0-judge-preregistration-2026-11-07.md",
        expected_sha256=(
            "4e8bea92afed0c345bad1a335e0833969f394ed152c24048f27500aadde9bc19"
        ),
        role="M0/E0 judge discrimination preregistration",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="docs/internal/solar-m0-llm-core-coverage-spec-2026-11-07.md",
        expected_sha256=(
            "eb1016f9a770afc14d080cd1d3a077ed09e327e6926b955d8a8dadf524405ec5"
        ),
        role="M0/C0 LLM_CORE coverage specification",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="docs/internal/solar-e1-c1-cpu-acceptance-2026-11-08.md",
        expected_sha256=(
            "63191ac46898b43c9e0061b689dc4a95475d473a573134a4a28c0497348b96eb"
        ),
        role="E1/C1 CPU acceptance record",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="docs/internal/solar-b1-execution-freeze-2026-11-08.md",
        expected_sha256=(
            "f557b1077b46d711e11c1d131ec5d29067891acf2b3b2d40bb7dccdeab027d62"
        ),
        role="B1 execution freeze record",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="docs/internal/solar-b1-refreeze-2026-11-08.md",
        expected_sha256=(
            "f69468480695e073a34fe26835270a033ae1cd45820e18dc5c9f284cf9dbd2b8"
        ),
        role="B1 revised freeze after CPU audit fix",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="docs/internal/solar-b2-checkpoint-2026-11-08.md",
        expected_sha256=(
            "63f37c1174f7fc006c9b15eb7cc626e83684e84281a4c0d00994c6b19b6a578b"
        ),
        role="B2 historical checkpoint before erratum",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="docs/internal/solar-b2-representative-erratum-2026-11-08.md",
        expected_sha256=(
            "31027b8ad9fbe9a59e6837307c17b89be22419bb018d7c493745d25a4ba29603"
        ),
        role="B2 approved representative expectation erratum",
    ),
    ArtifactExpectation(
        domain=LineDomain.BASELINE,
        relative_path="docs/internal/solar-b2-acceptance-2026-11-08.md",
        expected_sha256=(
            "da77fb07f373f4c431066b53ade5f58b16fc31ee5a36174e00d32f381550fe47"
        ),
        role="B2 acceptance and AKA qualification refresh",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="docs/internal/solar-e2-execution-2026-11-08.md",
        expected_sha256=(
            "0f2186d0fa820a12269365f2e33d236fa0de0ffc313b4b291c8aaf72687cdae2"
        ),
        role="Original E2 execution ledger (20 GiB quota stop)",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="docs/internal/solar-e2-successor-results-2026-11-19.md",
        expected_sha256=(
            "42669c1ad3253745ec6268f0ddaf257d397e3cc83d2ea72e9fde5852e67916dc"
        ),
        role="E2 PID-reaping successor terminal results",
    ),
    ArtifactExpectation(
        domain=LineDomain.JUDGE,
        relative_path="docs/internal/solar-e2-layernorm-holdout-results-2026-11-19.md",
        expected_sha256=(
            "4789fde20c704bb0e7fd057eec73bec16e27ec3109c7e5c7723c4fcc4cca4406"
        ),
        role="LayerNorm centered_fp64_v1 h0 holdout GPU results",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="docs/internal/solar-c2-c3-smoke-2026-11-19.md",
        expected_sha256=(
            "3b0817dc3064b3c2ecd550bebe40e5fded214ee0e82d4f8760b15abf87cf7fd7"
        ),
        role="C2 calibration and 40-row C3 r1 smoke checkpoint",
    ),
    ArtifactExpectation(
        domain=LineDomain.LLM_CORE,
        relative_path="docs/internal/solar-c2-c3-results-2026-11-20.md",
        expected_sha256=(
            "a0da91f109b805eb88d9a9703154e1ad069540dfca1ad20b398c5809292e8e63"
        ),
        role="C2 complete and bounded 12h C3 terminal report",
    ),
)

DEFAULT_CLAIMS: tuple[ClaimRecord, ...] = (
    ClaimRecord(
        claim_id="B-1",
        domain=LineDomain.BASELINE,
        verdict=ClaimVerdict.PASSED,
        statement=(
            "B2 representative regression (8/8 slots: 6 ready + 2 expected "
            "rejections) and refreshed AKA make_fx qualification chain "
            "(static 163 -> canary 112/112 -> full 163/163 ready) passed "
            "under frozen commit 043a6820 with GPU-free receipt replay."
        ),
        scope_boundary=(
            "Does not transfer performance authority (performance_authority="
            "false), full torchview AKA readiness (42 open gaps), or "
            "automatic LLM_CORE coverage."
        ),
        evidence_paths=(
            "solar-b2-20261108/representatives-verdict.json",
            "solar-b2-20261108/qualification/full/gate.json",
            "solar-b2-20261108/preflight/final-chain-cpu-replay.json",
        ),
    ),
    ClaimRecord(
        claim_id="B-2",
        domain=LineDomain.BASELINE,
        verdict=ClaimVerdict.UNPROVED_RETAINED,
        statement=(
            "Historical AKA external-reads closure stands at 127/128 complete "
            "G4 with l2n52_conv_activation_batchnorm-w4 (~23.6 GB unfold OOM) "
            "explicitly retained as an unproved replay memory gap."
        ),
        scope_boundary=(
            "Retained unproved in backlog; never collapsed into full formal "
            "analysis pass."
        ),
        evidence_paths=(
            "docs/internal/solar-m0-baseline-evidence-2026-11-07.md",
        ),
    ),
    ClaimRecord(
        claim_id="B-3",
        domain=LineDomain.BASELINE,
        verdict=ClaimVerdict.PASSED,
        statement=(
            "Commit b45e6f9a aligned RDNA4 WGP*2 CU resolution and bounded "
            "firmware VRAM carveout in hardware.py as the sole source change "
            "vs 622c6ba3, keeping standard-16gb.yaml byte-identical and "
            "passing B2 8/8 representatives before C2/C3."
        ),
        scope_boundary=(
            "Re-qualified 8/8 representatives for C2/C3; full 163-item AKA "
            "chain remains bound to 043a6820 / 622c6ba3."
        ),
        evidence_paths=(
            "solar-c2-c3-20261119-r2/preflight/source-diff-vs-prior.json",
            "solar-c2-c3-20261119-r2/preflight/representatives-verdict.json",
        ),
    ),
    ClaimRecord(
        claim_id="E-1",
        domain=LineDomain.JUDGE,
        verdict=ClaimVerdict.PASSED,
        statement=(
            "E2 successor executed the entire preregistered denominator "
            "(432/432 units, 1,296/1,296 correctness sessions, 144/144 online "
            "cost sessions, 72/72 offline replays) inside 2h59m59s GPU wall "
            "and 12.37 GiB evidence budget with 36,477 verified inputs."
        ),
        scope_boundary=(
            "Experimental validity passed without dropping any denominator row; "
            "separate from whether the judge met E0 pass criteria."
        ),
        evidence_paths=(
            (
                "solar-e2-successor-reaping-frozen-20261118/successor-confirmation/"
                "independent-evidence-audit.json"
            ),
            "solar-e2-successor-closeout-20261118/terminal-verification.json",
        ),
    ),
    ClaimRecord(
        claim_id="E-2",
        domain=LineDomain.JUDGE,
        verdict=ClaimVerdict.CRITERION_NOT_MET,
        statement=(
            "Current production judge detected 81/81 certified P1 violations "
            "with 0 confirmed numerical false rejections on 558 certified "
            "controls (plus 90 indeterminate LayerNorm controls and 3 timing "
            "REWARD_HACK rejections), yet failed E0 overall criteria because 15 "
            "LayerNorm L1 units failed 3-repeat consistency."
        ),
        scope_boundary=(
            "Historical E2 negative judge conclusion and all three exposed "
            "E2 h0 commitments remain permanently retired."
        ),
        evidence_paths=(
            (
                "solar-e2-successor-reaping-frozen-20261118/successor-confirmation/"
                "report/correctness.json"
            ),
            (
                "solar-e2-successor-closeout-20261118/"
                "counterexample-and-witness-index.json"
            ),
        ),
    ),
    ClaimRecord(
        claim_id="E-3",
        domain=LineDomain.JUDGE,
        verdict=ClaimVerdict.PASSED,
        statement=(
            "Diagnostics vs baseline session wall-clock overhead measured across "
            "66/72 comparable cost pairs: median 1.54x (+4.44s) GEMM, 1.16x "
            "(+0.98s) sum reduction, and 2.96x (+12.50s, range 2.88-18.84x) "
            "LayerNorm; 6 LayerNorm P3 pairs remained non-comparable."
        ),
        scope_boundary=(
            "Reflects cold per-session disk cache diagnostic overhead, not "
            "official candidate timing or speedup."
        ),
        evidence_paths=(
            (
                "solar-e2-successor-reaping-frozen-20261118/successor-confirmation/"
                "report/cost.json"
            ),
        ),
    ),
    ClaimRecord(
        claim_id="E-4",
        domain=LineDomain.JUDGE,
        verdict=ClaimVerdict.PASSED,
        statement=(
            "Opt-in research reference centered_fp64_v1 eliminated LayerNorm "
            "repeat inconsistency on CPU (3,669 retained calls), dev GPU "
            "(156/156 sessions), and independent h0 holdout GPU (186/186 "
            "sessions: 168/168 controls, 24/24 fixed/permuted/independent-draw "
            "groups, 18/18 expected negatives)."
        ),
        scope_boundary=(
            "Verified on three fixed LayerNorm shapes under research "
            "protocol; official contract promotion requires D0 follow-up F1."
        ),
        evidence_paths=(
            (
                "solar-e2-layernorm-reference-cpu-20261119/"
                "retained-call-comparison.json"
            ),
            "solar-e2-layernorm-gpu-dev-20261119/audit.json",
            "solar-e2-layernorm-gpu-holdout-20261119/audit.json",
        ),
    ),
    ClaimRecord(
        claim_id="C-1",
        domain=LineDomain.LLM_CORE,
        verdict=ClaimVerdict.PASSED,
        statement=(
            "C2 passed on RX 9060 XT (mem-12gib, 15.30 GB stable allocatable "
            "/ 17.10 GB visible) and froze identical 20-problem, 180-workload, "
            "360-row target view across r1 and r2 with 36 admission decisions."
        ),
        scope_boundary=(
            "Covers profile=core on rx9060xt/standard-16gb only; non-core "
            "definitions and multi-profile claims remain untested."
        ),
        evidence_paths=(
            "solar-c2-c3-20261119-r2/llm-core/capacity-evidence.json",
            "solar-c2-c3-20261119-r2/llm-core/c2-summary.json",
        ),
    ),
    ClaimRecord(
        claim_id="C-2",
        domain=LineDomain.LLM_CORE,
        verdict=ClaimVerdict.BOUNDED_INCOMPLETE,
        statement=(
            "C3 r2 reached lawful bounded INCOMPLETE at the 12h activity "
            "budget cap (43,200.38 s, 243.92 MiB evidence): 168/360 rows "
            "executed (40 passed, 111 failed, 16 unproved, 1 budget_exhausted, "
            "0 infrastructure_error) and 192 not_reached; clean GPU handback."
        ),
        scope_boundary=(
            "8/20 definitions completed all 9 slots; 12/20 completed smoke "
            "only. Cannot be reported as complete LLM_CORE coverage."
        ),
        evidence_paths=(
            "solar-c2-c3-20261119-r2/llm-core/coverage/target-coverage.json",
            "solar-c2-c3-20261119-r2/host-run/c3-resume-closeout.json",
            "solar-c2-c3-20261119-r2/llm-core/c3-summary.json",
        ),
    ),
    ClaimRecord(
        claim_id="C-3",
        domain=LineDomain.LLM_CORE,
        verdict=ClaimVerdict.PASSED,
        statement=(
            "Smoke does not generalize across scales (0/8 make_fx_aten "
            "complete definitions are 9-slot uniform: local_global, "
            "batched_bias, and batched_matmul regress on non-smoke shapes); "
            "formal:mapper consumed 93.2% of activity wall time (40,272.21 s)."
        ),
        scope_boundary=(
            "Dual-path comparison covers 180 workloads with only 4 common "
            "smoke passes (norm_activation); 96 pairs remain unreached."
        ),
        evidence_paths=(
            "solar-c2-c3-20261119-r2/llm-core/c3-summary.json",
            "docs/internal/solar-c2-c3-results-2026-11-20.md",
        ),
    ),
    ClaimRecord(
        claim_id="C-4",
        domain=LineDomain.LLM_CORE,
        verdict=ClaimVerdict.PASSED,
        statement=(
            "Device-free C3 audit passed across all 168 executed rows "
            "(authenticate_resumed_attempts + 0 non-timeout transient "
            "mismatches) and pinpointed row-167 nested-PGID post-timeout "
            "timeloop-mapper flush drift (+3.189 s) blocking standard resume."
        ),
        scope_boundary=(
            "Standard resume from the r2 coverage directory remains "
            "intentionally blocked until F2 lands."
        ),
        evidence_paths=(
            "solar-c2-c3-20261119-r2/llm-core/c3-final-audit.json",
            "solar-c2-c3-20261119-r2/llm-core/c3-exposure-ledger.json",
        ),
    ),
    ClaimRecord(
        claim_id="D-1",
        domain=LineDomain.LLM_CORE,
        verdict=ClaimVerdict.INVARIANT_PRESERVED,
        statement=(
            "All 12 deduplicated C3 gaps record l3d_frontend_candidate=false, "
            "so L3-d (Affine Loop / LLM / JAX frontends) remains deferred and "
            "official contracts, registries, and default backends stayed "
            "unchanged throughout M0-M3."
        ),
        scope_boundary=(
            "Deferral remains conditional until dual-path evidence isolates "
            "an irreducible common frontend extraction gap."
        ),
        evidence_paths=(
            "solar-c2-c3-20261119-r2/llm-core/c3-summary.json",
            "docs/internal/solar-c2-c3-results-2026-11-20.md",
        ),
    ),
)

DEFAULT_SELECTED_FOLLOWUPS: tuple[FollowupTask, ...] = (
    FollowupTask(
        task_id="F2",
        execution_order=1,
        priority="P0",
        title=(
            "Fix nested-PGID post-timeout cleanup and classify timeloop-mapper "
            "7,200s timeouts explicitly"
        ),
        consumer="C3 bounded resume campaigns & solar_bridge automation",
        owning_paths=(
            "src/sol_execbench/core/solar_bridge/runner.py",
            "src/solar/analysis/orojenesis/runner.py",
            "src/sol_execbench/core/solar_bridge/target_coverage.py",
        ),
        problem_statement=(
            "In C3 r2, row-167 timed out at 192.224 s and killed worker "
            "PGID_1 while grandchild timeloop-mapper in PGID_2 continued "
            "flushing .formal.*/timeloop-mapper.oaves.csv for +3.189 s, "
            "triggering coverage resume artifact drift; meanwhile row-082/083 "
            "7,200 s mapper timeouts surfaced as toolchain_unavailable (FAILED)."
        ),
        acceptance_criteria=(
            (
                "Ensure worker/orojenesis cancellation reaps the entire process "
                "tree before returning so no post-return .formal.* writes occur."
            ),
            (
                "Map Orojenesis TimeoutExpired deterministically to explicit "
                "mapper_timeout / BUDGET_EXHAUSTED rather than toolchain_unavailable."
            ),
            (
                "Pass CPU unit tests simulating nested-PGID child flush and "
                "7,200s mapper timeout followed by clean target-coverage --resume."
            ),
        ),
        exposure_constraint=(
            "Pure CPU fix; preserves immutable r1/r2 artifacts and adds no "
            "exposure."
        ),
    ),
    FollowupTask(
        task_id="F1",
        execution_order=2,
        priority="P1",
        title=(
            "Promote centered_fp64_v1 as canonical LayerNorm evaluation "
            "reference contract"
        ),
        consumer="Official SOL-ExecBench evaluator numerical judging",
        owning_paths=(
            "src/sol_execbench/core/bench/judge/",
            "src/sol_execbench/core/integrity/",
            "problems/AMD_AKA/torch2hip/l1n40_layernorm/",
        ),
        problem_statement=(
            "Successor E2 proved uncentered FP32 LayerNorm reference caused "
            "15 LayerNorm L1 repeat-inconsistency failures and 90 indeterminate "
            "controls; centered_fp64_v1 passed 3,669 CPU calls, 156/156 dev GPU "
            "sessions, and 186/186 h0 holdout GPU sessions."
        ),
        acceptance_criteria=(
            (
                "Propose and atomically bump the contract/schema version adopting "
                "centered_fp64_v1 for LayerNorm without mutating published "
                "gfx1200-official-score-v2."
            ),
            (
                "Pass full CPU numerical/contract/schema-version gates plus fresh "
                "preregistered LayerNorm GPU regression (excluding retired E2 and "
                "36385b8b... h0 commitments)."
            ),
        ),
        exposure_constraint=(
            "Retired E2/LayerNorm h0 commitments may never be reused as blind "
            "confirmation; future confirmation requires fresh instances."
        ),
    ),
    FollowupTask(
        task_id="F3",
        execution_order=3,
        priority="P1",
        title=(
            "Close make_fx_aten LLM_CORE incomplete tile-aware bounds, "
            "93.2% BMM mapper cost, and 3 deterministic formal proof gaps"
        ),
        consumer="SOLAR formal analysis & LLM_CORE core coverage on gfx1200",
        owning_paths=(
            "src/solar/analysis/",
            "src/solar/rocm/",
            "src/solar/verification/",
            "src/sol_execbench/core/solar_bridge/",
        ),
        problem_statement=(
            "Rank-1 gap incomplete_tile_aware_bound blocks 12 definitions / 63 "
            "rows (including cross-scale make_fx_aten regressions on "
            "local_global, batched_matmul, batched_bias), formal:mapper "
            "consumed 40,272.21 s (93.2%), and 3 deterministic make_fx gaps "
            "block sliding_window (arange rule), rope_* (index-read proof), "
            "and attention_sink (effect_boundary)."
        ),
        acceptance_criteria=(
            (
                "Run only after F2 lands; isolate why make_fx_aten drops from "
                f"{FORMAL_BOUND_KIND} on local_global and batched_*."
            ),
            (
                "Bound BMM Orojenesis search cost and close deterministic rules/"
                "proofs for sliding_window, rope_*, and attention_sink with CPU "
                "proof tests + fresh post-fix LLM_CORE smoke/dev validation."
            ),
        ),
        exposure_constraint=(
            "Train/validate strictly on smoke (20 UUIDs) and development "
            "workloads; treat the 32 reached holdout UUIDs as exposed and "
            "preserve the 48 unreached holdout UUIDs unexposed until final "
            "post-fix confirmation."
        ),
    ),
)

DEFAULT_DEFERRED_ITEMS: tuple[DeferredItem, ...] = (
    DeferredItem(
        item_id="DEF-L3D",
        title="L3-d new frontends (Affine Loop IR / LLM / JAX)",
        disposition="DEFERRED",
        reason=(
            "0/12 C3 gap_inventory buckets flagged l3d_frontend_candidate=true."
        ),
    ),
    DeferredItem(
        item_id="DEF-VRAM-OOM",
        title="16 GiB VRAM capacity OOMs & l2n52 unfold OOM",
        disposition="RETAINED_OUT_OF_SCOPE",
        reason=(
            "16 attention capacity-* unproved rows, 4 torchview capacity-high "
            "eager trace OOMs, and AKA l2n52 reflect hardware memory ceilings, "
            "not missing formal semantics."
        ),
    ),
    DeferredItem(
        item_id="DEF-TORCHVIEW",
        title="Secondary torchview_extended_einsum adapter gaps",
        disposition="DEFERRED_SECONDARY",
        reason=(
            "torchview remains comparison path per M0-C0; expand_as / "
            "sliding_window / attention_sink diffs wait until make_fx_aten "
            "mainline blockers close."
        ),
    ),
    DeferredItem(
        item_id="DEF-FUTURE-LINES",
        title=(
            "L3-b stage 2 symbolic Orojenesis, multi-input einsum DP, L5 "
            "gfx942, and start700 re-release"
        ),
        disposition="DEFERRED",
        reason=(
            "Blocked on explicit separate project charter per roadmap; none "
            "scheduled inside post-D0 3-item cap."
        ),
    ),
)


def _artifact_payload(artifact: VerifiedArtifact) -> dict[str, object]:
    return {
        "domain": artifact.domain.value,
        "relative_path": artifact.relative_path,
        "role": artifact.role,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "expected_sha256": artifact.expected_sha256,
        "digest_matched": artifact.digest_matched,
    }


def _claim_payload(claim: ClaimRecord) -> dict[str, object]:
    return {
        "claim_id": claim.claim_id,
        "domain": claim.domain.value,
        "verdict": claim.verdict.value,
        "statement": claim.statement,
        "scope_boundary": claim.scope_boundary,
        "evidence_paths": list(claim.evidence_paths),
    }


def _followup_payload(task: FollowupTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "execution_order": task.execution_order,
        "priority": task.priority,
        "title": task.title,
        "consumer": task.consumer,
        "owning_paths": list(task.owning_paths),
        "problem_statement": task.problem_statement,
        "acceptance_criteria": list(task.acceptance_criteria),
        "exposure_constraint": task.exposure_constraint,
    }


def _deferred_payload(item: DeferredItem) -> dict[str, object]:
    return {
        "item_id": item.item_id,
        "title": item.title,
        "disposition": item.disposition,
        "reason": item.reason,
    }


def _domain_summaries() -> dict[LineDomain, dict[str, object]]:
    return {
        LineDomain.BASELINE: {
            "line": "B",
            "m1_status": "complete",
            "planning_anchor_commit": PLANNING_ANCHOR_COMMIT,
            "b2_commit": "043a6820b263ae6b214ab3816f970dd7f64dee03",
            "c2_c3_commit": "b45e6f9af61bf1f669fd36f217a72f37c5757d61",
            "representatives_passed": "8/8",
            "aka_make_fx_chain": "static=163, canary=112/112, full=163/163",
            "retained_gap": "l2n52_conv_activation_batchnorm-w4 replay OOM",
        },
        LineDomain.JUDGE: {
            "line": "E",
            "e2_status": "complete_e0_not_met",
            "correctness_units_completed": 432,
            "correctness_sessions_completed": 1296,
            "cost_pairs_comparable": "66/72",
            "mandatory_p1_detected": "81/81",
            "certified_numerical_false_rejections": 0,
            "indeterminate_controls": 90,
            "timing_reward_hack_rejections": 3,
            "layernorm_repeat_inconsistent_units": 15,
            "layernorm_research_reference": "centered_fp64_v1",
            "layernorm_holdout_sessions_passed": "186/186",
            "official_contract_changed": False,
        },
        LineDomain.LLM_CORE: {
            "line": "C",
            "c2_status": "complete",
            "calibrated_capacity_class": "mem-12gib",
            "target_view_problems": 20,
            "target_view_workloads": 180,
            "planned_coverage_rows": 360,
            "c3_status": "bounded_incomplete_12h_budget",
            "elapsed_seconds": 43200.38107664397,
            "executed_rows": 168,
            "not_reached_rows": 192,
            "raw_counts": {
                "passed": 40,
                "failed": 111,
                "unproved": 16,
                "budget_exhausted": 1,
                "not_reached": 192,
                "infrastructure_error": 0,
            },
            "mapper_wall_seconds": 40272.21,
            "l3d_frontend_candidates_count": 0,
        },
    }


def validate_d0_inputs(inputs: D0Inputs) -> None:
    """Enforce bounds and selection invariants before verifying disk files."""
    if len(inputs.sealed_receipts) > MAX_ARTIFACT_EXPECTATIONS:
        raise ValueError(
            f"sealed_receipts count {len(inputs.sealed_receipts)} exceeds "
            f"limit {MAX_ARTIFACT_EXPECTATIONS}"
        )
    if len(inputs.tracked_reports) > MAX_REPORT_EXPECTATIONS:
        raise ValueError(
            f"tracked_reports count {len(inputs.tracked_reports)} exceeds "
            f"limit {MAX_REPORT_EXPECTATIONS}"
        )
    if len(inputs.claims) > MAX_CLAIM_RECORDS:
        raise ValueError(
            f"claims count {len(inputs.claims)} exceeds {MAX_CLAIM_RECORDS}"
        )
    if len(inputs.followups) > MAX_FOLLOWUP_ITEMS:
        raise ValueError(
            f"selected followups count {len(inputs.followups)} exceeds "
            f"hard cap {MAX_FOLLOWUP_ITEMS}"
        )
    if len(inputs.deferred_items) > MAX_DEFERRED_ITEMS:
        raise ValueError(
            f"deferred_items count {len(inputs.deferred_items)} exceeds "
            f"limit {MAX_DEFERRED_ITEMS}"
        )
    orders = tuple(task.execution_order for task in inputs.followups)
    if sorted(orders) != list(range(1, len(inputs.followups) + 1)):
        raise ValueError(
            f"followup execution_order must form 1..N, got {orders!r}"
        )
    if inputs.l3d_frontend_candidates_count != 0:
        raise ValueError(
            "L3-d cannot stay deferred with non-zero frontend candidates"
        )


def verify_artifact_expectation(
    base_dir: Path,
    expectation: ArtifactExpectation,
    *,
    max_bytes: int = MAX_REFERENCED_FILE_BYTES,
) -> VerifiedArtifact:
    """Verify file existence, bounded size, and exact SHA-256 digest."""
    rel = Path(expectation.relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(
            f"Artifact path must remain relative without '..': {rel}"
        )
    candidate = base_dir / rel
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing referenced artifact: {candidate}")
    size_bytes = candidate.stat().st_size
    if size_bytes > max_bytes:
        raise ValueError(
            f"Artifact {candidate} size {size_bytes} exceeds cap {max_bytes}"
        )
    actual_sha256 = sha256_file(candidate)
    if (
        expectation.expected_sha256 is not None
        and actual_sha256 != expectation.expected_sha256
    ):
        raise ValueError(
            f"SHA-256 mismatch for {expectation.relative_path}: "
            f"expected {expectation.expected_sha256}, got {actual_sha256}"
        )
    return VerifiedArtifact(
        domain=expectation.domain,
        relative_path=expectation.relative_path,
        role=expectation.role,
        sha256=actual_sha256,
        size_bytes=size_bytes,
        expected_sha256=expectation.expected_sha256,
        digest_matched=True,
    )


def verify_artifact_batch(
    base_dir: Path,
    expectations: Sequence[ArtifactExpectation],
    *,
    max_count: int = MAX_ARTIFACT_EXPECTATIONS,
    max_bytes: int = MAX_REFERENCED_FILE_BYTES,
) -> tuple[VerifiedArtifact, ...]:
    """Verify a bounded batch of artifact expectations in deterministic order."""
    if len(expectations) > max_count:
        raise ValueError(
            f"Artifact expectation batch size {len(expectations)} exceeds "
            f"cap {max_count}"
        )
    return tuple(
        verify_artifact_expectation(base_dir, item, max_bytes=max_bytes)
        for item in expectations
    )


def default_d0_inputs() -> D0Inputs:
    """Return the canonical M0-M2 receipts, reports, claims, and followups."""
    return D0Inputs(
        sealed_receipts=DEFAULT_SEALED_RECEIPTS,
        tracked_reports=DEFAULT_TRACKED_REPORTS,
        claims=DEFAULT_CLAIMS,
        followups=DEFAULT_SELECTED_FOLLOWUPS,
        deferred_items=DEFAULT_DEFERRED_ITEMS,
        l3d_frontend_candidates_count=0,
    )


def _write_indexed_json(
    output_dir: Path,
    rel_path: str,
    payload: dict[str, object],
    emitted_digests: dict[str, str],
) -> None:
    destination = output_dir / rel_path
    atomic_write_json_value(destination, payload, sort_keys=True)
    emitted_digests[rel_path] = sha256_file(destination)


def _emit_domain_indices(
    *,
    output_dir: Path,
    run_id: str,
    verified_receipts: Sequence[VerifiedArtifact],
    verified_reports: Sequence[VerifiedArtifact],
    emitted_digests: dict[str, str],
) -> None:
    domain_summaries = _domain_summaries()
    domain_dirs = {
        LineDomain.BASELINE: "baseline/index.json",
        LineDomain.JUDGE: "judge/index.json",
        LineDomain.LLM_CORE: "llm-core/index.json",
    }
    for domain, rel_out in domain_dirs.items():
        receipts_for_domain = [
            _artifact_payload(item)
            for item in verified_receipts
            if item.domain is domain
        ]
        reports_for_domain = [
            _artifact_payload(item)
            for item in verified_reports
            if item.domain is domain
        ]
        _write_indexed_json(
            output_dir,
            rel_out,
            {
                "bundle_kind": "d0_domain_index",
                "run_id": run_id,
                "domain": domain.value,
                "summary": domain_summaries[domain],
                "sealed_receipts": receipts_for_domain,
                "tracked_reports": reports_for_domain,
            },
            emitted_digests,
        )


def _emit_decision_files(
    *,
    output_dir: Path,
    run_id: str,
    inputs: D0Inputs,
    sealed_count: int,
    reports_count: int,
    emitted_digests: dict[str, str],
) -> tuple[str, ...]:
    ordered_followups = sorted(
        inputs.followups,
        key=lambda item: item.execution_order,
    )
    execution_order = tuple(task.task_id for task in ordered_followups)
    _write_indexed_json(
        output_dir,
        "decision/claims-matrix.json",
        {
            "bundle_kind": "d0_claims_matrix",
            "run_id": run_id,
            "claims_count": len(inputs.claims),
            "claims": [_claim_payload(claim) for claim in inputs.claims],
        },
        emitted_digests,
    )
    _write_indexed_json(
        output_dir,
        "decision/selected-followups.json",
        {
            "bundle_kind": "d0_selected_followups",
            "run_id": run_id,
            "hard_cap": MAX_FOLLOWUP_ITEMS,
            "selected_count": len(ordered_followups),
            "execution_order": list(execution_order),
            "l3d_frontend_candidates_count": (
                inputs.l3d_frontend_candidates_count
            ),
            "selected_followups": [
                _followup_payload(task) for task in ordered_followups
            ],
            "deferred_items": [
                _deferred_payload(item) for item in inputs.deferred_items
            ],
        },
        emitted_digests,
    )
    _write_indexed_json(
        output_dir,
        "decision/d0-manifest.json",
        {
            "bundle_kind": "d0_decision_manifest",
            "run_id": run_id,
            "all_referenced_artifacts_verified": True,
            "sealed_receipts_count": sealed_count,
            "tracked_reports_count": reports_count,
            "claims_count": len(inputs.claims),
            "selected_count": len(ordered_followups),
            "execution_order": list(execution_order),
            "domain_files": dict(sorted(emitted_digests.items())),
        },
        emitted_digests,
    )
    return execution_order


def synthesize_d0_bundle(
    *,
    repo_root: Path,
    output_dir: Path,
    run_id: str = DEFAULT_RUN_ID,
    inputs: D0Inputs | None = None,
) -> D0SynthesisResult:
    """Verify M0-M2 artifacts and atomically emit the 4-directory D0 bundle."""
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    resolved_inputs = default_d0_inputs() if inputs is None else inputs
    validate_d0_inputs(resolved_inputs)

    verified_receipts = verify_artifact_batch(
        repo_root / "data" / "outputs",
        resolved_inputs.sealed_receipts,
        max_count=MAX_ARTIFACT_EXPECTATIONS,
    )
    verified_reports = verify_artifact_batch(
        repo_root,
        resolved_inputs.tracked_reports,
        max_count=MAX_REPORT_EXPECTATIONS,
    )

    emitted_digests: dict[str, str] = {}
    _emit_domain_indices(
        output_dir=output_dir,
        run_id=run_id,
        verified_receipts=verified_receipts,
        verified_reports=verified_reports,
        emitted_digests=emitted_digests,
    )
    execution_order = _emit_decision_files(
        output_dir=output_dir,
        run_id=run_id,
        inputs=resolved_inputs,
        sealed_count=len(verified_receipts),
        reports_count=len(verified_reports),
        emitted_digests=emitted_digests,
    )
    return D0SynthesisResult(
        run_id=run_id,
        output_dir=output_dir,
        all_referenced_artifacts_verified=True,
        sealed_receipts_count=len(verified_receipts),
        tracked_reports_count=len(verified_reports),
        claims_count=len(resolved_inputs.claims),
        selected_count=len(execution_order),
        execution_order=execution_order,
        emitted_file_digests=dict(sorted(emitted_digests.items())),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint to generate and verify the M3/D0 evidence index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = (
        repo_root / "data" / "outputs" / "solar-next-iteration" / args.run_id
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    result = synthesize_d0_bundle(
        repo_root=repo_root,
        output_dir=output_dir,
        run_id=args.run_id,
    )
    for rel_path, digest in result.emitted_file_digests.items():
        print(f"{rel_path}: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
