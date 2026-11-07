"""Complete artifact reuse rejects identity drift without executing a reference."""

from dataclasses import replace
from pathlib import Path
from shutil import copytree

import pytest
import yaml

from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from solar.api import AnalysisRequest, AnalysisResult, analyze
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.bundle import BundleBudget
from solar.artifacts.names import (
    ANALYSIS_FILENAME,
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
)
from solar.ir.access_artifact import ACCESS_FILENAME
from solar.ir.contracts import IRPath
from solar.pipeline.verified_program import load_verified_program
from solar.reuse_contracts import VerifiedProgramRequest


@pytest.fixture(
    scope="module", params=list(IRPath), ids=lambda item: item.value
)
def built(request, tmp_path_factory):
    path = request.param
    root = tmp_path_factory.mktemp("verified-program") / "analysis"
    conversion = get_sample("F01-D").request(path, sha256_file(Path(__file__)))
    result = analyze(
        AnalysisRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=root,
            precision="fp32",
            require_verified_audit=False,
        )
    )
    assert isinstance(result, AnalysisResult), result
    receipt = yaml.safe_load(
        (root / CONVERSION_ATTESTATION_FILENAME).read_text()
    )
    identity = VerifiedProgramRequest(
        directory=root,
        manifest_sha256=sha256_file(root / MANIFEST_FILENAME),
        reference_sha256=conversion.reference_sha256,
        ir_path=path,
        verification=conversion.verification,
        execution_sha256=stable_json_checksum(
            receipt["predicate"]["execution"]
        ),
    )
    return root, identity


@pytest.fixture
def bundle(built, tmp_path):
    source, identity = built
    root = tmp_path / "bundle"
    copytree(source, root)
    return identity.model_copy(update={"directory": root})


def test_complete_bundle_loads_without_extraction_or_reference_execution(
    bundle, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "loading proof must not execute extraction or the reference"
        )

    monkeypatch.setattr(
        "solar.pipeline.stages.extract_request_graph", forbidden
    )
    monkeypatch.setattr("solar.verification.verify._load_module", forbidden)
    monkeypatch.setattr("solar.verification.verify._run_cases", forbidden)
    program = load_verified_program(bundle)
    assert program.prior_analysis.require_mapping("total")["macs"] == 60
    assert program.graph.path.name == bundle.ir_path.graph_filename
    assert program.access.source_storage is not None


@pytest.mark.parametrize(
    "damage", ["graph", "access", "attestation", "missing", "extra", "symlink"]
)
def test_changed_or_incomplete_bundle_is_rejected(bundle, damage):
    root = bundle.directory
    if damage == "graph":
        (root / bundle.ir_path.graph_filename).write_text("layers: {}")
    elif damage == "access":
        (root / ACCESS_FILENAME).write_text("{}")
    elif damage == "attestation":
        (root / CONVERSION_ATTESTATION_FILENAME).write_text(
            "predicate: {status: passed}"
        )
    elif damage == "missing":
        (root / ANALYSIS_FILENAME).unlink()
    elif damage == "extra":
        (root / "partial.tmp").write_text("unfinished")
    else:
        (root / "link").symlink_to(root / ANALYSIS_FILENAME)
    with pytest.raises(ValueError, match=r"SHA-256|inventory|symbolic"):
        load_verified_program(bundle)


@pytest.mark.parametrize(
    "damage", ["source", "path", "execution", "gradient-policy", "seed-policy"]
)
def test_caller_identity_changes_are_not_silently_reused(bundle, damage):
    update = {}
    if damage == "source":
        update["reference_sha256"] = "0" * 64
    elif damage == "path":
        update["ir_path"] = next(
            item for item in IRPath if item is not bundle.ir_path
        )
    elif damage == "execution":
        update["execution_sha256"] = "0" * 64
    elif damage == "gradient-policy":
        update["verification"] = replace(
            bundle.verification, verify_gradients=False
        )
    else:
        update["verification"] = replace(bundle.verification, seeds=(3, 5, 7))
    with pytest.raises(ValueError, match=r"identity|policy"):
        load_verified_program(bundle.model_copy(update=update))


def test_unknown_manifest_schema_precedes_malformed_business_fields(bundle):
    path = bundle.directory / MANIFEST_FILENAME
    path.write_text("schema_version: -1\nartifacts: invalid\n")
    changed = bundle.model_copy(update={"manifest_sha256": sha256_file(path)})
    with pytest.raises(ValueError, match="schema is not current"):
        load_verified_program(changed)


def test_bundle_file_budget_precedes_file_loading(bundle):
    changed = bundle.model_copy(update={"budget": BundleBudget(files=1)})
    with pytest.raises(ValueError, match="file count budget"):
        load_verified_program(changed)


def test_excessive_case_budget_is_rejected_before_manifest_or_list_expansion(
    bundle, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("case budget must precede manifest I/O")

    monkeypatch.setattr(
        "solar.pipeline.verified_program.read_regular_file", forbidden
    )
    changed = bundle.model_copy(
        update={
            "verification": replace(bundle.verification, seeds=range(10**15))
        }
    )
    with pytest.raises(ValueError, match="case budget"):
        load_verified_program(changed)


@pytest.mark.parametrize(
    "damage",
    [
        "verifier",
        "subjects",
        "results",
        "cases",
        "runtime",
        "gradient",
        "backend",
    ],
)
def test_resealed_receipt_still_requires_current_complete_semantic_proof(
    bundle, damage
):
    path = bundle.directory / CONVERSION_ATTESTATION_FILENAME
    receipt = yaml.safe_load(path.read_text())
    if damage == "verifier":
        receipt["predicate"]["verifier"] = "unsupported"
    elif damage == "subjects":
        receipt["subject"].pop()
    elif damage == "results":
        receipt["predicate"]["results"].pop()
    elif damage == "cases":
        receipt["predicate"]["cases"].pop()
    elif damage == "gradient":
        result = receipt["predicate"]["results"][0]
        result["gradient_inputs_verified"] = -1
        result.pop("gradient_error_semantics_verified", None)
    elif damage == "backend":
        receipt["predicate"]["execution"].pop("backend")
    else:
        receipt["predicate"]["execution"]["torch_version"] = "old"
    path.write_text(yaml.safe_dump(receipt))
    manifest_path = bundle.directory / MANIFEST_FILENAME
    manifest = yaml.safe_load(manifest_path.read_text())
    for item in manifest["artifacts"]:
        if item["path"] == CONVERSION_ATTESTATION_FILENAME:
            item["sha256"] = sha256_file(path)
    manifest_path.write_text(yaml.safe_dump(manifest))
    changed = bundle.model_copy(
        update={
            "manifest_sha256": sha256_file(manifest_path),
            "execution_sha256": stable_json_checksum(
                receipt["predicate"]["execution"]
            ),
        }
    )
    with pytest.raises(
        ValueError,
        match=r"schema|subjects|incomplete|case policy|runtime|gradient|execution",
    ):
        load_verified_program(changed)
