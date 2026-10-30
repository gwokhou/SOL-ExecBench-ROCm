"""Profile descriptions are provenance; parameters and audit pins are authority."""

from copy import deepcopy
from pathlib import Path

import pytest

from solar.artifacts import stable_json_checksum
from solar.pipeline.reuse_projection import configure_projection
from solar.reuse_contracts import ProfileScenario, ProjectionRequest
from solar.rocm import architecture
from solar.rocm.architecture import ArchitectureProfile


@pytest.fixture
def profile():
    return ArchitectureProfile.load("RX_9060_XT").to_dict()


def test_description_edit_revalidates_existing_audit(profile, tmp_path):
    requested = deepcopy(profile)
    requested["source"] = "Clarified description"
    requested["resource_limit_sources"]["valu"] = "Clarified citation"
    requested["memory_hierarchy"][0]["source"] = "Clarified memory provenance"
    result = configure_projection(
        ProjectionRequest(
            architecture=requested,
            output_dir=tmp_path / "projection",
            require_verified_audit=True,
        ),
        profile,
    )
    assert result.scenario is ProfileScenario.IDENTICAL
    assert result.audit_verified
    assert result.profile.audit_evidence == profile["audit_evidence"]
    assert result.identity["architecture"] == requested
    assert result.requested_sha256 == stable_json_checksum(requested)
    assert result.requested_sha256 != stable_json_checksum(profile)


@pytest.mark.parametrize(
    ("field", "path", "value"),
    [
        ("memory_bandwidth_bytes_per_second", (), 1e12),
        ("last_level_cache_bytes", (), 4096),
        ("compute_units", (), 64),
        ("clock_hz", (), 2e9),
        ("gfx_target", (), "gfx1201"),
        ("profile_revision", (), "another-revision"),
        ("peak_ops_per_second", ("fp32",), 1e12),
        ("resource_limits", ("valu", "fp32"), 1e12),
        ("memory_hierarchy", (0, "capacity_bytes"), 4096),
        ("memory_hierarchy", (1, "scope"), "device"),
        ("audit_evidence", ("sha256",), "0" * 64),
        ("audit_evidence", ("path",), "different.json"),
        ("audit_evidence", ("required_clocks_locked",), False),
        ("precision_support", ("fp32", "evidence"), "different.json"),
        ("calibration_exempt_modes", ("valu", "generic"), "Different basis"),
    ],
    ids=[
        "bandwidth",
        "cache",
        "units",
        "clock",
        "target",
        "revision",
        "peak",
        "resource-rate",
        "level-capacity",
        "level-scope",
        "audit-digest",
        "audit-path",
        "audit-policy",
        "support",
        "exemption",
    ],
)
def test_parameter_or_qualification_edits_cannot_inherit_audit(
    profile, tmp_path, field, path, value
):
    requested = deepcopy(profile)
    target = requested
    for key in (field, *path)[:-1]:
        target = target[key]
    assert target[(field, *path)[-1]] != value
    target[(field, *path)[-1]] = value
    request = ProjectionRequest(
        architecture=requested, output_dir=tmp_path / "p"
    )
    result = configure_projection(request, profile)
    assert result.scenario is ProfileScenario.HYPOTHETICAL
    assert not result.audit_verified
    assert result.profile.audit_evidence == {
        "status": "unavailable",
        "reason_code": "hypothetical_profile",
    }
    with pytest.raises(ValueError, match="hypothetical_profile"):
        configure_projection(
            request.model_copy(update={"require_verified_audit": True}), profile
        )


def test_description_edit_does_not_hide_audit_file_tampering(profile, tmp_path):
    source = (
        Path(architecture.__file__).parents[1]
        / profile["audit_evidence"]["path"]
    )
    evidence = tmp_path / "audit.json"
    evidence.write_bytes(source.read_bytes())
    profile["audit_evidence"]["path"] = str(evidence)
    requested = deepcopy(profile)
    requested["source"] = "Wording changed with intact audit pin"
    request = ProjectionRequest(
        architecture=requested,
        output_dir=tmp_path / "p",
        require_verified_audit=True,
    )
    assert configure_projection(request, profile).audit_verified
    evidence.write_bytes(evidence.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="audit evidence identity mismatch"):
        configure_projection(request, profile)


def test_explicit_hypothesis_still_drops_audit(profile, tmp_path):
    request = ProjectionRequest(
        architecture=profile, output_dir=tmp_path / "p", hypothetical=True
    )
    assert (
        configure_projection(request, profile).scenario
        is ProfileScenario.HYPOTHETICAL
    )
