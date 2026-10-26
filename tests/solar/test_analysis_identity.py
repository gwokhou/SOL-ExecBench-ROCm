"""Analysis reuse identities cover workflow code and bounded source work."""

import pytest

from solar.analysis import identity


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path / "solar"
    for folder in identity.RULE_SOURCE_FOLDERS:
        path = root / folder / "rule.py"
        path.parent.mkdir(parents=True)
        path.write_text("")
    for name in identity.RULE_SOURCE_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("")
    monkeypatch.setattr(
        identity, "__file__", str(root / "analysis" / "identity.py")
    )
    return root


def test_workflow_edits_invalidate_counts_and_downstream_reuse(sources):
    before = identity.analysis_rules_identity()
    (sources / "pipeline" / "rule.py").write_text("changed selection rule")
    assert (
        identity.analysis_rules_identity()[identity.RuleGroup.COUNTS]
        != before[identity.RuleGroup.COUNTS]
    )


def test_source_file_budget_rejects_excess_work(sources, monkeypatch):
    monkeypatch.setattr(identity, "MAX_RULE_FILES", 1)
    with pytest.raises(ValueError, match="file budget"):
        identity.analysis_rules_identity()


def test_zero_byte_budget_accepts_empty_sources_but_rejects_actual_bytes(
    sources, monkeypatch
):
    monkeypatch.setattr(identity, "MAX_RULE_BYTES", 0)
    assert set(identity.analysis_rules_identity()) == set(identity.RuleGroup)
    (sources / "pipeline" / "rule.py").write_text("x")
    with pytest.raises(ValueError, match="byte budget"):
        identity.analysis_rules_identity()
