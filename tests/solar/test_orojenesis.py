from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from solar.analysis import orojenesis


def _write_toolchain(tmp_path: Path, *, archive_sha256: str) -> tuple[Path, str]:
    mapper = tmp_path / "bin" / "timeloop-mapper"
    mapper.parent.mkdir()
    mapper.write_bytes(b"untrusted mapper")
    mapper.chmod(0o755)
    mapper_sha256 = hashlib.sha256(mapper.read_bytes()).hexdigest()
    provenance = {
        "schema_version": orojenesis.OROJENESIS_IDENTITY_SCHEMA_VERSION,
        "source": {
            "repository": orojenesis.OROJENESIS_REPOSITORY,
            "commit": orojenesis.OROJENESIS_COMMIT,
            "tree_git_oid": orojenesis.OROJENESIS_TREE_OID,
            "archive_sha256": archive_sha256,
        },
        "artifact": {"path": "bin/timeloop-mapper", "sha256": mapper_sha256},
        "build": {
            "compiler_wrapper_sha256": (orojenesis.OROJENESIS_COMPILER_WRAPPER_SHA256),
            "builder_image": orojenesis.OROJENESIS_BUILDER_IMAGE,
            "compiler": "test compiler",
        },
    }
    (tmp_path / orojenesis.OROJENESIS_PROVENANCE_FILENAME).write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    return tmp_path, mapper_sha256


def test_self_declared_mapper_digest_is_not_a_trust_anchor(tmp_path):
    home, _ = _write_toolchain(
        tmp_path, archive_sha256=orojenesis.OROJENESIS_SOURCE_ARCHIVE_SHA256
    )

    with pytest.raises(orojenesis.OrojenesisError, match="artifact is not trusted"):
        orojenesis.OrojenesisRunner(home)


def test_provenance_must_match_pinned_source_archive(tmp_path, monkeypatch):
    home, mapper_sha256 = _write_toolchain(tmp_path, archive_sha256="a" * 64)
    monkeypatch.setattr(
        orojenesis,
        "OROJENESIS_TRUSTED_MAPPER_SHA256",
        frozenset({mapper_sha256}),
    )

    with pytest.raises(orojenesis.OrojenesisError, match="source archive mismatch"):
        orojenesis.OrojenesisRunner(home)


def test_runner_requires_configured_home(monkeypatch):
    monkeypatch.delenv("SOLAR_OROJENESIS_HOME", raising=False)
    with pytest.raises(orojenesis.OrojenesisError, match="set --orojenesis-home"):
        orojenesis.OrojenesisRunner()


def test_runner_requires_executable_mapper(tmp_path):
    with pytest.raises(orojenesis.OrojenesisError, match="missing executable"):
        orojenesis.OrojenesisRunner(tmp_path)


def test_valid_provenance_manifest_is_returned_as_identity(tmp_path, monkeypatch):
    home, mapper_sha256 = _write_toolchain(
        tmp_path, archive_sha256=orojenesis.OROJENESIS_SOURCE_ARCHIVE_SHA256
    )
    monkeypatch.setattr(
        orojenesis, "OROJENESIS_TRUSTED_MAPPER_SHA256", frozenset({mapper_sha256})
    )
    runner = orojenesis.OrojenesisRunner(home, timeout_seconds=17)
    assert runner.timeout_seconds == 17
    assert runner.toolchain_identity["verification_mode"] == "provenance_manifest"
    assert len(runner.toolchain_identity["provenance_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(schema_version=2), "unsupported.*schema"),
        (
            lambda item: item["source"].update(repository="wrong"),
            "repository mismatch",
        ),
        (lambda item: item["source"].update(commit="wrong"), "revision mismatch"),
        (
            lambda item: item["source"].update(tree_git_oid="wrong"),
            "source tree mismatch",
        ),
        (
            lambda item: item["artifact"].update(path="../mapper"),
            "artifact path mismatch",
        ),
        (
            lambda item: item["artifact"].update(sha256="not-a-digest"),
            "lacks a binary",
        ),
        (
            lambda item: item["artifact"].update(sha256="0" * 64),
            "binary hash mismatch",
        ),
        (
            lambda item: item["build"].update(compiler_wrapper_sha256="wrong"),
            "compiler-wrapper mismatch",
        ),
        (
            lambda item: item["build"].update(builder_image="wrong"),
            "builder image mismatch",
        ),
        (lambda item: item["build"].update(compiler=""), "lacks build identity"),
    ],
)
def test_provenance_manifest_rejects_identity_drift(
    tmp_path, monkeypatch, mutation, message
):
    home, mapper_sha256 = _write_toolchain(
        tmp_path, archive_sha256=orojenesis.OROJENESIS_SOURCE_ARCHIVE_SHA256
    )
    monkeypatch.setattr(
        orojenesis, "OROJENESIS_TRUSTED_MAPPER_SHA256", frozenset({mapper_sha256})
    )
    path = home / orojenesis.OROJENESIS_PROVENANCE_FILENAME
    provenance = json.loads(path.read_text(encoding="utf-8"))
    mutation(provenance)
    path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(orojenesis.OrojenesisError, match=message):
        orojenesis.OrojenesisRunner(home)


@pytest.mark.parametrize("content", ["not-json", "[]"])
def test_provenance_manifest_must_be_an_object(tmp_path, monkeypatch, content):
    home, mapper_sha256 = _write_toolchain(
        tmp_path, archive_sha256=orojenesis.OROJENESIS_SOURCE_ARCHIVE_SHA256
    )
    monkeypatch.setattr(
        orojenesis, "OROJENESIS_TRUSTED_MAPPER_SHA256", frozenset({mapper_sha256})
    )
    (home / orojenesis.OROJENESIS_PROVENANCE_FILENAME).write_text(
        content, encoding="utf-8"
    )
    expected = "cannot parse" if content == "not-json" else "must be an object"
    with pytest.raises(orojenesis.OrojenesisError, match=expected):
        orojenesis.OrojenesisRunner(home)


def test_git_checkout_identity_fallback(tmp_path, monkeypatch):
    mapper = tmp_path / "bin" / "timeloop-mapper"
    mapper.parent.mkdir()
    mapper.write_bytes(b"mapper")
    mapper.chmod(0o755)
    mapper_sha256 = hashlib.sha256(mapper.read_bytes()).hexdigest()
    archive = b"canonical archive"
    monkeypatch.setattr(
        orojenesis, "OROJENESIS_TRUSTED_MAPPER_SHA256", frozenset({mapper_sha256})
    )
    monkeypatch.setattr(
        orojenesis,
        "OROJENESIS_SOURCE_ARCHIVE_SHA256",
        hashlib.sha256(archive).hexdigest(),
    )

    def fake_run(args, **kwargs):
        del kwargs
        if "archive" in args:
            return SimpleNamespace(stdout=archive)
        if args[-1] == "HEAD":
            return SimpleNamespace(stdout=orojenesis.OROJENESIS_COMMIT + "\n")
        if args[-1] == "HEAD^{tree}":
            return SimpleNamespace(stdout=orojenesis.OROJENESIS_TREE_OID + "\n")
        raise AssertionError(args)

    monkeypatch.setattr(orojenesis.subprocess, "run", fake_run)
    identity = orojenesis.OrojenesisRunner(tmp_path).toolchain_identity
    assert identity["verification_mode"] == "git_checkout"
    assert identity["artifact"]["sha256"] == mapper_sha256
