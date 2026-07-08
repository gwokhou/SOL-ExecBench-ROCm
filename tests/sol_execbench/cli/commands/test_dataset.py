from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli import dataset as cli_dataset
from sol_execbench.cli.main import cli


@dataclass
class _FakeDenominators:
    migrated_problems: int = 2
    discovered_problems: int = 3
    blockers: int = 1


@dataclass
class _FakeManifest:
    denominators: _FakeDenominators

    def to_json(self) -> str:
        return '{"schema_version":"fake.migration_manifest.v1"}\n'


def test_dataset_migrate_sol_writes_manifest_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    manifest = _FakeManifest(denominators=_FakeDenominators())
    calls: list[dict[str, object]] = []

    def fake_migrate_sol_execbench(*args, **kwargs):
        calls.append(
            {
                "source_root": args[0],
                "output_root": args[1],
                "categories": kwargs["categories"],
                "source_revision": kwargs["source_revision"],
            }
        )
        return manifest

    def fake_write_migration_manifest(manifest_arg, target):
        assert manifest_arg is manifest
        Path(target).write_text(manifest.to_json())

    monkeypatch.setattr(
        cli_dataset, "migrate_sol_execbench", fake_migrate_sol_execbench
    )
    monkeypatch.setattr(
        cli_dataset, "write_migration_manifest", fake_write_migration_manifest
    )

    result = CliRunner().invoke(
        cli,
        [
            "dataset",
            "migrate-sol",
            str(source),
            str(output),
            "--category",
            "level1",
            "--category",
            "level2",
            "--source-revision",
            "abc123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "source_root": source,
            "output_root": output,
            "categories": ("level1", "level2"),
            "source_revision": "abc123",
        }
    ]
    assert (output / "migration-manifest.json").read_text() == manifest.to_json()
    assert "Problems:" in result.output
    assert "2/3 migrated" in result.output


def test_dataset_migrate_sol_prints_json_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    explicit_manifest = tmp_path / "manifest.json"
    source.mkdir()
    manifest = _FakeManifest(denominators=_FakeDenominators())

    monkeypatch.setattr(
        cli_dataset,
        "migrate_sol_execbench",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        cli_dataset,
        "write_migration_manifest",
        lambda manifest_arg, target: Path(target).write_text(manifest_arg.to_json()),
    )

    result = CliRunner().invoke(
        cli,
        [
            "dataset",
            "migrate-sol",
            str(source),
            str(output),
            "--manifest",
            str(explicit_manifest),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == manifest.to_json()
    assert explicit_manifest.read_text() == manifest.to_json()


def test_dataset_migrate_flashinfer_writes_manifest_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    manifest = _FakeManifest(denominators=_FakeDenominators())
    calls: list[dict[str, object]] = []

    def fake_migrate_flashinfer_trace(*args, **kwargs):
        calls.append(
            {
                "source_root": args[0],
                "output_root": args[1],
                "source_revision": kwargs["source_revision"],
            }
        )
        return manifest

    monkeypatch.setattr(
        cli_dataset, "migrate_flashinfer_trace", fake_migrate_flashinfer_trace
    )
    monkeypatch.setattr(
        cli_dataset,
        "write_migration_manifest",
        lambda manifest_arg, target: Path(target).write_text(manifest_arg.to_json()),
    )

    result = CliRunner().invoke(
        cli,
        [
            "dataset",
            "migrate-flashinfer",
            str(source),
            str(output),
            "--source-revision",
            "abc123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "source_root": source,
            "output_root": output,
            "source_revision": "abc123",
        }
    ]
    assert (output / "migration-manifest.json").read_text() == manifest.to_json()
    assert "Problems:" in result.output
    assert "1 blocker(s)" in result.output


def test_dataset_migrate_flashinfer_prints_json_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    manifest = _FakeManifest(denominators=_FakeDenominators())

    monkeypatch.setattr(
        cli_dataset,
        "migrate_flashinfer_trace",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        cli_dataset,
        "write_migration_manifest",
        lambda manifest_arg, target: Path(target).write_text(manifest_arg.to_json()),
    )

    result = CliRunner().invoke(
        cli,
        [
            "dataset",
            "migrate-flashinfer",
            str(source),
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == manifest.to_json()
