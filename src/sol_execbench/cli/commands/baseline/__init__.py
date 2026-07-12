"""Baseline command package with explicit domain-service modules."""

from ._legacy import (
    _baseline_cli as _baseline_cli,
    baseline_cli,
)
from .build import ReleaseProvenance, build_release_baseline_bundle
from .authority import register_authority_commands
from .compare import register_compare_command
from .export import register_export_command
from .publication import register_publication_commands
from .release import register_release_commands
from .verify import verify_release_baseline_rerun

register_export_command(baseline_cli)
register_compare_command(baseline_cli)
register_authority_commands(baseline_cli)
register_release_commands(baseline_cli)
register_publication_commands(baseline_cli)

__all__ = [
    "ReleaseProvenance",
    "baseline_cli",
    "build_release_baseline_bundle",
    "verify_release_baseline_rerun",
]
