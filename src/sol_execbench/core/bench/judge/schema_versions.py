"""Owned diagnostic artifacts, separate from official evaluation contracts."""

from enum import StrEnum


class JudgeArtifactSchema(StrEnum):
    """Version of the independently serialized CPU development report."""

    DEVELOPMENT = "sol_execbench.judge_development.v1"
