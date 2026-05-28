from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = REPO_ROOT / "docker" / "Dockerfile"


def _dockerfile_lines() -> list[str]:
    return DOCKERFILE_PATH.read_text().splitlines()


def test_dockerfile_declares_rocm_base_args_before_first_from() -> None:
    lines = _dockerfile_lines()
    first_from_index = next(
        index for index, line in enumerate(lines) if line.startswith("FROM ")
    )
    pre_from_lines = lines[:first_from_index]

    assert "ARG ROCM_DOCKER_IMAGE=rocm/dev-ubuntu-24.04" in pre_from_lines
    assert "ARG ROCM_DOCKER_TAG=7.1.1-complete" in pre_from_lines
    assert lines[first_from_index] == (
        "FROM ${ROCM_DOCKER_IMAGE}:${ROCM_DOCKER_TAG} AS base"
    )


def test_dockerfile_keeps_existing_runtime_setup() -> None:
    dockerfile = DOCKERFILE_PATH.read_text()

    assert "ENV ROCM_PATH=/opt/rocm" in dockerfile
    assert "HIP_PLATFORM=amd" in dockerfile
    assert "ARG HOST_UID=1000" in dockerfile
    assert "ARG HOST_GID=1000" in dockerfile
    assert "ARG HOST_USER=sol-execbench" in dockerfile
    assert "COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv" in dockerfile
    assert "COPY docker/entrypoint.sh /entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["/entrypoint.sh"]' in dockerfile
