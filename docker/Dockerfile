# =============================================================================
# Adds ROCm runtime, tooling, and python dependencies. Used for benchmarking.
# =============================================================================

ARG ROCM_DOCKER_IMAGE=rocm/dev-ubuntu-24.04
ARG ROCM_DOCKER_TAG=7.1.1-complete
ARG PYTORCH_TORCH_VERSION=2.10.0+rocm7.1
ARG PYTORCH_TORCHVISION_VERSION=0.25.0+rocm7.1
ARG PYTORCH_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm7.1
ARG TRITON_ROCM_VERSION=3.6.0
ARG TRITON_ROCM_INDEX_URL=https://download.pytorch.org/whl/
FROM ${ROCM_DOCKER_IMAGE}:${ROCM_DOCKER_TAG} AS base

ARG PYTORCH_TORCH_VERSION
ARG PYTORCH_TORCHVISION_VERSION
ARG PYTORCH_ROCM_INDEX_URL
ARG TRITON_ROCM_VERSION
ARG TRITON_ROCM_INDEX_URL

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib

# System packages (changes infrequently - cache this layer well)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    curl git vim tmux gdb valgrind git-lfs wget \
    python3 python3-dev \
    build-essential libblas-dev liblapack-dev sudo && \
    hipcc --version && \
    command -v rocminfo && \
    command -v rocprofv3 && \
    command -v amd-smi && \
    command -v rocm-smi

ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=sol-execbench

ENV HOME=/home/${HOST_USER} \
    UV_CACHE_DIR=/home/${HOST_USER}/.cache/uv

# Copy uv binary early (changes rarely)
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

# Create user and setup directories. ROCm base images may already contain UID
# 1000, so reuse existing users/groups instead of assuming the host IDs are free.
RUN if [ "${HOST_UID}" = "0" ]; then \
        RUN_USER=root; \
    else \
        RUN_GROUP="$(getent group "${HOST_GID}" | cut -d: -f1 || true)" && \
        if [ -z "${RUN_GROUP}" ]; then \
            groupadd -g "${HOST_GID}" "${HOST_USER}" && RUN_GROUP="${HOST_USER}"; \
        fi && \
        RUN_USER="$(getent passwd "${HOST_UID}" | cut -d: -f1 || true)" && \
        if [ -z "${RUN_USER}" ]; then \
            useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}" && RUN_USER="${HOST_USER}"; \
        fi; \
    fi && \
    mkdir -p /sol-execbench /venv /home/${HOST_USER}/.cache/uv && \
    chown -R ${HOST_UID}:${HOST_GID} /sol-execbench /venv /home/${HOST_USER} && \
    AMD_SMI="$(command -v amd-smi)" && \
    { \
        echo "# SOL ExecBench ROCm clock commands"; \
        echo "${RUN_USER} ALL=(root) NOPASSWD: ${AMD_SMI} version, \\"; \
        echo "    ${AMD_SMI} set -l STABLE_PEAK, \\"; \
        echo "    ${AMD_SMI} set -l AUTO"; \
    } > /etc/sudoers.d/sol-execbench-amd-smi && \
    chmod 0440 /etc/sudoers.d/sol-execbench-amd-smi

# Copy and set up entrypoint while still root (avoid switching back)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Now switch to user for the rest of the build
WORKDIR /sol-execbench
USER ${HOST_UID}:${HOST_GID}

# Set up UV environment
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/venv

# Copy dependency files (changes less frequently than source code)
COPY --chown=${HOST_UID}:${HOST_GID} pyproject.toml uv.lock README.md ./

# Install base Python dependencies first (this is the slowest part - cache it well)
RUN --mount=type=cache,target=/home/${HOST_USER}/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv sync --frozen --no-install-project --no-dev

# Copy source code (changes frequently - do this last)
COPY --chown=${HOST_UID}:${HOST_GID} src/ ./src/

# Install the project itself (creates the sol-execbench console script).
RUN --mount=type=cache,target=/home/${HOST_USER}/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv sync --frozen --no-editable --no-dev

# The checked-in lock is the default ROCm 7.1 stack. Non-default Docker targets
# install their declared PyTorch ROCm wheels after the frozen sync so each image
# matches its target user-space version without mutating the repository lock.
RUN --mount=type=cache,target=/home/${HOST_USER}/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv pip install --python /venv/bin/python \
        --index-url "${PYTORCH_ROCM_INDEX_URL}" \
        --extra-index-url "${TRITON_ROCM_INDEX_URL}" \
        "torch==${PYTORCH_TORCH_VERSION}" \
        "torchvision==${PYTORCH_TORCHVISION_VERSION}" \
        "triton-rocm==${TRITON_ROCM_VERSION}"

# Set runtime environment.
# PYTHONPATH lets the volume-mounted src/ override the installed package at runtime.
ENV PATH="/venv/bin:${PATH}" \
    PYTHONPATH="/sol-execbench/src"

ENTRYPOINT ["/entrypoint.sh"]
