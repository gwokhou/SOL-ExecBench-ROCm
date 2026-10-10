# =============================================================================
# Adds ROCm runtime, tooling, and python dependencies. Used for benchmarking.
# =============================================================================

ARG ROCM_DOCKER_IMAGE=rocm/pytorch
ARG ROCM_DOCKER_TAG=rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0
ARG ROCM_DOCKER_DIGEST=sha256:68295d40dcdcb53d9b50828f30017faad8cf5ad9a8c7923c5a815b414b45e3e6
ARG PYTORCH_TORCH_VERSION=2.14.0+rocm10.1.0
ARG PYTORCH_TORCHVISION_VERSION=0.29.0a0+rocm10.1.0
ARG TRITON_ROCM_VERSION=3.8.0+git669b31ac.rocm10.1.0
FROM ${ROCM_DOCKER_IMAGE}:${ROCM_DOCKER_TAG}@${ROCM_DOCKER_DIGEST} AS base

ARG PYTORCH_TORCH_VERSION
ARG PYTORCH_TORCHVISION_VERSION
ARG TRITON_ROCM_VERSION

RUN python3 -c 'import platform, sys; assert sys.platform == "linux" and platform.machine() == "x86_64", "Only Linux x86-64 is supported"; assert sys.version_info[:2] == (3, 14), "The base image must provide Python 3.14"' && \
    ln -s "$(rocm-sdk path --root)" /opt/rocm

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib

# System packages (changes infrequently - cache this layer well)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    curl git vim tmux gdb valgrind git-lfs wget cmake util-linux zstd \
    python3-dev python3-venv \
    build-essential libblas-dev liblapack-dev sudo && \
    hipcc --version && \
    command -v rocminfo && \
    command -v rocprofv3 && \
    command -v amd-smi

ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=sol-execbench

ENV HOME=/home/${HOST_USER} \
    UV_CACHE_DIR=/home/${HOST_USER}/.cache/uv

# Copy uv binary early (changes rarely)
COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /usr/local/bin/uv

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
    SUDOERS_TMP="$(mktemp)" && \
    { \
        echo "# SOL ExecBench ROCm clock commands"; \
        echo "${RUN_USER} ALL=(root) NOPASSWD: ${AMD_SMI} version, \\"; \
        echo "    ${AMD_SMI} set -l STABLE_PEAK, \\"; \
        echo "    ${AMD_SMI} set -l AUTO"; \
    } > "${SUDOERS_TMP}" && \
    visudo -cf "${SUDOERS_TMP}" && \
    install -o root -g root -m 0440 "${SUDOERS_TMP}" \
        /etc/sudoers.d/sol-execbench-amd-smi && \
    visudo -cf /etc/sudoers.d/sol-execbench-amd-smi && \
    rm -f "${SUDOERS_TMP}"

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
    UV_NO_SYNC=1 \
    UV_PROJECT_ENVIRONMENT=/venv \
    ROCPROF_TMPDIR=/tmp

# Copy dependency files (changes less frequently than source code).
COPY --chown=${HOST_UID}:${HOST_GID} pyproject.toml uv.lock README.md .python-version ./

# Inherit the official GPU packages through system site-packages. Export only
# additional locked dependencies into /venv; never shadow or reinstall the
# base image's torch, torchvision, Triton, device wheels, or ROCm SDK.
RUN --mount=type=cache,target=/home/${HOST_USER}/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv venv --python python3.14 --system-site-packages /venv && \
    uv export --locked --no-dev --no-emit-project --no-hashes \
        --output-file /tmp/locked-requirements.txt && \
    sed -E '/^(torch|torchvision|triton|rocm(-[a-z0-9-]+)?|amd-(torch|torchvision)-device-[a-z0-9-]+)(\[.*\])?==/d' \
        /tmp/locked-requirements.txt > /tmp/project-requirements.txt && \
    uv pip install --python /venv/bin/python --no-deps \
        -r /tmp/project-requirements.txt

# Copy source code (changes frequently - do this last)
COPY --chown=${HOST_UID}:${HOST_GID} src/ ./src/
COPY --chown=${HOST_UID}:${HOST_GID} problems/ ./problems/
COPY --chown=${HOST_UID}:${HOST_GID} docker/rocm-targets.json ./docker/
COPY --chown=${HOST_UID}:${HOST_GID} docker/check_gpu_stack.py ./docker/

# Install the project itself (creates the sol-execbench console script).
RUN --mount=type=cache,target=/home/${HOST_USER}/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv pip install --python /venv/bin/python --no-deps .

# Compile the vendored AMD ISA JSON helper while the image has a C++ toolchain.
# XML specifications remain runtime-downloaded and integrity-checked by the tool.
RUN --mount=type=cache,target=/home/${HOST_USER}/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    /venv/bin/python -c "from sol_execbench.tools.amd_isa.helper import ensure_helper; ensure_helper()"

# Fail the build if the inherited framework differs from the declared target.
RUN /venv/bin/python -c 'import importlib.metadata as m, sys; expected = dict(zip(("torch", "torchvision", "triton"), sys.argv[1:], strict=True)); actual = {name: m.version(name) for name in expected}; assert actual == expected, (actual, expected); print(actual)' \
        "${PYTORCH_TORCH_VERSION}" "${PYTORCH_TORCHVISION_VERSION}" \
        "${TRITON_ROCM_VERSION}" && \
    /venv/bin/python docker/check_gpu_stack.py uv.lock && \
    /venv/bin/python -m pip check

# Set runtime environment.
# PYTHONPATH lets the volume-mounted src/ override the installed package at runtime.
ENV PATH="/venv/bin:${PATH}" \
    PYTHONPATH="/sol-execbench/src"

ENTRYPOINT ["/entrypoint.sh"]
