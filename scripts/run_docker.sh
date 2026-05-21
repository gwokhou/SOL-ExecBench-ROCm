#!/bin/bash
# Launch the sol-execbench Docker container with the right mounts.
#
# Usage:
#   ./scripts/run_docker.sh [--build] [docker-run-args...] [-- command...]
#
# Examples:
#   ./scripts/run_docker.sh                             # interactive shell
#   ./scripts/run_docker.sh --build                     # build image, then shell
#   ./scripts/run_docker.sh -- sol-execbench tests/sol_execbench/samples/rmsnorm --solution tests/sol_execbench/samples/rmsnorm/solution_python.json
#   ./scripts/run_docker.sh --device=/dev/kfd --device=/dev/dri -- bash
#
# Environment variables:
#   IMAGE_NAME          Docker image name    (default: sol-execbench)
#   IMAGE_TAG           Docker image tag     (default: latest)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-sol-execbench}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Container-side paths (must match Dockerfile / entrypoint expectations)
CONTAINER_PROJECT="/sol-execbench"

require_rocm_host_docker() {
    local context_name
    local docker_host

    context_name="$(docker context show 2>/dev/null || true)"
    docker_host="$(docker context inspect --format '{{ (index .Endpoints "docker").Host }}' 2>/dev/null || true)"

    if [[ "${context_name}" == "desktop-linux" || "${docker_host}" == *"/.docker/desktop/"* ]]; then
        cat >&2 <<EOF
ERROR: Docker context '${context_name}' points to Docker Desktop (${docker_host}).

ROCm device passthrough requires the native Linux Docker daemon so the
container can see /dev/kfd and /dev/dri.

Run:
  docker context use default
  unset DOCKER_HOST DOCKER_CONTEXT

Then retry this script.
EOF
        exit 1
    fi

    if [ ! -e /dev/kfd ]; then
        echo "ERROR: /dev/kfd is missing on the host. ROCm containers cannot access the AMD KFD device." >&2
        echo "Check that the amdgpu/ROCm kernel driver is loaded and that rocminfo works on the host." >&2
        exit 1
    fi

    if [ ! -d /dev/dri ]; then
        echo "ERROR: /dev/dri is missing on the host. ROCm containers cannot access DRM render devices." >&2
        echo "Check that the amdgpu DRM driver is loaded and that render nodes exist under /dev/dri." >&2
        exit 1
    fi
}

# Parse --build flag, then split remaining args on "--"
BUILD=false
DOCKER_ARGS=()
CMD=()
seen_separator=false
for arg in "$@"; do
    if [ "$arg" = "--build" ] && ! $seen_separator; then
        BUILD=true
        continue
    fi
    if [ "$arg" = "--" ]; then
        seen_separator=true
        continue
    fi
    if $seen_separator; then
        CMD+=("$arg")
    else
        DOCKER_ARGS+=("$arg")
    fi
done

require_rocm_host_docker

# Build the image if requested
if $BUILD; then
    echo "+ docker build -t ${IMAGE} -f ${REPO_ROOT}/docker/Dockerfile ${REPO_ROOT}"
    docker build \
        -t "${IMAGE}" \
        --build-arg HOST_UID="$(id -u)" \
        --build-arg HOST_GID="$(id -g)" \
        --build-arg HOST_USER="$(whoami)" \
        -f "${REPO_ROOT}/docker/Dockerfile" \
        "${REPO_ROOT}"
fi

LOCAL_FLASHINFER_TRACE_DIR="${REPO_ROOT}/data/flashinfer-trace"
if [ ! -d "${LOCAL_FLASHINFER_TRACE_DIR}" ]; then
    echo "WARNING: ${LOCAL_FLASHINFER_TRACE_DIR} does not exist"
    echo "       Run ./scripts/download_data.sh to download the flashinfer-trace dataset to run those problems."
fi
FLASHINFER_TRACE_DIR="/sol-execbench/data/flashinfer-trace"

# Default to interactive shell if no command given
if [ ${#CMD[@]} -eq 0 ]; then
    CMD=("bash")
fi

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_ARGS=(-it)
fi

DOCKER_CMD=(
    docker run --rm
    "${TTY_ARGS[@]}"
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --security-opt seccomp=unconfined
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${REPO_ROOT}:${CONTAINER_PROJECT}"
    -e "FLASHINFER_TRACE_DIR=${FLASHINFER_TRACE_DIR}"
    -e "SOL_EXECBENCH_GPU_CLK_MHZ=${SOL_EXECBENCH_GPU_CLK_MHZ:-}"
    -e "SOL_EXECBENCH_DRAM_CLK_MHZ=${SOL_EXECBENCH_DRAM_CLK_MHZ:-}"
    "${DOCKER_ARGS[@]}"
    "${IMAGE}"
    "${CMD[@]}"
)

echo "+ ${DOCKER_CMD[*]}"
exec "${DOCKER_CMD[@]}"
