#!/bin/bash
# Launch the sol-execbench Docker container with the right mounts.
#
# Usage:
#   ./scripts/run_docker.sh [--build] [--target <id>] [--allow-unknown-target] [--preflight-only] [docker-run-args...] [-- command...]
#
# Examples:
#   ./scripts/run_docker.sh                             # interactive shell
#   ./scripts/run_docker.sh --build                     # build image, then shell
#   ./scripts/run_docker.sh --target rocm-7.1.1         # select a declared ROCm Docker Target
#   ./scripts/run_docker.sh -- sol-execbench tests/sol_execbench/samples/rmsnorm --solution tests/sol_execbench/samples/rmsnorm/solution_python.json
#   ./scripts/run_docker.sh --device=/dev/kfd --device=/dev/dri -- bash
#
# Environment variables:
#   IMAGE_NAME          Docker image name    (default: sol-execbench)
#   IMAGE_TAG           Docker image tag     (default: latest)
#   ROCM_DOCKER_IMAGE   Unsafe unknown-target override repository
#   ROCM_DOCKER_TAG     Unsafe unknown-target override tag

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-sol-execbench}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Container-side paths (must match Dockerfile / entrypoint expectations)
CONTAINER_PROJECT="/sol-execbench"
DOCKER_TARGET=""
ALLOW_UNKNOWN_TARGET=false
PREFLIGHT_ONLY=false
DRY_RUN="${SOL_EXECBENCH_RUN_DOCKER_DRY_RUN:-0}"

matrix_json_value() {
    python -c 'import json, sys; data=json.loads(sys.argv[1]); value=data; [value := value[part] for part in sys.argv[2].split(".")]; print(value)' "$1" "$2"
}

resolve_docker_target_json() {
    local cmd=(
        python -m sol_execbench.core.docker_matrix preview
        --manifest "${REPO_ROOT}/docker/rocm-targets.json"
    )
    if [ -n "${DOCKER_TARGET}" ]; then
        cmd+=(--target "${DOCKER_TARGET}")
    fi
    if $ALLOW_UNKNOWN_TARGET; then
        cmd+=(
            --allow-unknown-target
            --override-image-repository "${ROCM_DOCKER_IMAGE:-rocm/dev-ubuntu-24.04}"
            --override-image-tag "${ROCM_DOCKER_TAG:-${DOCKER_TARGET}}"
        )
    fi
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}"
}

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

# Parse script flags, then split remaining args on "--"
BUILD=false
DOCKER_ARGS=()
CMD=()
seen_separator=false
while [ "$#" -gt 0 ]; do
    arg="$1"
    shift
    if ! $seen_separator; then
        case "$arg" in
            --build)
                BUILD=true
                continue
                ;;
            --target)
                if [ "$#" -eq 0 ]; then
                    echo "ERROR: --target requires a Target id." >&2
                    exit 2
                fi
                DOCKER_TARGET="$1"
                shift
                continue
                ;;
            --allow-unknown-target)
                ALLOW_UNKNOWN_TARGET=true
                continue
                ;;
            --preflight-only)
                PREFLIGHT_ONLY=true
                continue
                ;;
            --)
                seen_separator=true
                continue
                ;;
        esac
    fi
    if $seen_separator; then
        CMD+=("$arg")
    else
        DOCKER_ARGS+=("$arg")
    fi
done

TARGET_JSON="$(resolve_docker_target_json)"
ROCM_DOCKER_IMAGE_SELECTED="$(matrix_json_value "${TARGET_JSON}" "build_args.ROCM_DOCKER_IMAGE")"
ROCM_DOCKER_TAG_SELECTED="$(matrix_json_value "${TARGET_JSON}" "build_args.ROCM_DOCKER_TAG")"

if [ "${DRY_RUN}" != "1" ]; then
    require_rocm_host_docker
fi

# Build the image if requested
if $BUILD; then
    echo "+ docker build -t ${IMAGE} --build-arg ROCM_DOCKER_IMAGE=\"${ROCM_DOCKER_IMAGE_SELECTED}\" --build-arg ROCM_DOCKER_TAG=\"${ROCM_DOCKER_TAG_SELECTED}\" -f ${REPO_ROOT}/docker/Dockerfile ${REPO_ROOT}"
    if [ "${DRY_RUN}" != "1" ]; then
        docker build \
            -t "${IMAGE}" \
            --build-arg "ROCM_DOCKER_IMAGE=${ROCM_DOCKER_IMAGE_SELECTED}" \
            --build-arg "ROCM_DOCKER_TAG=${ROCM_DOCKER_TAG_SELECTED}" \
            --build-arg HOST_UID="$(id -u)" \
            --build-arg HOST_GID="$(id -g)" \
            --build-arg HOST_USER="$(whoami)" \
            -f "${REPO_ROOT}/docker/Dockerfile" \
            "${REPO_ROOT}"
    fi
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
if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi
exec "${DOCKER_CMD[@]}"
