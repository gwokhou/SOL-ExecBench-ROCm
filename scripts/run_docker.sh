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
#   SOL_EXECBENCH_DEPENDENCY_*  Dependency preflight observation overrides for tests/debugging

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

bool_text() {
    if [ "$1" = "1" ] || [ "$1" = "true" ]; then
        echo "true"
    else
        echo "false"
    fi
}

dev_dri_has_accessible_node() {
    [ -x /dev/dri ] || return 1
    find /dev/dri -maxdepth 1 -type c \( -name 'renderD*' -o -name 'card*' \) \
        -readable -writable -print -quit 2>/dev/null | grep -q .
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

preflight_override_present() {
    [ -n "${SOL_EXECBENCH_DOCKER_CONTEXT:-}" ] ||
        [ -n "${SOL_EXECBENCH_DOCKER_HOST:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEV_KFD_PRESENT:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEV_KFD_ACCESSIBLE:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEV_DRI_PRESENT:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEV_DRI_ACCESSIBLE:-}" ] ||
        [ -n "${SOL_EXECBENCH_GPU_ACCESSIBLE:-}" ]
}

dependency_preflight_override_present() {
    [ -n "${SOL_EXECBENCH_DEPENDENCY_TORCH_DISTRIBUTION_VERSION:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TORCH_VERSION:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TORCH_LOCAL_VERSION:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TORCH_ROCM_TARGET:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TORCH_HIP_VERSION:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TORCH_CUDA_VERSION:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TORCH_DEVICE_AVAILABLE:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TORCH_IMPORT_ERROR:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TORCHVISION_DISTRIBUTION_VERSION:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_DISTRIBUTION_VERSION:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_STATUS:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_HIPCC_VERSION:-}" ] ||
        [ -n "${SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION:-}" ]
}

append_dependency_arg_from_env() {
    local -n cmd_ref="$1"
    local env_name="$2"
    local flag="$3"
    local value="${!env_name:-}"
    if [ -n "${value}" ]; then
        cmd_ref+=("${flag}" "${value}")
    fi
}

classify_dependency_preflight_json() {
    local cmd=(
        python -m sol_execbench.core.dependency_matrix preflight
        --manifest "${REPO_ROOT}/docker/rocm-targets.json"
    )
    if [ -n "${DOCKER_TARGET}" ]; then
        cmd+=(--target "${DOCKER_TARGET}")
    fi
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_DISTRIBUTION_VERSION --torch-distribution-version
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_VERSION --torch-version
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_LOCAL_VERSION --torch-local-version
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_ROCM_TARGET --torch-rocm-target
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_HIP_VERSION --torch-hip-version
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_CUDA_VERSION --torch-cuda-version
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_DEVICE_AVAILABLE --torch-device-available
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_IMPORT_ERROR --torch-import-error
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCHVISION_DISTRIBUTION_VERSION --torchvision-distribution-version
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_DISTRIBUTION_VERSION --triton-rocm-distribution-version
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_STATUS --triton-rocm-status
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION --container-rocm-user-space-version
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_HIPCC_VERSION --hipcc-version
    append_dependency_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION --toolchain-rocm-version
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}"
}

docker_context_name() {
    if [ -n "${SOL_EXECBENCH_DOCKER_CONTEXT:-}" ]; then
        echo "${SOL_EXECBENCH_DOCKER_CONTEXT}"
    else
        docker context show 2>/dev/null || true
    fi
}

docker_context_host() {
    if [ -n "${SOL_EXECBENCH_DOCKER_HOST:-}" ]; then
        echo "${SOL_EXECBENCH_DOCKER_HOST}"
    else
        docker context inspect --format '{{ (index .Endpoints "docker").Host }}' 2>/dev/null || true
    fi
}

preflight_bool() {
    local env_name="$1"
    local fallback="$2"
    local value="${!env_name:-}"
    if [ -n "${value}" ]; then
        echo "${value}"
    else
        echo "${fallback}"
    fi
}

classify_docker_preflight_json() {
    local context_name
    local docker_host
    local dev_kfd_present
    local dev_kfd_accessible
    local dev_dri_present
    local dev_dri_accessible
    local cmd

    context_name="$(docker_context_name)"
    docker_host="$(docker_context_host)"
    dev_kfd_present="$(preflight_bool SOL_EXECBENCH_DEV_KFD_PRESENT "$(bool_text "$([ -e /dev/kfd ] && echo 1 || echo 0)")")"
    dev_kfd_accessible="$(preflight_bool SOL_EXECBENCH_DEV_KFD_ACCESSIBLE "$(bool_text "$([ -r /dev/kfd ] && [ -w /dev/kfd ] && echo 1 || echo 0)")")"
    dev_dri_present="$(preflight_bool SOL_EXECBENCH_DEV_DRI_PRESENT "$(bool_text "$([ -d /dev/dri ] && echo 1 || echo 0)")")"
    dev_dri_accessible="$(preflight_bool SOL_EXECBENCH_DEV_DRI_ACCESSIBLE "$(bool_text "$(dev_dri_has_accessible_node && echo 1 || echo 0)")")"

    cmd=(
        python -m sol_execbench.core.docker_matrix preflight
        --manifest "${REPO_ROOT}/docker/rocm-targets.json"
        --docker-context "${context_name}"
        --docker-host "${docker_host}"
        --dev-kfd-present "${dev_kfd_present}"
        --dev-kfd-accessible "${dev_kfd_accessible}"
        --dev-dri-present "${dev_dri_present}"
        --dev-dri-accessible "${dev_dri_accessible}"
    )
    if [ -n "${DOCKER_TARGET}" ]; then
        cmd+=(--target "${DOCKER_TARGET}")
    fi
    if [ -n "${SOL_EXECBENCH_GPU_ACCESSIBLE:-}" ]; then
        cmd+=(--gpu-accessible "${SOL_EXECBENCH_GPU_ACCESSIBLE}")
    fi
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}"
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
SELECTED_TARGET_ID="$(matrix_json_value "${TARGET_JSON}" "target_id")"

if [[ "${SELECTED_TARGET_ID}" == unsafe-untested-* ]]; then
    echo "${TARGET_JSON}"
    exit 1
fi

if [ "${DRY_RUN}" != "1" ] || $PREFLIGHT_ONLY || dependency_preflight_override_present; then
    DEPENDENCY_PREFLIGHT_JSON="$(classify_dependency_preflight_json)"
    DEPENDENCY_PREFLIGHT_STATUS="$(matrix_json_value "${DEPENDENCY_PREFLIGHT_JSON}" "status")"
    DEPENDENCY_PREFLIGHT_BENCHMARK_ALLOWED="$(matrix_json_value "${DEPENDENCY_PREFLIGHT_JSON}" "benchmark_allowed")"
    if $PREFLIGHT_ONLY; then
        if dependency_preflight_override_present; then
            echo "${DEPENDENCY_PREFLIGHT_JSON}"
            if [ "${DEPENDENCY_PREFLIGHT_STATUS}" = "mixed_version" ] ||
                [ "${DEPENDENCY_PREFLIGHT_STATUS}" = "pytorch_wheel_unavailable" ] ||
                [ "${DEPENDENCY_PREFLIGHT_BENCHMARK_ALLOWED}" != "True" ]; then
                exit 1
            fi
            exit 0
        fi
    elif [ "${DEPENDENCY_PREFLIGHT_STATUS}" = "mixed_version" ] ||
        [ "${DEPENDENCY_PREFLIGHT_STATUS}" = "pytorch_wheel_unavailable" ] ||
        [ "${DEPENDENCY_PREFLIGHT_BENCHMARK_ALLOWED}" != "True" ]; then
        echo "${DEPENDENCY_PREFLIGHT_JSON}"
        exit 1
    fi
fi

if [ "${DRY_RUN}" != "1" ] || $PREFLIGHT_ONLY || preflight_override_present; then
    PREFLIGHT_JSON="$(classify_docker_preflight_json)"
    PREFLIGHT_STATUS="$(matrix_json_value "${PREFLIGHT_JSON}" "status")"
    PREFLIGHT_BENCHMARK_ALLOWED="$(matrix_json_value "${PREFLIGHT_JSON}" "benchmark_allowed")"
    if $PREFLIGHT_ONLY; then
        echo "${PREFLIGHT_JSON}"
        if [ "${PREFLIGHT_STATUS}" = "runtime_unavailable" ] ||
            [ "${PREFLIGHT_BENCHMARK_ALLOWED}" != "True" ]; then
            exit 1
        fi
        exit 0
    fi
    if [ "${PREFLIGHT_STATUS}" = "runtime_unavailable" ] ||
        [ "${PREFLIGHT_BENCHMARK_ALLOWED}" != "True" ]; then
        echo "${PREFLIGHT_JSON}"
        exit 1
    fi
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
