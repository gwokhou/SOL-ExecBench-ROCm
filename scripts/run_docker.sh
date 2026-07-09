#!/bin/bash
# Launch the sol-execbench Docker container with the right mounts.
#
# Usage:
#   ./scripts/run_docker.sh [--build] [--target <id>] [--allow-unknown-target] [--allow-mixed-version-dependencies] [--allow-untested-target-smoke] [--record-container-validation] [--preflight-only] [--compatibility-entry <path>] [--compatibility-matrix <path>] [docker-run-args...] [-- command...]
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
#   IMAGE_TAG           Docker image tag     (default: rocm-<selected ROCm Docker tag>)
#   ROCM_DOCKER_IMAGE   Unsafe unknown-target override repository
#   ROCM_DOCKER_TAG     Unsafe unknown-target override tag
#   SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES=1  Allow dependency probe/smoke diagnostics for mixed-version stacks
#   SOL_EXECBENCH_ALLOW_UNTESTED_TARGET_SMOKE=1       Allow not_tested Targets to run smoke/E2E without validation claims
#   SOL_EXECBENCH_RECORD_CONTAINER_VALIDATION=1       Record a successful wrapper benchmark as container_validated evidence
#   SOL_EXECBENCH_HOST_PYTHON  Optional host Python executable override for wrapper helper commands
#   SOL_EXECBENCH_DEPENDENCY_*  Dependency preflight observation overrides for tests/debugging
#   SOL_EXECBENCH_COMPATIBILITY_ENTRY  Optional per-Target compatibility JSON sidecar path
#   SOL_EXECBENCH_COMPATIBILITY_MATRIX Optional aggregate compatibility matrix JSON path

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(uname -s)" != "Linux" ]; then
    echo "ERROR: scripts/run_docker.sh is supported only on Linux ROCm hosts." >&2
    exit 2
fi

IMAGE_NAME="${IMAGE_NAME:-sol-execbench}"
IMAGE_TAG="${IMAGE_TAG:-}"

# Container-side paths (must match Dockerfile / entrypoint expectations)
CONTAINER_PROJECT="/sol-execbench"
DOCKER_TARGET=""
ALLOW_UNKNOWN_TARGET=false
ALLOW_MIXED_VERSION_DEPENDENCIES=false
ALLOW_UNTESTED_TARGET_SMOKE=false
RECORD_CONTAINER_VALIDATION=false
PREFLIGHT_ONLY=false
DRY_RUN="${SOL_EXECBENCH_RUN_DOCKER_DRY_RUN:-0}"
COMPATIBILITY_ENTRY_PATH="${SOL_EXECBENCH_COMPATIBILITY_ENTRY:-}"
COMPATIBILITY_MATRIX_PATH="${SOL_EXECBENCH_COMPATIBILITY_MATRIX:-}"

run_host_python() {
    local pythonpath="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    if [ -n "${SOL_EXECBENCH_HOST_PYTHON:-}" ]; then
        PYTHONPATH="${pythonpath}" "${SOL_EXECBENCH_HOST_PYTHON}" "$@"
    else
        PYTHONPATH="${pythonpath}" uv run python "$@"
    fi
}

case "${SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES:-0}" in
    1 | true | TRUE | yes | YES)
        ALLOW_MIXED_VERSION_DEPENDENCIES=true
        ;;
esac

case "${SOL_EXECBENCH_ALLOW_UNTESTED_TARGET_SMOKE:-0}" in
    1 | true | TRUE | yes | YES)
        ALLOW_UNTESTED_TARGET_SMOKE=true
        ;;
esac

case "${SOL_EXECBENCH_RECORD_CONTAINER_VALIDATION:-0}" in
    1 | true | TRUE | yes | YES)
        RECORD_CONTAINER_VALIDATION=true
        ;;
esac

matrix_json_value() {
    run_host_python -c 'import json, sys; data=json.loads(sys.argv[1]); value=data; [value := value[part] for part in sys.argv[2].split(".")]; print(value)' "$1" "$2"
}

bool_text() {
    if [ "$1" = "1" ] || [ "$1" = "true" ]; then
        echo "true"
    else
        echo "false"
    fi
}

preflight_blocked() {
    local status="$1"
    local benchmark_allowed="$2"
    local hard_block_status="$3"
    if [ "${status}" = "${hard_block_status}" ] ||
        [ "${benchmark_allowed}" != "True" ]; then
        if $ALLOW_UNTESTED_TARGET_SMOKE && [ "${status}" = "not_tested" ]; then
            return 1
        fi
        return 0
    fi
    return 1
}

execution_preflight_blocked() {
    local status="$1"
    local benchmark_allowed="$2"
    local hard_block_status="$3"
    if preflight_blocked "${status}" "${benchmark_allowed}" "${hard_block_status}"; then
        if $RECORD_CONTAINER_VALIDATION && [ "${status}" = "not_tested" ]; then
            return 1
        fi
        return 0
    fi
    return 1
}

dev_dri_has_accessible_node() {
    [ -x /dev/dri ] || return 1
    find /dev/dri -maxdepth 1 -type c \( -name 'renderD*' -o -name 'card*' \) \
        -readable -writable -print -quit 2>/dev/null | grep -q .
}

resolve_docker_target_json() {
    local cmd=(
        -m sol_execbench.core.platform.docker_matrix preview
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
    run_host_python "${cmd[@]}"
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
        -m sol_execbench.core.platform.dependency_matrix preflight
        --manifest "${REPO_ROOT}/docker/rocm-targets.json"
    )
    if [ -n "${DOCKER_TARGET}" ]; then
        cmd+=(--target "${DOCKER_TARGET}")
    fi
    if $ALLOW_MIXED_VERSION_DEPENDENCIES; then
        cmd+=(--allow-mixed-version-debug)
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
    run_host_python "${cmd[@]}"
}

append_runtime_arg_from_env() {
    local -n cmd_ref="$1"
    local env_name="$2"
    local flag="$3"
    local value="${!env_name:-}"
    if [ -n "${value}" ]; then
        cmd_ref+=("${flag}" "${value}")
    fi
}

compatibility_sidecar_requested() {
    [ -n "${COMPATIBILITY_ENTRY_PATH}" ] || [ -n "${COMPATIBILITY_MATRIX_PATH}" ]
}

compatibility_entry_output_path() {
    if [ -n "${COMPATIBILITY_ENTRY_PATH}" ]; then
        echo "${COMPATIBILITY_ENTRY_PATH}"
    else
        echo "${COMPATIBILITY_MATRIX_PATH}.entry.json"
    fi
}

write_compatibility_sidecars() {
    compatibility_sidecar_requested || return 0

    local entry_path
    local cmd
    entry_path="$(compatibility_entry_output_path)"
    cmd=(
        -m sol_execbench.core.evidence.runtime_evidence collect-target
        --manifest "${REPO_ROOT}/docker/rocm-targets.json"
        --output "${entry_path}"
    )
    if [ -n "${DOCKER_TARGET}" ]; then
        cmd+=(--target "${DOCKER_TARGET}")
    fi
    if $ALLOW_MIXED_VERSION_DEPENDENCIES; then
        cmd+=(--allow-mixed-version-debug)
    fi
    append_runtime_arg_from_env cmd SOL_EXECBENCH_HOST_ROCM_VERSION --host-rocm-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_HOST_DRIVER_VERSION --host-driver-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEV_KFD_PRESENT --dev-kfd-present
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEV_KFD_ACCESSIBLE --dev-kfd-accessible
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEV_DRI_PRESENT --dev-dri-present
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEV_DRI_ACCESSIBLE --dev-dri-accessible
    append_runtime_arg_from_env cmd SOL_EXECBENCH_IMAGE_DIGEST --image-digest
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_DISTRIBUTION_VERSION --torch-distribution-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_VERSION --torch-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_LOCAL_VERSION --torch-local-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_ROCM_TARGET --torch-rocm-target
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_HIP_VERSION --torch-hip-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_CUDA_VERSION --torch-cuda-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_DEVICE_AVAILABLE --torch-device-available
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCH_IMPORT_ERROR --torch-import-error
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TORCHVISION_DISTRIBUTION_VERSION --torchvision-distribution-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_DISTRIBUTION_VERSION --triton-rocm-distribution-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_STATUS --triton-rocm-status
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION --container-rocm-user-space-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_HIPCC_VERSION --hipcc-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION --toolchain-rocm-version
    append_runtime_arg_from_env cmd SOL_EXECBENCH_RUNTIME_DEVICE_COUNT --device-count
    append_runtime_arg_from_env cmd SOL_EXECBENCH_RUNTIME_DEVICE_NAME --device-name
    append_runtime_arg_from_env cmd SOL_EXECBENCH_RUNTIME_GFX_ARCHITECTURE --gfx-architecture
    if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
        cmd+=(--visible-device-env "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}")
    fi
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
        cmd+=(--visible-device-env "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}")
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        cmd+=(--visible-device-env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
    fi
    if [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
        cmd+=(--visible-device-env "GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL}")
    fi
    if [ -n "${1:-}" ]; then
        cmd+=(--runtime-unavailable-reason "$1" --failure-category setup_runtime)
    elif [ -n "${2:-}" ]; then
        cmd+=(--failure-category dependency)
    fi
    run_host_python "${cmd[@]}" >/dev/null

    if [ -n "${COMPATIBILITY_MATRIX_PATH}" ]; then
        run_host_python -m sol_execbench.core.evidence.runtime_evidence aggregate \
            --output "${COMPATIBILITY_MATRIX_PATH}" "${entry_path}" >/dev/null
    fi
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
        -m sol_execbench.core.platform.docker_matrix preflight
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
    run_host_python "${cmd[@]}"
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
            --allow-mixed-version-dependencies)
                ALLOW_MIXED_VERSION_DEPENDENCIES=true
                continue
                ;;
            --allow-untested-target-smoke)
                ALLOW_UNTESTED_TARGET_SMOKE=true
                continue
                ;;
            --record-container-validation)
                RECORD_CONTAINER_VALIDATION=true
                continue
                ;;
            --preflight-only)
                PREFLIGHT_ONLY=true
                continue
                ;;
            --compatibility-entry)
                if [ "$#" -eq 0 ]; then
                    echo "ERROR: --compatibility-entry requires a path." >&2
                    exit 2
                fi
                COMPATIBILITY_ENTRY_PATH="$1"
                shift
                continue
                ;;
            --compatibility-matrix)
                if [ "$#" -eq 0 ]; then
                    echo "ERROR: --compatibility-matrix requires a path." >&2
                    exit 2
                fi
                COMPATIBILITY_MATRIX_PATH="$1"
                shift
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
if [ -z "${IMAGE_TAG}" ]; then
    IMAGE_TAG="rocm-${ROCM_DOCKER_TAG_SELECTED}"
fi
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

if [[ "${SELECTED_TARGET_ID}" == unsafe-untested-* ]]; then
    echo "${TARGET_JSON}"
    exit 1
fi

if $PREFLIGHT_ONLY || dependency_preflight_override_present; then
    DEPENDENCY_PREFLIGHT_JSON="$(classify_dependency_preflight_json)"
    DEPENDENCY_PREFLIGHT_STATUS="$(matrix_json_value "${DEPENDENCY_PREFLIGHT_JSON}" "status")"
    DEPENDENCY_PREFLIGHT_BENCHMARK_ALLOWED="$(matrix_json_value "${DEPENDENCY_PREFLIGHT_JSON}" "benchmark_allowed")"
    if $PREFLIGHT_ONLY; then
        if dependency_preflight_override_present; then
            if preflight_blocked "${DEPENDENCY_PREFLIGHT_STATUS}" "${DEPENDENCY_PREFLIGHT_BENCHMARK_ALLOWED}" "mixed_version" ||
                [ "${DEPENDENCY_PREFLIGHT_STATUS}" = "pytorch_wheel_unavailable" ]; then
                write_compatibility_sidecars "" "dependency"
            else
                write_compatibility_sidecars "" ""
            fi
            echo "${DEPENDENCY_PREFLIGHT_JSON}"
            if preflight_blocked "${DEPENDENCY_PREFLIGHT_STATUS}" "${DEPENDENCY_PREFLIGHT_BENCHMARK_ALLOWED}" "mixed_version" ||
                [ "${DEPENDENCY_PREFLIGHT_STATUS}" = "pytorch_wheel_unavailable" ]; then
                exit 1
            fi
            exit 0
        fi
    elif [ "${DEPENDENCY_PREFLIGHT_STATUS}" = "pytorch_wheel_unavailable" ]; then
        write_compatibility_sidecars "" "dependency"
        echo "${DEPENDENCY_PREFLIGHT_JSON}"
        exit 1
    elif execution_preflight_blocked "${DEPENDENCY_PREFLIGHT_STATUS}" "${DEPENDENCY_PREFLIGHT_BENCHMARK_ALLOWED}" "mixed_version" &&
        ! { $ALLOW_MIXED_VERSION_DEPENDENCIES && [ "${DEPENDENCY_PREFLIGHT_STATUS}" = "mixed_version" ]; }; then
        write_compatibility_sidecars "" "dependency"
        echo "${DEPENDENCY_PREFLIGHT_JSON}"
        exit 1
    fi
fi

if [ "${DRY_RUN}" != "1" ] || $PREFLIGHT_ONLY || preflight_override_present; then
    PREFLIGHT_JSON="$(classify_docker_preflight_json)"
    PREFLIGHT_STATUS="$(matrix_json_value "${PREFLIGHT_JSON}" "status")"
    PREFLIGHT_BENCHMARK_ALLOWED="$(matrix_json_value "${PREFLIGHT_JSON}" "benchmark_allowed")"
    PREFLIGHT_REASON="$(matrix_json_value "${PREFLIGHT_JSON}" "reason")"
    if $PREFLIGHT_ONLY; then
        if preflight_blocked "${PREFLIGHT_STATUS}" "${PREFLIGHT_BENCHMARK_ALLOWED}" "runtime_unavailable"; then
            write_compatibility_sidecars "${PREFLIGHT_REASON}" ""
        else
            write_compatibility_sidecars "" ""
        fi
        echo "${PREFLIGHT_JSON}"
        if preflight_blocked "${PREFLIGHT_STATUS}" "${PREFLIGHT_BENCHMARK_ALLOWED}" "runtime_unavailable"; then
            exit 1
        fi
        exit 0
    fi
    if execution_preflight_blocked "${PREFLIGHT_STATUS}" "${PREFLIGHT_BENCHMARK_ALLOWED}" "runtime_unavailable"; then
        write_compatibility_sidecars "${PREFLIGHT_REASON}" ""
        echo "${PREFLIGHT_JSON}"
        exit 1
    fi
fi

# Build the image if requested
if $BUILD; then
    PYTORCH_TORCH_VERSION_SELECTED="$(matrix_json_value "${TARGET_JSON}" "build_args.PYTORCH_TORCH_VERSION")"
    PYTORCH_TORCHVISION_VERSION_SELECTED="$(matrix_json_value "${TARGET_JSON}" "build_args.PYTORCH_TORCHVISION_VERSION")"
    PYTORCH_ROCM_INDEX_URL_SELECTED="$(matrix_json_value "${TARGET_JSON}" "build_args.PYTORCH_ROCM_INDEX_URL")"
    TRITON_ROCM_VERSION_SELECTED="$(matrix_json_value "${TARGET_JSON}" "build_args.TRITON_ROCM_VERSION")"
    TRITON_ROCM_INDEX_URL_SELECTED="$(matrix_json_value "${TARGET_JSON}" "build_args.TRITON_ROCM_INDEX_URL")"
    echo "+ docker build -t ${IMAGE} --build-arg ROCM_DOCKER_IMAGE=\"${ROCM_DOCKER_IMAGE_SELECTED}\" --build-arg ROCM_DOCKER_TAG=\"${ROCM_DOCKER_TAG_SELECTED}\" --build-arg PYTORCH_TORCH_VERSION=\"${PYTORCH_TORCH_VERSION_SELECTED}\" --build-arg PYTORCH_TORCHVISION_VERSION=\"${PYTORCH_TORCHVISION_VERSION_SELECTED}\" --build-arg PYTORCH_ROCM_INDEX_URL=\"${PYTORCH_ROCM_INDEX_URL_SELECTED}\" --build-arg TRITON_ROCM_VERSION=\"${TRITON_ROCM_VERSION_SELECTED}\" --build-arg TRITON_ROCM_INDEX_URL=\"${TRITON_ROCM_INDEX_URL_SELECTED}\" -f ${REPO_ROOT}/docker/Dockerfile ${REPO_ROOT}"
    if [ "${DRY_RUN}" != "1" ]; then
        docker build \
            -t "${IMAGE}" \
            --build-arg "ROCM_DOCKER_IMAGE=${ROCM_DOCKER_IMAGE_SELECTED}" \
            --build-arg "ROCM_DOCKER_TAG=${ROCM_DOCKER_TAG_SELECTED}" \
            --build-arg "PYTORCH_TORCH_VERSION=${PYTORCH_TORCH_VERSION_SELECTED}" \
            --build-arg "PYTORCH_TORCHVISION_VERSION=${PYTORCH_TORCHVISION_VERSION_SELECTED}" \
            --build-arg "PYTORCH_ROCM_INDEX_URL=${PYTORCH_ROCM_INDEX_URL_SELECTED}" \
            --build-arg "TRITON_ROCM_VERSION=${TRITON_ROCM_VERSION_SELECTED}" \
            --build-arg "TRITON_ROCM_INDEX_URL=${TRITON_ROCM_INDEX_URL_SELECTED}" \
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

DOCKER_COMMON_ARGS=(
    docker run --rm
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
)

container_repo_path() {
    local path="$1"
    case "${path}" in
        "${REPO_ROOT}"/*)
            echo "${CONTAINER_PROJECT}/${path#"${REPO_ROOT}/"}"
            ;;
        "${REPO_ROOT}")
            echo "${CONTAINER_PROJECT}"
            ;;
        *)
            echo "${path}"
            ;;
    esac
}

run_container_dependency_preflight_json() {
    local cmd=(
        "${DOCKER_COMMON_ARGS[@]}"
        --entrypoint python
        "${IMAGE}"
        -m sol_execbench.core.platform.dependency_matrix preflight
        --manifest "${CONTAINER_PROJECT}/docker/rocm-targets.json"
    )
    if [ -n "${DOCKER_TARGET}" ]; then
        cmd+=(--target "${DOCKER_TARGET}")
    fi
    "${cmd[@]}"
}

write_container_validated_sidecars() {
    compatibility_sidecar_requested || return 0

    local entry_path
    local container_entry_path
    local cmd
    entry_path="$(compatibility_entry_output_path)"
    container_entry_path="$(container_repo_path "${entry_path}")"
    cmd=(
        "${DOCKER_COMMON_ARGS[@]}"
        --entrypoint python
        "${IMAGE}"
        -m sol_execbench.core.evidence.runtime_evidence collect-target
        --manifest "${CONTAINER_PROJECT}/docker/rocm-targets.json"
        --output "${container_entry_path}"
        --container-validated
        --dev-kfd-present "$(bool_text "$([ -e /dev/kfd ] && echo 1 || echo 0)")"
        --dev-kfd-accessible "$(bool_text "$([ -r /dev/kfd ] && [ -w /dev/kfd ] && echo 1 || echo 0)")"
        --dev-dri-present "$(bool_text "$([ -d /dev/dri ] && echo 1 || echo 0)")"
        --dev-dri-accessible "$(bool_text "$(dev_dri_has_accessible_node && echo 1 || echo 0)")"
    )
    if [ -n "${DOCKER_TARGET}" ]; then
        cmd+=(--target "${DOCKER_TARGET}")
    fi
    "${cmd[@]}" >/dev/null
    if [ -n "${COMPATIBILITY_MATRIX_PATH}" ]; then
        run_host_python -m sol_execbench.core.evidence.runtime_evidence aggregate \
            --output "${COMPATIBILITY_MATRIX_PATH}" "${entry_path}" >/dev/null
    fi
}

if [ "${DRY_RUN}" != "1" ] && $RECORD_CONTAINER_VALIDATION; then
    CONTAINER_DEPENDENCY_PREFLIGHT_JSON="$(run_container_dependency_preflight_json)"
    CONTAINER_DEPENDENCY_PREFLIGHT_STATUS="$(matrix_json_value "${CONTAINER_DEPENDENCY_PREFLIGHT_JSON}" "status")"
    CONTAINER_DEPENDENCY_PREFLIGHT_BENCHMARK_ALLOWED="$(matrix_json_value "${CONTAINER_DEPENDENCY_PREFLIGHT_JSON}" "benchmark_allowed")"
    if execution_preflight_blocked "${CONTAINER_DEPENDENCY_PREFLIGHT_STATUS}" "${CONTAINER_DEPENDENCY_PREFLIGHT_BENCHMARK_ALLOWED}" "mixed_version"; then
        write_compatibility_sidecars "" "dependency"
        echo "${CONTAINER_DEPENDENCY_PREFLIGHT_JSON}"
        exit 1
    fi
fi

DOCKER_CMD=(
    "${DOCKER_COMMON_ARGS[@]}"
    "${TTY_ARGS[@]}"
    "${IMAGE}"
    "${CMD[@]}"
)

echo "+ ${DOCKER_CMD[*]}"
if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi
set +e
"${DOCKER_CMD[@]}"
status="$?"
set -e
if [ "${status}" -eq 0 ]; then
    if $RECORD_CONTAINER_VALIDATION; then
        write_container_validated_sidecars
    elif compatibility_sidecar_requested; then
        write_compatibility_sidecars "" ""
    fi
    exit 0
fi
exit "${status}"
