#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly PATCH_ID="gfx1200-sq-wave-cycles-stable-peak-v1"
readonly PATCH_HOME="${SOL_EXECBENCH_PATCH_HOME:-${HOME}}"
readonly BIN_DIR="${PATCH_HOME}/.local/bin"
readonly STATE_DIR="${PATCH_HOME}/.local/share/sol-execbench/patches/gfx1200-sq-wave-cycles"
readonly WRAPPER_PATH="${BIN_DIR}/rocprofv3-gfx1200-patched"
readonly ROLLBACK_PATH="${BIN_DIR}/rollback-rocprofv3-gfx1200-patch"
readonly WRAPPER_CHECKSUM_PATH="${STATE_DIR}/wrapper.sha256"
readonly ROLLBACK_CHECKSUM_PATH="${STATE_DIR}/rollback.sha256"
readonly AMD_SMI_BIN="${AMD_SMI:-/opt/rocm/bin/amd-smi}"
readonly SUDO_BIN="${SOL_EXECBENCH_SUDO:-sudo}"
readonly LOCK_ROOT="${XDG_RUNTIME_DIR:-/tmp}"
readonly LOCK_PATH="${LOCK_ROOT}/sol-execbench-${PATCH_ID}-${UID}.lock"

usage() {
  cat <<'EOF'
Remove the gfx1200 SQ_WAVE_CYCLES local workaround safely.

Usage:
  rollback-rocprofv3-gfx1200-patch
  rollback-rocprofv3-gfx1200-patch --help

What it does:
  - Verifies the installed wrapper and rollback command against their recorded
    SHA-256 checksums before deleting either file.
  - Refuses to run while the patched profiler is active.
  - Refuses to delete an installed file that has been locally modified.
  - Resets any remaining STABLE_PEAK state to AUTO on every visible AMD GPU
    and verifies the resulting state before deleting the patch.
  - Removes the user-local patch files and checksum state.

What it leaves unchanged:
  - /opt/rocm and the official rocprofv3 installation.
  - The amd-smi sudoers rule, because normal benchmark clock locking shares it.

Default installed paths:
  ~/.local/bin/rocprofv3-gfx1200-patched
  ~/.local/bin/rollback-rocprofv3-gfx1200-patch

If rollback reports `file_checksum_mismatch`, inspect the named file instead
of deleting it blindly. If it reports `patch_is_running`, wait for profiling
and automatic AUTO restoration to finish, then retry.

Optional deployment/test overrides:
  SOL_EXECBENCH_PATCH_HOME, AMD_SMI, SOL_EXECBENCH_SUDO,
  XDG_RUNTIME_DIR
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "rollback_error=unknown_argument value=$1" >&2
      usage >&2
      exit 64
      ;;
  esac
fi

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "rollback_error=patch_is_running lock=${LOCK_PATH}" >&2
  exit 75
fi

verify_managed_file() {
  local target=$1
  local checksum_path=$2
  [[ -e "${target}" ]] || return 0
  if [[ ! -f "${checksum_path}" ]]; then
    echo "rollback_error=missing_checksum path=${checksum_path}" >&2
    exit 65
  fi
  local expected actual
  expected=$(sed -n '1p' "${checksum_path}")
  actual=$(sha256sum "${target}" | awk '{print $1}')
  if [[ -z "${expected}" || "${actual}" != "${expected}" ]]; then
    echo "rollback_error=file_checksum_mismatch path=${target}" >&2
    exit 65
  fi
}
verify_managed_file "${WRAPPER_PATH}" "${WRAPPER_CHECKSUM_PATH}"
verify_managed_file "${ROLLBACK_PATH}" "${ROLLBACK_CHECKSUM_PATH}"

perf_levels() {
  "${AMD_SMI_BIN}" metric -l 2>&1
}

all_levels_are() {
  local output=$1
  local expected=$2
  local level_lines
  level_lines=$(grep "PERF_LEVEL:" <<<"${output}") || return 1
  [[ -n "${level_lines}" ]] || return 1
  ! grep -vq "${expected}" <<<"${level_lines}"
}

if [[ -x "${AMD_SMI_BIN}" ]]; then
  current_level=$(perf_levels) || {
    echo "rollback_error=clock_state_unavailable output=${current_level}" >&2
    exit 69
  }
  if grep -q "PERF_LEVEL_STABLE_PEAK" <<<"${current_level}"; then
    reset_output=$("${SUDO_BIN}" -n "${AMD_SMI_BIN}" set -l AUTO 2>&1) || {
      echo "rollback_error=clock_reset_failed output=${reset_output}" >&2
      exit 70
    }
    reset_state=$(perf_levels) || true
    if ! all_levels_are "${reset_state}" "AMDSMI_DEV_PERF_LEVEL_AUTO"; then
      echo "rollback_error=clock_reset_unverified output=${reset_state}" >&2
      exit 70
    fi
  fi
fi

if [[ -e "${WRAPPER_PATH}" ]]; then
  rm -f -- "${WRAPPER_PATH}"
fi

rm -f -- "${WRAPPER_CHECKSUM_PATH}" "${ROLLBACK_CHECKSUM_PATH}"
rmdir -- "${STATE_DIR}" 2>/dev/null || true
if [[ "${0}" != "${ROLLBACK_PATH}" ]]; then
  rm -f -- "${ROLLBACK_PATH}"
else
  trap 'rm -f -- "${ROLLBACK_PATH}"' EXIT
fi
echo "rolled_back=${WRAPPER_PATH}"
