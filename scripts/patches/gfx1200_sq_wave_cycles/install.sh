#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly PATCH_HOME="${SOL_EXECBENCH_PATCH_HOME:-${HOME}}"
readonly BIN_DIR="${PATCH_HOME}/.local/bin"
readonly STATE_DIR="${PATCH_HOME}/.local/share/sol-execbench/patches/gfx1200-sq-wave-cycles"
readonly WRAPPER_PATH="${BIN_DIR}/rocprofv3-gfx1200-patched"
readonly ROLLBACK_PATH="${BIN_DIR}/rollback-rocprofv3-gfx1200-patch"
readonly WRAPPER_CHECKSUM_PATH="${STATE_DIR}/wrapper.sha256"
readonly ROLLBACK_CHECKSUM_PATH="${STATE_DIR}/rollback.sha256"
readonly SOURCE_WRAPPER="${SCRIPT_DIR}/rocprofv3-gfx1200-patched"
readonly SOURCE_ROLLBACK="${SCRIPT_DIR}/rollback.sh"
readonly REAL_ROCPROF="${ROCPROFV3_REAL:-/opt/rocm/bin/rocprofv3}"
readonly AMD_SMI_BIN="${AMD_SMI:-/opt/rocm/bin/amd-smi}"
readonly SUDO_BIN="${SOL_EXECBENCH_SUDO:-sudo}"

usage() {
  cat <<'EOF'
Install the gfx1200 SQ_WAVE_CYCLES local workaround.

Usage:
  install.sh
  install.sh --help

What it does:
  - Installs ~/.local/bin/rocprofv3-gfx1200-patched.
  - Installs ~/.local/bin/rollback-rocprofv3-gfx1200-patch.
  - Records SHA-256 checksums under ~/.local/share/sol-execbench/patches/.
  - Does not modify /opt/rocm or replace the official rocprofv3.

Prerequisite:
  The exact amd-smi path must have passwordless coverage for `version`,
  `set -l STABLE_PEAK`, and `set -l AUTO`. These set commands affect every
  visible AMD GPU. From the repository root:

    sudo .venv/bin/python scripts/setup_rocm_clock_sudoers.py \
      --mode install --user "${USER}" --amd-smi /opt/rocm/bin/amd-smi

After installation:
  Run the patched profiler with normal rocprofv3 arguments:

    ~/.local/bin/rocprofv3-gfx1200-patched \
      --kernel-trace --pmc SQ_WAVE_CYCLES -- <workload>

  Show wrapper-specific help without changing GPU state:

    ~/.local/bin/rocprofv3-gfx1200-patched --patch-help

  Roll back in one command:

    ~/.local/bin/rollback-rocprofv3-gfx1200-patch

Re-running this installer upgrades only checksum-matching managed files. It
refuses to overwrite an unmanaged or locally modified file.

Optional deployment/test overrides:
  SOL_EXECBENCH_PATCH_HOME, ROCPROFV3_REAL, AMD_SMI, SOL_EXECBENCH_SUDO
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "install_error=unknown_argument value=$1" >&2
      usage >&2
      exit 64
      ;;
  esac
fi

for executable in "${SOURCE_WRAPPER}" "${SOURCE_ROLLBACK}" "${REAL_ROCPROF}" "${AMD_SMI_BIN}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "install_error=missing_executable path=${executable}" >&2
    exit 66
  fi
done
for command in \
  "version" \
  "set -l STABLE_PEAK" \
  "set -l AUTO"; do
  read -r -a arguments <<<"${command}"
  if ! "${SUDO_BIN}" -n -l -- "${AMD_SMI_BIN}" "${arguments[@]}" >/dev/null 2>&1; then
    echo "install_error=amd_smi_sudoers_missing path=${AMD_SMI_BIN} command=${command}" >&2
    exit 77
  fi
done

mkdir -p "${BIN_DIR}" "${STATE_DIR}"
verify_managed_file() {
  local target=$1
  local checksum_path=$2
  [[ -e "${target}" ]] || return 0
  if [[ ! -f "${checksum_path}" ]]; then
    echo "install_error=unmanaged_existing_file path=${target}" >&2
    exit 65
  fi
  local expected actual
  expected=$(sed -n '1p' "${checksum_path}")
  actual=$(sha256sum "${target}" | awk '{print $1}')
  if [[ -z "${expected}" || "${actual}" != "${expected}" ]]; then
    echo "install_error=existing_file_checksum_mismatch path=${target}" >&2
    exit 65
  fi
}
verify_managed_file "${WRAPPER_PATH}" "${WRAPPER_CHECKSUM_PATH}"
verify_managed_file "${ROLLBACK_PATH}" "${ROLLBACK_CHECKSUM_PATH}"

wrapper_tmp="${WRAPPER_PATH}.tmp.$$"
rollback_tmp="${ROLLBACK_PATH}.tmp.$$"
wrapper_checksum_tmp="${WRAPPER_CHECKSUM_PATH}.tmp.$$"
rollback_checksum_tmp="${ROLLBACK_CHECKSUM_PATH}.tmp.$$"
cleanup() {
  rm -f -- \
    "${wrapper_tmp}" "${rollback_tmp}" \
    "${wrapper_checksum_tmp}" "${rollback_checksum_tmp}"
}
trap cleanup EXIT HUP INT TERM

install -m 0755 "${SOURCE_WRAPPER}" "${wrapper_tmp}"
install -m 0755 "${SOURCE_ROLLBACK}" "${rollback_tmp}"
sha256sum "${wrapper_tmp}" | awk '{print $1}' >"${wrapper_checksum_tmp}"
sha256sum "${rollback_tmp}" | awk '{print $1}' >"${rollback_checksum_tmp}"
mv -f -- "${wrapper_tmp}" "${WRAPPER_PATH}"
mv -f -- "${rollback_tmp}" "${ROLLBACK_PATH}"
mv -f -- "${wrapper_checksum_tmp}" "${WRAPPER_CHECKSUM_PATH}"
mv -f -- "${rollback_checksum_tmp}" "${ROLLBACK_CHECKSUM_PATH}"

echo "installed=${WRAPPER_PATH}"
echo "rollback=${ROLLBACK_PATH}"
echo "hint=invoke ${WRAPPER_PATH} with the same arguments as rocprofv3"
