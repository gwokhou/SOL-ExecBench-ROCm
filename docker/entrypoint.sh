#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Thin wrapper that locks GPU/DRAM clocks before the Python server starts
# and guarantees unlock on exit (even SIGTERM/crash).  All real logic lives
# in sol_execbench.core.bench.clock_lock.
# ---------------------------------------------------------------------------

lock_clocks() {
    local status
    status="$(python - <<'ROCPY'
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
from sol_execbench.core.bench.clock_lock import lock_clocks

hip_version = torch.version.hip
if hip_version:
    print(f'Detected HIP backend: {hip_version}')
else:
    print('WARNING: PyTorch HIP backend not detected')

device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
if not device:
    print('WARNING: No ROCm GPU detected — continuing without clock lock')
    print('CLOCKS_LOCKED=0')
    raise SystemExit(0)

print(f'Detected GPU: {device}')
if not lock_clocks(device):
    print('WARNING: Clock locking unavailable — continuing unlocked')
    print('CLOCKS_LOCKED=0')
    raise SystemExit(0)

print('CLOCKS_LOCKED=1')
ROCPY
)"
    if echo "$status" | grep -q 'CLOCKS_LOCKED=1'; then
        export SOL_EXECBENCH_CLOCKS_LOCKED=1
    else
        export SOL_EXECBENCH_CLOCKS_LOCKED=0
    fi
    printf '%s\n' "$status"
}

cleanup() {
    if [ "${SOL_EXECBENCH_CLOCKS_LOCKED}" = "1" ]; then
        python -c '
from sol_execbench.core.bench.clock_lock import unlock_clocks
print("Unlocking clocks...")
unlock_clocks()
print("Clocks unlocked")
'
    fi
}

# check if flashinfer-trace directory is mounted
if [ ! -d "${FLASHINFER_TRACE_DIR}" ]; then
    echo "WARNING: FLASHINFER_TRACE_DIR is not mounted"
    echo "         Continuing without flashinfer-trace; dependency smoke tests are still allowed."
fi

lock_clocks
trap 'cleanup' EXIT
trap 'exit' TERM INT

"$@"
