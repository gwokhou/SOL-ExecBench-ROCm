#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Thin wrapper that locks GPU/DRAM clocks before the Python server starts
# and guarantees unlock on exit (even SIGTERM/crash).  All real logic lives
# in sol_execbench.core.bench.clock_lock.
# ---------------------------------------------------------------------------

acquire_gpu_lock() {
    local lock_dir="${SOL_EXECBENCH_GPU_LOCK_DIR:-/tmp/sol-execbench-locks}"
    local lock_path="${lock_dir}/gpu-0.lock"
    local timeout="${SOL_EXECBENCH_GPU_LOCK_TIMEOUT_SECONDS:-60}"
    mkdir -p "${lock_dir}"
    exec 9>>"${lock_path}"
    if ! flock -w "${timeout}" 9; then
        echo "ERROR: GPU 0 is busy; lock wait exceeded ${timeout} seconds" >&2
        return 75
    fi
    printf 'entrypoint_pid=%s\n' "$$" >"${lock_path}"
    export SOL_EXECBENCH_GPU_LOCK_FD=9
}

lock_clocks() {
    local status
    status="$(python - <<'ROCPY'
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
from sol_execbench.core.bench.clock_lock import acquire_clock_lock

clock_lock = acquire_clock_lock()
if not clock_lock.locked:
    print('WARNING: Clock locking unavailable — continuing unlocked')
    print('CLOCKS_LOCKED=0')
    print('CLOCK_LOCK_ACQUIRED=0')
    raise SystemExit(0)

print('CLOCKS_LOCKED=1')
print(f'CLOCK_LOCK_ACQUIRED={int(clock_lock.acquired)}', flush=True)
if clock_lock.acquired:
    clock_lock.detach()
ROCPY
)"
    if echo "$status" | grep -q 'CLOCKS_LOCKED=1'; then
        export SOL_EXECBENCH_CLOCKS_LOCKED=1
    else
        export SOL_EXECBENCH_CLOCKS_LOCKED=0
    fi
    if echo "$status" | grep -q 'CLOCK_LOCK_ACQUIRED=1'; then
        export SOL_EXECBENCH_CLOCK_LOCK_ACQUIRED=1
    else
        export SOL_EXECBENCH_CLOCK_LOCK_ACQUIRED=0
    fi
    printf '%s\n' "$status"
}

cleanup() {
    if [ "${SOL_EXECBENCH_CLOCK_LOCK_ACQUIRED}" = "1" ]; then
        python -c '
from sol_execbench.core.bench.clock_lock import unlock_clocks
print("Unlocking clocks...")
if not unlock_clocks():
    raise SystemExit("Clock reset to AUTO failed or could not be verified")
print("Clocks unlocked")
'
    fi
}

export SOL_EXECBENCH_CLOCKS_LOCKED=0
export SOL_EXECBENCH_CLOCK_LOCK_ACQUIRED=0
export SOL_EXECBENCH_SANDBOXED=1
trap 'cleanup' EXIT
trap 'exit' TERM INT

# check if flashinfer-trace directory is mounted
if [ ! -d "${FLASHINFER_TRACE_DIR}" ]; then
    echo "WARNING: FLASHINFER_TRACE_DIR is not mounted"
    echo "         Continuing without flashinfer-trace; dependency smoke tests are still allowed."
fi

acquire_gpu_lock
lock_clocks

"$@"
