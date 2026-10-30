#!/bin/bash
set -e

# Consume the host's native UUID owner. The host retains cleanup/restoration;
# this entrypoint only checks its declared performance policy before exec.
verify_host_managed_clocks() {
    if python - "${HOST_CLOCKS_DECLARED}" <<'ROCPY'
import sys
from sol_execbench.core.bench.clock_lock import observe_clock_levels
from sol_execbench.core.bench.gpu_lock_paths import configured_gpu_locks

expected = {"0": "AMDSMI_DEV_PERF_LEVEL_AUTO", "1": "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"}.get(sys.argv[1])
if expected is None or observe_clock_levels(configured_gpu_locks().uuid) != (expected,):
    raise SystemExit(1)
ROCPY
    then
        export SOL_EXECBENCH_CLOCKS_LOCKED="${HOST_CLOCKS_DECLARED}"
        echo "CLOCKS_LOCKED=${SOL_EXECBENCH_CLOCKS_LOCKED}"
        return
    fi
    echo "ERROR: native host clock policy failed container verification" >&2
    return 1
}

HOST_CLOCKS_DECLARED="${SOL_EXECBENCH_CLOCKS_LOCKED:-0}"
export SOL_EXECBENCH_CLOCKS_LOCKED=0
export SOL_EXECBENCH_SANDBOXED=1
# This container consumes the existing native host owner; it never creates a
# private lock inode or performs an all-device clock write.
python - <<'ROCPY'
from sol_execbench.core.bench.gpu_lock import verify_host_gpu_owner
from sol_execbench.core.bench.gpu_lock_paths import configured_gpu_locks, prepare_gpu_locks
paths = configured_gpu_locks()
if not paths.host_managed:
    raise SystemExit("Use scripts/run_docker.sh with a native host GPU owner")
prepare_gpu_locks(paths)
verify_host_gpu_owner(paths)
ROCPY

# check if flashinfer-trace directory is mounted
if [ ! -d "${FLASHINFER_TRACE_DIR}" ]; then
    echo "WARNING: FLASHINFER_TRACE_DIR is not mounted"
    echo "         Continuing without flashinfer-trace; dependency smoke tests are still allowed."
fi

if [ "${SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST:-0}" != "1" ]; then
    echo "ERROR: native host clock ownership is required" >&2
    exit 1
fi
verify_host_managed_clocks
exec "$@"
