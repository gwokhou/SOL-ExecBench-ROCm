# Phase 175: PID Lock Module - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 5
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/bench/pid_lock.py` | utility | request-response | `src/sol_execbench/core/bench/clock_lock.py` | exact |
| `tests/sol_execbench/core/bench/test_pid_lock.py` | test | CRUD | `tests/sol_execbench/core/bench/test_clock_lock.py` | exact |
| `scripts/run_rdna4_profiler_timing_batch.py` | script | batch-processing | `scripts/run_rdna4_profiler_timing_batch.py` | self-integration |
| `scripts/run_rdna4_profiler_overhead_calibration.py` | script | batch-processing | `scripts/run_rdna4_profiler_timing_batch.py` | role-match |
| `scripts/run_derived_isolated.py` | script | batch-processing | `scripts/run_derived_isolated.py` | self-integration |

## Pattern Assignments

### `src/sol_execbench/core/bench/pid_lock.py` (utility, request-response)

**Analog:** `src/sol_execbench/core/bench/clock_lock.py`

**Imports pattern** (lines 24-31):
```python
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)
```

**Module constants pattern** (lines 33-38):
```python
VERIFY_DELAY_S = 3
AMD_SMI_FAILURE_MARKERS = (
    "unable to set performance level",
    "failed to set",
    "error:",
)
```

**Function structure pattern** (lines 78-87):
```python
def probe_clock_lock_available() -> bool:
    """Probe whether ``amd-smi`` is available via passwordless sudo."""
    try:
        probe = subprocess.run(
            ["sudo", "-n", _amd_smi_executable(), "version"],
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    return probe.returncode == 0
```

**Error handling pattern** (lines 117-119):
```python
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    logger.warning("Failed to lock clocks (STABLE_PEAK): %s", e)
    return False
```

**Best-effort cleanup pattern** (lines 172-178):
```python
def unlock_clocks() -> None:
    """Reset GPU clocks to auto via ``amd-smi set -l AUTO``.

    Best-effort; errors are logged but not raised.
    """
    try:
        subprocess.run(
            ["sudo", "-n", _amd_smi_executable(), "set", "-l", "AUTO"],
            capture_output=True,
        )
    except Exception as e:
        logger.warning("Failed to unlock clocks: %s", e)
```

**Environment variable check pattern** (lines 181-184):
```python
def are_clocks_locked() -> bool:
    """Check whether clocks were locked successfully before evaluation."""
    return os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
```

---

### `tests/sol_execbench/core/bench/test_pid_lock.py` (test, CRUD)

**Analog:** `tests/sol_execbench/core/bench/test_clock_lock.py`

**Imports pattern** (lines 7-18):
```python
from unittest.mock import MagicMock, patch

import pytest

from sol_execbench.core.bench import clock_lock_module
from sol_execbench.core.bench.clock_lock import (
    are_clocks_locked,
    lock_clocks,
    probe_clock_lock_available,
    unlock_clocks,
    verify_clocks,
)
```

**Test fixture pattern** (lines 28-33):
```python
@pytest.fixture(autouse=True)
def _mock_tool_paths(monkeypatch):
    def _which(tool):
        return {"rocm-smi": "rocm-smi", "amd-smi": "amd-smi"}.get(tool)

    monkeypatch.setattr(clock_lock_module.shutil, "which", _which)
```

**Test class structure pattern** (lines 36-55):
```python
class TestProbeClockLockAvailable:
    def test_returns_true_when_amd_smi_version_succeeds(self):
        probe_result = MagicMock(returncode=0)
        with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
            result = probe_clock_lock_available()

        assert result is True
        mock_run.assert_called_once_with(
            ["sudo", "-n", "amd-smi", "version"], capture_output=True
        )
```

**Mock subprocess pattern** (lines 43-45):
```python
with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
    result = probe_clock_lock_available()

    assert result is True
    mock_run.assert_called_once_with(...)
```

**Subprocess-based integration test pattern** (lines 194-200):
```python
def test_does_not_raise_on_failure(self):
    with patch(f"{_MODULE}.subprocess.run", side_effect=Exception("no sudo")):
        unlock_clocks()
```

---

### `scripts/run_rdna4_profiler_timing_batch.py` (script, batch-processing)

**Analog:** Self-integration - existing script that needs lock integration

**Current main entry point** (lines 1377-1413):
```python
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_timing_dirs = (
        tuple(args.source_timing_dir)
        if args.source_timing_dir
        else (DEFAULT_SOURCE_TIMING_DIR,)
    )
    skip_problem = tuple(args.skip_problem) + _load_problem_id_file(
        args.skip_problem_file
    )
    try:
        return run_batch(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            # ... other args
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
```

**Integration point:** Add `with acquire_pid_lock(args.output_dir):` wrapper around `run_batch()` call, immediately after argument parsing.

**Error handling pattern** (lines 1410-1412):
```python
except (OSError, ValueError, json.JSONDecodeError) as exc:
    print(f"ERROR: {exc}")
    return 2
```

---

### `scripts/run_rdna4_profiler_overhead_calibration.py` (script, batch-processing)

**Analog:** `scripts/run_rdna4_profiler_timing_batch.py`

**Status:** ⚠️ **SCRIPT MISSING** - Only `.pyc` bytecode exists in codebase

**Expected pattern** (based on timing_batch analog):
```python
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    
    # MANDATORY: Acquire lock before any work
    with acquire_pid_lock(args.output_dir):
        return run_calibration(
            output_dir=args.output_dir,
            # ... other args
        )
```

**Planner note:** Script file doesn't exist in source tree. Integration task should either:
1. Skip this integration if script is deprecated
2. Note that script will be created in future phase
3. Create placeholder stub if script is expected to exist

---

### `scripts/run_derived_isolated.py` (script, batch-processing)

**Analog:** Self-integration - existing script that needs optional lock

**Current argparse pattern** (lines 240-281):
```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    # ... existing args ...
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args(argv)
```

**Current main entry point** (lines 284-319):
```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = discover_problems(args.benchmark_dir, args.category)
    problem_id_filter = load_problem_id_filter(args.problem_id_file)
    completed = load_completed(args.status_jsonl) if args.resume else set()
    failures = 0

    for problem_dir in problems:
        # ... problem processing logic ...
        if status.status != "ok":
            failures += 1
            if not args.continue_on_failure:
                return status.returncode or 1
    return 1 if failures else 0
```

**Integration point:** 
1. Add `--pid-lock` argparse flag in `parse_args()`
2. Conditionally wrap main loop in `with acquire_pid_lock(args.output_dir):` based on flag

**Optional lock pattern** (based on research):
```python
from contextlib import nullcontext

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    
    # OPTIONAL: Conditionally acquire lock
    if args.pid_lock:
        acquire = acquire_pid_lock(args.output_dir)
    else:
        acquire = nullcontext()  # No-op context manager
    
    with acquire:
        # Existing script logic
        problems = discover_problems(args.benchmark_dir, args.category)
        # ... rest of processing ...
```

---

## Shared Patterns

### Module License Header
**Source:** All files in `src/sol_execbench/`
**Apply to:** All new source files
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### Future Imports
**Source:** All Python files in codebase
**Apply to:** All new Python files
```python
from __future__ import annotations
```

### Logging Pattern
**Source:** `clock_lock.py` line 31
**Apply to:** `pid_lock.py`
```python
import logging
logger = logging.getLogger(__name__)
```

### Path Type Annotations
**Source:** All modern Python files in codebase
**Apply to:** All new Python files
```python
from pathlib import Path
def function(output_dir: Path) -> bool:
```

### Context Manager Cleanup
**Source:** `clock_lock.py` unlock pattern (lines 167-178)
**Apply to:** `pid_lock.py` acquire_pid_lock function
```python
# Best-effort cleanup with try/finally
try:
    # ... acquire lock ...
    yield
except BlockingIOError:
    # ... handle contention ...
    sys.exit(1)
finally:
    fd.close()  # Kernel auto-releases on close
```

### Argparse Help Text Pattern
**Source:** `run_derived_isolated.py` (lines 261-269)
**Apply to:** `run_derived_isolated.py` --pid-lock flag
```python
parser.add_argument(
    "--pid-lock",
    action="store_true",
    help="Acquire exclusive process lock to prevent concurrent runs",
)
```

### Error Exit Pattern
**Source:** `run_rdna4_profiler_timing_batch.py` (lines 1410-1412)
**Apply to:** All script integrations
```python
except (OSError, ValueError, json.JSONDecodeError) as exc:
    print(f"ERROR: {exc}")
    return 2
```

### Subprocess Mocking in Tests
**Source:** `test_clock_lock.py` (lines 38-45)
**Apply to:** `test_pid_lock.py` subprocess-based tests
```python
def test_exclusive_acquire(self):
    probe_result = MagicMock(returncode=0)
    with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
        result = function_under_test()
    
    assert result is True
    mock_run.assert_called_once_with(...)
```

## No Analog Found

No files without analogs — all new files have strong matches in existing codebase.

## Metadata

**Analog search scope:** 
- `src/sol_execbench/core/bench/` (all bench utility modules)
- `scripts/` (batch processing scripts)
- `tests/sol_execbench/core/bench/` (bench module tests)

**Files scanned:** 13 bench modules, 2 scripts, 9 test files
**Pattern extraction date:** 2026-06-10

**Key pattern observations:**
1. **Bench modules use plain functions** - No classes, just functions with clear docstrings
2. **Best-effort cleanup** - Errors logged but not raised in cleanup functions
3. **Subprocess mocking** - Tests use `unittest.mock.patch` for subprocess calls
4. **Context manager integration** - Scripts use `with` statements for resource management
5. **Argparse patterns** - Scripts use `action="store_true"` for boolean flags
