from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
PATCH_DIR = ROOT / "scripts/patches/gfx1200_sq_wave_cycles"
WRAPPER = PATCH_DIR / "rocprofv3-gfx1200-patched"
INSTALL = PATCH_DIR / "install.sh"
ROLLBACK = PATCH_DIR / "rollback.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_environment(tmp_path: Path, *, initial: str = "AUTO") -> dict[str, str]:
    state = tmp_path / "perf-level"
    state.write_text(initial, encoding="utf-8")
    amd_smi = tmp_path / "amd-smi"
    rocprofv3 = tmp_path / "rocprofv3"
    sudo = tmp_path / "sudo"
    _write_executable(
        amd_smi,
        """#!/usr/bin/env bash
set -euo pipefail
state=${AMD_SMI_STATE_FILE:?}
if [[ $1 == version ]]; then echo fake-version; exit 0; fi
if [[ $1 == metric ]]; then
  if [[ ${PERF_LEVEL_OVERRIDE:-} == MIXED ]]; then
    echo "PERF_LEVEL: AMDSMI_DEV_PERF_LEVEL_AUTO"
    echo "PERF_LEVEL: AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"
  else
    echo "PERF_LEVEL: AMDSMI_DEV_PERF_LEVEL_${PERF_LEVEL_OVERRIDE:-$(cat "$state")}";
  fi
  exit 0
fi
if [[ $1 == set && $2 == -l ]]; then printf '%s' "$3" >"$state"; exit 0; fi
exit 2
""",
    )
    _write_executable(
        rocprofv3,
        """#!/usr/bin/env bash
printf 'rocprof:%s\n' "$*"
exit "${FAKE_ROCPROF_EXIT:-0}"
""",
    )
    _write_executable(
        sudo,
        """#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == -n ]] && shift
if [[ -n ${SUDO_LOG:-} ]]; then printf '%s\n' "$*" >>"${SUDO_LOG}"; fi
if [[ ${1:-} == -l ]]; then
  shift
  [[ ${1:-} == -- ]] && shift
  exit 0
fi
exec "$@"
""",
    )
    return {
        **os.environ,
        "AMD_SMI": str(amd_smi),
        "AMD_SMI_STATE_FILE": str(state),
        "ROCPROFV3_REAL": str(rocprofv3),
        "SOL_EXECBENCH_SUDO": str(sudo),
        "XDG_RUNTIME_DIR": str(tmp_path),
        "STATE_PATH": str(state),
    }


def test_wrapper_locks_runs_and_restores_auto(tmp_path: Path) -> None:
    env = _fake_environment(tmp_path)

    result = subprocess.run(
        [WRAPPER, "--kernel-trace", "--", "workload"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "rocprof:--kernel-trace -- workload" in result.stdout
    assert "patch_clock_state=acquired_stable_peak" in result.stderr
    assert Path(env["STATE_PATH"]).read_text(encoding="utf-8") == "AUTO"


def test_wrapper_preserves_external_stable_peak(tmp_path: Path) -> None:
    env = _fake_environment(tmp_path, initial="STABLE_PEAK")

    result = subprocess.run(
        [WRAPPER, "--version"], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0
    assert "patch_clock_state=preserved_external_stable_peak" in result.stderr
    assert Path(env["STATE_PATH"]).read_text(encoding="utf-8") == "STABLE_PEAK"


def test_wrapper_restores_auto_after_profiler_failure(tmp_path: Path) -> None:
    env = {**_fake_environment(tmp_path), "FAKE_ROCPROF_EXIT": "23"}

    result = subprocess.run(
        [WRAPPER, "--version"], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 23
    assert Path(env["STATE_PATH"]).read_text(encoding="utf-8") == "AUTO"


def test_wrapper_rejects_manual_state_without_changing_it(tmp_path: Path) -> None:
    env = _fake_environment(tmp_path, initial="MANUAL")

    result = subprocess.run(
        [WRAPPER, "--version"], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 78
    assert "patch_error=unsupported_initial_perf_level" in result.stderr
    assert Path(env["STATE_PATH"]).read_text(encoding="utf-8") == "MANUAL"


def test_wrapper_rejects_mixed_multi_gpu_state(tmp_path: Path) -> None:
    env = {**_fake_environment(tmp_path), "PERF_LEVEL_OVERRIDE": "MIXED"}

    result = subprocess.run(
        [WRAPPER, "--version"], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 78
    assert "patch_error=unsupported_initial_perf_level" in result.stderr


def test_embedded_help_is_complete_and_has_no_gpu_side_effect(tmp_path: Path) -> None:
    env = {
        **_fake_environment(tmp_path),
        "SOL_EXECBENCH_PATCH_HOME": str(tmp_path / "home"),
    }
    commands = (
        (WRAPPER, "--patch-help"),
        (INSTALL, "--help"),
        (ROLLBACK, "--help"),
    )

    for script, help_flag in commands:
        result = subprocess.run(
            [script, help_flag],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0
        assert "STABLE_PEAK" in result.stdout
        assert "AUTO" in result.stdout
        assert "rocprofv3-gfx1200-patched" in result.stdout
        assert "rollback-rocprofv3-gfx1200-patch" in result.stdout

    assert Path(env["STATE_PATH"]).read_text(encoding="utf-8") == "AUTO"
    assert not (tmp_path / "home/.local/bin").exists()


def test_install_and_checksum_guarded_rollback(tmp_path: Path) -> None:
    sudo_log = tmp_path / "sudo.log"
    env = {
        **_fake_environment(tmp_path),
        "SOL_EXECBENCH_PATCH_HOME": str(tmp_path / "home"),
        "SUDO_LOG": str(sudo_log),
    }
    install_result = subprocess.run(
        [INSTALL], env=env, text=True, capture_output=True, check=False
    )
    installed = tmp_path / "home/.local/bin/rocprofv3-gfx1200-patched"
    installed_rollback = tmp_path / "home/.local/bin/rollback-rocprofv3-gfx1200-patch"

    assert install_result.returncode == 0
    assert installed.is_file()
    assert installed_rollback.is_file()
    policy_queries = sudo_log.read_text(encoding="utf-8").splitlines()
    assert policy_queries == [
        f"-l -- {env['AMD_SMI']} version",
        f"-l -- {env['AMD_SMI']} set -l STABLE_PEAK",
        f"-l -- {env['AMD_SMI']} set -l AUTO",
    ]

    rollback_result = subprocess.run(
        [ROLLBACK], env=env, text=True, capture_output=True, check=False
    )

    assert rollback_result.returncode == 0
    assert not installed.exists()
    assert not installed_rollback.exists()


def test_rollback_refuses_to_remove_modified_wrapper(tmp_path: Path) -> None:
    env = {
        **_fake_environment(tmp_path),
        "SOL_EXECBENCH_PATCH_HOME": str(tmp_path / "home"),
    }
    subprocess.run([INSTALL], env=env, check=True, capture_output=True)
    installed = tmp_path / "home/.local/bin/rocprofv3-gfx1200-patched"
    with installed.open("a", encoding="utf-8") as stream:
        stream.write("# local modification\n")

    result = subprocess.run(
        [ROLLBACK], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 65
    assert "rollback_error=file_checksum_mismatch" in result.stderr
    assert installed.exists()


def test_rollback_resets_stable_peak_before_removing_patch(tmp_path: Path) -> None:
    env = {
        **_fake_environment(tmp_path),
        "SOL_EXECBENCH_PATCH_HOME": str(tmp_path / "home"),
    }
    subprocess.run([INSTALL], env=env, check=True, capture_output=True)
    Path(env["STATE_PATH"]).write_text("STABLE_PEAK", encoding="utf-8")

    result = subprocess.run(
        [ROLLBACK], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0
    assert Path(env["STATE_PATH"]).read_text(encoding="utf-8") == "AUTO"
    assert not (tmp_path / "home/.local/bin/rocprofv3-gfx1200-patched").exists()


def test_shell_scripts_pass_syntax_check() -> None:
    for script in (WRAPPER, INSTALL, ROLLBACK):
        subprocess.run(["bash", "-n", script], check=True)
