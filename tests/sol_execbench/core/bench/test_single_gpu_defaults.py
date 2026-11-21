# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for boundary-only single-GPU host defaults."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from sol_execbench.core.bench import single_gpu_defaults
from sol_execbench.core.bench.clock_ownership import ClockDevice

BDF = "0000:05:00.0"
UNIQUE_ID = "0000000000000001"
ROCR_UUID = f"GPU-{UNIQUE_ID}"
SMI_UUID = "00000000-0000-1000-8000-000000000001"


def _make_pci_root(
    tmp_path: Path, entries: tuple[tuple[str, str], ...] = ((BDF, UNIQUE_ID),)
) -> Path:
    pci_root = tmp_path / "pci"
    for bdf, uid in entries:
        device_dir = pci_root / bdf
        device_dir.mkdir(parents=True)
        (device_dir / "unique_id").write_text(f"{uid}\n", encoding="utf-8")
    return pci_root


def test_derives_all_host_defaults_for_unique_gpu(tmp_path, monkeypatch):
    pci_root = _make_pci_root(tmp_path)
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))

    derived = single_gpu_defaults.derive_single_gpu_host_defaults(
        {},
        pci_root=pci_root,
        device_observer=lambda: (ClockDevice(bdf=BDF, uuid=SMI_UUID),),
    )

    expected_lock_dir = str(fake_home / ".local" / "state")
    assert derived == {
        "GPU_LOCK_DIR": expected_lock_dir,
        "SOL_EXECBENCH_CLOCK_STATE_DIR": str(
            Path(expected_lock_dir) / "gpu-leases" / "sol"
        ),
        "GPU_LOCK_UUID": SMI_UUID,
        "ROCR_VISIBLE_DEVICES": ROCR_UUID,
    }


def test_uses_xdg_state_home_when_absolute(tmp_path):
    pci_root = _make_pci_root(tmp_path)
    state_home = tmp_path / "xdg-state"

    derived = single_gpu_defaults.derive_single_gpu_host_defaults(
        {"XDG_STATE_HOME": str(state_home)},
        pci_root=pci_root,
        device_observer=lambda: (ClockDevice(bdf=BDF, uuid=SMI_UUID),),
    )

    assert derived["GPU_LOCK_DIR"] == str(state_home)
    assert derived["SOL_EXECBENCH_CLOCK_STATE_DIR"] == str(
        state_home / "gpu-leases" / "sol"
    )


def test_partial_derivation_fills_missing_pair_only(tmp_path):
    pci_root = _make_pci_root(
        tmp_path,
        entries=(
            ("0000:05:00.0", "1111111111111111"),
            ("0000:c1:00.0", "2222222222222222"),
        ),
    )
    devices = (
        ClockDevice(
            bdf="0000:05:00.0",
            uuid="11111111-0000-1000-8000-000000000001",
        ),
        ClockDevice(
            bdf="0000:c1:00.0",
            uuid="22222222-0000-1000-8000-000000000002",
        ),
    )
    base_env = {
        "GPU_LOCK_DIR": str(tmp_path / "locks"),
        "SOL_EXECBENCH_CLOCK_STATE_DIR": str(tmp_path / "private"),
    }

    from_rocr = single_gpu_defaults.derive_single_gpu_host_defaults(
        {**base_env, "ROCR_VISIBLE_DEVICES": "GPU-2222222222222222"},
        pci_root=pci_root,
        device_observer=lambda: devices,
    )
    assert from_rocr == {
        "GPU_LOCK_UUID": "22222222-0000-1000-8000-000000000002"
    }

    from_uuid = single_gpu_defaults.derive_single_gpu_host_defaults(
        {
            **base_env,
            "GPU_LOCK_UUID": "11111111-0000-1000-8000-000000000001",
        },
        pci_root=pci_root,
        device_observer=lambda: devices,
    )
    assert from_uuid == {"ROCR_VISIBLE_DEVICES": "GPU-1111111111111111"}


def test_multi_gpu_without_selector_remains_fail_closed(tmp_path):
    pci_root = _make_pci_root(
        tmp_path,
        entries=(
            ("0000:05:00.0", "1111111111111111"),
            ("0000:c1:00.0", "2222222222222222"),
        ),
    )
    devices = (
        ClockDevice(
            bdf="0000:05:00.0",
            uuid="11111111-0000-1000-8000-000000000001",
        ),
        ClockDevice(
            bdf="0000:c1:00.0",
            uuid="22222222-0000-1000-8000-000000000002",
        ),
    )

    derived = single_gpu_defaults.derive_single_gpu_host_defaults(
        {"GPU_LOCK_DIR": str(tmp_path / "locks")},
        pci_root=pci_root,
        device_observer=lambda: devices,
    )

    assert "GPU_LOCK_UUID" not in derived
    assert "ROCR_VISIBLE_DEVICES" not in derived


def test_container_and_host_managed_envs_are_left_untouched(tmp_path):
    pci_root = _make_pci_root(tmp_path)
    for blocked_env in (
        {"GPU_LOCK_HELD_BY_HOST": "1"},
        {"SOL_EXECBENCH_SANDBOXED": "1"},
        {"GPU_LOCK_FD": "9"},
        {"SOL_EXECBENCH_GPU_LOCK_DIR": "/tmp/legacy"},
    ):
        assert (
            single_gpu_defaults.derive_single_gpu_host_defaults(
                blocked_env,
                pci_root=pci_root,
                device_observer=lambda: (ClockDevice(bdf=BDF, uuid=SMI_UUID),),
            )
            == {}
        )


def test_context_manager_restores_process_environment(tmp_path, monkeypatch):
    pci_root = _make_pci_root(tmp_path)
    monkeypatch.setattr(single_gpu_defaults, "PCI_DEVICES_ROOT", pci_root)
    monkeypatch.setattr(
        single_gpu_defaults,
        "observe_clock_devices",
        lambda: (ClockDevice(bdf=BDF, uuid=SMI_UUID),),
    )
    for key in (
        "GPU_LOCK_DIR",
        "GPU_LOCK_UUID",
        "ROCR_VISIBLE_DEVICES",
        "SOL_EXECBENCH_CLOCK_STATE_DIR",
    ):
        monkeypatch.delenv(key, raising=False)

    with single_gpu_defaults.host_single_gpu_defaults() as derived:
        assert derived["GPU_LOCK_UUID"] == SMI_UUID
        assert os.environ["GPU_LOCK_UUID"] == SMI_UUID
        assert os.environ["ROCR_VISIBLE_DEVICES"] == ROCR_UUID

    assert "GPU_LOCK_UUID" not in os.environ
    assert "ROCR_VISIBLE_DEVICES" not in os.environ


def test_cli_prepare_locks_creates_regular_files_and_prints_lines(
    tmp_path, monkeypatch, capsys
):
    pci_root = _make_pci_root(tmp_path)
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(single_gpu_defaults, "PCI_DEVICES_ROOT", pci_root)
    monkeypatch.setattr(
        single_gpu_defaults,
        "observe_clock_devices",
        lambda: (ClockDevice(bdf=BDF, uuid=SMI_UUID),),
    )
    for key in (
        "GPU_LOCK_DIR",
        "GPU_LOCK_UUID",
        "ROCR_VISIBLE_DEVICES",
        "SOL_EXECBENCH_CLOCK_STATE_DIR",
        "XDG_STATE_HOME",
    ):
        monkeypatch.delenv(key, raising=False)

    exit_code = single_gpu_defaults.main(["--prepare-locks"])

    assert exit_code == 0
    lines = dict(
        line.split("=", 1) for line in capsys.readouterr().out.splitlines()
    )
    lock_dir = Path(lines["GPU_LOCK_DIR"])
    assert (
        stat.S_IMODE(
            (lock_dir / f"gpu-{SMI_UUID}.L1-ownership.lock").stat().st_mode
        )
        == 0o600
    )
    assert (lock_dir / f"gpu-{SMI_UUID}.L2-execution.lock").is_file()
    assert Path(lines["SOL_EXECBENCH_CLOCK_STATE_DIR"]).is_dir()


def test_ignores_igpu_without_pci_unique_id_when_single_dgpu_present(
    tmp_path, monkeypatch
):
    pci_root = _make_pci_root(tmp_path)
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))

    derived = single_gpu_defaults.derive_single_gpu_host_defaults(
        {},
        pci_root=pci_root,
        device_observer=lambda: (
            ClockDevice(bdf=BDF, uuid=SMI_UUID),
            ClockDevice(
                bdf="0000:14:00.0",
                uuid="99999999-0000-1000-8000-000000000099",
            ),
        ),
    )

    assert derived == {
        "GPU_LOCK_DIR": str(fake_home / ".local" / "state"),
        "SOL_EXECBENCH_CLOCK_STATE_DIR": str(
            fake_home / ".local" / "state" / "gpu-leases" / "sol"
        ),
        "GPU_LOCK_UUID": SMI_UUID,
        "ROCR_VISIBLE_DEVICES": ROCR_UUID,
    }


def test_derives_partitioned_kfd_topology_pair(tmp_path):
    pci_root = _make_pci_root(
        tmp_path, entries=(("0000:05:00.0", "aaaa000000000001"),)
    )
    kfd_root = tmp_path / "kfd-nodes"
    for idx, uid_int in ((1, 0xAAAA000000000001), (2, 0xAAAA000000000002)):
        node = kfd_root / str(idx)
        node.mkdir(parents=True)
        (node / "properties").write_text(
            f"simd_count 64\ndomain 0\nlocation_id 1280\nunique_id {uid_int}\n",
            encoding="utf-8",
        )
    devices = (
        ClockDevice(
            bdf="0000:05:00.0",
            uuid="11111111-0000-1000-8000-000000000001",
        ),
        ClockDevice(
            bdf="0000:05:00.0",
            uuid="11111111-0000-1000-8000-000000000002",
        ),
    )
    base_env = {
        "GPU_LOCK_DIR": str(tmp_path / "locks"),
        "SOL_EXECBENCH_CLOCK_STATE_DIR": str(tmp_path / "private"),
    }

    from_uuid = single_gpu_defaults.derive_single_gpu_host_defaults(
        {
            **base_env,
            "GPU_LOCK_UUID": "11111111-0000-1000-8000-000000000002",
        },
        pci_root=pci_root,
        kfd_root=kfd_root,
        device_observer=lambda: devices,
    )
    assert from_uuid == {"ROCR_VISIBLE_DEVICES": "GPU-aaaa000000000002"}

    from_rocr = single_gpu_defaults.derive_single_gpu_host_defaults(
        {**base_env, "ROCR_VISIBLE_DEVICES": "GPU-aaaa000000000002"},
        pci_root=pci_root,
        kfd_root=kfd_root,
        device_observer=lambda: devices,
    )
    assert from_rocr == {
        "GPU_LOCK_UUID": "11111111-0000-1000-8000-000000000002"
    }
