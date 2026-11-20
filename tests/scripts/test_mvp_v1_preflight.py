"""Tests for cloud GPU preflight twin, connectivity probe, one-shot bundle, and evidence gate."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import socket
import tarfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.mvp_v1.analytical_roofline import (
    EmpiricalCeiling,
    VendorBackend,
    load_hardware_ceiling,
)
from scripts.mvp_v1.anti_distortion import DVFSWindowReceipt, TelemetrySample
from scripts.mvp_v1.cloud_preflight import (
    ConnectivityVerdict,
    build_oneshot_cloud_script,
    list_local_ssh_fingerprints,
    main as preflight_main,
    probe_remote_connectivity,
    validate_downloaded_bundle,
    verify_python312_twin,
)
from scripts.mvp_v1.cloud_runner import (
    main as runner_main,
    run_benchmark_campaign,
)
from scripts.mvp_v1.kernels import KernelBackend
from scripts.mvp_v1.report import (
    WorkloadMeasurementRow,
    compare_summaries_markdown,
    write_mvp_outputs,
)
from scripts.mvp_v1.workloads import SweepMode
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded


@contextmanager
def _loopback_server(
    handler: Callable[[socket.socket], None],
) -> Iterator[int]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = int(srv.getsockname()[1])
    stop = threading.Event()

    def _serve() -> None:
        srv.settimeout(0.05)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=handler, args=(conn,), daemon=True).start()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        stop.set()
        srv.close()
        thread.join(timeout=1.0)


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_python312_twin_verifies_mvp_and_cdna3_imports() -> None:
    res = verify_python312_twin(target_id="MI300X")
    assert res.passed is True
    assert res.target_id == "MI300X"
    assert res.checked_modules >= 100
    assert res.smoke_rows == 6


def test_ssh_fingerprints_and_connectivity_probe_verdicts(
    tmp_path: Path,
) -> None:
    raw_blob = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00 test-key-blob"
    expected_b64 = (
        base64.b64encode(hashlib.sha256(raw_blob).digest())
        .decode("ascii")
        .rstrip("=")
    )
    pub_file = tmp_path / "id_ed25519.pub"
    pub_file.write_text(
        f"ssh-ed25519 {base64.b64encode(raw_blob).decode('ascii')} m1@local\n",
        encoding="utf-8",
    )
    fps = list_local_ssh_fingerprints(tmp_path)
    assert fps == (f"id_ed25519.pub: SHA256:{expected_b64} (m1@local)",)

    def _ssh_banner_handler(conn: socket.socket) -> None:
        with conn:
            conn.sendall(b"SSH-2.0-OpenSSH_9.6_Mock\r\n")

    def _silent_stall_handler(conn: socket.socket) -> None:
        with conn:
            conn.settimeout(0.8)
            try:
                conn.recv(256)
                threading.Event().wait(0.8)
            except OSError:
                pass

    def _http_ok_handler(conn: socket.socket) -> None:
        with conn:
            conn.recv(512)
            conn.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")

    closed_port = _unused_loopback_port()
    with (
        _loopback_server(_ssh_banner_handler) as ssh_ready_port,
        _loopback_server(_silent_stall_handler) as ssh_stall_port,
        _loopback_server(_http_ok_handler) as http_open_port,
    ):
        ready = probe_remote_connectivity(
            "127.0.0.1",
            ssh_port=ssh_ready_port,
            http_port=http_open_port,
            timeout_s=0.4,
            ssh_dir=tmp_path,
        )
        assert ready.verdict is ConnectivityVerdict.SSH_READY
        assert ready.ssh_banner_received is True
        assert ready.http_reachable is True
        assert ready.local_ssh_fingerprints == fps

        blocked = probe_remote_connectivity(
            "127.0.0.1",
            ssh_port=ssh_stall_port,
            http_port=http_open_port,
            timeout_s=0.25,
            ssh_dir=tmp_path,
        )
        assert blocked.verdict is (
            ConnectivityVerdict.SSH_BANNER_BLOCKED_HTTP_OPEN
        )
        assert blocked.ssh_tcp_connected is True
        assert blocked.ssh_banner_received is False
        assert blocked.http_reachable is True

        dead = probe_remote_connectivity(
            "127.0.0.1",
            ssh_port=closed_port,
            http_port=closed_port,
            timeout_s=0.25,
            ssh_dir=tmp_path,
        )
        assert dead.verdict is ConnectivityVerdict.UNREACHABLE


def test_oneshot_cloud_script_end_to_end_cpu_smoke_and_verify(
    tmp_path: Path,
) -> None:
    import sys

    script_path = tmp_path / "oneshot.sh"
    out_dir = tmp_path / "cloud_out"
    build_oneshot_cloud_script(
        script_path,
        mode=SweepMode.SMOKE,
        remote_out_dir=str(out_dir),
    )
    assert os.access(script_path, os.X_OK)

    env = dict(os.environ)
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "MVP_MODE": "smoke",
            "MVP_DEVICE": "cpu",
            "MVP_COUNTERS_FLAG": "--no-collect-counters",
            "MVP_CLOCKS_FLAG": "--no-lock-clocks",
            "MVP_EXTRA_ARGS": "--target-override MI300X",
            "MVP_OUTPUT_DIR": str(out_dir),
        }
    )
    completed = run_in_process_group_bounded(
        ["bash", str(script_path)],
        timeout=120.0,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    match = re.search(
        r"=== MVP_BUNDLE_SHA256 ([0-9a-f]{64}) (\S+) ===",
        completed.stdout,
    )
    assert match is not None
    expected_sha, tar_path = match.group(1), Path(match.group(2))
    assert "=== MVP_REPORT_BEGIN ===" in completed.stdout
    assert "AMD Instinct MI300X" in completed.stdout

    receipt = validate_downloaded_bundle(tar_path, expected_sha256=expected_sha)
    assert receipt.target_id == "MI300X"
    assert receipt.row_count == 6
    assert receipt.all_passed is True
    assert receipt.sha256 == expected_sha

    dir_receipt = validate_downloaded_bundle(out_dir)
    assert dir_receipt.row_count == 6
    assert dir_receipt.all_passed is True
    assert (
        preflight_main(
            ["verify", str(tar_path), "--expected-sha256", expected_sha]
        )
        == 0
    )


def test_evidence_gate_rejects_html_forbidden_and_sha_mismatch(
    tmp_path: Path,
) -> None:
    html_file = tmp_path / "login.tar.gz"
    html_file.write_text("<!DOCTYPE HTML><html>login</html>", encoding="utf-8")
    with pytest.raises(ValueError, match="HTML/JSON error page"):
        validate_downloaded_bundle(html_file)

    json_file = tmp_path / "forbidden.tar.gz"
    json_file.write_text('{"message": "Forbidden"}', encoding="utf-8")
    with pytest.raises(ValueError, match="HTML/JSON error page"):
        validate_downloaded_bundle(json_file)

    valid_tar = tmp_path / "empty_valid.tar.gz"
    dummy = tmp_path / "note.txt"
    dummy.write_text("hello", encoding="utf-8")
    with tarfile.open(valid_tar, mode="w:gz") as tf:
        tf.add(dummy, arcname="note.txt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_downloaded_bundle(valid_tar, expected_sha256="0" * 64)


def test_compare_summaries_markdown_accepts_report_md_and_json_equivalently(
    tmp_path: Path,
) -> None:
    pt_dir = tmp_path / "pytorch"
    port_dir = tmp_path / "portable"
    assert (
        runner_main(
            [
                "--mode",
                "smoke",
                "--device",
                "cpu",
                "--backend",
                "pytorch",
                "--target-override",
                "MI300X",
                "--output-dir",
                str(pt_dir),
            ]
        )
        == 0
    )
    assert (
        runner_main(
            [
                "--mode",
                "smoke",
                "--device",
                "cpu",
                "--backend",
                "portable",
                "--target-override",
                "MI300X",
                "--output-dir",
                str(port_dir),
            ]
        )
        == 0
    )
    from_json = compare_summaries_markdown(
        pt_dir / "summary.json", port_dir / "summary.json"
    )
    from_md = compare_summaries_markdown(
        pt_dir / "report.md", port_dir / "report.md"
    )
    assert from_md == from_json
    assert (
        "ROCm Vertical Stack SOL Diagnosis — MI300X (pytorch vs portable)"
        in (from_json)
    )

    saved_mi300x = Path("data/outputs/mi300x_devcloud/report.md")
    if saved_mi300x.is_file():
        hyphen_cmp = compare_summaries_markdown(saved_mi300x, saved_mi300x)
        assert "ROCm Vertical Stack SOL Diagnosis — MI300X" in hyphen_cmp
        assert "gqa_paged_decode_bf16_b16_kv32768" in hyphen_cmp


def test_explicit_tier_unavailable_and_three_way_vertical_diagnosis(
    tmp_path: Path,
    capsys,
) -> None:
    for backend in (KernelBackend.TIER2_VLLM, KernelBackend.TIER3_AITER):
        unavail_paths = run_benchmark_campaign(
            mode=SweepMode.SMOKE,
            device="cpu",
            target_override="MI300X",
            backend=backend,
            output_dir=tmp_path / f"{backend.value}_unavail",
            collect_counters=False,
            lock_clocks=False,
        )
        unavail_summary = json.loads(
            unavail_paths["summary_json"].read_text(encoding="utf-8")
        )
        assert unavail_summary["backend"] == backend.value
        assert all(
            r["implementation"].startswith("unavailable:")
            and r["correctness_passed"] is False
            and r["counter_status"] == "skipped_unavailable"
            for r in unavail_summary["rows"]
        )
        assert "| - |" in unavail_paths["report_md"].read_text("utf-8")
        traces = [
            Trace.model_validate_json(line)
            for line in unavail_paths["traces_jsonl"]
            .read_text("utf-8")
            .splitlines()
            if line.strip()
        ]
        assert all(
            t.evaluation is not None and not t.is_successful() for t in traces
        )

    mi300x = load_hardware_ceiling("MI300X")
    dvfs_paths = run_benchmark_campaign(
        mode=SweepMode.SMOKE,
        device="cpu",
        target_override="MI300X",
        backend=KernelBackend.PYTORCH,
        output_dir=tmp_path / "seed",
        collect_counters=False,
        lock_clocks=False,
    )
    seed_payload = json.loads(dvfs_paths["summary_json"].read_text("utf-8"))
    tier_dirs = []
    specs_map = {
        "w_gap": (0.10, 0.08, 0.04, 60.0, 1.05, "aiter_ok"),
        "w_low": (0.20, 0.19, 0.18, 12.0, 1.80, "aiter_amp"),
        "w_miss": (0.30, 0.25, None, None, None, "unavailable:aiter_gqa"),
        "w_ok": (0.12, 0.06, 0.06, 72.0, 1.02, "aiter_fast"),
    }
    emp_c = EmpiricalCeiling(
        hbm_bandwidth_bytes_per_s=seed_payload["empirical_ceiling"][
            "hbm_bandwidth_bytes_per_s"
        ],
        bf16_flops_per_s=seed_payload["empirical_ceiling"]["bf16_flops_per_s"],
    )
    dvfs_r = DVFSWindowReceipt(
        lock_requested=False,
        lock_acquired=False,
        lock_note="telemetry_only",
        before=TelemetrySample(vendor=VendorBackend.CPU),
    )
    for tier_name, idx in (("portable", 0), ("vllm", 1), ("aiter", 2)):
        rows = []
        for wid, (m0, m1, m2, emp, amp, impl) in specs_map.items():
            med = (m0, m1, m2)[idx]
            is_unavail = med is None
            rows.append(
                WorkloadMeasurementRow(
                    workload_id=wid,
                    operator="skinny_gemm",
                    precision="bf16",
                    implementation=impl if idx == 2 else f"{tier_name}_impl",
                    correctness_passed=not is_unavail,
                    max_abs_err=0.0,
                    max_rel_err=0.0,
                    median_ms=0.0 if is_unavail else float(med),
                    mean_ms=0.0 if is_unavail else float(med),
                    p90_ms=0.0 if is_unavail else float(med),
                    flops=1e9,
                    compulsory_bytes=1e6,
                    arithmetic_intensity=10.0,
                    t_sol_ms_spec=0.02,
                    sol_pct_spec=0.0 if is_unavail else 50.0,
                    bottleneck_spec="memory",
                    t_sol_ms_empirical=0.024,
                    sol_pct_empirical=emp if idx == 2 else 25.0,
                    bottleneck_empirical="memory",
                    counter_status="collected"
                    if not is_unavail
                    else "skipped_unavailable",
                    observed_mem_bytes=1e6,
                    traffic_amplification=amp if idx == 2 else 1.05,
                    mfma_instructions=None,
                )
            )
        tier_dirs.append(
            write_mvp_outputs(
                tmp_path / tier_name,
                ceiling=mi300x,
                empirical=emp_c,
                dvfs=dvfs_r,
                rows=rows,
                traces=[],
                backend_name=tier_name,
            )
        )

    from_json = compare_summaries_markdown(
        *(d["summary_json"] for d in tier_dirs)
    )
    from_md = compare_summaries_markdown(*(d["report_md"] for d in tier_dirs))
    assert from_json == from_md
    assert (
        "ROCm Vertical Stack SOL Diagnosis — MI300X (portable vs vllm vs aiter)"
        in (from_json)
    )
    assert "DISPATCH_GAP (2.00x)" in from_json
    assert "LOW_EMP_SOL+HIGH_AMP" in from_json
    assert "TIER_UNAVAILABLE(aiter)" in from_json
    assert "| `w_ok` |" in from_json
    assert "OK" in from_json

    capsys.readouterr()
    assert (
        runner_main(["--compare", *(str(d["summary_json"]) for d in tier_dirs)])
        == 0
    )
    assert "DISPATCH_GAP (2.00x)" in capsys.readouterr().out
