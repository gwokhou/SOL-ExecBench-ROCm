"""Tests for cloud GPU preflight twin, connectivity probe, one-shot bundle, and evidence gate."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import socket
import tarfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.mvp_v1.cloud_preflight import (
    ConnectivityVerdict,
    build_oneshot_cloud_script,
    list_local_ssh_fingerprints,
    main as preflight_main,
    probe_remote_connectivity,
    validate_downloaded_bundle,
    verify_python312_twin,
)
from scripts.mvp_v1.cloud_runner import main as runner_main
from scripts.mvp_v1.report import compare_summaries_markdown
from scripts.mvp_v1.workloads import SweepMode
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
    mi_dir = tmp_path / "mi300x"
    b2_dir = tmp_path / "b200"
    assert (
        runner_main(
            [
                "--mode",
                "smoke",
                "--device",
                "cpu",
                "--target-override",
                "MI300X",
                "--output-dir",
                str(mi_dir),
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
                "--target-override",
                "B200",
                "--output-dir",
                str(b2_dir),
            ]
        )
        == 0
    )
    from_json = compare_summaries_markdown(
        mi_dir / "summary.json", b2_dir / "summary.json"
    )
    from_md = compare_summaries_markdown(
        mi_dir / "report.md", b2_dir / "report.md"
    )
    assert from_md == from_json

    saved_mi300x = Path("data/outputs/mi300x_devcloud/report.md")
    if saved_mi300x.is_file():
        hyphen_cmp = compare_summaries_markdown(saved_mi300x, saved_mi300x)
        assert "MI300X vs MI300X" in hyphen_cmp
        assert "gqa_paged_decode_bf16_b16_kv32768" in hyphen_cmp
