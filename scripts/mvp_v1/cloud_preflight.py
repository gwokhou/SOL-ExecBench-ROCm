# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight twin verification, connectivity probe, one-shot bundle, and evidence gate."""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import os
import socket
import struct
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from scripts.mvp_v1.workloads import SweepMode
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.integrity import sha256_bytes, sha256_file
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

_BUNDLE_ROOTS = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "src",
    "scripts/mvp_v1",
)
_SKIP_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache"})
_SKIP_SUFFIXES = (".pyc", ".pyo")
_GZIP_MAGIC = b"\x1f\x8b"
_ERROR_PAYLOAD_PREFIXES = (b"<", b"{", b"[")

_TWIN_SUBPROCESS_SCRIPT = """
import __future__, ast, importlib.abc, importlib.machinery, importlib.util, json, pathlib, sys, tempfile

repo_root = pathlib.Path(sys.argv[1]).resolve()
target_id = sys.argv[2]
compat_path = repo_root / "scripts" / "mvp_v1" / "py312_compat.py"
compat_spec = importlib.util.spec_from_file_location("_py312_bootstrap", compat_path)
compat_mod = importlib.util.module_from_spec(compat_spec)
compat_spec.loader.exec_module(compat_mod)
rewrite_except = compat_mod.rewrite_pep758_except_clauses

class _UncachedPy312Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = pathlib.Path(self.path).read_text(encoding="utf-8")
        rewritten = rewrite_except(source)
        ast.parse(rewritten, filename=self.path, feature_version=(3, 12))
        return compile(
            rewritten,
            self.path,
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )

class _UncachedPy312Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.partition(".")[0] not in {"sol_execbench", "solar", "scripts"}:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            return None
        spec.loader = _UncachedPy312Loader(spec.loader.name, spec.loader.path)
        return spec

sys.meta_path.insert(0, _UncachedPy312Finder())

from scripts.mvp_v1 import triton_ops_ext as _triton_ops_ext
from scripts.mvp_v1 import vendor_adapters_ext as _vendor_adapters_ext
from scripts.mvp_v1.cloud_runner import main as runner_main

del _triton_ops_ext, _vendor_adapters_ext

cdna3_path = repo_root / "scripts" / "internal" / "cdna3" / "run_cdna3_diagnostic_calibration.py"
if cdna3_path.is_file():
    cdna3_loader = _UncachedPy312Loader("_cdna3_twin_check", str(cdna3_path))
    cdna3_spec = importlib.util.spec_from_loader("_cdna3_twin_check", cdna3_loader)
    cdna3_loader.exec_module(importlib.util.module_from_spec(cdna3_spec))

with tempfile.TemporaryDirectory(prefix="mvp-py312-twin-") as tmp_dir:
    rc = runner_main(
        [
            "--mode",
            "smoke",
            "--device",
            "cpu",
            "--target-override",
            target_id,
            "--output-dir",
            tmp_dir,
        ]
    )
    if rc != 0:
        raise SystemExit(rc)
    summary = json.loads((pathlib.Path(tmp_dir) / "summary.json").read_text(encoding="utf-8"))

loaded = [
    name
    for name, mod in sys.modules.items()
    if name.partition(".")[0] in {"sol_execbench", "solar", "scripts"}
    and getattr(mod, "__file__", None)
]
print(json.dumps({"checked_modules": len(loaded), "smoke_rows": len(summary["rows"])}))
"""

_ONESHOT_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(mktemp -d)"
cleanup() { rm -rf "${WORKDIR}"; }
trap cleanup EXIT

base64 -d <<'__SOL_EXECBENCH_ARCHIVE_EOF__' | tar -xzf - -C "${WORKDIR}"
__ARCHIVE_B64__
__SOL_EXECBENCH_ARCHIVE_EOF__

cd "${WORKDIR}"
find . -name "__pycache__" -exec rm -rf {} +

if [ -n "${PYTHON_BIN:-}" ]; then
    PY="${PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    PY="$(command -v python)"
fi

export PYTHONPATH="${WORKDIR}/src:${WORKDIR}${PYTHONPATH:+:${PYTHONPATH}}"
REQ_MODE="${MVP_MODE:-__DEFAULT_MODE__}"
OUT_BASE="${MVP_OUTPUT_DIR:-__DEFAULT_OUT_DIR__}"
EXTRA_ARGS=()
if [ -n "${MVP_EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS <<< "${MVP_EXTRA_ARGS}"
fi

SMOKE_OUT="${OUT_BASE}"
if [ "${REQ_MODE}" != "smoke" ]; then
    SMOKE_OUT="${OUT_BASE}_smoke"
fi

echo "=== Gate 1: Cloud Smoke Preflight ==="
PYTHON_BIN="${PY}" \\
MVP_MODE=smoke \\
MVP_DEVICE="${MVP_DEVICE:-auto}" \\
MVP_BACKEND="${MVP_BACKEND:-auto}" \\
MVP_COUNTERS_FLAG="${MVP_COUNTERS_FLAG:---collect-counters}" \\
MVP_CLOCKS_FLAG="${MVP_CLOCKS_FLAG:---lock-clocks}" \\
MVP_OUTPUT_DIR="${SMOKE_OUT}" \\
bash "${WORKDIR}/scripts/mvp_v1/run_cloud_15min.sh" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

FINAL_OUT="${SMOKE_OUT}"
if [ "${REQ_MODE}" != "smoke" ]; then
    echo "=== Gate 2: Cloud Full Capture ==="
    PYTHON_BIN="${PY}" \\
    MVP_MODE="${REQ_MODE}" \\
    MVP_DEVICE="${MVP_DEVICE:-auto}" \\
    MVP_BACKEND="${MVP_BACKEND:-auto}" \\
    MVP_COUNTERS_FLAG="${MVP_COUNTERS_FLAG:---collect-counters}" \\
    MVP_CLOCKS_FLAG="${MVP_CLOCKS_FLAG:---lock-clocks}" \\
    MVP_OUTPUT_DIR="${OUT_BASE}" \\
    bash "${WORKDIR}/scripts/mvp_v1/run_cloud_15min.sh" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    FINAL_OUT="${OUT_BASE}"
fi

FINAL_TAR="${FINAL_OUT}.tar.gz"
FINAL_SHA="$(sha256sum "${FINAL_TAR}" | awk '{print $1}')"
echo "=== MVP_BUNDLE_SHA256 ${FINAL_SHA} ${FINAL_TAR} ==="
echo "=== MVP_REPORT_BEGIN ==="
cat "${FINAL_OUT}/report.md"
echo "=== MVP_REPORT_END ==="
"""


class ConnectivityVerdict(StrEnum):
    """Classification of remote SSH vs HTTP reachability."""

    SSH_READY = "ssh_ready"
    SSH_BANNER_BLOCKED_HTTP_OPEN = "ssh_banner_blocked_http_open"
    HTTP_ONLY = "http_only"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True, slots=True, kw_only=True)
class TwinVerificationResult:
    """Outcome of uncached Python 3.12 syntax + PEP 649 twin verification."""

    target_id: str
    checked_modules: int
    smoke_rows: int
    passed: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ConnectivityReport:
    """Result of 3-second SSH/HTTP pre-flight connectivity classification."""

    host: str
    ssh_port: int
    http_port: int
    proxy: str | None
    verdict: ConnectivityVerdict
    ssh_tcp_connected: bool
    ssh_banner_received: bool
    http_reachable: bool
    local_ssh_fingerprints: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True, kw_only=True)
class BundleVerificationReceipt:
    """Hard evidence gate receipt for a downloaded or local MVP bundle."""

    bundle_path: str
    sha256: str | None
    target_id: str
    row_count: int
    all_passed: bool
    report_md_path: str
    summary_json_path: str
    traces_jsonl_path: str


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verify_python312_twin(
    *,
    target_id: str = "MI300X",
    repo_root: Path | None = None,
) -> TwinVerificationResult:
    """Verify Python 3.12 AST + PEP 649 semantics without stale .pyc caches."""
    root = (repo_root or _default_repo_root()).resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root / 'src'}:{root}"
    completed = run_in_process_group_bounded(
        [sys.executable, "-c", _TWIN_SUBPROCESS_SCRIPT, str(root), target_id],
        timeout=60.0,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Python 3.12 twin verification failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    ]
    if not lines:
        raise RuntimeError("Python 3.12 twin emitted no JSON summary")
    payload = json.loads(lines[-1])
    return TwinVerificationResult(
        target_id=target_id,
        checked_modules=int(payload["checked_modules"]),
        smoke_rows=int(payload["smoke_rows"]),
        passed=True,
    )


def _ssh_key_fingerprint(raw_line: str, label: str) -> str | None:
    parts = raw_line.strip().split()
    if len(parts) < 2 or parts[0].startswith("#"):
        return None
    try:
        blob = base64.b64decode(parts[1].encode("ascii"), validate=True)
    except ValueError:
        return None
    digest = bytes.fromhex(sha256_bytes(blob))
    b64 = base64.b64encode(digest).decode("ascii").rstrip("=")
    comment = parts[2] if len(parts) > 2 else ""
    suffix = f" ({comment})" if comment else ""
    return f"{label}: SHA256:{b64}{suffix}"


def list_local_ssh_fingerprints(
    ssh_dir: Path | None = None,
) -> tuple[str, ...]:
    """Return OpenSSH SHA256 fingerprints for local ``*.pub`` identities."""
    root = ssh_dir or (Path.home() / ".ssh")
    if not root.is_dir():
        return ()
    entries: list[str] = []
    for pub_path in sorted(root.glob("*.pub")):
        for line in pub_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            fp = _ssh_key_fingerprint(line, pub_path.name)
            if fp is not None:
                entries.append(fp)
    return tuple(entries)


def _open_socks5_tunnel(
    proxy_host: str,
    proxy_port: int,
    target_host: str,
    target_port: int,
    timeout_s: float,
) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout_s)
    sock.settimeout(timeout_s)
    try:
        sock.sendall(b"\x05\x01\x00")
        greeting = sock.recv(2)
        if greeting != b"\x05\x00":
            raise OSError(f"SOCKS5 auth rejected: {greeting!r}")
        try:
            addr_part = b"\x01" + socket.inet_aton(target_host)
        except OSError:
            host_bytes = target_host.encode("utf-8")
            addr_part = b"\x03" + bytes([len(host_bytes)]) + host_bytes
        sock.sendall(
            b"\x05\x01\x00" + addr_part + struct.pack("!H", target_port)
        )
        reply = sock.recv(10)
        if len(reply) < 4 or reply[1] != 0:
            code = reply[1] if len(reply) >= 2 else -1
            raise OSError(f"SOCKS5 connect failed with code {code}")
    except Exception:
        sock.close()
        raise
    return sock


def _open_http_connect_tunnel(
    proxy_host: str,
    proxy_port: int,
    target_host: str,
    target_port: int,
    timeout_s: float,
) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout_s)
    sock.settimeout(timeout_s)
    try:
        req = (
            f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
            f"Host: {target_host}:{target_port}\r\n\r\n"
        ).encode("ascii")
        sock.sendall(req)
        resp = sock.recv(1024)
        status_line = resp.split(b"\r\n", 1)[0]
        if b" 200 " not in status_line:
            raise OSError(
                f"HTTP CONNECT rejected: {status_line.decode('latin-1', 'replace')}"
            )
    except Exception:
        sock.close()
        raise
    return sock


def _open_tcp_or_proxy(
    host: str,
    port: int,
    *,
    proxy: str | None,
    timeout_s: float,
) -> socket.socket:
    if not proxy:
        sock = socket.create_connection((host, port), timeout=timeout_s)
        sock.settimeout(timeout_s)
        return sock
    parsed = urlparse(proxy)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"Unsupported proxy URL: {proxy!r}")
    scheme = parsed.scheme.lower()
    if scheme in {"socks5", "socks5h"}:
        return _open_socks5_tunnel(
            parsed.hostname, parsed.port, host, port, timeout_s
        )
    if scheme in {"http", "https"}:
        return _open_http_connect_tunnel(
            parsed.hostname, parsed.port, host, port, timeout_s
        )
    raise ValueError(f"Unsupported proxy scheme {scheme!r} in {proxy!r}")


def _probe_ssh_banner(
    host: str,
    port: int,
    *,
    proxy: str | None,
    timeout_s: float,
) -> tuple[bool, bool, str]:
    try:
        sock = _open_tcp_or_proxy(host, port, proxy=proxy, timeout_s=timeout_s)
    except OSError as exc:
        return False, False, f"ssh_connect_failed:{exc}"
    try:
        sock.sendall(b"SSH-2.0-SOL_Preflight_1.0\r\n")
        banner = sock.recv(256)
        if banner.startswith(b"SSH-"):
            text = banner.splitlines()[0].decode("ascii", errors="replace")
            return True, True, f"ssh_banner:{text}"
        return True, False, f"ssh_no_banner:{banner[:32]!r}"
    except OSError as exc:
        return True, False, f"ssh_banner_timeout:{exc}"
    finally:
        sock.close()


def _probe_http_endpoint(
    host: str,
    port: int,
    *,
    proxy: str | None,
    timeout_s: float,
) -> tuple[bool, str]:
    parsed = urlparse(proxy) if proxy else None
    use_http_proxy = (
        parsed is not None
        and parsed.scheme.lower() in {"http", "https"}
        and parsed.hostname is not None
        and parsed.port is not None
    )
    try:
        if use_http_proxy and parsed is not None and parsed.port is not None:
            sock = socket.create_connection(
                (str(parsed.hostname), parsed.port), timeout=timeout_s
            )
            sock.settimeout(timeout_s)
            req = (
                f"GET http://{host}:{port}/ HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\nConnection: close\r\n\r\n"
            ).encode("ascii")
        else:
            sock = _open_tcp_or_proxy(
                host, port, proxy=proxy, timeout_s=timeout_s
            )
            req = (
                f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
            ).encode("ascii")
        with sock:
            sock.sendall(req)
            resp = sock.recv(512)
            first_line = resp.split(b"\r\n", 1)[0]
            if first_line.startswith(b"HTTP/"):
                return True, first_line.decode("latin-1", errors="replace")
            return False, f"non_http_response:{first_line[:40]!r}"
    except OSError as exc:
        return False, f"http_probe_failed:{exc}"


def probe_remote_connectivity(
    host: str,
    *,
    proxy: str | None = None,
    ssh_port: int = 22,
    http_port: int = 80,
    timeout_s: float = 3.0,
    ssh_dir: Path | None = None,
) -> ConnectivityReport:
    """Classify SSH vs HTTP reachability within a bounded timeout."""
    fingerprints = list_local_ssh_fingerprints(ssh_dir)
    ssh_tcp, ssh_banner, ssh_detail = _probe_ssh_banner(
        host, ssh_port, proxy=proxy, timeout_s=timeout_s
    )
    http_ok, http_detail = _probe_http_endpoint(
        host, http_port, proxy=proxy, timeout_s=timeout_s
    )
    if ssh_banner:
        verdict = ConnectivityVerdict.SSH_READY
    elif http_ok and ssh_tcp:
        verdict = ConnectivityVerdict.SSH_BANNER_BLOCKED_HTTP_OPEN
    elif http_ok:
        verdict = ConnectivityVerdict.HTTP_ONLY
    else:
        verdict = ConnectivityVerdict.UNREACHABLE
    return ConnectivityReport(
        host=host,
        ssh_port=ssh_port,
        http_port=http_port,
        proxy=proxy,
        verdict=verdict,
        ssh_tcp_connected=ssh_tcp,
        ssh_banner_received=ssh_banner,
        http_reachable=http_ok,
        local_ssh_fingerprints=fingerprints,
        detail=f"{ssh_detail}; {http_detail}",
    )


def _atomic_write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        Path(tmp_name).replace(path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(part in _SKIP_DIR_NAMES for part in parts):
        return None
    if info.name.endswith(_SKIP_SUFFIXES):
        return None
    return info


def _pack_repo_archive_b64(repo_root: Path) -> str:
    buf = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tf,
    ):
        for rel in _BUNDLE_ROOTS:
            target = repo_root / rel
            if target.exists():
                tf.add(target, arcname=rel, filter=_tar_filter)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return "\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76))


def build_oneshot_cloud_script(
    output_script: Path,
    *,
    mode: SweepMode = SweepMode.FULL,
    remote_out_dir: str = "/tmp/mvp_cloud_out",
    repo_root: Path | None = None,
) -> Path:
    """Emit a self-extracting one-shot cloud runner (smoke gate -> full + SHA256)."""
    root = (repo_root or _default_repo_root()).resolve()
    archive_b64 = _pack_repo_archive_b64(root)
    script_text = (
        _ONESHOT_TEMPLATE.replace("__ARCHIVE_B64__", archive_b64)
        .replace("__DEFAULT_MODE__", mode.value)
        .replace("__DEFAULT_OUT_DIR__", remote_out_dir)
    )
    _atomic_write_text(output_script, script_text, mode=0o755)
    return output_script


def _check_tar_member_safe(member: tarfile.TarInfo) -> None:
    rel = Path(member.name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe tar member path: {member.name!r}")


def _locate_bundle_files(root: Path) -> tuple[Path, Path, Path]:
    summaries = sorted(root.rglob("summary.json"))
    if len(summaries) != 1:
        raise ValueError(
            f"Expected exactly 1 summary.json under {root}, found {len(summaries)}"
        )
    bundle_dir = summaries[0].parent
    report_path = bundle_dir / "report.md"
    traces_path = bundle_dir / "traces.jsonl"
    if not report_path.is_file() or not traces_path.is_file():
        raise ValueError(
            f"Bundle directory {bundle_dir} missing report.md or traces.jsonl"
        )
    return summaries[0], report_path, traces_path


def _validate_bundle_payload(
    summary_path: Path,
    report_path: Path,
    traces_path: Path,
) -> tuple[str, int, bool]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target = summary.get("target")
    rows = summary.get("rows")
    if not isinstance(target, dict) or not isinstance(
        target.get("target_id"), str
    ):
        raise ValueError("summary.json missing target.target_id")
    if not isinstance(rows, list) or not rows:
        raise ValueError("summary.json has empty or invalid rows")
    report_text = report_path.read_text(encoding="utf-8").strip()
    if not report_text.startswith("#"):
        raise ValueError("report.md does not start with a Markdown heading")
    traces = [
        Trace.model_validate_json(line)
        for line in traces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(traces) != len(rows):
        raise ValueError(
            f"traces.jsonl count ({len(traces)}) != summary rows ({len(rows)})"
        )
    executed_rows = [
        (row, trace)
        for row, trace in zip(rows, traces, strict=True)
        if not str(row.get("implementation", "")).startswith("unavailable:")
    ]
    all_passed = bool(executed_rows) and all(
        bool(row.get("correctness_passed")) and trace.is_successful()
        for row, trace in executed_rows
    )
    return str(target["target_id"]), len(rows), all_passed


def validate_downloaded_bundle(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> BundleVerificationReceipt:
    """Reject HTML/JSON error responses, hash mismatches, or invalid Trace bundles."""
    if not path.exists():
        raise ValueError(f"Bundle path does not exist: {path}")
    actual_sha: str | None = None
    if path.is_dir():
        summary_p, report_p, traces_p = _locate_bundle_files(path)
        target_id, row_count, all_passed = _validate_bundle_payload(
            summary_p, report_p, traces_p
        )
        return BundleVerificationReceipt(
            bundle_path=str(path),
            sha256=None,
            target_id=target_id,
            row_count=row_count,
            all_passed=all_passed,
            report_md_path=str(report_p),
            summary_json_path=str(summary_p),
            traces_jsonl_path=str(traces_p),
        )
    with path.open("rb") as handle:
        head = handle.read(512)
    if not head:
        raise ValueError(f"Bundle file is empty: {path}")
    stripped = head.lstrip()
    if stripped.startswith(_ERROR_PAYLOAD_PREFIXES) or not head.startswith(
        _GZIP_MAGIC
    ):
        raise ValueError(
            "Downloaded file appears to be an HTML/JSON error page, not a gzip bundle"
        )
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256.lower():
        raise ValueError(
            f"SHA256 mismatch: expected {expected_sha256.lower()}, got {actual_sha}"
        )
    with tempfile.TemporaryDirectory(prefix="mvp-verify-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        with tarfile.open(path, mode="r:gz") as tf:
            for member in tf.getmembers():
                _check_tar_member_safe(member)
            tf.extractall(tmp_path, filter="data")
        summary_p, report_p, traces_p = _locate_bundle_files(tmp_path)
        target_id, row_count, all_passed = _validate_bundle_payload(
            summary_p, report_p, traces_p
        )
    return BundleVerificationReceipt(
        bundle_path=str(path),
        sha256=actual_sha,
        target_id=target_id,
        row_count=row_count,
        all_passed=all_passed,
        report_md_path=str(report_p.relative_to(tmp_path)),
        summary_json_path=str(summary_p.relative_to(tmp_path)),
        traces_jsonl_path=str(traces_p.relative_to(tmp_path)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_twin = sub.add_parser("twin", help="Run uncached Python 3.12 twin gate.")
    p_twin.add_argument("--target", default="MI300X")

    p_probe = sub.add_parser(
        "probe", help="Run 3s SSH/HTTP connectivity probe."
    )
    p_probe.add_argument("host")
    p_probe.add_argument("--proxy", default=None)
    p_probe.add_argument("--ssh-port", type=int, default=22)
    p_probe.add_argument("--http-port", type=int, default=80)
    p_probe.add_argument("--timeout", type=float, default=3.0)

    p_bundle = sub.add_parser(
        "bundle", help="Build self-extracting cloud script."
    )
    p_bundle.add_argument("--output", type=Path, required=True)
    p_bundle.add_argument(
        "--mode",
        choices=[m.value for m in SweepMode],
        default=SweepMode.FULL.value,
    )
    p_bundle.add_argument("--remote-out-dir", default="/tmp/mvp_cloud_out")

    p_verify = sub.add_parser("verify", help="Verify downloaded bundle or dir.")
    p_verify.add_argument("path", type=Path)
    p_verify.add_argument("--expected-sha256", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for cloud preflight verification and bundling."""
    args = _build_parser().parse_args(argv)
    if args.command == "twin":
        res = verify_python312_twin(target_id=args.target)
        print(json.dumps(asdict(res), indent=2))
        return 0 if res.passed else 1
    if args.command == "probe":
        report = probe_remote_connectivity(
            args.host,
            proxy=args.proxy,
            ssh_port=args.ssh_port,
            http_port=args.http_port,
            timeout_s=args.timeout,
        )
        print(json.dumps(asdict(report), indent=2))
        return 0
    if args.command == "bundle":
        out = build_oneshot_cloud_script(
            args.output,
            mode=SweepMode(args.mode),
            remote_out_dir=args.remote_out_dir,
        )
        print(json.dumps({"script": str(out), "sha256": sha256_file(out)}))
        return 0
    receipt = validate_downloaded_bundle(
        args.path, expected_sha256=args.expected_sha256
    )
    print(json.dumps(asdict(receipt), indent=2))
    return 0 if receipt.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
