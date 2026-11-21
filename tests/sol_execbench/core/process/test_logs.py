from __future__ import annotations

import subprocess
import time
from pathlib import Path

from sol_execbench.core.process.logs import (
    redacted_file_tail,
    redacted_text_tail,
    run_command_to_files,
    temporary_stream_path,
)


def test_redacted_text_tail_covers_common_credentials() -> None:
    value = (
        "AWS_SECRET_ACCESS_KEY=abc\n"
        "Authorization: Bearer bearer-token\n"
        "HF_TOKEN = hf_123\n"
        "PASSWORD: pass_123"
    )

    result = redacted_text_tail(value)

    assert "abc" not in result
    assert "bearer-token" not in result
    assert "hf_123" not in result
    assert "pass_123" not in result
    assert result.count("<redacted>") == 4
    assert redacted_text_tail(value, 0) == ""


def test_redacted_file_tail_handles_long_lines_and_split_secrets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.log"
    token = "HF_TOKEN=secret_split_tail"
    path.write_text(
        "x" * (8192 - len("HF_TOKEN=secret")) + token,
        encoding="utf-8",
    )

    result = redacted_file_tail(path, 200)

    assert "secret_split_tail" not in result
    assert "split_tail" not in result
    assert "<redacted>" in result


def test_redacted_file_tail_handles_non_positive_limit_and_missing_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.log"

    assert redacted_file_tail(missing) == ""
    assert redacted_file_tail(missing, 0) == ""


def test_temporary_stream_path_sanitizes_name_and_supports_prefix(
    tmp_path: Path,
) -> None:
    path = temporary_stream_path(
        tmp_path,
        "cpu safe/check",
        "stdout",
        name_prefix="sol_execbench_",
    )
    try:
        assert path.parent == tmp_path
        assert path.name.startswith("sol_execbench_cpu_safe_check_stdout_")
        assert path.suffix == ".log"
    finally:
        path.unlink(missing_ok=True)


def test_run_command_to_files_mirrors_output_from_test_runner(
    tmp_path: Path,
) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    def fake_runner(command: list[str], **kwargs: object):
        assert command == ["demo"]
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            command,
            3,
            stdout="captured stdout",
            stderr="captured stderr",
        )

    completed = run_command_to_files(
        ["demo"],
        stdout_path,
        stderr_path,
        runner=fake_runner,
    )

    assert completed.returncode == 3
    assert stdout_path.read_text(encoding="utf-8") == "captured stdout"
    assert stderr_path.read_text(encoding="utf-8") == "captured stderr"


def test_redacted_tails_cover_json_quotes_standalone_prefixes_and_linear_time(
    tmp_path: Path,
) -> None:
    payload = (
        '{"api_key": "json-secret-1", "nested": {"token": "json-secret-2"}}\n'
        "url=https://ghp_1234567890abcdef@github.com/repo\n"
        "pat=github_pat_abcdef123456_789\n"
        "bare hf_AbCdEf12345678 and sk-proj-1234567890abcdef1234\n"
        "HF_HOME=/cache/huggingface"
    )
    redacted = redacted_text_tail(payload, 4000)
    for secret in (
        "json-secret-1",
        "json-secret-2",
        "ghp_1234567890abcdef",
        "github_pat_abcdef123456_789",
        "hf_AbCdEf12345678",
        "sk-proj-1234567890abcdef1234",
    ):
        assert secret not in redacted
    assert "HF_HOME=/cache/huggingface" in redacted

    adversarial = "A" * (1024 * 1024) + "_TOKEN"
    start = time.perf_counter()
    tail = redacted_text_tail(adversarial, 4000)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 50.0
    assert len(tail) == 4000

    large_log = tmp_path / "large.log"
    large_log.write_text(
        "A" * (1024 * 1024) + '\n{"secret": "eof-secret-val"}\n',
        encoding="utf-8",
    )
    file_tail = redacted_file_tail(large_log, 128)
    assert "eof-secret-val" not in file_tail
    assert '"secret": "<redacted>"' in file_tail
