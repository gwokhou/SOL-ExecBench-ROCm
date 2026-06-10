# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Run derived dataset reporting as isolated, resumable per-problem jobs.

This wrapper keeps long-tail derived sidecar generation out of the calling
Codex/process tree as much as the local platform allows. It delegates the real
work to scripts/run_dataset.py, records per-problem status, and can resume after
OOM kills or shell/session restarts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from sol_execbench.core.bench.pid_lock import acquire_pid_lock


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DATASET = REPO_ROOT / "scripts" / "run_dataset.py"
CATEGORIES = {"L1", "L2", "FlashInfer-Bench", "Quant"}

LaunchMode = Literal["direct", "prlimit", "systemd"]


@dataclass(frozen=True)
class ProblemStatus:
    problem_id: str
    status: str
    returncode: int | None
    started_at: str
    finished_at: str
    elapsed_seconds: float
    command: list[str]
    log_ref: str


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def discover_problems(benchmark_dir: Path, categories: list[str] | None) -> list[Path]:
    roots: list[Path]
    if (benchmark_dir / "definition.json").exists() and (
        benchmark_dir / "workload.jsonl"
    ).exists():
        roots = [benchmark_dir]
    else:
        selected = categories or sorted(CATEGORIES)
        roots = []
        for category in selected:
            category_dir = benchmark_dir / category
            if category_dir.exists():
                roots.extend(sorted(category_dir.iterdir()))
    return [
        path
        for path in sorted(roots)
        if path.is_dir()
        and (path / "definition.json").exists()
        and (path / "workload.jsonl").exists()
    ]


def problem_id_for(benchmark_dir: Path, problem_dir: Path) -> str:
    try:
        return problem_dir.relative_to(benchmark_dir).as_posix()
    except ValueError:
        return f"{problem_dir.parent.name}/{problem_dir.name}"


def load_completed(status_path: Path) -> set[str]:
    completed: set[str] = set()
    if not status_path.exists():
        return completed
    with status_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ok" and isinstance(
                payload.get("problem_id"), str
            ):
                completed.add(payload["problem_id"])
    return completed


def load_problem_id_filter(path: Path | None) -> set[str] | None:
    """Load newline-delimited problem IDs for targeted retries."""
    if path is None:
        return None
    problem_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        problem_ids.add(value)
    return problem_ids


def base_run_dataset_command(args: argparse.Namespace, problem_dir: Path) -> list[str]:
    command = [
        str(args.uv_bin),
        "run",
        str(RUN_DATASET),
        str(problem_dir),
        "--phase",
        "derived",
        "-o",
        str(args.output_dir),
    ]
    if args.long_tail_exclusions is not None:
        command.extend(["--long-tail-exclusions", str(args.long_tail_exclusions)])
    if args.amd_score_report is not None:
        command.extend(["--amd-score-report", str(args.amd_score_report)])
    if args.amd_sol_bound_dir is not None:
        command.extend(["--amd-sol-bound-dir", str(args.amd_sol_bound_dir)])
    if args.solar_derivation is not None:
        command.extend(["--solar-derivation", str(args.solar_derivation)])
    if args.scoring_baseline is not None:
        command.extend(["--scoring-baseline", str(args.scoring_baseline)])
    if args.gpu_architecture is not None:
        command.extend(["--gpu-architecture", args.gpu_architecture])
    return command


def wrap_command(
    command: list[str],
    *,
    launch_mode: LaunchMode,
    memory_max: str | None,
    memory_swap_max: str | None,
    unit_name: str,
) -> list[str]:
    if launch_mode == "direct":
        return command
    if launch_mode == "prlimit":
        if memory_max is None:
            raise ValueError("--memory-max is required for --launch-mode prlimit")
        return ["prlimit", f"--as={memory_max}", "--"] + command
    if launch_mode == "systemd":
        wrapped = [
            "systemd-run",
            "--user",
            "--wait",
            "--collect",
            "--unit",
            unit_name,
        ]
        if memory_max is not None:
            wrapped.extend(["--property", f"MemoryMax={memory_max}"])
        if memory_swap_max is not None:
            wrapped.extend(["--property", f"MemorySwapMax={memory_swap_max}"])
        wrapped.extend(["--same-dir", "--"])
        wrapped.extend(command)
        return wrapped
    raise ValueError(f"unknown launch mode: {launch_mode}")


def environment_with_uv_cache(uv_cache_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if uv_cache_dir is not None:
        env["UV_CACHE_DIR"] = str(uv_cache_dir)
    return env


def append_status(status_path: Path, status: ProblemStatus) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(status), sort_keys=True) + "\n")


def should_skip(
    problem_id: str,
    *,
    completed: set[str],
    start_at: str | None,
    start_after: str | None,
) -> bool:
    if problem_id in completed:
        return True
    if start_at is not None:
        return problem_id < start_at
    if start_after is not None:
        return problem_id <= start_after
    return False


def run_problem(
    args: argparse.Namespace,
    *,
    problem_id: str,
    problem_dir: Path,
    log_path: Path,
) -> ProblemStatus:
    started = time.monotonic()
    started_at = utc_timestamp()
    command = base_run_dataset_command(args, problem_dir)
    wrapped = wrap_command(
        command,
        launch_mode=args.launch_mode,
        memory_max=args.memory_max,
        memory_swap_max=args.memory_swap_max,
        unit_name="sol-derived-" + problem_id.replace("/", "-").replace("_", "-")[:48],
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n===== {started_at} DERIVED {problem_id} =====\n")
        log_handle.write("$ " + shlex.join(wrapped) + "\n")
        log_handle.flush()
        result = subprocess.run(
            wrapped,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment_with_uv_cache(args.uv_cache_dir),
        )
    elapsed = time.monotonic() - started
    return ProblemStatus(
        problem_id=problem_id,
        status="ok" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        started_at=started_at,
        finished_at=utc_timestamp(),
        elapsed_seconds=round(elapsed, 3),
        command=wrapped,
        log_ref=str(log_path),
    )


def _partition_problems_by_index(
    problems: list[Path],
    max_workers: int,
) -> list[list[Path]]:
    """Pre-partition problems for exclusive worker ownership."""
    if not problems:
        return []
    chunk_size = (len(problems) + max_workers - 1) // max_workers
    chunks = []
    for i in range(0, len(problems), chunk_size):
        chunks.append(problems[i : i + chunk_size])
    return chunks


def _process_problem_chunk(
    chunk: list[Path],
    *,
    args: argparse.Namespace,
    benchmark_dir: Path,
    status_lock: threading.Lock,
) -> list[ProblemStatus]:
    """Process a chunk of problems serially within a worker thread."""
    chunk_results = []
    for problem_dir in chunk:
        problem_id = problem_id_for(benchmark_dir, problem_dir)
        if (
            args.problem_id_filter is not None
            and problem_id not in args.problem_id_filter
        ):
            continue
        if should_skip(
            problem_id,
            completed=args.completed,
            start_at=args.start_at,
            start_after=args.start_after,
        ):
            continue
        print(f"DERIVED {problem_id}", flush=True)
        status = run_problem(
            args,
            problem_id=problem_id,
            problem_dir=problem_dir,
            log_path=args.log_file,
        )
        with status_lock:
            append_status(args.status_jsonl, status)
        print(
            f"{problem_id}: {status.status} rc={status.returncode} "
            f"elapsed={status.elapsed_seconds:.1f}s",
            flush=True,
        )
        chunk_results.append(status)

        # Handle --continue-on-failure
        if status.status != "ok" and not args.continue_on_failure:
            # Early exit: return what we have, signal failure via status
            return chunk_results
    return chunk_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--category", action="append", choices=sorted(CATEGORIES))
    parser.add_argument("--long-tail-exclusions", type=Path)
    parser.add_argument("--amd-score-report", type=Path)
    parser.add_argument("--amd-sol-bound-dir", type=Path)
    parser.add_argument("--solar-derivation", type=Path)
    parser.add_argument("--scoring-baseline", type=Path)
    parser.add_argument("--gpu-architecture")
    parser.add_argument("--uv-bin", type=Path, default=Path("uv"))
    parser.add_argument("--uv-cache-dir", type=Path)
    parser.add_argument(
        "--launch-mode",
        choices=["direct", "prlimit", "systemd"],
        default="direct",
        help=(
            "direct keeps subprocesses in the current process tree; prlimit caps "
            "virtual memory; systemd puts each problem in a transient user unit."
        ),
    )
    parser.add_argument(
        "--memory-max",
        help="Memory limit for prlimit or systemd, e.g. 20G.",
    )
    parser.add_argument(
        "--memory-swap-max",
        help="systemd MemorySwapMax, e.g. 0 or 2G.",
    )
    parser.add_argument("--status-jsonl", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--start-at")
    parser.add_argument("--start-after")
    parser.add_argument(
        "--problem-id-file",
        type=Path,
        help="Optional newline-delimited problem IDs to run, e.g. L1/problem_name.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--pid-lock",
        action="store_true",
        help="Acquire exclusive process lock to prevent concurrent runs",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 1, 4),
        help="Number of concurrent jobs (default: min(cpu_count, 4))",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Conditional PID lock based on --pid-lock flag
    if args.pid_lock:
        acquire = acquire_pid_lock(args.output_dir)
    else:
        acquire = nullcontext()

    with acquire:
        problems = discover_problems(args.benchmark_dir, args.category)
        problem_id_filter = load_problem_id_filter(args.problem_id_file)
        completed = load_completed(args.status_jsonl) if args.resume else set()

        # Attach filter and completed set to args for worker access
        args.problem_id_filter = problem_id_filter
        args.completed = completed

        # Pre-partition problems for parallel execution
        problem_chunks = _partition_problems_by_index(problems, args.jobs)
        status_lock = threading.Lock()

        try:
            all_results = []
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = {
                    executor.submit(
                        _process_problem_chunk,
                        chunk,
                        args=args,
                        benchmark_dir=args.benchmark_dir,
                        status_lock=status_lock,
                    ): chunk
                    for chunk in problem_chunks
                }

                for future in as_completed(futures.keys()):
                    chunk_results = future.result()
                    all_results.extend(chunk_results)

                    # Early exit on failure if --continue-on-failure not set
                    if chunk_results and chunk_results[-1].status != "ok":
                        if not args.continue_on_failure:
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            return chunk_results[-1].returncode or 1

            # Sort results by problem_id for deterministic output
            all_results.sort(key=lambda s: s.problem_id)
            failures = sum(1 for s in all_results if s.status != "ok")
            return 1 if failures else 0

        except KeyboardInterrupt:
            # Exit code 130 for SIGINT (128 + 2)
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
