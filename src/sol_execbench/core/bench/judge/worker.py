"""Private CPU worker; never initializes the GPU evaluator or mapper."""

import argparse
import os
import resource
from pathlib import Path

import torch

from sol_execbench.core.bench.judge.contracts import development_cases
from sol_execbench.core.bench.judge.development import run_development
from sol_execbench.core.bench.judge.executor import MAX_MEMORY_BYTES
from sol_execbench.core.data.json_utils import load_json_value


def main() -> None:
    """Run one owned development case under a conservative address-space cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    request = load_json_value(parser.parse_args().request)
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = min(MAX_MEMORY_BYTES, physical // 2)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    torch.set_num_threads(1)
    run_development(
        Path(request["repository"]),
        Path(request["output"]),
        cases=(development_cases()[request["index"]],),
        run_nonce=request["run_nonce"],
    )


if __name__ == "__main__":
    main()
