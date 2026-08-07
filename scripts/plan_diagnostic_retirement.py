#!/usr/bin/env python3
"""Print the audit-only dry-run retirement plan for the resolved targets.

This is a thin CLI wrapper over the lifecycle package planner. The exact
inventory, byte totals, and reachability proof live in
``sol_execbench.core.bench.performance_model.lifecycle.retirement`` and are
also exposed as ``sol-execbench diagnostics lifecycle retirement-plan``.
The planner never deletes or moves data.
"""

from __future__ import annotations

import json

from sol_execbench.core.bench.performance_model.lifecycle import (
    plan_retirement,
    repo_root,
    resolved_retirement_targets,
    store_root,
)


def main() -> int:
    """Print the retirement plan as JSON and return a CI exit code."""
    plan = plan_retirement(
        store_root_path=store_root(),
        targets=resolved_retirement_targets(repo_root()),
        repo_root=repo_root(),
    )
    print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))
    if plan.reachable_targets:
        print(
            f"{plan.reachable_targets} target(s) are registry-reachable; "
            "deletion is refused",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
