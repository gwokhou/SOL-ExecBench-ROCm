# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Offline generation of reviewable, numerically verified handler candidates."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from solar.einsum.llm_agent import AgentConfig, NodeTypeConversionAgent
from solar.einsum.node_type_registry import NodeTypeHandlerFactory, NodeTypeRegistry


def learn_handler_candidate(
    *,
    node_type: str,
    sample_node_data: Mapping[str, Any],
    output_dir: str | Path,
    model: str = "gpt-4",
) -> dict[str, Any]:
    """Generate a verified candidate without making it formally trusted."""
    if not node_type.isidentifier():
        raise ValueError("node_type must be a Python identifier")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for handler learning")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        agent = NodeTypeConversionAgent(
            AgentConfig(
                api_key=api_key,
                model=model,
                cache_dir=str(staging),
                fail_closed=True,
            )
        )
        code, metadata = agent.generate_conversion_code(
            node_type, dict(sample_node_data)
        )
        handler = NodeTypeHandlerFactory.create_handler_from_code(
            node_type, code, metadata
        )
        NodeTypeRegistry(cache_dir=str(staging)).save_generated_handler(
            node_type, handler
        )
        artifact = _candidate_manifest(node_type, staging, metadata)
        (staging / "candidate.yaml").write_text(
            yaml.safe_dump(artifact, sort_keys=False)
        )
        staging.replace(output)
        return artifact
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _candidate_manifest(
    node_type: str, staging: Path, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    files = sorted(
        path
        for path in staging.iterdir()
        if path.is_file() and path.name != "candidate.yaml"
    )
    return {
        "schema_version": 1,
        "status": "verified_candidate_unreviewed",
        "node_type": node_type,
        "verification": metadata.get("verification_details"),
        "files": [
            {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in files
        ],
        "formal_use": "forbidden_until_reviewed_and_committed_under_solar/handlers",
    }


__all__ = ["learn_handler_candidate"]
