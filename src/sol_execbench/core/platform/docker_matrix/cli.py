"""Declared ROCm Docker Target selection and diagnostic preflight helpers."""

from __future__ import annotations

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sol_execbench.core.utils import parse_bool as _parse_bool
from sol_execbench.core.platform.docker_matrix.models import (
    DEFAULT_DOCKER_TARGET_MANIFEST,
    DockerPreflightObservation,
)
from sol_execbench.core.platform.docker_matrix.preflight import (
    classify_docker_preflight,
)
from sol_execbench.core.platform.docker_matrix.targets import (
    docker_build_args_for_target,
    preview_docker_target_selection,
    select_docker_target,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview = subparsers.add_parser("preview")
    preview.add_argument(
        "--manifest", type=Path, default=DEFAULT_DOCKER_TARGET_MANIFEST
    )
    preview.add_argument("--target")
    preview.add_argument("--allow-unknown-target", action="store_true")
    preview.add_argument("--override-image-repository")
    preview.add_argument("--override-image-tag")
    preview.add_argument("--image-digest")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument(
        "--manifest", type=Path, default=DEFAULT_DOCKER_TARGET_MANIFEST
    )
    preflight.add_argument("--target")
    preflight.add_argument("--docker-context")
    preflight.add_argument("--docker-host")
    preflight.add_argument("--dev-kfd-present", required=True, type=_parse_bool)
    preflight.add_argument("--dev-kfd-accessible", required=True, type=_parse_bool)
    preflight.add_argument("--dev-dri-present", required=True, type=_parse_bool)
    preflight.add_argument("--dev-dri-accessible", required=True, type=_parse_bool)
    preflight.add_argument("--gpu-accessible", type=_parse_bool)
    preflight.add_argument("--image-digest")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Emit shell-consumable Docker Matrix JSON."""

    args = _build_parser().parse_args(argv)
    if args.command == "preview":
        payload = preview_docker_target_selection(
            target_id=args.target,
            manifest_path=args.manifest,
            allow_unknown_override=args.allow_unknown_target,
            override_image_repository=args.override_image_repository,
            override_image_tag=args.override_image_tag,
            image_digest=args.image_digest,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.command == "preflight":
        selection = select_docker_target(args.target, manifest_path=args.manifest)
        observation = DockerPreflightObservation(
            docker_context=args.docker_context,
            docker_host=args.docker_host,
            dev_kfd_present=args.dev_kfd_present,
            dev_kfd_accessible=args.dev_kfd_accessible,
            dev_dri_present=args.dev_dri_present,
            dev_dri_accessible=args.dev_dri_accessible,
            gpu_accessible=args.gpu_accessible,
            selected_target=selection.target,
            image_repository=selection.target.docker_image_repository,
            image_tag=selection.target.docker_image_tag,
            image_digest=args.image_digest,
            build_args=docker_build_args_for_target(selection.target),
        )
        payload = classify_docker_preflight(observation).to_preview_payload()
        print(json.dumps(payload, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
