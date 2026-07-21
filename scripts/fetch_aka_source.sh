#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Fetch a pinned, shallow clone of AMD AgentKernelArena (Apache-2.0) into the
# gitignored data/AgentKernelArena/ tree. The converted problem set derives its
# raw inputs from this clone; the clone itself is local-only (covered by the
# /data/* ignore rule) and is never committed.

set -euo pipefail

REPO="https://github.com/AMD-AGI/AgentKernelArena"
REV="869228138e07e773b61dd7fc1d8cdc0435c7b405"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/AgentKernelArena"

mkdir -p "${ROOT}/data"

if [ ! -d "${DEST}/.git" ]; then
  echo "==> cloning ${REPO} into ${DEST}"
  git clone --filter=blob:none "${REPO}" "${DEST}"
fi

echo "==> fetching pinned revision ${REV}"
git -C "${DEST}" fetch --depth 1 origin "${REV}"
git -C "${DEST}" checkout -q "${REV}"

HEAD="$(git -C "${DEST}" rev-parse HEAD)"
if [ "${HEAD}" != "${REV}" ]; then
  echo "ERROR: checkout drifted: ${HEAD} != ${REV}" >&2
  exit 1
fi

printf '%s\n' "${HEAD}" > "${DEST}/.aka-head"
echo "==> AgentKernelArena ready at ${DEST} (@${HEAD:0:12})"
