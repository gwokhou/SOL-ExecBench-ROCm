#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
# One-click 15-minute cloud capture and tarball packager for v0.1 3-operator MVP.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -n "${PYTHON_BIN:-}" ]; then
    PY="${PYTHON_BIN}"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PY="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    PY="$(command -v python)"
fi

if ! "${PY}" -c "import pydantic, yaml; assert int(pydantic.__version__.split('.')[0]) >= 2" >/dev/null 2>&1; then
    "${PY}" -m pip install -q "pydantic>=2.6" pyyaml
fi

MVP_MODE="${MVP_MODE:-full}"
MVP_DEVICE="${MVP_DEVICE:-auto}"
MVP_BACKEND="${MVP_BACKEND:-auto}"
MVP_COUNTERS_FLAG="${MVP_COUNTERS_FLAG:---collect-counters}"
MVP_CLOCKS_FLAG="${MVP_CLOCKS_FLAG:---lock-clocks}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${MVP_OUTPUT_DIR:-${REPO_ROOT}/data/outputs/mvp_v1_cloud_${TS}}"

echo "=== v0.1 MVP Cloud Preflight ==="
"${PY}" -c '
import torch
print("torch:", torch.__version__, "hip:", getattr(torch.version, "hip", None), "cuda_avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device[0]:", p.name, "gfx:", getattr(p, "gcnArchName", "-"), "CUs/SMs:", getattr(p, "multi_processor_count", "-"), "VRAM_GiB:", round(p.total_memory / (1024 ** 3), 2))
'

"${PY}" -m scripts.mvp_v1.cloud_runner \
    --mode "${MVP_MODE}" \
    --device "${MVP_DEVICE}" \
    --backend "${MVP_BACKEND}" \
    ${MVP_COUNTERS_FLAG} \
    ${MVP_CLOCKS_FLAG} \
    --output-dir "${OUT_DIR}" \
    "$@"

TAR_PATH="${OUT_DIR}.tar.gz"
tar -C "$(dirname "${OUT_DIR}")" -czf "${TAR_PATH}" "$(basename "${OUT_DIR}")"
TAR_SHA256="$(sha256sum "${TAR_PATH}" | awk '{print $1}')"

echo ""
echo "=== Cloud Capture Complete ==="
echo "Report:  ${OUT_DIR}/report.md"
echo "Bundle:  ${TAR_PATH}"
echo "SHA256:  ${TAR_SHA256}"
echo "Compare: python -m scripts.mvp_v1.cloud_runner --compare <portable/summary.json> <vllm/summary.json> [aiter/summary.json]"
