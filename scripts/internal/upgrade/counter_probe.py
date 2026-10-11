# SPDX-License-Identifier: Apache-2.0
"""Small trusted replay probe, including a state-changing in-place dispatch."""

import json

import torch

from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock

ELEMENTS = 4096


def main() -> None:
    """Detect replay that accidentally applies an in-place update twice."""
    with acquire_gpu_lock(0, timeout_seconds=30):
        if torch.cuda.device_count() != 1:
            raise RuntimeError("counter probe requires one visible GPU")
        if (
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            != "gfx1200"
        ):
            raise RuntimeError("counter probe requires gfx1200")
        left = torch.ones(ELEMENTS, device="cuda")
        right = torch.full_like(left, 2)
        output = torch.empty_like(left)
        torch.add(left, right, out=output)
        left.add_(1)
        torch.cuda.synchronize()
        if not bool(output.cpu().eq(3).all()):
            raise RuntimeError("vector-add result changed")
        if not bool(left.cpu().eq(2).all()):
            raise RuntimeError("replay repeated the in-place update")
        print(
            json.dumps(
                {
                    "correct": True,
                    "elements": ELEMENTS,
                    "inplace_update_applied_once": True,
                }
            )
        )


if __name__ == "__main__":
    main()
