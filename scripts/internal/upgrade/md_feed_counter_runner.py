"""M-d counter-collection mirror of the L2 event_chain case (spec M4).

Runs the same warmup + dual-stream dispatch sequence as
``torch_scope_workload.execute("event_chain")`` so a rocprofv3 counter
session and the L2 scope capture observe one dispatch-id space. Usage:
``rocprofv3 --pmc <counters> --output-format csv -- <python> <this> <repo>``.
The 2026-11-07 attempt log lives in the M-d HANDSOFF entry; the --pmc form
exited rc=1 before producing CSV output and needs the app-side traceback
captured next.
"""

"""Mirror the L2 event_chain dispatch sequence for counter collection."""
import json
import sys
from pathlib import Path

import torch

repo = Path(sys.argv[1])
definition = (
    repo / "problems/AMD_AKA/torch2hip/3267_doubled_matmul/definition.json"
)
namespace = {}
exec(  # noqa: S102 - audited repository reference
    compile(
        json.loads(definition.read_text())["reference"], str(definition), "exec"
    ),
    namespace,
)
function = namespace["run"]
torch.manual_seed(20261011)
m = k = n = 4096
a_cpu = torch.randn(m, k, dtype=torch.float32)
b_cpu = torch.randn(k, n, dtype=torch.float32)
a, b = a_cpu.to(device="cuda"), b_cpu.to(device="cuda")
producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
event = torch.cuda.Event()
for _ in range(2):
    function(a, b)
torch.cuda.synchronize()
with torch.cuda.stream(producer):
    result = function(a, b)
    event.record(producer)
with torch.cuda.stream(consumer):
    consumer.wait_event(event)
    result = result * 2
torch.cuda.synchronize()
torch.cuda.synchronize()
print("done", float(result.sum()))
