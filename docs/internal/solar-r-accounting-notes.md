# R accounting interpretation and read domains

Dense contractions retain the convention `1 MAC = 2 FLOPs`. Resource counts
follow the fixed executable operator graph: a fused `addmm` or convolution may
initialize its accumulator with bias; the dense `2 * MAC` count does not require
an additional VALU bias instruction. An explicit graph `add` is charged to VALU.
Thus MLP paths with separate `bias_add` versus fused `addmm` need not have equal
VALU counts or unfused traffic. Compare MACs, explicit graph granularity and
model assumptions; do not add fictitious mandatory work merely to equalize paths.
These are model bounds, not GPU instruction measurements or a proof of fusion.

Conversion identity is checked using full normalized dtypes before grouping
integer resource rates. A width or signedness change is a conversion, even when
both rates use `integer`. This resource model charges one conversion per output;
its generic rate is a conservative ceiling, not measured conversion latency.

External read proofs additionally cover unit-scaled addmm, and positive-shape,
ungrouped 1x1 conv2d with stride/dilation one, padding zero, no transpose and
matching output geometry. Other convolution domains or non-default addmm
coefficients remain unproved. Pure relu/elu preserve in-place and out mutations.
Indexed rank names such as `B0` use the existing einsum tokenizer; diagonal or
compound accesses are not inferred as full operand reads.

A pure `where` is not atomic. Its condition is read, but external value branches
are conditional, so they remain unproved; internal already-materialized branches
can pass. `ones_like` uses shape/dtype metadata and does not read input values.
`scalar_tensor` with a numeric literal and no tensor inputs reads no tensor data;
unknown constructors and tensor-dependent arguments do not inherit this proof.
Dynamic gather/index operations, copies, mutation and unknown operations keep
explicit refusal. This is not a general permission to price every pure function.

Changes in native semantic metadata invalidate old semantic implementation
identities. New CPU receipts must be labeled CPU; the 2026-10-29 d55d2a34 GPU
receipts stay historical. Fresh GPU replay is required before claiming that the
changed semantic implementation has passed on RX 9060 XT. No mapper capacity or
formal qualification follows from CPU accounting alone.
