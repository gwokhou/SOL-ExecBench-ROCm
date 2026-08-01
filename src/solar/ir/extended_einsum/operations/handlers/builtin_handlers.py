"""Explicit built-in handler inventory and precedence."""

from __future__ import annotations

from solar.ir.extended_einsum.operations.handlers.attention_ops import (
    FlexAttentionHandler,
    MultiHeadAttentionHandler,
    ScaledDotProductAttentionHandler,
)
from solar.ir.extended_einsum.operations.handlers.base import EinsumOpHandler
from solar.ir.extended_einsum.operations.handlers.conv_ops import (
    Conv1dHandler,
    Conv2dHandler,
    Conv3dHandler,
    ConvTranspose1dHandler,
    ConvTranspose2dHandler,
    ConvTranspose3dHandler,
)
from solar.ir.extended_einsum.operations.handlers.cumulative_ops import (
    CumulativeHandler,
)
from solar.ir.extended_einsum.operations.handlers.elementwise_ops import (
    BinaryElementwiseHandler,
    UnaryElementwiseHandler,
)
from solar.ir.extended_einsum.operations.handlers.loss_ops import LossHandler
from solar.ir.extended_einsum.operations.handlers.matmul_ops import (
    BmmHandler,
    LinearHandler,
    MatmulHandler,
)
from solar.ir.extended_einsum.operations.handlers.misc_ops import (
    CrossEntropyHandler,
    EmbeddingHandler,
    GRUHandler,
    LSTMHandler,
    PairwiseLossHandler,
    RNNHandler,
    TopKHandler,
    TrivialOpsHandler,
)
from solar.ir.extended_einsum.operations.handlers.norm_ops import (
    NormalizationHandler,
)
from solar.ir.extended_einsum.operations.handlers.pooling_ops import (
    PoolingHandler,
)
from solar.ir.extended_einsum.operations.handlers.reduction_ops import (
    ReductionHandler,
)
from solar.ir.extended_einsum.operations.handlers.shape_ops import (
    MatrixStructureHandler,
    TensorManipulationHandler,
)

# Later handlers intentionally take precedence for overlapping operation names.
# In particular, specialized loss handlers override the generic LossHandler.
BUILTIN_HANDLER_CLASSES: tuple[type[EinsumOpHandler], ...] = (
    ScaledDotProductAttentionHandler,
    FlexAttentionHandler,
    MultiHeadAttentionHandler,
    Conv1dHandler,
    Conv2dHandler,
    Conv3dHandler,
    ConvTranspose1dHandler,
    ConvTranspose2dHandler,
    ConvTranspose3dHandler,
    CumulativeHandler,
    UnaryElementwiseHandler,
    BinaryElementwiseHandler,
    LossHandler,
    MatmulHandler,
    LinearHandler,
    BmmHandler,
    EmbeddingHandler,
    GRUHandler,
    LSTMHandler,
    RNNHandler,
    CrossEntropyHandler,
    PairwiseLossHandler,
    TopKHandler,
    TrivialOpsHandler,
    NormalizationHandler,
    PoolingHandler,
    ReductionHandler,
    TensorManipulationHandler,
    MatrixStructureHandler,
)

# These specialized handlers intentionally replace the generic loss mapping.
BUILTIN_HANDLER_OVERRIDE_OPS: dict[
    type[EinsumOpHandler],
    frozenset[str],
] = {
    CrossEntropyHandler: frozenset(
        {
            "cross_entropy",
            "nll_loss",
        },
    ),
    PairwiseLossHandler: frozenset(
        {
            "binary_cross_entropy",
            "cosine_embedding_loss",
            "huber_loss",
            "kl_div",
            "l1_loss",
            "mse_loss",
            "smooth_l1_loss",
        },
    ),
}

__all__ = ["BUILTIN_HANDLER_CLASSES", "BUILTIN_HANDLER_OVERRIDE_OPS"]
