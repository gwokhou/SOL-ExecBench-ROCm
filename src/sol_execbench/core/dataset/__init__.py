# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Problem-corpus contracts and AKA compatibility support."""

from sol_execbench.core.dataset.aka_corpus import (
    AKACorpusEntry,
    AKACorpusManifest,
)
from sol_execbench.core.dataset.corpus import (
    load_corpus_manifest,
    select_corpus,
    static_selection_reason,
    validate_corpus,
)
from sol_execbench.core.dataset.corpus_models import (
    CorpusEntry,
    CorpusManifest,
    CorpusProfile,
    StaticTargetDescriptor,
)

__all__ = [
    "AKACorpusEntry",
    "AKACorpusManifest",
    "CorpusEntry",
    "CorpusManifest",
    "CorpusProfile",
    "StaticTargetDescriptor",
    "load_corpus_manifest",
    "select_corpus",
    "static_selection_reason",
    "validate_corpus",
]
