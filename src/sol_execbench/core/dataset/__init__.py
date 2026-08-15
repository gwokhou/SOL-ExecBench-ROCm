# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Problem-corpus contracts and AKA compatibility support."""

from sol_execbench.core.dataset.aka_corpus import (
    AKACorpusEntry,
    AKACorpusManifest,
)
from sol_execbench.core.dataset.corpus import (
    generate_corpus,
    load_corpus_manifest,
    validate_corpus,
)
from sol_execbench.core.dataset.corpus_models import (
    CorpusEntry,
    CorpusManifest,
    CorpusProfile,
    CorpusTargetViewManifest,
    StaticTargetDescriptor,
)

__all__ = [
    "AKACorpusEntry",
    "AKACorpusManifest",
    "CorpusEntry",
    "CorpusManifest",
    "CorpusProfile",
    "CorpusTargetViewManifest",
    "StaticTargetDescriptor",
    "generate_corpus",
    "load_corpus_manifest",
    "validate_corpus",
]
