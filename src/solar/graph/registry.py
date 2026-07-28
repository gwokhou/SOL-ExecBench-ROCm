# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Lazy backend registry for SOLAR graph extractors."""

from __future__ import annotations

from collections.abc import Callable

from solar.graph.contracts import (
    ExtractionKind,
    GraphBackend,
    normalize_extraction_kind,
)


def _load_make_fx_backend() -> GraphBackend:
    from solar.graph.make_fx_backend import backend

    return backend


def _load_torchview_backend() -> GraphBackend:
    from solar.graph.torchview.extraction import backend

    return backend


_EXTRACTION_LOADERS: dict[ExtractionKind, Callable[[], GraphBackend]] = {
    ExtractionKind.MAKE_FX_REFERENCE: _load_make_fx_backend,
    ExtractionKind.TORCHVIEW: _load_torchview_backend,
}


def extraction_backend(kind: ExtractionKind | str) -> GraphBackend:
    """Return the backend implementing the requested graph extraction."""
    return _EXTRACTION_LOADERS[normalize_extraction_kind(kind)]()


def extraction_backends() -> tuple[GraphBackend, ...]:
    """Return every registered graph-extraction backend."""
    return tuple(loader() for loader in _EXTRACTION_LOADERS.values())


__all__ = ["extraction_backend", "extraction_backends"]
