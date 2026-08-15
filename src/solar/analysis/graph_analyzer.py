# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyze a SOLAR IR graph into hardware-independent compute and I/O metrics.

This is the second SOLAR pipeline stage, converting an IR graph artifact into
``analysis.yaml``. It emits per-layer and graph totals plus conservative formal
fusion and Orojenesis evidence. See ``SOL_GUIDE.md`` for the memory models and
formal-evidence contract.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from solar.analysis.formal_analysis import FormalBoundAnalyzer
from solar.analysis.graph_accounting import LayerAccountant
from solar.analysis.graph_context import (
    AnalysisJob,
    PathLike,
)
from solar.analysis.graph_loading import GraphLoader
from solar.analysis.graph_models import (
    AnalysisAccumulator,
)
from solar.analysis.graph_validation import (
    GraphValidator,
    accept_prevalidated_graph,
)
from solar.analysis.orojenesis.runner import (
    OrojenesisRunner,
)
from solar.analysis.orojenesis_evidence import OrojenesisEvidenceEvaluator
from solar.analysis.reporting import build_analysis_result, write_analysis
from solar.composition import BoundComponent, component_attribute
from solar.ir.extended_einsum.operations.analyzer import EinsumAnalyzer
from solar.precision import (
    BYTES_PER_ELEMENT,
    DEFAULT_PRECISION,
)
from solar.rocm.architecture import ArchitectureProfile


class IRGraphAnalyzer:
    """Analyze a SOLAR IR graph and write `analysis.yaml`."""

    def __init__(
        self,
        debug: bool = False,
        *,
        validator: GraphValidator = accept_prevalidated_graph,
        einsum_analyzer: EinsumAnalyzer | None = None,
    ) -> None:
        """Initialize graph analysis and operation-cost helpers."""
        self.debug = debug
        self.validator = validator
        self.einsum_analyzer = einsum_analyzer or EinsumAnalyzer(debug=debug)
        self._components: tuple[BoundComponent, ...] = (
            GraphLoader(self),
            LayerAccountant(self),
            OrojenesisEvidenceEvaluator(self),
            FormalBoundAnalyzer(self),
        )

    def __getattr__(self, name: str) -> Any:
        """Resolve private workflow behavior from composed components."""
        return component_attribute(self._components, name)

    def analyze_graph(
        self,
        graph_path: PathLike,
        output_dir: PathLike,
        *,
        precision: str = DEFAULT_PRECISION,
        copy_graph: bool = True,
        strict: bool = False,
        architecture: str | Path | ArchitectureProfile | None = None,
        orojenesis_runner: OrojenesisRunner | None = None,
        require_orojenesis: bool = False,
    ) -> dict[str, Any] | None:
        """Analyze a SOLAR IR graph and write `analysis.yaml`.

        Args:
            graph_path: Path to the IR graph artifact.
            output_dir: Directory to write `analysis.yaml` into.
            precision: Tensor precision for byte calculations (e.g., fp32, bf16).
            copy_graph: If True, copy the IR graph into output dir under its
                source file name.
            strict: Reject unsupported layers and every implicit dtype fallback.
            architecture: Architecture profile or path used for formal bounds.
            orojenesis_runner: Optional configured tile-evidence runner.
            require_orojenesis: Require complete tile evidence when true.

        Returns:
            Analysis dict, or None on failure.

        """
        return self._analyze_job(
            AnalysisJob(
                graph_path=graph_path,
                output_dir=output_dir,
                precision=precision,
                copy_graph=copy_graph,
                strict=strict,
                architecture=architecture,
                orojenesis_runner=orojenesis_runner,
                require_orojenesis=require_orojenesis,
            ),
        )

    def _analyze_job(self, job: AnalysisJob) -> dict[str, Any] | None:
        """Run validated graph analysis through explicit accounting stages."""
        prepared = self._prepare_analysis(job)
        if prepared is None:
            return None
        topology = self._build_graph_topology(
            prepared.all_layers,
            prepared.declared_graph_outputs,
        )
        accumulator = AnalysisAccumulator()
        for layer_id, layer in topology.layers.items():
            self._analyze_layer(
                layer_id,
                layer,
                topology,
                prepared,
                accumulator,
            )
        io_totals = self._graph_io_totals(accumulator)
        formal = self._run_formal_analysis(
            prepared,
            topology,
            io_totals,
            job.orojenesis_runner,
            require_orojenesis=job.require_orojenesis,
        )
        lower_bound = self._lower_bound(
            prepared,
            accumulator,
            formal,
            require_orojenesis=job.require_orojenesis,
        )
        analysis = build_analysis_result(
            prepared,
            accumulator,
            io_totals,
            formal,
            lower_bound,
            start_node_count=len(topology.start_node_ids),
            intermediate_tensor_count=len(topology.intermediate_tensors),
        )
        output_path = write_analysis(prepared.output_dir, analysis)
        if self.debug:
            print(f"✅ Wrote analysis: {output_path}")
        return analysis

    _QUANT_DTYPE_MAP = MappingProxyType(
        {
            "nvfp4": "nvfp4",
            "float4_e2m1fn_x2": "nvfp4",
            "fp8": "fp8",
            "float8_e4m3fn": "fp8",
            "float8_e5m2": "fp8",
            "float8_e4m3fnuz": "fp8",
            "float8_e5m2fnuz": "fp8",
        },
    )

    def _resolve_quant_precision(self, graph_path: Path) -> str | None:
        """Search for metadata.yaml near the IR graph and return quant precision.

        Walks up from the IR graph path looking for metadata.yaml
        (max 3 levels). Picks highest-throughput quant dtype (nvfp4 > fp8).
        """
        search_dir = graph_path.parent
        for _ in range(3):
            candidate = search_dir / "metadata.yaml"
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        meta = yaml.safe_load(f) or {}
                except (OSError, UnicodeError, yaml.YAMLError):
                    return None

                best = None
                for conv in meta.get("dtype_conversions") or []:
                    orig = str(conv.get("orig_dtypes", "")).lower()
                    for keyword, prec in self._QUANT_DTYPE_MAP.items():
                        if keyword in orig:
                            if best is None or BYTES_PER_ELEMENT.get(
                                prec,
                                99,
                            ) < BYTES_PER_ELEMENT.get(best, 99):
                                best = prec
                            break
                return best
            search_dir = search_dir.parent
        return None


__all__ = ["IRGraphAnalyzer"]
