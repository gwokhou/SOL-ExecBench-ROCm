# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
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

"""Reward hack defenses for SOL ExecBench evaluation.

Provides detection functions for four common reward-hacking patterns.
The identity of torch.cuda.Event.elapsed_time is captured at module load
time — before any user code is imported — so patching after the fact is
detected.
"""

from __future__ import annotations

from sol_execbench.core.bench.reward_hack.models import (
    RewardHackDetected,
    SourceReview,
    SourceReviewIssue,
    SourceReviewSeverity,
)
from sol_execbench.core.bench.reward_hack.runtime import (
    _ELAPSED_TIME_ADDR as _ELAPSED_TIME_ADDR,
    check_eval_integrity,
    check_lazy_outputs,
    check_monkey_patch,
    check_runtime_integrity,
    check_thread_injection,
    snapshot_critical_functions,
    snapshot_runtime_integrity,
)
from sol_execbench.core.bench.reward_hack.static_review import review_solution_sources

__all__ = [
    "RewardHackDetected",
    "SourceReview",
    "SourceReviewIssue",
    "SourceReviewSeverity",
    "_ELAPSED_TIME_ADDR",
    "check_eval_integrity",
    "check_lazy_outputs",
    "check_monkey_patch",
    "check_runtime_integrity",
    "check_thread_injection",
    "review_solution_sources",
    "snapshot_critical_functions",
    "snapshot_runtime_integrity",
]
