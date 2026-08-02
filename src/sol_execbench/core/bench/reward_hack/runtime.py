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

import _thread
import threading
from collections.abc import Iterator
from typing import Any, Self

import torch

from sol_execbench.core.bench.reward_hack.models import RewardHackError


class _Sealed:
    """Base for holders that freeze their state at construction.

    Subclasses populate fields via ``object.__setattr__(self, name, value)``
    in ``__init__`` and then call :meth:`_seal`.  Afterwards any attribute
    mutation raises :class:`AttributeError`.  This denies a candidate reaching
    the holder via ``sys.modules['__main__']`` the trivial ``holder.x = forged``
    rewrite (audit findings dlopen-b1 / static-b2); only an
    ``object.__setattr__`` bypass defeats it, a substantially higher bar layered
    behind the static-source review.
    """

    __slots__ = ("_sealed",)

    def _seal(self) -> None:
        """Lock the holder against further attribute mutation."""
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("sealed holder is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("sealed holder is immutable")


class _SealedReference(_Sealed):
    """Immutable, identity-comparable holder for a captured function reference.

    Reward-hack defenses previously stored ``id()`` integers in mutable module
    globals.  A native ``__attribute__((constructor))`` loaded during candidate
    ``dlopen`` can both replace ``torch.cuda.Event.elapsed_time`` and rewrite a
    reachable module global, neutralizing an id()-based guard (audit finding
    ``runtime.py:40`` / dlopen-b1).  This holder instead keeps a **strong
    reference** to the pristine object captured before any candidate code runs
    and compares with identity (``is``), which also defeats CPython address
    reuse (audit finding ``runtime.py:302``).
    """

    __slots__ = ("_reference",)

    _reference: Any

    def __init__(self, reference: Any) -> None:
        """Store *reference* strongly, then seal the holder against mutation."""
        object.__setattr__(self, "_reference", reference)
        self._seal()

    @property
    def reference(self) -> Any:
        """Return the captured pristine object."""
        return self._reference

    def is_intact(self, current: Any) -> bool:
        """Return True only when *current* is the exact captured object."""
        return current is self._reference


def _resolve_elapsed_time() -> Any:
    """Return the live ``torch.cuda.Event.elapsed_time``, or None if unavailable."""
    try:
        import torch.cuda

        return torch.cuda.Event.elapsed_time
    except (AttributeError, RuntimeError):
        return None


# Sealed strong reference captured at import, before any candidate code runs.
# This is the authoritative guard used by check_monkey_patch / verify below.
_TIMING_GUARD = _SealedReference(_resolve_elapsed_time())

# Deprecated diagnostic alias for the captured id(); retained for back-compat
# with tooling and tests. NOT security-critical: check_monkey_patch reads the
# sealed strong reference above, not this integer, so rewriting the module
# global cannot defeat the guard.
_ELAPSED_TIME_ADDR: int | None = (
    id(_TIMING_GUARD.reference) if _TIMING_GUARD.reference is not None else None
)


def timing_function_available() -> bool:
    """Return True when a pristine timing function was captured at import."""
    return _TIMING_GUARD.reference is not None


def check_monkey_patch() -> None:
    """Detect if torch.cuda.Event.elapsed_time has been patched.

    Compares the **live** function object against the pristine strong reference
    captured at module import (before any candidate code runs).  Must be called
    before every timed section.

    Raises:
        RewardHackError: If the timing function identity has changed, or if it
            only became available after the guard was captured.

    """
    current = _resolve_elapsed_time()
    pristine = _TIMING_GUARD.reference
    if current is None:
        return  # torch.cuda unavailable on this target (e.g. CPU-only run)
    if pristine is None:
        raise RewardHackError(
            "torch.cuda.Event.elapsed_time became available after the integrity "
            "guard was captured; timing function cannot be trusted",
        )
    if not _TIMING_GUARD.is_intact(current):
        raise RewardHackError(
            "torch.cuda.Event.elapsed_time has been monkey-patched",
        )


def verify_timing_function_intact() -> None:
    """Re-confirm the timing function survived candidate import / dlopen.

    Call immediately after :func:`load_user_function` so a patch installed by a
    native ``__attribute__((constructor))`` during ``dlopen`` is caught before
    any workload is timed.  Functionally equivalent to :func:`check_monkey_patch`
    (the per-workload check would also detect it); named separately to make the
    post-load verification point explicit at call sites.
    """
    check_monkey_patch()


def check_thread_injection(threads_before: int, threads_after: int) -> None:
    """Detect if user code spawned background threads.

    Capture ``threading.active_count()`` before and after the user call and
    pass both values here.

    Raises:
        RewardHackError: If the thread count increased.

    """
    if threads_after > threads_before:
        raise RewardHackError(
            f"Thread injection detected: "
            f"{threads_after} threads after call vs {threads_before} before",
        )


class ThreadInjectionMonitor:
    """Detect Python thread starts and sample native/Python thread count.

    Paper §4.4.1 "Thread Injection": a worker hides work on a Python thread
    during the timed region. A single before/after ``active_count()`` delta
    (:func:`check_thread_injection`) misses workers that exit before the
    post-call check. Standard ``threading`` and ``_thread`` entry points are
    guarded synchronously, so even a worker that starts and exits between two
    samples is recorded. A daemon sampler remains as defense in depth for
    longer-lived threads created through an already-captured entry point.

    The daemon sleeps between samples (releasing the GIL) to avoid perturbing
    GPU timing, and excludes itself from the sampled count so it does not
    inflate the baseline.
    """

    def __init__(self, interval_s: float = 0.001) -> None:
        """Initialize a monitor with a positive sampling interval."""
        if interval_s <= 0:
            raise ValueError("thread sampling interval must be positive")
        self._interval = interval_s
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0
        self._peak = 0
        self._starts_observed = 0
        self._original_thread_start: Any = None
        self._original_raw_start: Any = None
        self._original_threading_raw_start: Any = None
        self._thread_start_wrapper: Any = None

    def __enter__(self) -> Self:
        """Install thread-start guards and begin sampling."""
        # Baseline is captured before the monitor thread starts, so it does not
        # include the monitor itself.
        self._baseline = threading.active_count()
        self._peak = self._baseline
        self._starts_observed = 0
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=max(self._interval * 10, 0.05)):
            self._stop.set()
            self._thread.join(timeout=0.05)
            raise RuntimeError("thread injection monitor failed to start")
        self._install_start_guards()
        return self

    def _install_start_guards(self) -> None:
        self._original_thread_start = threading.Thread.start
        self._original_raw_start = _thread.start_new_thread
        self._original_threading_raw_start = getattr(
            threading,
            "_start_new_thread",
            None,
        )
        monitor = self

        def guarded_thread_start(
            thread: threading.Thread,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return monitor._guard_thread_start(thread, *args, **kwargs)

        self._thread_start_wrapper = guarded_thread_start
        setattr(  # noqa: B010 -- Guard intentionally replaces a runtime hook
            threading.Thread,
            "start",
            guarded_thread_start,
        )
        setattr(  # noqa: B010 -- Guard intentionally replaces a runtime hook
            _thread,
            "start_new_thread",
            self._guard_raw_start,
        )
        if self._original_threading_raw_start is not None:
            setattr(  # noqa: B010 -- Guard intentionally replaces a runtime hook
                threading,
                "_start_new_thread",
                self._guard_raw_start,
            )

    def _restore_start_guards(self) -> None:
        if self._original_thread_start is not None:
            setattr(  # noqa: B010 -- Restore the guarded runtime hook
                threading.Thread,
                "start",
                self._original_thread_start,
            )
        if self._original_raw_start is not None:
            setattr(  # noqa: B010 -- Restore the guarded runtime hook
                _thread,
                "start_new_thread",
                self._original_raw_start,
            )
        if self._original_threading_raw_start is not None:
            setattr(  # noqa: B010 -- Restore the guarded runtime hook
                threading,
                "_start_new_thread",
                self._original_threading_raw_start,
            )

    def _guard_thread_start(
        self,
        thread: threading.Thread,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._starts_observed += 1
        return self._original_thread_start(thread, *args, **kwargs)

    def _guard_raw_start(
        self,
        function: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self._starts_observed += 1
        return self._original_raw_start(function, args, kwargs or {})

    def _sample(self) -> None:
        while not self._stop.is_set():
            # Exclude this monitor thread from the count.
            current = threading.active_count() - 1
            self._peak = max(self._peak, current)
            self._ready.set()
            if self._stop.wait(self._interval):
                break

    def __exit__(self, *exc: object) -> bool:
        """Stop sampling, restore thread entry points, and propagate errors."""
        self._restore_start_guards()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval * 10, 0.05))
        return False

    @property
    def baseline(self) -> int:
        """Return the active-thread count captured before monitoring."""
        return self._baseline

    @property
    def peak(self) -> int:
        """Return the largest active-thread count observed."""
        return self._peak

    @property
    def starts_observed(self) -> int:
        """Return the number of guarded thread starts observed."""
        return self._starts_observed


def check_thread_injection_from_monitor(
    monitor: ThreadInjectionMonitor,
) -> None:
    """Raise if the monitor observed more threads than its baseline.

    Companion to :class:`ThreadInjectionMonitor`: compares the baseline
    captured at monitor entry against the peak sampled during the timed region.

    Raises:
        RewardHackError: If the peak thread count exceeded the baseline.

    """
    if monitor.starts_observed > 0:
        raise RewardHackError(
            "Thread injection detected: "
            f"{monitor.starts_observed} thread start event(s) during timed execution",
        )
    if monitor.peak > monitor.baseline:
        raise RewardHackError(
            f"Thread injection detected: peak {monitor.peak} threads during "
            f"timed execution vs {monitor.baseline} at baseline",
        )


def check_lazy_outputs(outputs: list[Any]) -> None:
    """Detect lazy/proxy tensors in the user output.

    Uses strict ``type()`` equality — not ``isinstance`` — so any subclass
    (including ``FakeTensor``) is rejected.

    Raises:
        RewardHackError: If any output is not exactly ``torch.Tensor``.

    """
    for t in outputs:
        if type(t) is not torch.Tensor:
            raise RewardHackError(
                f"Lazy evaluation detected: output is {type(t).__name__}, not torch.Tensor",
            )


class _IntegritySnapshot(_Sealed):
    """Frozen, identity-checked snapshot of critical eval functions.

    Holds **strong references** (not ``id()`` integers) to every snapshotted
    function captured before candidate code runs.  ``check_eval_integrity``
    compares each live namespace entry against the stored reference with
    identity (``is``), defeating both CPython address reuse and the
    "rewrite-the-snapshot-in-__main__" attack (audit finding ``runtime.py:322``
    / static-b2): the holder is sealed and exposes a read-only
    :class:`types.MappingProxyType`, so a candidate reaching it via
    ``sys.modules['__main__']`` cannot mutate the stored references to match
    patched live functions without ``object.__setattr__`` (layered behind the
    static-source review).
    """

    __slots__ = ("_proxy", "_references")

    _proxy: Any
    _references: dict[str, Any]

    def __init__(self, references: dict[str, Any]) -> None:
        """Store a private copy of *references* and expose a read-only view."""
        from types import MappingProxyType

        object.__setattr__(self, "_references", dict(references))
        object.__setattr__(self, "_proxy", MappingProxyType(self._references))
        self._seal()

    def __contains__(self, name: object) -> bool:
        return name in self._proxy

    def __iter__(self) -> Iterator[str]:
        return iter(self._proxy)

    def __len__(self) -> int:
        return len(self._proxy)

    def reference_of(self, name: str) -> Any:
        """Return the captured reference for *name*."""
        return self._references[name]


def snapshot_critical_functions(
    namespace: dict[str, Any],
    names: list[str],
) -> _IntegritySnapshot:
    """Capture strong references to named callables from a namespace.

    Call this **before** user code is imported.  Pass the returned snapshot to
    :func:`check_eval_integrity` after user code runs.

    Args:
        namespace: The globals dict to snapshot (typically ``globals()``).
        names: Function names to capture.

    Returns:
        Frozen snapshot mapping each present name to its captured reference.

    """
    return _IntegritySnapshot(
        {name: namespace[name] for name in names if name in namespace},
    )


def check_eval_integrity(
    snapshot: dict[str, Any] | _IntegritySnapshot,
    namespace: dict[str, Any],
) -> None:
    """Verify that critical eval-driver functions have not been replaced.

    Compares the **live** object for each snapshotted name against the stored
    strong reference using identity (``is``).  Accepts the
    :class:`_IntegritySnapshot` returned by :func:`snapshot_critical_functions`
    or a plain mapping (an empty dict always passes, for test/initial use).

    Args:
        snapshot: The snapshot returned by :func:`snapshot_critical_functions`.
        namespace: The current globals dict to check.

    Raises:
        RewardHackError: If any function identity has changed or been deleted.

    """
    if isinstance(snapshot, _IntegritySnapshot):
        names = list(snapshot)
        get_reference = snapshot.reference_of
    else:
        names = list(snapshot.keys())
        get_reference = snapshot.__getitem__  # type: ignore[union-attr]
    for name in names:
        current = namespace.get(name)
        if current is None or current is not get_reference(name):
            raise RewardHackError(
                f"Eval driver integrity violated: '{name}' has been monkey-patched",
            )


def _runtime_integrity_namespace(
    driver_namespace: dict[str, Any],
) -> dict[str, Any]:
    import importlib
    import sys

    namespace = {
        name: value
        for name, value in driver_namespace.items()
        if callable(value) and not name.startswith("_")
    }
    modules = (
        ("eval_correctness", "sol_execbench.core.bench.eval_correctness"),
        ("eval_runtime", "sol_execbench.core.bench.eval_runtime"),
        ("eval_timing", "sol_execbench.core.bench.eval_timing"),
        ("eval_execution", "sol_execbench.core.bench.eval_workload_execution"),
        ("timing", "sol_execbench.core.bench.timing"),
    )
    for prefix, module_name in modules:
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if callable(value) and not name.startswith("__"):
                namespace[f"{prefix}.{name}"] = value
    return namespace


def snapshot_runtime_integrity(
    driver_namespace: dict[str, Any],
) -> _IntegritySnapshot:
    """Snapshot driver and cross-module timing/correctness functions."""
    namespace = _runtime_integrity_namespace(driver_namespace)
    return snapshot_critical_functions(namespace, list(namespace))


def check_runtime_integrity(
    snapshot: dict[str, Any] | _IntegritySnapshot,
    driver_namespace: dict[str, Any],
) -> None:
    """Detect monkey-patching across the complete evaluation call graph."""
    check_eval_integrity(
        snapshot,
        _runtime_integrity_namespace(driver_namespace),
    )
