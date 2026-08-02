# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""End-to-end coverage for the native dlopen constructor defense (P0-3).

The eval driver calls ``verify_timing_function_intact()`` immediately after
``load_user_function`` so a ``__attribute__((constructor))`` in a compiled
extension that swaps ``torch.cuda.Event.elapsed_time`` is caught before any
workload is timed. These tests compile a real native extension whose
constructor performs that swap and confirm the sealed guard detects it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sol_execbench.core.bench.reward_hack import RewardHackError

_EVIL_CONSTRUCTOR_CPP = """\
#include <Python.h>

static PyObject* _fake_elapsed_time(PyObject* self, PyObject* args) {
    return PyFloat_FromDouble(0.001);
}

static PyMethodDef _fake_method = {
    "_fake_elapsed_time",
    (PyCFunction)_fake_elapsed_time,
    METH_VARARGS,
    "",
};

__attribute__((constructor))
static void _patch_elapsed_time(void) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* torch_mod = PyImport_ImportModule("torch");
    PyObject* cuda_mod = torch_mod ? PyObject_GetAttrString(torch_mod, "cuda") : NULL;
    PyObject* event_cls = cuda_mod ? PyObject_GetAttrString(cuda_mod, "Event") : NULL;
    if (event_cls) {
        PyObject* fake = PyCFunction_New(&_fake_method, NULL);
        if (fake) {
            PyObject_SetAttrString(event_cls, "elapsed_time", fake);
            Py_DECREF(fake);
        }
        Py_DECREF(event_cls);
    }
    Py_XDECREF(cuda_mod);
    Py_XDECREF(torch_mod);
    PyGILState_Release(gstate);
}

static PyModuleDef _module_def = {
    PyModuleDef_HEAD_INIT,
    "evil_elapsed_time_ctor",
    NULL,
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_evil_elapsed_time_ctor(void) {
    return PyModule_Create(&_module_def);
}
"""


@pytest.mark.cpp
@pytest.mark.requires_rocm_dev
def test_native_constructor_elapsed_time_patch_is_detected(
    tmp_path: Path,
) -> None:
    """A native constructor that swaps elapsed_time during dlopen is caught.

    Mirrors the eval-driver defense chain (static review -> compile -> dlopen
    via ``load_user_function`` -> ``verify_timing_function_intact``) for audit
    finding dlopen-b1 / P0-3: a constructor that patches the timing function
    must be detected before any workload is timed.
    """
    from torch.utils.cpp_extension import load

    from sol_execbench.core.bench.reward_hack import (
        verify_timing_function_intact,
    )

    source = tmp_path / "evil_elapsed_time_ctor.cpp"
    source.write_text(_EVIL_CONSTRUCTOR_CPP, encoding="utf-8")

    original = torch.cuda.Event.elapsed_time
    try:
        load(
            name="evil_elapsed_time_ctor",
            sources=[str(source)],
            # Isolate the compile so parallel xdist workers cannot collide on
            # the shared ~/.cache/torch_extensions namespace.
            build_directory=str(tmp_path),
            verbose=False,
        )
        # The constructor ran during load and replaced elapsed_time.
        assert torch.cuda.Event.elapsed_time is not original
        with pytest.raises(RewardHackError, match="elapsed_time"):
            verify_timing_function_intact()
    finally:
        torch.cuda.Event.elapsed_time = original
