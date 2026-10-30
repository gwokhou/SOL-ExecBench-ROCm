"""CPU substitutes prove preparation order, strict reuse and GPU cleanup boundaries."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
    "scripts/internal/rdna4/run_qualified_rdna4_resource_peak_calibration.py"
)
PRODUCER = "scripts/internal/rdna4/run_rdna4_resource_peak_calibration.py"


@pytest.fixture
def qualified(load_script, monkeypatch, tmp_path):
    module = load_script(SCRIPT)
    active = False
    events = []
    compiler = tmp_path / "hipcc"
    compiler.write_text("CPU compiler fixture")
    for name in ("a.hip", "b.hip"):
        (tmp_path / name).write_text(name)

    @contextmanager
    def lease(*_, **__):
        nonlocal active
        if _ and _[0] == "cpu":
            yield
            return
        assert not active
        events.append("acquire")
        active = True
        try:
            yield
        finally:
            events.append("cleanup")
            active = False

    def compile_probe(source, workdir, *_args, **_kwargs):
        assert not active, "compilation held the GPU lease"
        events.append("compile")
        path = workdir / "binaries" / source.stem / f"{source.stem}-default.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.read_text())
        return path

    def device():
        assert active, "device query escaped the lease"
        events.append("device")
        return {"gfx_target": "gfx1200", "device_name": "CPU fixture"}

    def clock():
        assert active
        events.append("clock")
        return {"clock_locked_verified": True}

    def sample(*_args, **_kwargs):
        assert active
        events.append("sample")
        return SimpleNamespace(to_dict=lambda: {"samples": [1.0] * 7})

    legacy = {
        "PROBE_DIR": tmp_path,
        "PROBES": ({"source": "a.hip"}, {"source": "b.hip"}),
        "_required_rocm_tool": lambda _: str(compiler),
        "_run": lambda _: SimpleNamespace(
            stdout="CPU compiler fixture version"
        ),
        "_git_revision": lambda: "fixture",
        "_compile_probe": compile_probe,
        "_compile_calibration_binaries": lambda gfx, workdir, hipcc, **kwargs: {
            name: {None: compile_probe(tmp_path / name, workdir, hipcc, gfx)}
            for name in ("a.hip", "b.hip")
        },
        "_device_identity": device,
        "_clock_state": clock,
        "_run_sample_batch": sample,
    }
    monkeypatch.setattr(module, "_legacy", lambda: legacy)
    monkeypatch.setattr(module, "_CANARY_PROBES", ("a.hip",))
    monkeypatch.setattr(module, "native_gpu_execution", lease)
    verify = module.verify_qualification_artifact

    def verify_cpu(*args):
        assert not active, "artifact hashing held the GPU lease"
        events.append("verify")
        return verify(*args)

    monkeypatch.setattr(module, "verify_qualification_artifact", verify_cpu)
    common = [
        "--qualification-root",
        str(tmp_path / "qualification"),
        "--out",
        str(tmp_path / "result.json"),
    ]
    return module, legacy, events, common


def _complete_chain(qualified):
    module, _, _, common = qualified
    for stage in ("static", "canary", "full"):
        assert module.main([f"qualify-{stage}", *common]) == 0


def test_chain_queries_device_once_per_gpu_stage_and_reuses(qualified, capsys):
    module, _, events, common = qualified
    _complete_chain(qualified)
    assert events.count("compile") == 2
    assert events.count("device") == events.count("clock") == 2
    assert events.count("sample") == 3
    assert module.main(["qualify-full", *common]) == 0
    assert events.count("sample") == 3
    assert (
        events.count("device") == 3
    )  # Fresh readback even for cached GPU gate.
    assert events.count("acquire") == events.count("cleanup") == 3
    before = list(events)
    assert module.main(["qualify-full", *common, "--check-only"]) == 0
    assert events[len(before) :] and set(events[len(before) :]) == {"verify"}
    output = capsys.readouterr()
    assert '"device_readback_performed": false' in output.out
    assert '"cpu_prepare"' in output.err and '"cpu_finalize"' in output.err


@pytest.mark.parametrize(
    "drift", ["binary", "receipt", "compiler", "source", "configuration"]
)
def test_stale_chain_rejected_before_lease(qualified, tmp_path, drift):
    module, _, events, common = qualified
    _complete_chain(qualified)
    if drift == "binary":
        path = tmp_path / "qualification/static/binaries/a/a-default.bin"
    elif drift == "receipt":
        path = tmp_path / "qualification/canary/a.hip/evidence.json"
    elif drift == "compiler":
        path = tmp_path / "hipcc"
    elif drift == "source":
        path = tmp_path / "a.hip"
    else:
        path = None
        common += ["--measurement-batches", "8"]
    if path:
        path.write_text(path.read_text() + " ")
    events.clear()
    with pytest.raises(ValueError, match="drift"):
        module.main(["qualify-full", *common])
    assert "acquire" not in events and "sample" not in events


def test_device_mismatch_and_cancellation_keep_cleanup(
    qualified, monkeypatch, tmp_path, capsys
):
    module, legacy, events, common = qualified
    module.main(["qualify-static", *common])
    original_device = legacy["_device_identity"]
    legacy["_device_identity"] = lambda: {"gfx_target": "gfx9999"}
    with pytest.raises(RuntimeError, match="visible GPU"):
        module.main(["qualify-canary", *common])
    assert events[-1] == "cleanup" and "sample" not in events
    legacy["_device_identity"] = original_device

    def cancel(*_args, **_kwargs):
        raise KeyboardInterrupt

    legacy["_run_sample_batch"] = cancel
    with pytest.raises(KeyboardInterrupt):
        module.main(["qualify-canary", *common])
    assert events[-1] == "cleanup"
    assert not (tmp_path / "qualification/canary/gate.json").exists()
    assert (
        '"gpu_lifecycle_including_wait_and_cleanup": false'
        in capsys.readouterr().err
    )


def test_static_compile_failure_never_acquires(qualified):
    module, legacy, events, common = qualified

    def fail(*_args, **_kwargs):
        raise RuntimeError("compile failed")

    legacy["_compile_calibration_binaries"] = fail
    with pytest.raises(RuntimeError, match="compile failed"):
        module.main(["qualify-static", *common])
    assert not events


@pytest.mark.parametrize("entry", ["direct", "qualified"])
def test_formal_compiles_before_lease_and_preserves_cleanup(
    entry, load_script, monkeypatch, tmp_path
):
    producer = load_script(PRODUCER)
    module = producer if entry == "direct" else load_script(SCRIPT)
    active = False
    compiled = []
    events = []
    monkeypatch.setattr(
        producer,
        "PROBES",
        (
            {
                "source": "a.hip",
                "tuning": producer.TuningSpec("unroll", "UNROLL", (1, 2, 4)),
            },
            {"source": "b.hip"},
        ),
    )
    monkeypatch.setattr(producer, "_required_rocm_tool", lambda _: "cpu-hipcc")
    monkeypatch.setattr(producer, "_isa_spec_evidence", lambda _: {})

    def compile_probe(source, workdir, *_args, compiler_defines, **_kwargs):
        assert not active
        path = workdir / f"{source.name}-{tuple(compiler_defines.values())}.bin"
        path.write_text("CPU executable fixture")
        compiled.append(path)
        return path

    monkeypatch.setattr(producer, "_compile_probe", compile_probe)

    @contextmanager
    def lease(*_, **__):
        nonlocal active
        assert len(compiled) == 4
        active = True
        events.append("acquire")
        try:
            yield
        finally:
            active = False
            events.append("cleanup")

    def calibrate(_args, _workdir, prepared):
        assert active
        prepared.verify()
        assert len(prepared.binaries["a.hip"]) == 3
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "native_gpu_execution", lease)
    monkeypatch.setattr(producer, "_calibrate", calibrate)
    if entry == "qualified":
        monkeypatch.setattr(
            module,
            "_legacy",
            lambda: {
                "_prepare_calibration": producer._prepare_calibration,
                "_calibrate": calibrate,
            },
        )
        monkeypatch.setattr(
            module, "_verify_qualification", lambda *_, **__: None
        )
        monkeypatch.setattr(module, "_check_device", lambda *_: None)
        args_stub = SimpleNamespace(gfx="gfx1200")
        prepared = producer._prepare_calibration(args_stub, tmp_path)
        monkeypatch.setattr(
            module, "_prepare_qualified_calibration", lambda *_: prepared
        )
    args = ["--workdir", str(tmp_path), "--out", str(tmp_path / "formal.json")]
    if entry == "qualified":
        args = ["run", "--qualification-root", str(tmp_path / "qual"), *args]
    with pytest.raises(KeyboardInterrupt):
        module.main(args)
    assert events == ["acquire", "cleanup"]


def test_prepared_binary_drift_is_rejected(load_script, tmp_path):
    module = load_script(PRODUCER)
    binary = tmp_path / "probe.bin"
    binary.write_bytes(b"before")
    prepared = module.CalibrationPreparation(
        "hipcc", {}, {binary: module.sha256_file(binary)}, {}
    )
    binary.write_bytes(b"after")
    with pytest.raises(ValueError, match="prepared calibration binary drift"):
        prepared.verify()


def test_parent_replaced_during_gpu_phase_is_not_rebound(qualified, tmp_path):
    module, legacy, events, common = qualified
    module.main(["qualify-static", *common])
    original = legacy["_run_sample_batch"]
    parent = tmp_path / "qualification/static/gate.json"

    def replace_parent(*args, **kwargs):
        batch = original(*args, **kwargs)
        parent.write_text(parent.read_text() + " ")
        return batch

    legacy["_run_sample_batch"] = replace_parent
    with pytest.raises(ValueError, match="parent changed"):
        module.main(["qualify-canary", *common])
    assert events[-1] == "cleanup"
    assert not (tmp_path / "qualification/canary/gate.json").exists()


def test_formal_uses_only_static_bound_binaries(
    qualified, load_script, monkeypatch, tmp_path
):
    module, legacy, _, common = qualified
    producer = load_script(PRODUCER)
    _complete_chain(qualified)
    args = module._parse_args(["run", *common])
    gates = []
    module._verify_qualification(
        args, module.BatchGPUQualificationStage.FULL, gates=gates
    )
    legacy.update(
        {
            "_calibration_binary_paths": lambda root: {
                "a.hip": {None: root / "binaries/a/a-default.bin"},
                "b.hip": {None: root / "binaries/b/b-default.bin"},
            },
            "CalibrationPreparation": producer.CalibrationPreparation,
            "_isa_spec_evidence": lambda _: {},
        }
    )
    prepared = module._prepare_qualified_calibration(args, gates, legacy)
    assert len(prepared.binary_hashes) == 2
    # No compiler is available in this consumer: the successful preparation above
    # can only consume the already bound static artifacts.
    del legacy["_compile_probe"]
    del legacy["_compile_calibration_binaries"]
    module._prepare_qualified_calibration(args, gates, legacy)
    binary = tmp_path / "qualification/static/binaries/a/a-default.bin"
    binary.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="binary drift"):
        module._prepare_qualified_calibration(args, gates, legacy)


def test_isa_preparation_never_enables_downloads(load_script, monkeypatch):
    module = load_script(PRODUCER)
    observed = []

    def inspect(_arch, _checks, *, allow_download):
        observed.append(allow_download)
        return SimpleNamespace(supported_instructions=(), to_dict=dict)

    monkeypatch.setattr(module, "helper_path", lambda: Path("/bin/true"))
    monkeypatch.setattr(module, "inspect_isa_requirements", inspect)
    module._isa_spec_evidence("gfx1200")
    assert observed == [False]


def test_precompilation_preserves_counterbalanced_tuning(
    load_script, monkeypatch, tmp_path
):
    module = load_script(PRODUCER)
    tuning = module.TuningSpec("unroll", "UNROLL", (1, 2, 4))
    binaries = {value: tmp_path / str(value) for value in tuning.candidates}
    order = []

    def sample(binary, batch, **_kwargs):
        value = int(binary.name)
        order.append((batch, value))
        return module.SampleBatch(batch, (float(value),) * 7)

    def candidate(_tuning, value, _binary, _batches):
        return {
            "selection_result": 10 if value in (2, 4) else 1,
            "configuration": {"unroll": value},
        }

    monkeypatch.setattr(module, "_run_sample_batch", sample)
    monkeypatch.setattr(module, "_tuning_candidate_entry", candidate)
    prepared = module._tune_probe(
        {"tuning": tuning}, tmp_path / "probe.hip", binaries, 2
    )
    assert order == [(0, 1), (0, 2), (0, 4), (1, 4), (1, 2), (1, 1)]
    assert prepared.selected_configuration == {"unroll": 2}
    assert (
        prepared.tuning_evidence["held_out_samples_used_for_selection"] is False
    )


def test_missing_isa_helper_fails_without_build_or_gpu(
    load_script, monkeypatch, tmp_path
):
    module = load_script(PRODUCER)
    monkeypatch.setattr(module, "helper_path", lambda: tmp_path / "absent")
    monkeypatch.setattr(
        module,
        "inspect_isa_requirements",
        lambda *_args, **_kwargs: pytest.fail("implicit ISA preparation"),
    )
    with pytest.raises(FileNotFoundError, match="prebuilt AMD ISA helper"):
        module._isa_spec_evidence("gfx1200")
