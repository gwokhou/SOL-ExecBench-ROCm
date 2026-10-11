# SOLAR upgrade: official-source check

Checked 2026-10-11 against official documentation and tagged source. Repository HEAD during this check: `1b431d7455a4ba045fbb9646a448b3b0f758aeb4`. This supplements the October 10 report; it does not reuse that report's old dependency baseline. No production changes, package installation, tests, or GPU runs were performed for these notes.

## PyTorch shape declarations

PyTorch 2.14 introduces the experimental `ShapesSpec` / `ShapeVar` / `dynamic_spec` interface across compile, export and tracing. Shared dimensions, derived expressions and assumptions are supported. `make_fx` integration requires fake tracing. Declared dimensions become unbacked symbols: shape-dependent branches may fail instead of specializing. Combining a decorator with explicit `dynamic_shapes`, or a spec with `prefer_deferred_runtime_asserts_over_guards=True`, is rejected. This is a new declaration interface, not the invention of dynamic shapes, FakeTensor, export or compilation. [PyTorch 2.14 announcement](https://pytorch.org/blog/pytorch-2-14-release-blog/#declarative-dynamic-shapes-with-dynamic_spec)

Tagged source confirms that an explicit spec does **not** override an attached decorator: the ambiguity produces `ValueError`. `ShapesSpec` assumptions are `SymBool` expressions and use deferred runtime assertions. `to_jsonable()` serializes assumptions through `repr`; it supplies neither a SOLAR-owned schema nor a demonstrated stable replay contract. `ShapeVar` derives from `IntVar`; leaf declarations distinguish dynamic variables, derived `SymInt`, concrete integers and unspecified/static slots. [v2.14.0 dynamic_spec.py](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/fx/experimental/dynamic_spec.py)

Engineering inference: keep a SOLAR-owned, validated shape-domain contract and explicitly reject unsupported branches/constraints. A successful fake trace does not prove a bound for every point in that domain. Upstream API capability also does not establish that an AMD downstream wheel contains the identical implementation. The tagged `proxy_tensor.py` URL did not load through the browsing tool; the fake-mode restriction above is confirmed by the official release announcement, not an executed local probe.

## ROCm profiler boundaries

ROCm 10.1 adds HIP event barrier activities; `--hip-runtime-trace` enables them automatically, but their native outputs are JSON/rocpd. CSV requires database conversion. It fixes gfx12 ATT occupancy mode and a PyTorch-bundled SDK/rocprofv3 initialization conflict. Compute Profiler adds interactive roofline output and removes its CSV **profile backend**; this does not mean every rocprofv3 CSV output disappears. Kernel replay is beta; rocprofv3 restricts it to counters with explicit opt-in and excludes ATT, PC sampling and SPM combinations. [Versioned ROCm 10.1 release notes](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)

Current SDK documentation separately describes a more general replay callback domain. Snapshot coverage excludes managed, host, stream-ordered/pool allocations and HIP graph-managed memory. HIP graphs run once with a warning; host-memory pressure can also decline replay. Agent serialization and unfenced asynchronous copies matter. These details come from mutable `latest` documentation, so verify the installed SDK before relying on them. [SDK kernel replay guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-kernel-replay.html)

Engineering inference: event analysis requires JSON/rocpd ingestion; CSV collection success alone is insufficient. The coordinating current-source audit reports that local, uncommitted `hip_events.py` and JSON/rocpd parsers already implement this path, so this is a contract to validate, not a claim that today's checkout lacks a parser. ATT occupancy repair does not establish that the separate SQ_WAVE_CYCLES AUTO-clock issue is repaired. Replay must report actual pass coverage and state fidelity; it cannot silently substitute for canonical unperturbed timing. These capabilities explain measured gaps around SOLAR rather than strengthen its mathematical bound automatically.

## Origami and installation identity

Origami's 2.14 change is default enablement for ROCm max-autotune, using a latency model for GEMM tile selection. It is not a universal eager-mode optimization or a formal lower-bound method. Grouped GEMM/CK improvements belong to compiler execution paths. No gfx1200 benefit is established by these release statements alone. [PyTorch ROCm changes](https://pytorch.org/blog/pytorch-2-14-release-blog/#rocm)

The AMD installation page has moved to AI Ecosystem. Its current destination explicitly lists `rocm/pytorch:rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0` and gfx1200-specific `torch[device-gfx1200]==2.14.0+rocm10.1.0` wheels. The old URL now contains a redirect notice and stale material. Use the new destination when citing this combination; no image digest or wheel contents were checked here. [AMD installation instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

## Python 3.14

`InterpreterPoolExecutor` and `Executor.map(buffersize=...)` are new in 3.14. The former uses threads with separate interpreters/GILs and serialized work; the latter bounds submitted results awaiting consumption. [concurrent.futures](https://docs.python.org/3.14/library/concurrent.futures.html)

Subinterpreters and per-interpreter parallelism predate 3.14; the convenient standard-library interface is new. Extension compatibility is not universal and in-process interpreter isolation is not strict process isolation. [concurrent.interpreters](https://docs.python.org/3.14/library/concurrent.interpreters.html)

Linux's applicable POSIX default multiprocessing start method changes from fork to forkserver. Code explicitly using another context must be assessed separately; direct subprocess launch is not changed by this default. [multiprocessing](https://docs.python.org/3.14/library/multiprocessing.html)

Free threading requires an appropriate build/runtime; installing ordinary Python 3.14 does not disable the GIL. [Free-threading HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html)

Engineering inference: first evaluate bounded scheduling for offline, serialized analysis. Do not infer that Torch graph extraction, extension imports, external Orojenesis workers, or GPU state can safely migrate to subinterpreters. Benchmark before claiming speedup; preserve subprocess isolation where it owns cleanup and failure handling.
