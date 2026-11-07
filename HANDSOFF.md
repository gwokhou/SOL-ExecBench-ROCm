# Project handoff and active follow-ups

Last audited: 2026-08-13 against source revision
`fd7970520e205318e2e63d8a96d5203e1d5972c3`, the live ignored evidence/store,
the remote P0 conformance Release metadata, and the completed start-460 /
start640 rejections described below.

This file records unresolved repository-level work, the decisions that
constrain it, and the evidence that closed the lifecycle production-topology
P0. Generated run state belongs in the diagnostic lifecycle registry; detailed
completed investigations and one-time inventories belong in Git history.

## Current state

- The official gfx1200 score is published: `0.497818327` for the
  `rx9060xt-gfx1200-reference-v2` baseline against the
  `rx9060xt-gfx1200-eager-reference-self-eval` candidate over 43 problems and
  163 workloads. `RELEASE/release-bundle.json` is the publish marker; GitHub
  Release `gfx1200-official-score-v2` is complete and not part of the backlog.
- Current performance-diagnostic contracts: validation corpus v9, performance
  diagnostic v7, calibration v8, inference profile v10, lifecycle run v3,
  BenchmarkConfig v2, reference IPC v2, ROCm event timing v4. Superseded schema
  readers are intentionally absent.
- HEAD implements mandatory pre-collection qualification: an isolated,
  content-bound chain of `qualify-static` (zero-GPU, all 660 design cases),
  `qualify-canary` (per-axis extrema), then `qualify-full` (every workload in
  the selected role), each writing hashed family receipts and binding
  design/contract/collector/source identity. Any drift blocks `collect` before
  its first case. This vocabulary governs every current large-batch GPU
  producer; all qualification timing is non-authoritative.
- Successor history (all terminal, detailed evidence in Git history and the
  lifecycle registry):
  - **start-460** — first full lifecycle; acceptance `accepted=false`
    (median APE 13.0% but P90 61.6%, `restore_wmma_path` recall 0.55). Held-out
    corpus exposed and ineligible for reuse.
  - **Cycle 2 / Cycle 3 (start-220)** — Cycle 2 revealed the working-set
    coordinate change; Cycle 3 froze `M=1032` (beyond the 1024 candidate limit)
    and stopped terminally incomplete. Design
    `2bcfa7fc7feadb39a165b03d6a855d73045b9ae536dabda445a1cdbb1dbc60ee`.
  - **start-280 / start-340 / start-400** — exploratory repairs; start-340 and
    start-400 closed the elementwise capacity gap but stopped pre-verdict
    (`calibration_out_of_range:working_set_bytes`). The conservative reuse unit
    is a family: an exposed case taints its 20-case family; the other families
    reuse by exact artifact identity.
  - **start-640 (`p1-successor-start640-pcie5x16-r1`)** — closed terminal.
    Full lifecycle at `8848d605` via recovery chain `d1ee4324→18535cf→8848d605`;
    acceptance `9f740c5bc9a971e736f0f735161bf823a61e071898b9fcfac835a9c711839327`
    recorded `accepted=false` (model-generalization, same class as start-460).
    No publication/release/tag exists; tag
    `gfx1200-diagnostics-v7-production-v1` must not be created.
- **The start640 coverage failure was root-caused as a cross-design
  distribution shift**, not a model bug: `composite_graph` held-out fell 11/20
  outside the reused development measured range (hard extrapolation), while the
  elementwise family (in-range) reached 100% coverage. The conformal interval is
  a single `solar_lower_bound_ms` point model scaled by `exp(q95)`; the reused
  development `q95` sits far below the held-out P90 residual, and a
  leave-one-out residual over the full development corpus (scheme A) was
  falsified offline. The fix is a single i.i.d.-split design, not a conformal
  code change.
- A fully fresh successor **`p1-successor-start700`** was authored (commit
  `cb35de63`) with capacity-bounded `elementwise`/`transpose` shapes and a
  disjoint 15-neighborhood transformer schedule, reusing the attested
  `d1ee4324` VRAM policy (`dbac7df4`) via the same recovery chain and a fresh
  `source-review-18535cf-to-cb35de63.json`. Design ID
  `410042044bc9c67fa82e048ba49de54662fff27c27a233579658d4cc13f2d1e6`.
- Commit `cbaa3822` drops the rocprofv3 `rocpd` SQLite profiling database from
  counter collection (it was never parsed — the 6-counter model consumes only
  the counter CSV; marker alignment uses the marker CSV; publication omits it).
  Measured effect: collect ~2.5 → ~2.15 min/case (~13%); each new case stops
  writing ~13 MB, and ~3.0 GB of historical `.db.gz` no longer accumulates.
- The start700 qualification chain was re-run at `cbaa3822` (the commit changed
  the bound `source_revision`); all five gates verify (static `6fe2bb7959`,
  canary/full development and held-out). Development collection then produced
  **69 of 440** cases before the operator paused it. **The RDNA4 diagnostic
  production re-release is deliberately on hold**: no frozen development corpus,
  SOLAR, inference profile, held-out collection, or acceptance verdict exists,
  and no publication/release/tag exists. Resumption is
  `collect --role development` from the paused root without re-qualifying.
- The production-topology lifecycle P0 closed on 2026-08-09 (conformance run
  `c96d65d534bdfe51ac23b0fda026721a614fa6f3e03388fa1ca3da533a922096`, GitHub
  Release `diagnostic-lifecycle-p0-conformance-v1`). It proves the mechanism
  but does not itself authorize production acceptance or publication.
- Do not remove `microarchitecture-diagnostics-v7/` or
  `microarchitecture-diagnostics-v7-cycle2/` merely because CAS import
  completed; expanded source-tree retirement needs a reviewed plan.
- Real-device evidence is strongest on one RX 9060 XT/gfx1200 host; multi-GPU
  and CDNA coverage is narrower. Static AMDGPU metadata extraction does not
  parse the CCOB manifest.

## Active backlog

<a id="solar-evolution"></a>

### SOLAR fake-static backend adoption (线B) — 2026-11-07

The L3-a adopt decision landed: the production make_fx backend now traces
under FakeTensorMode with the workload's concrete shapes
([spec + erratum](docs/internal/solar-lineb-fake-production-acceptance-2026-11-07.md);
evidence `data/outputs/solar-lineb-fake-20261107/`; merged `8be1e62a`/`e0345df7`).

- **B1 full-corpus equality**: 163/163 scored workloads produce
  byte-identical `operator_graph.yaml` versus the pristine real backend
  (dual-tree campaign; real side = main at `39329249`, fake side via
  PYTHONPATH worktree shadow). First pass 159/163; the 4 `rope_thd_fwd_bf16`
  slots convert a tensor-derived scalar and fake metadata cannot propagate.
- **Declared real fallback**: `DataDependentOutputException`/
  `DynamicOutputShapeException` retry on the incumbent real-tensor trace —
  pre-adoption behavior, not a verification bypass; the 4 slots' graphs are
  byte-identical to real (B1 rerun). nonzero-style ops keep their
  pre-adoption extraction capability through the same fallback.
- **B3 GPU validation**: 3/3 representative slots (R06 dynamic index, R04
  attention, l2n52 conv) ready on cuda:0 with the fake backend.
- Pilot module `make_fx_fake.py` deleted (production owns the path);
  importers atomically updated. Suite 1870 passed on main after merge;
  ruff/ty clean. Benefit (L3-a): extraction wall 11–196× faster,
  verification replay still real by design.
- Execution notes: `_RELEASE_SOURCE_PATHS` covers `scripts` — the lineC
  chain window froze the main tree, so lineB developed in a git worktree
  and merged after `09b8c55e`; a second venv exceeded disk quota, solved
  with a PYTHONPATH shadow over the main venv.

### SOLAR full-denominator qualification chain — 2026-11-07

The L1-deferred full-denominator qualification chain closed on the
make_fx_aten denominator
([spec](docs/internal/solar-linec-qualification-acceptance-2026-11-07.md);
evidence `data/outputs/solar-qualification-mfx-20261107/`).

- **Chain**: `qualify-static → qualify-canary → qualify-full` via the
  production CLI on a workspace rebuilt at HEAD (`39329249`; the L1
  workspace pinned `12859e3e` and was correctly refused — fail-closed
  revision binding). Gates hash-chained: static 163 items → canary 112
  (risk-first) → full 163.
- **Results**: static pass; canary readiness **112/112 ready**; full
  readiness **163/163 ready** — the L1-era canary wall (33 torchview
  failures) does not exist on the make_fx denominator. Identity
  parameters must match across stages (first canary attempt failed
  `identity drift: static` on a timeout-parameter mismatch — rerun with
  matched `--timeout 3600`).
- **Meaning**: with make_fx readiness at 163/163 and Line A's
  external-reads denominator cleared, the corpus publication
  qualification is achieved for the make_fx path under the chain's own
  semantics; the torchview denominator chain remains open (42 readiness
  gaps, L3-c matrix) and is not required for this closure.
- Roadmap L1-d deferred item is now settled: full-denominator chain
  complete (make_fx), recorded against the frozen corpus
  sha `24cee562…`, pinned mapper `18591892…`, gfx1200 device.

### SOLAR L2 schedule-causality execution (线B 受限域闭合) — 2026-11-07

Line B (调度因果, roadmap L2) closed its restricted-domain objective: within
the audited eager executor scope, `schedule_complete` is qualified for the
first time, with the copy dependency domain and cross-lane event edges
verified ([spec](docs/internal/solar-l2-schedule-causality-acceptance-2026-11-06.md);
evidence `data/outputs/solar-l2-schedule-20261106/`, GPU-a30fa29c1cca1511,
ROCm 10.1 / Torch 2.14).

- **Ledger v2** (`sol_execbench.hip_sdk_ledger.v2`): the collector now
  records structured engine-copy completions (`memcpy` rows: corr, direction,
  bytes, GPU timestamps, agents — the SDK record carries no queue) and the
  raw `hipStream_t` argument of the frozen streamed set (4 async memcpys +
  3 launch APIs; handle zero is the legacy default stream, SDK ordinals
  never carry default-stream meaning). rocprofv3 export gained
  `--memory-copy-trace` via the opt-in `include_memory_copy` request field,
  enabling a real three-channel (ledger/export/API) copy check on device.
- **Copy domain** (`hip_sdk_memcpy.py`): whole-capture multiset bijection;
  chunked large copies share one correlation and must sum to the host
  `sizeBytes`; lane binding is co-observed (stream ordinal → queue from
  kernels on the same stream); H2D/D2H/D2D only, frozen vocabulary, honest
  reason codes (`copy_api_unsupported`, `copy_direction_unsupported`,
  `copy_queue_unresolved`, `copy_stream_conflict`,
  `default_stream_implicit_sync_not_inferred`).
- **Scope artifacts v2** (`hip_torch_scope.v2` / `hip_evaluate_scope.v2`):
  carry `copy_domain` + `schedule` qualification; `schedule_complete` flips
  to true only with every in-scope copy lane-bound, all event links
  verified, and no default-stream lane mixing; `performance_authority` and
  `adversarial_attestation` stay literally false; teardown allow-list gained
  the pinned-copy path's zero-dispatch event-pool calls.
- **Schedule evidence v3** (`performance_schedule_evidence.v3`): new
  `event_wait` edge reason and an explicit
  `cross_lane_event_dependencies` state (verified/unverified/absent);
  `build_schedule_evidence` publishes cross-lane edges only from an
  independently qualified scope (temporal adjacency is still never causal);
  `predict_hw` consumes verified edges to serialize inter-lane critical
  paths and is bit-identical when none are supplied. **M-d production feed
  pipeline is recorded as follow-up work** (scope↔counter dispatch-identity
  bridge + explicit opt-in; spec §7.4) — not silently folded into this
  closure.
- **GPU denominator 18/18** (six cases × 3 processes, all under the
  STABLE_PEAK host guard): the three 2026-10-11 regression cases keep their
  baselines; `copy_lane` (pinned H2D + D2H engine copies lane-bound),
  `copy_event` (engine copy is the verified event-link predecessor —
  `predecessor_copy_correlation_id` set, dispatch endpoint None),
  `default_mix` (correctly rejected 3/3 with
  `default_stream_implicit_sync_not_inferred`). 15/18 runs hold
  `schedule_complete=true`; copy domain qualified on every run.
- **Mechanism facts recorded** (spec §7.3): chunked large copies (one corr,
  multiple completion records), the dual stream-identity spaces (ordinal vs
  handle; torch explicit streams can take ordinal 0), the WAIT-barrier race
  window (4096³ fp32 matmul ≈ 0.5 ms on RX 9060 XT — producer work must
  exceed the host submission gap for an observable barrier), and the
  pinned-path teardown surface.
- Tests: 33 new (memcpy domain 19, torch-scope schedule/copy semantics 5,
  schedule-evidence v3 branch 5, prediction verified-edge 4); affected
  bench/cli/driver suites 1537 passed. Ruff/format/`ty` clean on every
  touched path; repo schema check clean for the `sol_execbench` families.
  Pre-existing reds outside this line unchanged (gpu-admission fixture
  file, residue-audit l3a strings, solar_bridge `scripts.internal` import
  errors, SOLAR readability baseline, solar-internal schema strings).
- Roadmap §0 B-group row and the L2 section were updated to the closed
  state; scope-causality acceptance docs and the user sidecar doc were
  re-versioned atomically.

### SOLAR external-reads batch 8 (线A 收官) — 2026-11-06

Batch 8 (conv2d 窗口覆盖泛化) closed Line A: the 128-row inventory
denominator is fully cleared
([spec](docs/internal/solar-l3e-batch8-acceptance-2026-11-06.md); evidence
`batch8-*` under `data/outputs/solar-l3e-external-reads-20261106/`).

- The batch-7 per-dim receptive-field coverage predicate was extracted as
  the shared `_windows_cover` and generalized to `conv2d` (batch-8 spec):
  kernel from the weight's spatial dims, stride/padding/dilation decoded
  from conv positionals, groups allowed (channel partitioning preserves
  the union), transposed/output_padding kept unproved, and the 1×1
  contraction special case retained via OR. External conv reads are a
  unique-element union, so coverage decides the read exactly.
- Gates: G3 exact — 14/14 must-flip rows flipped (l1n63_conv2d w0–w2,
  l1n82_conv_depthwise w0–w2 groups=8, l2n17_instancenorm_fused w0–w2,
  l2n52_conv_activation_batchnorm w0–w4, all make_fx), zero regressions,
  **128-row denominator now fully blocked-free**; G1/G2 — 6 new unit
  tests (covered dense/depthwise/tail-exact; stride-gap, transposed and
  the batch-1 strided negative unchanged); G5 — all external-reads and
  access suites green, ruff clean, `ty` diagnostics outside this line
  (5 pre-existing in r16/shapes-domain tests and the concurrent L2 WIP).
- **G4 honest miss (pre-declared clause)**: 13/14 analyzed with totals;
  `l2n52...-w4` (B=64,H=W=128) is memory-bound — its explicit-unfold
  verification replay needs a ~23.6 GB intermediate (64×576×126×126 fp32)
  and was kernel-OOM-killed at 26 GB anon RSS on the 29 GB host
  (`batch8-g4-summary.json` carries the three-piece evidence). The
  static predicate is shape-parametric and identical to the completed
  w0–w3 siblings; the miss is the end-to-end replay confirmation only,
  recorded without any relaxation. Safety note for future conv G4 runs
  on this host: run conv-heavy slots under a cgroup MemoryMax guard
  (per the batch-1 method note); never bare.
- **Line A final state**: external-reads gap 128 → 0; 127 of 128 rows
  hold G4-verified formal bounds (batches 1–8), 1 row (l2n52-w4) holds
  the static bound with a recorded memory-bound replay miss. Full-run
  suite gates: `tests/solar`+`tests/scripts` **1871 passed** at
  `5cfdef71` (one G5 atomic update: the default-API unproved-conv refusal
  test re-pinned to a stride-gap conv — the old padding-1 shape's windows
  are covered and now prove their reads; the shared `_conv_request` gained
  stride/padding parameters with the default shape unchanged for its
  other two consumers).

### SOLAR external-reads batch 7 — 2026-11-06

Batch 7 (pool 窗口覆盖 + 融合/索引化 loss + torchview de-opaque) closed
with all gates green
([spec](docs/internal/solar-l3e-batch7-acceptance-2026-11-06.md); evidence
`batch7-*` under `data/outputs/solar-l3e-external-reads-20261106/`).

- Three frozen proof classes, no new access representation: (1) fused
  losses read both tensor operands exactly (`cross_entropy` = log_softmax
  full read ⊕ per-class subset gather ⊕ full target read; `kl_div`
  elementwise-then-reduce) — a third tensor operand (weight) is a per-class
  gather and stays rejected; (2) decomposed `nll_loss_forward` proves the
  target full read and requires every non-target tensor operand internal
  (corpus shape: log_softmax output); an external data operand stays
  `dynamic_index`; (3) pool window coverage decides the unique-element
  union read for `max_pool2d`/`max_pool2d_with_indices`/`avg_pool2d`
  (per-dim receptive-field math; `ceil_mode=True` stays unproved).
  torchview registry: `kl_div`/`cross_entropy`/`max_pool2d` de-opaqued in
  the reviewed-pure class; `nll_loss` stays opaque.
- Gates: G3 exact — 23/23 must-flip rows flipped (make_fx 11 + torchview
  12), zero regressions, remaining blocked rows exactly the 14 conv2d
  rows; G4 — 23/23 analyzed with totals; G1/G2 — 13 new unit tests
  covering all three classes plus stride/tail/ceil-mode and
  external-data/weight-operand negatives; G5 — `tests/solar`+
  `tests/scripts` 1865 passed, ruff/ty clean.
- Pre-declared atomic update: batch-1's `test_indexed_loss_still_blocks`
  became `test_indexed_loss_external_data_still_blocks` — same test now
  pins the still-true external-data negative (the original graph's
  internal-data shape is batch-7's proven positive).
- **Line A cumulative: 128 → 14 inventory-blocked rows** (91 rows hold
  G4-verified formal bounds). Remaining: `conv2d` 14 (window coverage,
  batch-8).

### SOLAR external-reads batch 6 — 2026-11-06

Batch 6 (transparent_alias 内部物化输出别名) closed with all gates green
([spec](docs/internal/solar-l3e-batch6-acceptance-2026-11-06.md); evidence
`batch6-*` under `data/outputs/solar-l3e-external-reads-20261106/`).

- Pre-spec recon overturned the batch-2 characterization: all 42
  `<transparent_alias>` entries (38 rows) are `identity`/`detach`/`view`
  hops over **untracked internal-materialized** inputs — declared outputs
  aliasing graph-owned fresh storage, not external storage. Frozen gate
  (`transparent_declared_output_aliases_external`): a transparent declared
  output passes only when every tensor input is untracked with reason
  `MATERIALIZED`; a tracked input (external-reaching view chain) or any
  other untracked reason keeps `unproven_alias`. The inventory diagnostic
  mirrors the same predicate.
- Gates: G3 exact — 22/22 must-flip rows flipped (rmsnorm_bwd w0–w3,
  dynamic_mxfp8_quant w0–w7, gpumode_positionwise_ffn w0–w3,
  l1n40_layernorm w0–w2, l2n37_groupnorm_fused w0–w2, all make_fx), zero
  regressions, and the 16 mixed rows lost the transparent component with
  pool/loss/conv components unchanged; G4 — 22/22 analyzed with totals
  (zero-total norm/mxfp8 families match the batches 1–4 precedent); G1/G2
  12 new unit tests plus preserved negatives (external detach/view still
  alias); G5 — in-scope suites green, ruff/ty clean.
- **Line A cumulative: 128 → 37 inventory-blocked rows** (69 rows from
  batches 1–5 plus these 22 hold G4-verified formal bounds). Remaining
  vocabulary: `conv2d` 14 (window coverage), indexed/fused loss 14
  (`nll_loss_forward` 5 make_fx, `cross_entropy` 5 + `kl_div` 4 torchview
  effect_boundary), pools 9 (`max_pool2d_with_indices` 3 + `avg_pool2d` 3
  make_fx, `max_pool2d` 3 torchview effect_boundary).
- G5 environment note (recorded, not adjudicated): the full-suite run needs
  `GPU_LOCK_DIR`/`GPU_LOCK_UUID` exported or collection dies in
  `configured_gpu_locks`; its 15 failures are all outside this line — 13 in
  the concurrent L2 WIP modules (hip_sdk/hip_causality working-tree edits),
  1 pre-existing from the L3-a commit (`l3a_measure.py` CUDA docstring
  residue), 1 load-flaky test passing solo.

### SOLAR external-reads batch 5 — 2026-11-06

Batch 5 (make_fx view 未声明别名) closed, all gates green
([spec](docs/internal/solar-l3e-batch5-acceptance-2026-11-06.md); evidence
`batch5-*` under `data/outputs/solar-l3e-external-reads-20261106/`).

- `Tensor.view`（含 `_unsafe_view` 规范化）的别名性来自语义而非 schema
  声明——tracked 连续输入经 `reshape_access` 证明纯视图，未跟踪（内部）
  输入按内部数据移动记 `MATERIALIZED`，非连续输入保守维持
  `UNSUPPORTED_VIEW`（view 兼容性一般规则留待后续）。mingpt 的 transpose
  blocker 是 `_unsafe_view` 继承，同根清除。
- G3 exact 4/4（l3n44_mingpt make_fx w0–w3）；G4 4/4 analyzed with
  formal totals；suite 1841 passed（+1 view 单元测试，无旧语义固定）。
- **Line A cumulative: 128 → 31 rows**（69 rows hold formal bounds）.
  Remaining: `<transparent_alias>` 42（界语义域）、conv2d 14（区域机器）、
  索引化 loss 14（nll 5/cross_entropy 5/kl_div 4）、view 非连续分量
  （混合行内，随批次清）。

### SOLAR external-reads batch 4 — 2026-11-06

Batch 4 (layer_norm torchview de-opaque) closed with all gates green
([spec](docs/internal/solar-l3e-batch4-acceptance-2026-11-06.md); evidence
`batch4-*` under `data/outputs/solar-l3e-external-reads-20261106/`).

- torchview native registry moved `layer_norm` from opaque `_LIBRARY` to
  reviewed `_NO_EFFECTS` (same class as batch-3 gelu; read vocabulary landed
  in batch-3). G3 exact — 3/3 l1n40_layernorm torchview rows flipped, zero
  beyond; G4 — 3/3 analyzed with formal totals; suite 1839 passed, no test
  pinned the old opaque flag (atomic-update list empty).
- Implementation note: one silent no-op replace (formatted-text drift) was
  caught by asserting the pattern and re-verifying the registry lookup
  before relaunching the campaign.
- **Line A cumulative: 128 → 35 rows** (65 rows hold formal bounds from
  batches 1–4). Remaining vocabulary: `<transparent_alias>` 42 (output
  aliasing domain), make_fx `view`/`_unsafe_view` inheritance 20 + mingpt
  transpose 4 (undeclared-alias view family — one root, batch-5 design:
  torch `Tensor.view` is unconditionally a view, stride-decidable via the
  batch-2 machinery), `conv2d` 14 (window coverage, region machinery),
  `nll_loss` 5 / `cross_entropy` 5 / `kl_div` 4 (indexed-loss semantics),
  `max_pool` 3.

### SOLAR external-reads batch 3 — 2026-11-06

Batch 3 (norm 词汇补全 + gelu 复核) closed with all corrected gates green
([spec + S3 erratum 15→12](docs/internal/solar-l3e-batch3-acceptance-2026-11-06.md);
evidence `batch3-*` under `data/outputs/solar-l3e-external-reads-20261106/`).

- **Production changes**: `layer_norm`/`batch_norm`/`group_norm` joined
  `FULL_OPERAND_READS` (normalization reads every tensor operand; zero
  standalone flips expected and observed — family completion for mixed
  rows). The torchview native registry moved **gelu** from the opaque
  `_LIBRARY` effects to reviewed `_NO_EFFECTS` (same class as relu/elu);
  kl_div/cross_entropy/nll_loss intentionally stay opaque (indexed-loss
  semantics unproved).
- **Gates**: G3 exact — 12/12 gelu rows flipped, zero beyond-S3; G4 —
  12/12 analyzed on CPU with formal totals; suite 1839 passed (one
  pre-declared atomic test update: batch-1's layer_norm negative became a
  vocabulary positive, output-alias dimension excluded).
- **Erratum**: frozen S3 count 15 included 3 l1n40_layernorm torchview rows
  whose blocker target is layer_norm (opaque `_LIBRARY`), not gelu — the
  spec's own definition yields 12. Those 3 rows are batch-4 candidates
  (norm de-opaque), alongside conv2d window coverage (needs region
  machinery), make_fx `view` inheritance ×20, and `<transparent_alias>` ×42.
- **Cumulative**: external-reads gap 128 → **38 rows** (batch-1 38 +
  batch-2 12 + batch-3 12 = 62 rows now hold formal bounds from this line;
  plus the 38 alias rows and others remaining).

### SOLAR external-reads batch 2 — 2026-11-06

Batch 2 (contiguous/reshape 条件别名判定) closed with all gates green
([spec](docs/internal/solar-l3e-batch2-acceptance-2026-11-06.md);
evidence under `data/outputs/solar-l3e-external-reads-20261106/`,
`batch2-*` files).

- **Production semantics**: torch's conditional reshape/contiguous alias is
  now decided statically from the tracked input's row-major strides —
  contiguous inputs prove a pure view (`reshape_access`), any other tracked
  input forces a copy whose output is recorded `MATERIALIZED` (fixing the
  copy_boundary inheritance that blocked downstream views); an untracked
  (internal) input is pure internal data movement (no external-read
  question). Clone outputs are likewise `MATERIALIZED` instead of
  `copy_boundary`. Analysis side: `contiguous`/`reshape` joined
  `FULL_OPERAND_READS`, the COPY rejection no longer covers `contiguous`,
  and resolved `STATIC` views pass like materialized ones (unsupported or
  dynamic reasons still block).
- **Gates**: G3 exact — 12/12 must-flip rows flipped, zero beyond-S2 flips,
  4 partial rows (mingpt layer_norm/transpose) stay blocked; G4 — 12/12
  flipped rows analyzed on CPU with formal totals; G1/G2 unit coverage for
  both stride branches plus negatives; G5 — suite 1839 passed with the
  pre-declared atomic updates (`test_copy_reads` 14 expectations,
  `test_access_graph` 2 tests; one precise torchview skip for a
  transposed-reshape composition it cannot convert).
- **Cumulative**: external-reads gap now 128 → 50 rows (38 batch-1 + 12
  batch-2 + 38 baseline-passing? — precisely: 50 rows still blocked).
  Residual vocabulary: `<transparent_alias>` 42, `view` 20 (make_fx
  schema-alias inheritance, separate root), `layer_norm` 18, `conv2d` 14,
  `gelu`-effect 12, `batch_norm` 8, `nll_loss_forward` 5, `cross_entropy` 5.
  Batch 3 (norm/conv/pool window reads) and the transparent-alias/output
  aliasing domain remain open.
- Two implementation notes: the first G3 pass missed because the
  analysis-side STATIC_VIEW acceptance still excluded resolved STATIC
  views, and mxfp8's reshapes sit on untracked internal inputs — both were
  spec-internal completions (S2 stayed the fixed oracle; no thresholds
  moved). Campaign `--floor` is now parameterized (0 = require all).

### SOLAR external-reads batch 1 — 2026-11-06

The L3-c follow-up "external-reads 语义扩展" closed its **batch 1** with all
corrected gates green
([spec + dated erratum](docs/internal/solar-l3e-external-reads-acceptance-2026-11-06.md);
evidence `data/outputs/solar-l3e-external-reads-20261106/`).

- **Production change**: `FULL_OPERAND_READS` extended by the frozen
  15-target batch-1 vocabulary (elementwise math + `cat` + `expand`/`repeat`)
  — mechanically identical proof, no control-flow or schema changes; no
  existing test pinned the old semantics (atomic-update list empty).
- **Gap reduction**: of the 128 unproved-external-reads rows, **38 now
  obtain formal bounds** (verified by full CPU analyses with totals;
  three-way set equality: corrected S1 == re-inventory zero-blocker set ==
  G4 analyzed set, all 38 row-identical). 82 rows remain blocked with an
  exact residual vocabulary: `view` 36, `reshape` 24, `layer_norm` 18,
  `conv2d` 14, `contiguous` 12, `gelu`-effect 12, `alias/detach` 16,
  `batch_norm` 8, `nll_loss_forward` 5 (+torchview effect-boundary).
- **Erratum (dated, prior-evidence)**: the inventory diagnostic under-modeled
  the production transparent-layer ALIAS path; corrected model adds
  `<transparent_alias>` markers. Original S1=46 corrected to **38**; the 8
  `dynamic_mxfp8_quant` rows fail `unproven_alias` (batch-2 domain,
  fail-closed untouched).
- **Method notes**: the campaign inventory runs extraction+conversion only
  (verification replay omitted — CPU conv replay via explicit unfold can
  exceed host memory, twice kernel-OOM at 27 GB before this fix; per-slot
  subprocess isolation retained; RLIMIT_AS rejected: it caps VA, breaking
  torch imports). G4 workers also fixed two record-assembly bugs (success
  path lacks `reason_code`; parent overwrite condition inverted).
- Tests: 16 new (`tests/solar/test_external_reads_batch1.py`); suite 1838
  passed; ruff/ty clean. Batches 2 (view/copy alias semantics) and 3
  (norm/conv/pool window reads) remain, each with residual counts above.

### SOLAR L3-c execution — 2026-11-06

L3-c (2.14 算子覆盖差分) closed: the full-corpus dual-path coverage matrix is
delivered with pre-fixed gates all green
([spec](docs/internal/solar-l3c-coverage-matrix-acceptance-2026-11-06.md);
evidence `data/outputs/solar-l3c-20261106/`). No allowlist or registry was
modified.

- **Denominator & gates**: 163 scored workloads × 2 paths = 326 rows; gate
  passed with zero failures; L1-canary reconciliation exact (112 items,
  0 drift on status/stage/reason — HEAD `7e069bd2` reproduces the canary
  wall); 7/8 roadmap families present, `mutation` honestly absent (no
  corpus workload carries schema mutation effects).
- **Headline differential**: `make_fx_aten` covers readiness **163/163**;
  `torchview_extended_einsum` covers 121/163 (audit gaps: 14
  `native_operation_unsupported` + 10 `native_attribute_unsupported` +
  14 `gradient_verification_failed` + 4 `graph_extraction_failed`). The
  aten path strictly dominates corpus readiness coverage.
- **Formal-bound layer (new datum beyond L1)**: of ready slots, make_fx
  87/163 and torchview 64/121 produce formal bounds; 128 rows fail with
  `external reads are unproved` (families: normalization 47, convolution
  16, low_precision 16, …). Conv family on torchview additionally hits the
  pinned mapper's 7200 s budget (5/5 `l2n52` slots → `toolchain_unavailable`
  after full searches, ~43 CPU-hours each) — conv+batchnorm analysis
  throughput on this host is infeasible within production bounds.
- **Dual-path totals**: 64 slots analyzed on both paths; 51/64 totals
  equal; 13 differ, all routing/quant/silu families, differing only in
  `num_layers` granularity (aten decomposition vs einsum view; flops/macs
  agree at 0) — recorded, not adjudicated (spec §6).
- **Actionable gap list** (the matrix product, `canary-wall-attribution.json`):
  torchview conversion coverage 24 items (op+attribute), torchview gradient
  verification 14, extraction 4, both-path external-reads semantics 128,
  conv mapper budget 5. Full-denominator qualification chain remains the
  L1-deferred follow-up, now unblocked by this matrix.
- Campaign: `scripts/internal/solar/run_l3c_coverage_matrix.py` (audit via
  production `audit_corpus_stage_readiness`, analysis via production
  `analyze_workload`, resume-able; slow-slot mapper timeouts resolved by
  the production runner's own 7200 s bound). Tests: 10 new
  (`tests/scripts/test_l3c_coverage_matrix.py`); suite 1822 passed;
  ruff/ty clean.

### SOLAR L3-b execution — 2026-11-05

L3-b (形状域契约，阶段 1 具体实例化) closed. Acceptance pre-fixed in
[solar-l3b-shape-domain-acceptance-2026-11-05.md](docs/internal/solar-l3b-shape-domain-acceptance-2026-11-05.md);
one authoring correction made **before** campaign execution (K range widened
to [2,256] so the `K<=E` assumption is non-degenerate and the pre-fixed
assumption-violation probe can exist — the original [2,8] made it vacuous).

- **Own constraint AST**: new production package `src/solar/shapes/`
  (`ShapeSymbol` range/divisibility/provenance, integer-affine `DimExpr`,
  `TensorBinding`, `Assumption` ≤/==, `IntervalDim`, `ShapeDomain`) with
  canonical serialization and `stable_identity` checksum; fully SOLAR-owned
  (no upstream ShapesSpec/SymBool runtime dependency). Validation returns
  only the pre-fixed reason codes; stage-1 tensor-shape binding solves
  unit/affine dims and refuses to guess (`shape_mismatch`).
- **R06 first real consumer**: frozen domain `r06-moe-topk-softmax-shape-v1`
  (`T` ∈ [64,1024]%64, `E` ∈ [64,256]%64, `K` ∈ [2,256]%2, `K<=E`,
  gating/bias/topk/route_scale/weights/ids bindings). All 4 workloads +
  boundary-max + interior-non-corpus (T=128,E=192,K=6) validated; negative
  probes out-of-range/divisibility/assumption rejected with exact codes.
- **Structure certification**: R06 graphs at w0/w1/interior are
  `structure_stable` — the single constant divergence across instances is
  `topk.k` (8 vs 6), explained by the K symbol; the `shape_branch` sample
  certifies `shape_branch_rejected`. Domain-wide reuse claims stay bounded
  by this finite-instance evidence (per-instance validation ≠ whole-domain
  proof, per spec).
- **Instance provenance through existing seams**: R06 w0 analyzed (make_fx,
  CPU, pinned mapper) with `analysis_metadata.shape_domain`
  (domain_id + domain_sha256 + assignment + derived tensor shapes);
  `solar-analysis.yaml` carries the block and `manifest.yaml` binds its
  sha256. No production schema changed (metadata is the caller-authored
  seam); ConversionRequest-level shape fields remain stage-2 work.
- **Pricing separation invariance**: with-domain vs control analyses are
  identical after removing the block (`invariance_diff: null`) — the shape
  domain adds zero tensor accounting.
- Evidence: `data/outputs/solar-l3b-20261105/` (`l3b-summary.json` gate all
  green; campaign `scripts/internal/solar/run_l3b_shape_domain.py`).
  Tests: `tests/solar/test_shapes_domain.py` (23, full vocabulary);
  suite 1812 passed; ruff/ty clean.

### SOLAR L3-a execution — 2026-11-05

L3-a (静态 fake 取图试点) closed, with L4 item 1 (stage measurement report)
delivered in the same campaign. Acceptance was pre-fixed before execution in
[solar-l3a-fake-static-pilot-acceptance-2026-11-05.md](docs/internal/solar-l3a-fake-static-pilot-acceptance-2026-11-05.md);
one dated erratum appended mid-campaign declaring R08/make_fx as a second
expected failure (formal-bound extraction rejects unproved external reads —
fail-closed policy; L1 evidence never claimed R08 priceable).

- **Measurement (M gate passed, 16/16 slots)**: first per-stage
  wall/peak-RSS/GPU-allocator report over the frozen R16 denominator × both
  IR paths on gfx1200 (`data/outputs/solar-l3a-20261105/measurement/`).
  Headline medians: `conversion_verification` dominates single-slot wall
  (make_fx 1.55 s, torchview 1.10 s ≈ 2/3 of pipeline), `graph_extraction`
  0.58/0.66 s, `ir_conversion` 0.03/0.05 s, `formal_analysis` 0.15/0.09 s,
  publication ≤ 0.013 s. Worker process peak RSS ≈ 2.9 GB (torch import
  ~1.0 s per isolated worker). Orojenesis mapper at R16 scale: 30 runs,
  8.2 s total, sampled peak 27 MB — the mapper is not the bottleneck at
  this denominator, so L4's thread-count question ranks below verification
  batch cost.
- **Pilot (D1–D4 all pass, adopt=true)**: new unregistered module
  `src/solar/graph/make_fx_fake.py` (FakeTensorMode, concrete shapes,
  reuses `trace_make_fx_reference`; default paths untouched, verified by an
  automated `git status --porcelain -- src` structural check). All 8
  make_fx R-slots: O1 semantic-projection equal AND O2 exact-document
  equal AND O3 downstream aten conversion + real-replay verification
  passed. Focused samples: mutation effects parity, scalar-constant
  arguments parity, data-dependent branch rejected under fake
  (`DataDependentOutputException` on `_local_scalar_dense`). O5: extraction
  wall 11–196× faster (median ~100×); GPU allocator peak never higher
  (R08 38→4 MB; unchanged slots are dominated by the real input tensors
  the factory allocates — full GPU savings require factory-side fake
  inputs, a production-reform consideration, not L3-a scope).
- **Decision recorded**: adopt (D1 8/8, D2 3/3, D3 8/8+8/8, D4 structural
  check). Wiring fake-static extraction into the production make_fx path is
  a follow-up line with its own pre-fixed acceptance (fake-capable input
  factory, torchview path untouched, EROFS/container preconditions); it
  does not modify the closed L3-a pilot.
- Tests: `tests/solar/test_make_fx_fake.py` +
  `tests/scripts/test_l3a_stage_measurement.py`; solar suite unchanged at
  1595 passed; ruff and `ty` clean. Campaign reproducible via
  `scripts/internal/solar/run_l3a_stage_measurement.py --resume`.

### SOLAR L1 execution — 2026-11-05

L1 progress on the [roadmap](docs/internal/solar-evolution-roadmap-2026-11-05.md):

- **L1-c closed** (`12859e3e` "Prove frozen top-k gather reads for external
  reads"): new frozen vocabulary `solar/ir/topk_gather.py` proves a gather whose
  index reduces to a top-k index output over an identically shaped operand adds
  no external reads; `softmax`/`topk` joined full-operand reads. R06
  (`aka-moe_topk_softmax-w0`) flipped from `unavailable [dynamic_index,
  read_semantics_unproved]`/diagnostic to **exact (16640 elements / 33792 B,
  oracle-matched) / roofline_eq1_v1 / all three views ready on both IR paths**.
  Solar suite 1595 passed; F07 keeps its dynamic-index rejection.
- **L1-a verified**: EROFS was specific to the external hardened controller
  (`--read-only` root). Host and repo-container default Triton caches are
  writable (probe evidence in `data/outputs/solar-l1-eroofs-verification-20261105/`).
### SOLAR L1 execution — 2026-11-05

**L1 update (later this day): L1-d closed via the curated-denominator path.**
The full-denominator canary wall stands (L3-c domain, unchanged), but the
roadmap-sanctioned option (a) was executed with the existing readiness seam:
`scripts/internal/solar/run_r16_curated_qualification.py` audits the frozen
R16 denominator with R08 as a declared expected-rejection sentinel. Both IR
paths pass the curated gate — 15 ready + 1 declared expected-rejection =
16/16 terminal. R06 additionally holds formal
`capacity_constrained_tile_aware_v1` bounds on both paths (the L1-c
vocabulary in effect; external reads exact at 33792 B). R08's make_fx slot
upgraded from the historical gradient-verification rejection to ready:
graph extraction, IR conversion and conversion verification all genuinely
passed on the integrated tree. Full-denominator qualify-canary/full remain
INCOMPLETE and deferred to L3-c. Evidence:
`data/outputs/solar-l1-release-20261105/r16-terminal-ledger.json`. The L1
line is closed under its acceptance alternative (itemized terminal ledger +
per-slot GPU evidence); full-corpus publication qualification is now an
L3-c-gated follow-up, not an L1 deliverable.

### SOLAR L1 execution (earlier) — 2026-11-05
  (git-revision bound), `qualify-static` gate passed; `qualify-canary` returns
  INCOMPLETE — 112 audited, 79 ready, 33 failed = 3 R08-expected rejections
  (consistent with the R16 ledger) + 20 native-registry coverage gaps (L3-c
  domain) + 10 axis-point numerical mismatches (R04-w0 itself ready, matching
  `validated_prior_ready`). The chain's full-scored-denominator readiness gate
  cannot complete without L3-c coverage work or a curated SOLAR qualification
  denominator. Campaign evidence:
  `data/outputs/solar-l1-release-20261105/campaign-summary.json`.
- Environment fix during the campaign: an orphaned, broken
  `.venv/bin/amd-smi` console shim (amdsmi module not importable) shadowed
  `/usr/bin/amd-smi` and failed `observe_clock_devices()`; renamed aside
  (`environment-fix.json` in the campaign directory).

R16 after L1-c: 12 complete + 2 R06 priceable-pending-rerun + 2 R08
expected-rejection = 16 slots with terminal or decided paths. L1 is not fully
closed; the remaining decision (coverage expansion vs curated denominator) is
recorded in the roadmap L1-d entry.

### SOLAR evolution roadmap L0 — 2026-11-05

The evolution line is designed and committed
([roadmap](docs/internal/solar-evolution-roadmap-2026-11-05.md), `9efa3659`).
This entry performs L0 state reconciliation; production code is untouched.

Integration preimage verdict: the 176-path integration rehearsal landed as
`f9d0d994`. Re-verified against the handoff's `candidate-manifest.json`
(1,373 files): 157/176 integration paths are byte-identical today; all 19
drifted paths are attributable to `cb41d916`/`fbc148b8`; of the 85
preserved-main paths, 81 are unchanged and 4 changed via
`97355a7a`/`e75daf3c`/`fbc148b8`; no planned path is missing. The
"integration applied" claim is verified, no longer inferred.

Main registrations since 2026-10-30: `f9d0d994` (integration landing),
`cb41d916`, `fbc148b8`, `e75daf3c`, `89cf9c10`, then `24ef5850`, `97355a7a`
(HIP event causality and controlled scope qualification), `44f8b6b1`,
`b4e89ccc`, `9b48c89f`, `9efa3659`.

External evidence registration (decision 2026-11-05: register path and
digest only; do not copy files into this repository):

| Artifact (canonical location) | sha256 |
| --- | --- |
| `~/Documents/Codex/2026-10-26/task/solar-integration-prep-20261029/INTEGRATION.md` | `0c63e7673d9f8c4e62c3099cb913fdc929f433eea21b242fbe4179796f716b39` |
| `~/Documents/Codex/2026-10-26/task/solar-integration-prep-20261029/FINAL.md` | `7f3188ce13a4ac515c8763afb7383ab0a628b9bc26ae0d784fc552662a9ed5d8` |
| `~/Documents/Codex/2026-10-26/task/solar-integration-prep-20261029/integration-plan.json` | `d8c536310b553c18ade2b2d344a5a53160ad71c6af47d7df4a3478afff990b00` |
| `~/Documents/Codex/2026-10-26/task/solar-integration-prep-20261029/candidate-manifest.json` | `07c37b722b344cad05dfa808ec6bf708e247cad6568d9a999bfaedacfa88961a` |
| `~/Documents/Codex/2026-10-26/task/solar-stages1-5/data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/CHECKPOINT-0528.md` | `fe065d8873fa4a0b13a643e94243701d143ae03c2c5898b4d8230b4d13c909b6` |
| `~/Documents/Codex/2026-10-26/task/solar-stages1-5/data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/CHECKPOINT-0553.md` | `7bd325f1977f098838ad5bb2e57f96815ef0b1ec47a84780f76071f173ebdc40` |

Earlier entries cite `data/outputs/solar-resume-20261026/...` relative to
this repository; those files do not exist here. The external worktree paths
above are the canonical locations.

R-slot terminal ledger (source: the rehearsal's `ACCEPTANCE-LEDGER.json`,
analysis source `9e8cc0e6…`, GPU readiness source `5d37dcc4…` via strict
semantic-runtime-identity reuse):

- R04 `torch2hip/dot_product_attention`, both IR paths:
  `validated_prior_ready`, priceable, cache-aware. The 2026-10-26 container
  blocker (read-only `/home/sol-execbench/.triton`, EROFS errno 30 during a
  fresh extraction attempt) is not a slot blocker; readiness stands on strict
  reuse of the 5d37 receipts. Any fresh GPU rerun of the qualification chain
  in containers still needs the bounded tmpfs fix — an L1-d prerequisite,
  not an R04 defect.
- R06 `torch2flydsl/moe_topk_softmax`, both IR paths: ready, not priceable,
  cache-aware unavailable, external reads `unavailable`. Decision 2026-11-05:
  build the frozen top-k bounded-read proof vocabulary (roadmap L1-c);
  fallback is explicit rejection with recorded reason.
- R08 `instruction2triton/rmsnorm_bwd`: torchview path `expected_rejection`
  at graph extraction; make_fx path `expected_rejection` at conversion
  verification (gradient check). Terminal as legitimate boundary rejections,
  retained in the 16-slot denominator; roadmap L1-b determination is closed
  with no code change.

R16 final: 12 complete + 2 ready-unpriced (R06) + 2 expected-rejection
(R08) = 16. The GPU canary passed both paths (CHECKPOINT-0553, 2026-10-26
05:53 UTC); the mapper-followup document's "not executed" narrative is
superseded by a dated addendum there.

### SOLAR integration preparation — 2026-10-29

The current scientific snapshot is `9e8cc0e615716fced80ca5998bfe7da17b028aa9943ebb9d76a845afa68f7876`.
The fixed R set retains 16 slots: 14 readiness receipts, 12 priceable and 12
complete cache-aware model results. R06 remains two ready but unpriced dynamic
read cases; R08 retains the extraction and gradient-verification boundary
rejections. All final counts and required mapper evidence were recalculated
with this analysis implementation. Semantic runtime identity `005d6c03…82d1d`
is unchanged, permitting strict reuse of the original 5d37 GPU receipts;
this does not relabel their source or replay the GPU.

The 32 MiB point is nominal profile/model capacity, not measured performance
or whole-project formal qualification. Stages 4–5 remain bounded research
interfaces with their explicit domain and numerical policies. No speedup,
complete-corpus qualification, deployment, or start700 restart is claimed.

The integration rehearsal starts from all current Git-visible main bytes,
including its uncommitted work, and preserves the newer HIP scope,
profile-summary v5 and allocator VMM workaround. It reconciles 176 paths;
main and the research worktree remain untouched during preparation. The
rehearsal CPU suite passed 3821 tests (2 skipped, 74 environment/hardware
exclusions; 229.68 seconds). Exact file preconditions, rollback, same-source
ledger, canonical bundles and final validation are in the local handoff
`/home/guohao/Documents/Codex/2026-10-26/task/solar-integration-prep-20261029/`.
Apply only after rechecking that handoff's main preimage; integration is not
performed by this document. Preserve all original run evidence.

The following 2026-10-26 entries are historical snapshots, not current results.

### Historical P1/P2 — SOLAR: stages 1–5 CPU implementation on an isolated branch

The user explicitly renewed authorization to continue stages 1–5 after the
BASE-01-only safe split. This branch, `codex/solar-stages1-5-resume`, starts at
`edc9ad211857f208e73efdd631ec2e472c11a581` and adapts reviewed modules from the
saved stage commit `c9c0e5b6`; it does not merge the foundation's 86 unrelated
changes. Main and both archive branches remain unchanged. No push or main merge
was performed. The original split was a scope decision, not a claim that all
archived code was defective.

Stages 1–5 CPU checks passed respectively 919, 1140, 1233, 1364 and 1445 tests.
The final stage suite also passed Ruff, format, ty, coupling, readability and
schema checks, with identical before/after source identity
`ec176be36d93db331f71a18a27426a08c82691f28f39ad56a8f2114603263277`.
Each phase retains its own source archive, command logs, JUnit and earlier
failed attempts under ignored `data/outputs/solar-resume-20261026/`.
Final F32 collection and documentation closure are recorded in the
[current implementation report](docs/internal/solar-resume-2026-10-26.md).

The fixed installed image uses CPU only, four CPU quota, 8 GiB memory, one test
worker and no network. No software was installed. The primary worktree's 86
existing edits are checked against a content/mode/status protection manifest.
The prior BASE-01 and safe-split reports describe their historical source
snapshots; their counts are not substituted for these runs.

A subsequent same-source F03-D real-mapper check passed both IR paths at
64/4096 B capacity (420/196 B traffic). See the
[mapper follow-up and proposed GPU validation](docs/internal/solar-mapper-followup-2026-10-26.md).
Broader real-mapper coverage, all 16 real-workload R slots, GPU validation
and formal publication qualification remain open. CPU fixtures and model-only
contraction search do not establish device speedups. The operator-paused
`p1-successor-start700` batch remains paused and outside this work. Next bounded
validation steps and approval/resource requirements are in the current report.

### P1 — start-640 PCIe5.0x16 successor path (closed terminal)

`p1-successor-start640-pcie5x16-r1` completed at `8848d605` and is **closed
terminal evidence**: acceptance `accepted=false` (model-generalization), no
publication/release/tag. Its `accepted=false` held-out corpus must not be
reused; tag `gfx1200-diagnostics-v7-production-v1` must not be created.

### P1 — Build a fully fresh successor after the start-460 rejection

Production publication remains blocked by start-460 `accepted=false`. Tag
`gfx1200-diagnostics-v7-production-v1` stays reserved until a later immutable
run records `accepted=true` and completes external receipt ingestion. None of
the start-460 corpus may be reused; the next inference profile must be fit from
a separately frozen development split and judged on 220 entirely fresh
held-out pairs.

The start640 failure was root-caused as a cross-design distribution shift, so a
single i.i.d.-split design is the fix. That successor is
**`p1-successor-start700`** (design `410042044bc9c67fa82e048ba49de54662fff27c27a233579658d4cc13f2d1e6`,
authored at `cb35de63`, re-qualified at `cbaa3822`). It is **paused by the
operator before acceptance**: qualification complete, development collection
stopped at 69/440, no SOLAR/freeze/inference/held-out/acceptance. Resumption is
`collect --role development` without re-qualifying, then the same sequence:
development collect → SOLAR → freeze → inference profile → held-out collect →
SOLAR → freeze → acceptance once.

The acceptance thresholds remain unchanged: at least 90% interval coverage per
family, median APE at most 15%, P90 APE at most 30%, and at least one enabled
code-changing action with at least 10 held-out positives, 90% precision, and
70% recall.

### P1 — Author and validate a separate MI300X capacity policy

The current total-memory selection admits only gfx1200 8 GiB / 16 GiB classes
and rejects gfx942/MI300X. A separately versioned CDNA3 policy must bind
observed total HBM, exact gfx942 device/software identity, topology/isolation,
a justified probe working set and applicability range, and real MI300X
qualification/calibration receipts. It must fail closed for unknown MI300X
variants and not change existing gfx1200 digest semantics. Runtime free memory,
capacity ratios, simulators, or schema-only tests are insufficient.

Authoritative surfaces:
`src/sol_execbench/core/bench/performance_model/vram_policy.py`,
`calibration.py`, `prediction.py`,
`src/sol_execbench/data/hardware_calibration_probes/diagnostic_microarchitecture.hip`,
`scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py`.

### P1 — Validate the non-formal gfx942 (CDNA3) adaptation on real hardware

The code-level non-formal gfx942 path is authored and verified on gfx1200 only:
`gfx942` now runs materialize → evaluate → `solar analyze` through the
non-formal bridge (`analyze_workload_diagnostic`), but every gfx942-specific
artifact is spec-derived and `inexact`. Formal publication stays gfx1200-only.
Re-derive the following on a real gfx942 (MI300X/MI308X) host before treating
any gfx942 claim as validated:

- Confirm the `gfx942_v1.yaml` counter spellings via
  `rocprofv3-avail -d 0 info --pmc`. The `cache_lds` group reuses RDNA
  spellings as placeholders; CDNA3 exposes different L2/TCP counters. Update the
  manifest and re-bind `counter_semantics_sha256` via
  `build_diagnostic_model_identity(model_version, counter_resource="gfx942_v1.yaml")`.
- Compile and run the CDNA3 MFMA probes and record `V_MFMA_*` ISA evidence via
  `scripts/internal/cdna3/run_cdna3_diagnostic_calibration.py`. Author the
  deferred fp8 (FNUZ encoding) and int8 (32x32x8 output layout) MFMA probes
  on-device.
- Confirm the `MI300X.yaml` roofline profile (304 CUs, 192 GiB HBM, ~5.3 TB/s,
  FP16/BF16/INT8 MFMA peaks, L2/L3 sizes) via `rocminfo` and replace the
  provisional values.
- Derive the `cdna3_total_memory_class.v1` probe working set (currently a
  provisional 8 GiB) with a wave64 `diagnostic_microarchitecture.hip` variant;
  a capacity-ratio/simulator justification is insufficient (see the MI300X
  capacity-policy P1 above).
- Run `uv run pytest -m requires_cdna3 -n 0` and record the full evidence chain.

Authoritative surfaces:
`src/sol_execbench/data/rocprofv3_counters/gfx942_v1.yaml`,
`src/solar/rocm/profiles/MI300X.yaml`,
`src/sol_execbench/core/bench/performance_model/vram_policy.py`,
`scripts/internal/cdna3/run_cdna3_diagnostic_calibration.py`,
`src/sol_execbench/core/solar_bridge/analyzer.py`.

Completion evidence must name the exact GPU, ROCm/PyTorch stack, test set, and
skipped prerequisites; results remain engineering/inexact until a formal
resource-peak calibration receipt is produced.

### P1 — Expand empirical hardware and isolation coverage

- Run `test_real_multi_gpu_candidate_device_switch_is_rejected` with at least
  two visible ROCm GPUs.
- Reacquire the dataset/source for revision `d56fadca` and rerun the six
  historical timeout observations on exact gfx942 (`FlashInfer-Bench/014`,
  `019`, `L2/040`, `L2/055`).
- Validate CDNA4 NVFP4/MXFP4 adaptation on representative hardware.

Completion evidence must name the exact GPU, ROCm/PyTorch stack, test set, and
skipped prerequisites.

### P2 — Close the expanded historical-store archive boundary

`data/store-control-plane-v2-historical-20260809/` is a 7.4 MiB historical P0
store rejected by current readers. `check_non_canonical_artifacts.py` reports
five unmarked v2 run manifests there. Do not migrate or rewrite them. Completion
requires either archiving the tree into `data/cold-archive/` with a reviewed
inventory/digest + deletion approval, or retaining it with a `NON_CANONICAL.md`
marker; the non-canonical check must then pass.

### P2 — Resolve the compressed code-object metadata boundary

Choose one: implement bounded CCOB manifest parsing with exact target-member
selection and real fixtures, or classify full CCOB parsing as unsupported and
keep heuristic gzip/zlib scanning as best-effort. The current scan must not be
described as complete CCOB support.

Authoritative surfaces:
`src/sol_execbench/core/bench/static_kernel/amdgpu_metadata.py`,
`tests/sol_execbench/core/bench/test_amdgpu_metadata.py`,
`docs/user/static_kernel_evidence.md`.

### P2 — Classify the Torchview backward-reference boundary

Four SOLAR comparison cases remain outside the 41-workload denominator because
the forward-only extractor cannot represent backward references. Choose one: a
representation preserving upstream-gradient dependencies (with conversion and
denominator tests), or an exact unsupported reason code + documented denominator
policy that keeps all four cases visible.

Authoritative surfaces: `src/solar/graph/torchview/extraction.py`,
`src/solar/pipeline/`, `scripts/internal/solar/run_cross_path_focus.py`,
`tests/solar/test_pipeline_integration.py`,
`docs/user/CROSS-PATH-COMPARISON.md`.

## Invariants

- Performance diagnostics never change canonical Trace timing, `T_SOL`, SOL
  Score, leaderboard values, or rewards.
- Canonical execution precedes profiler replay. Profiler duration, achieved
  throughput, and measured candidate runtime never become prediction features.
- Evidence identity, schema version, calibration range, artifact hashes, and
  lifecycle parents fail closed. Old-schema compatibility readers are forbidden.
- `L` remains unavailable without an explicitly supplied trusted frontier.
- Partial or ungoverned diagnostics cannot request kernel code changes.
- Tuning and parameter-estimation samples cannot enter held-out acceptance.
- Candidate inputs use per-run entropy and per-invocation trusted-reference
  validation. The candidate process receives neither nonce nor expected output.
- Publication evaluation remains networkless, capability-free, and private-IPC.
- Mutable process evidence stays under ignored `data/outputs/`; immutable
  diagnostic release projections stay under ignored `data/publications/`.
- Current `sol_execbench.*` schema identifiers are defined only in
  `src/sol_execbench/core/integrity/schema_versions.py`; current SOLAR versions
  are defined only in `src/solar/schema_versions.py`.

## Verification before handoff

Run the checks relevant to the changed surface, including:

```bash
uv run python scripts/check_current_docs.py
uv run python scripts/check_schema_versions.py
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
uv run ty check
uv run python scripts/check_coupling.py
uv run python scripts/check_readability.py
uv run python scripts/check_production_reachability.py
uv run python scripts/check_non_canonical_artifacts.py
uv run python scripts/check_diagnostic_store_consistency.py
uv run python scripts/check_python_reuse.py
uv run pytest tests/
git diff --check
```

Hardware claims additionally require the precisely marked ROCm tests on the
named device. Never treat a skip as passing hardware evidence.

At this audit every command above that was run passed except
`check_non_canonical_artifacts.py`, whose five historical v2 findings are the
explicit expanded-store archive-boundary backlog above.

### Handoff continuity for coding-agent switch

Bootstrap checks before any GPU work resumes:

1. Repository identity and local edit scope:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
```

Expected: branch `main`, `HEAD`
`fd7970520e205318e2e63d8a96d5203e1d5972c3`, plus the uncommitted CDNA3
gfx942 non-formal adaptation (see the P1 validation backlog above).

2. No accidental lifecycle run for the active source revision:

```bash
rg -n 'cbaa382289418bda706cabe7df964b24943c927a' data/store/orchestrations data/store/attempts -g '*.json'
```

Expected empty (no `p1-successor-start700` acceptance/publication evidence).

3. Controlled bootstrap state from the output tree:

```bash
ls -1 data/outputs/p1-successor-start700
ls -1 data/outputs/p1-successor-start700/corpus-qualification
ls -1 data/outputs/p1-successor-start700/corpus/cases/point_fit/elementwise | head
```

Expected: five qualification gates (`static`, `development/canary`,
`development/full`, `held_out/canary`, `held_out/full`) and a partially
collected development tree (69/440, no frozen corpus, no SOLAR).

4. Before resuming, run lifecycle admission checks with the reviewed command
contract from `docs/performance-diagnostics.md`:

```bash
uv run sol-execbench --format json diagnostics lifecycle status --run-id <RUN_ID> --store-root data/store
uv run sol-execbench --format json diagnostics lifecycle resume --run-id <RUN_ID> --store-root data/store
uv run sol-execbench --format json diagnostics lifecycle plan ...
uv run sol-execbench --format json diagnostics lifecycle run --plan PLAN.json --store-root data/store
```

5. Keep `source-review-18535cf-to-cb35de63.json` immutable (the start700
recovery preregistration proof). start700 is fully fresh — no
`record-exposure -> freeze-fragment -> compose-held-out` reuse route applies,
and no held-out payload may be composed from any prior `accepted=false` corpus.

Historical local continuation (2026-10-26 05:28 UTC): real GPU canary exposed an internal readiness-to-IPC bridge defect, now repaired with focused CPU regression passing. Full final-source CPU rerun and GPU canary/R16 remain pending; all owned GPU resources are released. See [safe continuation checkpoint](data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/CHECKPOINT-0528.md). Old source-bound CPU/F32/mapper receipts remain historical.

Historical local checkpoint (2026-10-26 05:53 UTC): final source 455e281d passes 1449 CPU tests and F32 32/32; GPU canary passes both paths. R16 retains 12 ready, two R08 boundary rejections and two R04 cache-write blockers (Triton EROFS). All owned GPU resources and the shared lock are released. See [current execution checkpoint](data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/CHECKPOINT-0553.md); do not treat readiness as full accounting or formal qualification.
