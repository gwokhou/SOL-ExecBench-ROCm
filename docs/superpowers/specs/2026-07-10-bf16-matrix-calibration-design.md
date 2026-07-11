# BF16 Matrix Calibration Probe Design

## Goal

Replace the calibration backend's deliberately unsupported BF16 matrix
placeholder with real, architecture-specific GPU microbenchmarks.  A supported
adapter may produce a validated calibration artifact only after its exact BF16
matrix instruction path compiles, executes, passes numerical validation, and
produces stable throughput samples.

## Scope

- `gfx94*` and `gfx95*` use a BF16 matrix-FMA (MFMA) probe.
- `gfx12*` uses a BF16 WMMA probe, because RDNA4 exposes that distinct matrix
  instruction family rather than MFMA.  Its profile key is
  `compute.matrix.bf16.bf16.wmma`.
- Each probe compiles only for its declared ISA family.  It distinguishes
  conclusively unsupported paths from paths that could not be assessed; the
  precise rules are defined below.
- Matrix probes retain the current seven-sample conservative statistic and
  stability validation.  They report TFLOP/s, counting BF16 fused
  multiply-add as two floating-point operations.

This change does not add a fabricated fallback, change packaged models, or
promote a model without live calibration evidence.

## Architecture and Data Flow

`ArchitectureAdapter` will declare the matrix key appropriate to its family:

| Family | Key path | Probe implementation |
| --- | --- | --- |
| `gfx12*` | `wmma` | BF16 WMMA HIP kernel |
| `gfx94*` | `mfma` | BF16 MFMA HIP kernel |
| `gfx95*` | `mfma` | BF16 MFMA HIP kernel |

`HipCommandBackend.compile()` selects the kernel source from the complete
profile key.  It passes the adapter's target architecture to `hipcc`, so an
incorrect host/device path cannot accidentally compile a generic FP32 source.
The generated program performs a tiled BF16 matrix product with an FP32
accumulator, emits seven throughput samples, copies a deterministic output
tile back to the host, and validates it against a CPU FP32 reference within a
documented BF16-appropriate tolerance.

`execute()` continues to parse only positive finite `RESULT` lines.  A
successful process therefore reaches the existing correctness/stability seams
and `CalibrationCandidate` statistics unchanged.

### Evidence-state rules

`unavailable` and `unknown` have intentionally non-overlapping meanings:

| State | Meaning | BF16 matrix examples |
| --- | --- | --- |
| `measured` | The exact adapter path compiled, ran, produced valid samples, and passed numerical and stability checks. | A `gfx942` MFMA or `gfx1200` WMMA probe completes and its output matches the CPU reference. |
| `unavailable` | A completed capability check conclusively established that this device/ISA does not expose the exact declared instruction path. It is not a toolchain, launch, or correctness error. | A successfully queried device reports that BF16 MFMA/WMMA is unsupported for its selected architecture. |
| `unknown` | The system could not determine support reliably. It must never be interpreted as unsupported or as a zero-value measurement. | `hipcc` missing or failing, an unrecognised compiler diagnostic, process/driver failure, malformed benchmark output, numerical mismatch, or unstable samples. |

The implementation may return `unavailable` only from an explicit,
architecture-aware capability check whose result is independent of a transient
tool or execution failure.  In particular, a failed compilation is `unknown`;
the compile command alone is insufficient evidence that a live device lacks
the instruction.  Every non-measured candidate carries the stable reason code
that identifies this distinction.

## Validation Semantics

The adapter's declared set is its required calibrated matrix.  On a hardware
and ROCm combination that supports the corresponding instruction path, all
declared candidates, including BF16 matrix, must be measured before the
existing clock-provenance checks can yield `validated`.  On a combination that
cannot support or cannot reliably compile the matrix path, calibration remains
diagnostic (`provisional`) and model build continues to reject it.  This keeps
official-score authority strict while making validated artifacts attainable on
the intended supported hardware.

## Tests

Unit coverage will prove:

1. adapters declare WMMA for `gfx12*` and MFMA for `gfx94*`/`gfx95*`;
2. the production backend selects an architecture-specific BF16 source and
   passes the matching offload architecture to `hipcc`;
3. an explicit capability-negative result is `unavailable`, whereas a missing
   or failing compiler, an unrecognised diagnostic, failed launch, malformed
   output, numerical mismatch, and unstable samples are each `unknown`;
4. a successful BF16 program result is measured, including numerical and
   stability checks;
5. a complete RDNA4 probe (including BF16 WMMA) produces validated provenance,
   whereas missing BF16 evidence remains provisional.

Marker-gated live tests will compile and execute each available architecture's
actual matrix probe: `requires_rdna4`, `requires_cdna3`, and `requires_cdna4`.
They do not run on unavailable hardware and do not claim cross-architecture
measurement from unit-test doubles.

## Non-goals

- Supporting unrelated BF16 paths or data types.
- Using rocWMMA as a runtime dependency for the shipped probe.
- Changing hardware-model authority gates to accept unavailable candidates.
