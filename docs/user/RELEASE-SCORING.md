# Release and Official Score Workflow

Canonical Trace JSONL is the evaluation artifact. A numeric formula result is
not an official score unless every input crosses an independently verifiable
release authority boundary.

## Current v3 status

Official scoring is unavailable for the checked-in 14-workload RX 9060 XT
corpus. The corpus manifest records
`reason_code: release_authority_not_published`. The current CLI exposes only
`score status`; no official scorer is implemented. Caller-authored measurement,
baseline, or SOLAR JSON is not accepted as authority.

The packaged architecture profile also records that its claimed locked-clock
resource-audit artifact was not published. Formal SOLAR analysis therefore
fails at the architecture stage; diagnostic candidate evaluation does not
upgrade this state.

## Evidence required for a future release

A release may enable official scoring only after pinning all of the following
to the immutable corpus identity:

1. A content-addressed release baseline with exact 14-workload coverage,
   solution identity, environment identity, locked clocks, and the paper timing
   protocol.
2. A passing independent rerun that verifies every baseline workload and its
   canonical trace.
3. A trusted candidate execution attestation bound to canonical traces,
   solution content, the same environment, isolation policy, and timing
   protocol. A self-declared JSON field is not an attestation.
4. One reviewed SOLAR manifest per scored workload, with exact manifest and
   artifact digests, reference identity, architecture identity, conversion
   proof, analysis contract, and formal bound.
5. The verified architecture resource-audit file cited by the packaged
   profile, including its content digest and locked-clock provenance.

The release verifier must reject missing workloads, sentinels in the score
denominator, duplicate UUIDs, non-finite latency, mismatched identities,
unlocked clocks, diagnostic timing, unsafe local execution, failed reruns, and
candidate runtimes below the formal bound.

## Local analysis

`sol_execbench.core.scoring.formula.sol_score` is the formula helper for audited local
inputs. Its output must be described as local or diagnostic while official
authority is unavailable. Incorrect candidates score zero; correct candidates
must satisfy `T_b > T_SOL` and `T_k >= T_SOL`.
