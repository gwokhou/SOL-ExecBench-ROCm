# Official SOL score release — AMD Radeon RX 9060 XT (gfx1200)

This directory holds the repository publish marker for the gfx1200 official SOL
score. [`release-bundle.json`](release-bundle.json) is the content-addressed
release bundle: it binds the corpus manifest, baseline, candidate, and SOLAR
statements by SHA-256 and byte size. Its presence is what flips
`sol-execbench score status` to `published`.

## Published score

| field | value |
| --- | --- |
| Official SOL score | **0.497818327** |
| Candidate | `rx9060xt-gfx1200-eager-reference-self-eval` |
| Baseline | `rx9060xt-gfx1200-reference-v2` |
| Source revision | `82abdc0bcc4385f2ebc74085fff2e8f5e0f2b310` |
| Container image | `sol-execbench:rocm-7.2-complete` (`sha256:476d88704a51c55459d28286073f53b41fefcc8018f1d8f219f291536a7e5321`) |
| Coverage | 43 scored problems, 163 scored workloads |

The candidate is the canonical eager-PyTorch reference scored against itself,
so each workload satisfies `T_k ≈ T_b` and the suite score is ≈ 0.5. Baseline
and candidate are independent locked-clock (`STABLE_PEAK`) GPU runs under the
paper timing protocol (warmup 10, timed 50, 3 trials, mean, 2× detected-L2
cache clear), recorded inside the pinned hardened container.

## Release evidence + verification

The full verified evidence set is a deterministic zstd archive produced by the
`score release-package` CLI command and distributed via the GitHub Release
tagged `gfx1200-official-score-v2` (asset
`gfx1200-official-score-release.tar.zst`, ~84 MB; `attestation.json` alongside).

Reproduce the official score from the distributed archive:

```bash
uv run sol-execbench score release-verify gfx1200-official-score-release.tar.zst \
  --expected-sha256 "$(python -c 'import json;print(json.load(open("attestation.json"))["archive"]["sha256"])')"
# -> Official SOL score: 0.497818327
```

`score release-verify` extracts the archive and re-runs the fail-closed
verifier (`verify_and_score_release`), which checks the repository-pinned corpus
hash, corpus-pinned baseline ID, canonical eager baseline, one shared source
revision, identical validated runtime environments, full problem/workload
coverage, passing baseline traces, complete candidate traces, the paper timing
protocol, and the exact SOLAR artifact denominator. Raw caller-supplied timing
values are not accepted. See `docs/user/RELEASE-SCORING.md` for the full
publication cutover and the `Score Release` GitHub Actions workflow.
