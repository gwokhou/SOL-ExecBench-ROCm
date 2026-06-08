---
status: passed
verified_at: "2026-06-08T16:11:40Z"
---

# Phase 167 Verification

## Checks

- `rocm-smi --showclocks --showperflevel --showproductname --showdriverversion --showhw`
- `rocm-smi --showpower --showtemp --showuse`
- `sudo rocm-smi --setperflevel manual`
- `sudo rocm-smi --setsclk 2`
- `sudo rocm-smi --setmclk 5`
- `sudo rocm-smi --resetclocks`
- `rocm-smi --showclocks --showperflevel --showclkfrq --showproductname --showdriverversion --showhw`

## Result

Passed for evidence collection and guardrail enforcement.

The command path and reset path were verified. The final read-only state showed
`Performance Level: auto`, so the GPU was not left in manual clock mode.

## Residual Risk

Observed post-set clocks did not match requested maximum SCLK/MCLK levels under
idle low-power conditions. RDNA4 timing must remain non-authoritative until a
future benchmark-window run records stable observed locked clocks and reset
evidence.
