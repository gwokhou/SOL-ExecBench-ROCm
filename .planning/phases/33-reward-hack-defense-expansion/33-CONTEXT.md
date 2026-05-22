# Phase 33 Context: Reward-Hack Defense Expansion

**Date:** 2026-05-22
**Status:** Complete

## Problem

The evaluation driver already caught several runtime reward hacks, including
timing monkey-patches, thread injection, lazy/proxy outputs, eval-driver
function replacement, and runtime `load_inline()` calls. It did not have a
structured static review layer for source patterns that can execute at import
time or distort timing/correctness before runtime checks run.

## Relevant Code

- `src/sol_execbench/core/bench/reward_hack.py` contains reward-hack detection
  helpers.
- `src/sol_execbench/driver/templates/eval_driver.py` emits traces and applies
  runtime checks during evaluation.
- `tests/sol_execbench/core/bench/test_reward_hack.py` covers pure reward-hack
  helpers.
- `tests/sol_execbench/driver/test_eval_driver.py` covers subprocess driver
  behavior.

## Constraints

- Legitimate PyTorch, Triton, and HIP/C++ submissions must continue to pass by
  default.
- Canonical trace schema remains unchanged.
- Static review must be testable without GPU hardware.
