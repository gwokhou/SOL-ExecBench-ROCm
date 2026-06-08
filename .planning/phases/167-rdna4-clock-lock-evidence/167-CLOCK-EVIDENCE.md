# Phase 167 Clock Evidence

## Verdict

Phase 167 collected usable RDNA4 host clock-control evidence, but it does not
support upgrading RDNA4 timing to benchmark-grade authoritative timing on this
host.

The command path is available:

- `sudo rocm-smi --setperflevel manual` succeeded.
- `sudo rocm-smi --setsclk 2` succeeded and reported SCLK bitmask `0x4`.
- `sudo rocm-smi --setmclk 5` succeeded and reported MCLK bitmask `0x20`.
- `sudo rocm-smi --resetclocks` succeeded and reset performance level to
  `auto`.

The observed clocks did not prove stable maximum-frequency lock:

- Requested SCLK level: `2` / `2780Mhz`.
- Observed post-set SCLK level: `1` / `0Mhz`.
- Requested MCLK level: `5` / `1258Mhz`.
- Observed post-set MCLK level: `4` / `1124Mhz`.
- Host was idle and `rocm-smi` repeatedly warned that the AMD GPU was in a
  low-power state.

## Host And Device

- GPU: AMD Radeon Graphics
- GFX version: `gfx1200`
- Device ID: `0x7590`
- Bus: `0000:03:00.0`
- VBIOS: `113-R906XGOI-F1`
- Driver version: `6.17.0-35-generic`
- Final evidence timestamp: `2026-06-08T16:11:40Z`

## Pre-State

Initial read-only evidence showed:

- Performance level: `auto`
- MCLK: level `0`, `96Mhz`
- SCLK: level `1`, `0Mhz`
- Temperature: edge `41C`, junction `43C`, memory `64C`
- Power: `12W`
- GPU utilization: `0%`
- Low-power warning present

## Command Sequence

Commands executed:

```bash
rocm-smi --showclocks --showperflevel --showproductname --showdriverversion --showhw
rocm-smi --showpower --showtemp --showuse
sudo rocm-smi --setperflevel manual
rocm-smi --showclocks --showperflevel --showclkfrq
sudo rocm-smi --setsclk 2
sudo rocm-smi --setmclk 5
rocm-smi --showclocks --showperflevel --showclkfrq --showproductname --showdriverversion --showhw
rocm-smi --showpower --showtemp --showuse
sudo rocm-smi --resetclocks
rocm-smi --showclocks --showperflevel --showclkfrq --showproductname --showdriverversion --showhw
rocm-smi --showpower --showtemp --showuse
```

## Post-Set State

After successful manual/per-clock commands:

- Performance level: `manual`
- Supported MCLK levels included `5: 1258Mhz`
- Supported SCLK levels included `2: 2780Mhz`
- Observed MCLK: level `4`, `1124Mhz`
- Observed SCLK: level `1`, `0Mhz`
- FCLK: `1860Mhz`
- SOCCLK: `1200Mhz`
- Temperature: edge `42C`, junction `45C`, memory `65C`
- Power: `23W`
- GPU utilization: `0%`

## Reset State

`sudo rocm-smi --resetclocks` reported:

- `Successfully reset clocks`
- `Performance level reset to auto`
- `set_overdrive_level, Not supported on the given system`

Final read-only evidence showed:

- Performance level: `auto`
- MCLK: level `0`, `96Mhz`
- SCLK: level `1`, `0Mhz`
- FCLK: `666Mhz`
- SOCCLK: `417Mhz`
- Temperature: edge `41C`, junction `44C`, memory `65C`
- Power: `12W`
- GPU utilization: `0%`
- Low-power warning present

## Claim Boundary

For this `gfx1200` host, Phase 167 supports the following claim:

> RDNA4 clock-control commands and reset were exercised and recorded, but
> stable maximum clock lock was not proven under the observed idle/low-power
> state.

It does not support:

- benchmark-grade authoritative timing;
- leaderboard-grade RDNA4 timing comparisons;
- paper-parity timing claims;
- clock-locked timing claims for CDNA3, CDNA4, or other AMD devices.

## Phase 168 Input

Phase 168 should include this artifact in the release evidence bundle and keep
RDNA4 timing categorized below authoritative timing unless a later host run
records stable observed clocks during the benchmark window and reset evidence
after the run.
