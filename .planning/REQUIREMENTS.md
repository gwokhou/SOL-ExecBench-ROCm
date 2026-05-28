# Requirements: SOL ExecBench ROCm Port

## Current Requirements

No active milestone requirements are open.

The most recent shipped requirements are archived at
`.planning/milestones/v1.18-REQUIREMENTS.md`.

Start the next milestone with `/gsd-new-milestone` to define the next active
requirements set.

## Deferred Requirements

### Native Host Matrix

- **HOST-01**: Native host validation can be run on separate machines or
  reinstalled hosts for ROCm 7.0.x, 7.1.x, and 7.2.x.
- **HOST-02**: Native host validation can compare direct host results against
  Docker user-space results for the same Target.

### Extended Hardware Coverage

- **HW-01**: CDNA 3 and CDNA 4 compatibility Matrix Entries can be marked
  `host_validated` or `container_validated` only when archived real-hardware
  evidence exists.
- **HW-02**: Matrix reports can aggregate compatibility status by architecture
  family after multiple hardware Targets have evidence.

### Matrix Tooling

- **TOOL-01**: Matrix reports can be diffed across runs to highlight status,
  dependency, image, or runtime changes.
- **TOOL-02**: Compatibility JSON schemas can be exported for external CI or
  downstream consumers.
