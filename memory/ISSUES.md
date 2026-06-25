# Open Issues

## ISSUE-2026-06-25: Terminology Refactor for Graph / Path / Trace

Status: done

Context:

- User-facing terminology now treats **Graph** as the real dependency graph
  captured from instrumentation/failing-test execution, **Path** as the issue
  symptom-to-root-cause subgraph/path, and **Trace** as the model/agent
  execution trajectory.
- Dashboard labels, docs, and non-persisted helper/variable names use this
  split. Legacy artifact/DB keys such as `call_graph_*`, `chain_*`, and
  `dynamic_traceable_*` remain readable and writable only through compatibility
  alias layers.

Completed scope:

- Renamed dashboard/source helper paths, comments, tests, and documentation to
  align with Graph / Path / Trace.
- Added backward-compatible adapters for existing artifact/DB JSON keys where
  field names cannot be changed in-place.
- Kept backward-compatible reads/writes for older artifact fields.
- Updated README, proposal/report text, and dashboard copy in the same change.
- Kept scorer behavior unchanged apart from current-name aliases.

Acceptance criteria:

- User-facing dashboard labels consistently use Graph for instrumented
  dependency graph metrics, Path for symptom-to-root-cause Path metrics, and
  Trace for agent execution trajectories.
- Source comments and non-persisted variable/helper names follow the same
  terminology.
- Persisted legacy keys are isolated behind compatibility mapping code and are
  not reused as new public terminology.
- Tests assert the new terms and cover old-artifact compatibility.
