# Open Issues

## ISSUE-2026-06-25: Terminology Refactor for Graph / Path / Trace

Status: open

Context:

- User-facing terminology now treats **Graph** as the real dependency graph
  captured from instrumentation/failing-test execution, **Path** as the issue
  symptom-to-root-cause subgraph/path, and **Trace** as the model/agent
  execution trajectory.
- Dashboard labels and docs should use this split. Existing implementation
  field names such as `hit_precision`, `hit_recall`, `hit_f1`,
  `chain_read_precision`, and `chain_*` are legacy/internal.
- The current dashboard copy has been partially aligned, but Python/JavaScript
  variable names, helper names, comments, test names, and persisted JSON keys
  still mix legacy `trace`, `chain`, `call_graph`, and `path` wording.

Follow-up scope:

- Rename dashboard and source variables, helper names, comments, test names, and
  documentation to align with Graph / Path / Trace.
- Introduce backward-compatible adapters for existing artifact/DB JSON keys
  where field names cannot be changed in-place.
- Keep backward-compatible reads for older artifact fields.
- Update README, proposal/report HTML, and dashboard copy in the same change.
- Avoid changing scorer behavior while doing the terminology refactor.

Acceptance criteria:

- User-facing dashboard labels consistently use Graph for instrumented
  dependency graph metrics, Path for symptom-to-root-cause dependency-path
  metrics, and Trace for agent execution trajectories.
- Source comments and non-persisted variable/helper names follow the same
  terminology.
- Persisted legacy keys are isolated behind compatibility mapping code and are
  not reused as new public terminology.
- Tests assert the new terms and cover old-artifact compatibility.
