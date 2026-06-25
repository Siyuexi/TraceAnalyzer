# Project Memory

- Dashboard semantics must stay aligned with P2A functionality and public
  research terminology. Trace labels, KPI names, legends, reports, README text,
  and proposal/report claims should not invent meanings that are absent from
  `p2a/core.py`, `p2a/eval_fault_localization.py`, or
  `p2a/dashboard_adapter.py`.
- For read/write/error/root-cause dashboard features, first add or reuse backend
  parser/scorer fields, then render those fields in the frontend. Frontend-only
  inference is acceptable only as a compatibility fallback for older artifacts.
- The eval SQLite DB is raw capture and run status by default. It should store
  raw rollout content, issue descriptions, golden patches, token/runtime data,
  and artifact references; dashboard metrics and trace pattern states should be
  recomputed from raw DB content plus bonus maps. Any future dashboard score
  persistence is one-way dashboard -> DB, and default reads should still
  recompute rather than trust stale DB score fields.
- Dashboard issue descriptions and golden patches may be filled from local
  dataset parquet files when old DB rows lack those raw fields. Node Source
  should come from the explicit or inferred P2A bonus-map directory; DB
  `source_preview` is only an old-artifact fallback.
- The unified dashboard must remain compatible with local training, local
  inference, and third-party API inference artifacts, and new fields should
  degrade cleanly when older artifacts do not contain them.
- Execution failure is a structured runtime/tool-result concept, not a raw text
  keyword search over source code or observations. Miracle/reverse metrics and
  trace pattern markers must be derived from the same backend fields.
- A single read step can hit multiple bonus-map roles. The dashboard should
  render split step colors for symptom/intermediate/root-cause role sets, while
  same-callable symptom+root-cause nodes use a diagonal split. Miracle detection
  must treat same-step symptom/intermediate/root observations as simultaneous
  evidence, not as root-before-evidence.
- Terminology policy: use **Graph** for the real dependency graph captured from
  instrumentation/failing-test execution, **Path** for the issue
  symptom-to-root-cause subgraph/path, and **Trace** only for the model/agent
  execution trajectory. User-facing dashboard labels, legends, README text, and
  proposal/report text should follow this Graph / Path / Trace split. Existing
  code field names such as `hit_precision`, `hit_recall`, and
  `chain_read_precision` are legacy/internal until a dedicated terminology
  refactor.
