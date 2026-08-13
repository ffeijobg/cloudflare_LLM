-- Migration number: 0001 2026-08-12T00:00:00.000Z

-- One row per (run, job, step) delivered via the workflow_job webhook.
-- Durable, queryable history for trend/regression comparisons — the
-- ChatAgent's DO state only keeps the last 20/30 runs/jobs for quick
-- lookup and isn't meant as the source of truth for history.
CREATE TABLE IF NOT EXISTS step_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  repo TEXT NOT NULL,
  workflow TEXT NOT NULL,
  run_id INTEGER NOT NULL,
  job_id INTEGER,
  step TEXT NOT NULL,
  duration_seconds INTEGER,
  status TEXT,
  timestamp TEXT NOT NULL
);

-- Matches the trend query's WHERE (repo, workflow, step) + ORDER BY timestamp.
CREATE INDEX IF NOT EXISTS idx_step_runs_trend
  ON step_runs (repo, workflow, step, timestamp);

CREATE INDEX IF NOT EXISTS idx_step_runs_run_id
  ON step_runs (run_id);
