// GitHub webhook payload verification/parsing, sanitization, D1 trend
// storage, and log fetching — used by both the (thin) webhook handler in
// server.ts and GithubRunWorkflow, which does the actual
// fetch/parse/categorize/diagnose/store work. Kept free of Agents SDK /
// Durable Object imports so neither caller creates an import cycle with
// the other.

export interface WorkflowRunSummary {
  action?: string;
  repo?: string;
  workflowName?: string;
  runId?: number;
  runNumber?: number;
  status?: string;
  conclusion?: string | null;
  htmlUrl?: string;
  headSha?: string;
  updatedAt?: string;
}

export interface StepTrend {
  averageDurationSeconds: number;
  sampleSize: number;
  isRegression: boolean;
}

export interface JobStepSummary {
  name?: string;
  status?: string;
  conclusion?: string | null;
  number?: number;
  durationSeconds?: number;
  trend?: StepTrend;
}

export type JobCategory = "failure" | "regression" | "success";

export interface JobSummary {
  jobId?: number;
  runId?: number;
  jobName?: string;
  workflowName?: string;
  repo?: string;
  conclusion?: string | null;
  htmlUrl?: string;
  steps: JobStepSummary[];
  // Tail of the job's raw log, only captured when a step failed — lets the
  // LLM diagnose *why*, not just which step was slow.
  failureExcerpt?: string;
  category?: JobCategory;
  // Short plain-language diagnosis + next steps, generated once by
  // GithubRunWorkflow for anything that isn't a plain "success".
  diagnosis?: string;
  updatedAt?: string;
}

// Every GitHub webhook delivery is recorded onto this single, well-known
// agent instance so any chat session can read it back, regardless of
// which per-session ChatAgent the user is connected to.
export const GITHUB_MONITOR_AGENT_NAME = "github-monitor";

const MAX_FIELD_LENGTH = 300;

// Built from char codes (0, 31, 127) rather than regex escapes to avoid
// ambiguity with literal control characters in source.
const CONTROL_CHAR_PATTERN = new RegExp(
  "[" +
    String.fromCharCode(0) +
    "-" +
    String.fromCharCode(31) +
    String.fromCharCode(127) +
    "]",
  "g"
);

// Webhook-derived strings (workflow name, repo, commit SHA...) are
// attacker-influenceable (e.g. via a PR from a fork) and get replayed into
// every future chat session's model context through getGithubWorkflowRuns.
// Strip control characters and cap length so a poisoned field can't smuggle
// large or non-printable content into that shared memory.
export function sanitizeField(
  value: string | undefined | null
): string | undefined {
  if (!value) return undefined;
  const cleaned = value.replace(CONTROL_CHAR_PATTERN, " ").trim();
  if (!cleaned) return undefined;
  return cleaned.length > MAX_FIELD_LENGTH
    ? `${cleaned.slice(0, MAX_FIELD_LENGTH)}…`
    : cleaned;
}

export function computeStepDuration(
  startedAt?: string | null,
  completedAt?: string | null
): number | undefined {
  if (!startedAt || !completedAt) return undefined;
  const start = Date.parse(startedAt);
  const end = Date.parse(completedAt);
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) return undefined;
  return Math.round((end - start) / 1000);
}

// Raw job logs can run to MBs; slice the tail before doing any per-char work
// so a huge log can't burn CPU on sanitization it'll just get truncated
// away anyway — errors are almost always near the end of a failed job's log.
const RAW_LOG_TAIL_CHARS = 20_000;
const MAX_LOG_EXCERPT_CHARS = 4000;

export function sanitizeLogExcerpt(raw: string): string | undefined {
  const tail =
    raw.length > RAW_LOG_TAIL_CHARS ? raw.slice(-RAW_LOG_TAIL_CHARS) : raw;
  let cleaned = "";
  for (const ch of tail) {
    const code = ch.charCodeAt(0);
    if (code === 10 || (code >= 32 && code !== 127)) cleaned += ch;
  }
  cleaned = cleaned.trim();
  if (!cleaned) return undefined;
  return cleaned.length > MAX_LOG_EXCERPT_CHARS
    ? cleaned.slice(-MAX_LOG_EXCERPT_CHARS)
    : cleaned;
}

// GitHub has no per-step log API — only a whole-job log. We only fetch it
// when a step actually failed, and only keep the tail, to bound cost.
// Throws instead of swallowing errors — this now runs inside a Workflow
// step.do() call, which retries on a thrown error; swallowing here would
// silently defeat that retry.
export async function fetchJobLogExcerpt(
  pat: string | undefined,
  repoFullName: string | undefined,
  jobId: number | undefined
): Promise<string | undefined> {
  if (!pat || !repoFullName || !jobId) return undefined;
  const res = await fetch(
    `https://api.github.com/repos/${repoFullName}/actions/jobs/${jobId}/logs`,
    {
      headers: {
        Authorization: `Bearer ${pat}`,
        Accept: "application/vnd.github+json",
        "User-Agent": "github-action-assistant",
        "X-GitHub-Api-Version": "2022-11-28"
      }
    }
  );
  if (!res.ok) {
    throw new Error(
      `GitHub job log fetch failed: ${res.status} for job ${jobId}`
    );
  }
  return sanitizeLogExcerpt(await res.text());
}

export async function verifyGitHubSignature(
  secret: string,
  rawBody: string,
  signatureHeader: string | null
): Promise<boolean> {
  if (!signatureHeader?.startsWith("sha256=")) return false;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const mac = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(rawBody)
  );
  const expected = `sha256=${[...new Uint8Array(mac)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")}`;
  if (expected.length !== signatureHeader.length) return false;
  let mismatch = 0;
  for (let i = 0; i < expected.length; i++) {
    mismatch |= expected.charCodeAt(i) ^ signatureHeader.charCodeAt(i);
  }
  return mismatch === 0;
}

export function parseGitHubPayload(
  rawBody: string,
  contentType: string
): unknown {
  if (contentType.includes("application/json")) {
    return JSON.parse(rawBody);
  }
  // GitHub webhook configured with content type "form" sends the JSON
  // payload URL-encoded under a single "payload" field.
  const encoded = new URLSearchParams(rawBody).get("payload");
  if (!encoded) throw new Error("Missing payload field in form body");
  return JSON.parse(encoded);
}

interface RawWorkflowRunPayload {
  action?: string;
  repository?: { full_name?: string };
  workflow_run?: {
    id?: number;
    run_number?: number;
    name?: string;
    status?: string;
    conclusion?: string | null;
    html_url?: string;
    head_sha?: string;
    updated_at?: string;
  };
}

export function parseWorkflowRunPayload(payload: unknown): WorkflowRunSummary {
  const p = payload as RawWorkflowRunPayload;
  const run = p.workflow_run;
  return {
    action: sanitizeField(p.action),
    repo: sanitizeField(p.repository?.full_name),
    workflowName: sanitizeField(run?.name),
    runId: run?.id,
    runNumber: run?.run_number,
    status: sanitizeField(run?.status),
    conclusion: sanitizeField(run?.conclusion),
    htmlUrl: sanitizeField(run?.html_url),
    headSha: sanitizeField(run?.head_sha),
    updatedAt: sanitizeField(run?.updated_at)
  };
}

interface RawWorkflowJobPayload {
  action?: string;
  repository?: { full_name?: string };
  workflow_job?: {
    id?: number;
    run_id?: number;
    name?: string;
    workflow_name?: string;
    conclusion?: string | null;
    html_url?: string;
    steps?: Array<{
      name?: string;
      status?: string;
      conclusion?: string | null;
      number?: number;
      started_at?: string | null;
      completed_at?: string | null;
    }>;
  };
}

export interface ParsedWorkflowJob {
  jobId?: number;
  runId?: number;
  jobName?: string;
  workflowName?: string;
  repo?: string;
  conclusion?: string | null;
  htmlUrl?: string;
  updatedAt: string;
  steps: JobStepSummary[];
  hasFailedStep: boolean;
}

// Returns null when the job isn't in its final "completed" state yet — step
// start/end times aren't final until then, so there's nothing meaningful to
// process for queued/in_progress deliveries.
export function parseWorkflowJobPayload(
  payload: unknown
): ParsedWorkflowJob | null {
  const p = payload as RawWorkflowJobPayload;
  if (p.action !== "completed") return null;
  const job = p.workflow_job;
  const steps: JobStepSummary[] = (job?.steps ?? []).map((step) => ({
    name: sanitizeField(step.name),
    status: sanitizeField(step.status),
    conclusion: sanitizeField(step.conclusion) ?? null,
    number: step.number,
    durationSeconds: computeStepDuration(step.started_at, step.completed_at)
  }));
  const hasFailedStep = steps.some(
    (step) => step.conclusion === "failure" || step.conclusion === "timed_out"
  );
  return {
    jobId: job?.id,
    runId: job?.run_id,
    jobName: sanitizeField(job?.name),
    workflowName: sanitizeField(job?.workflow_name),
    repo: sanitizeField(p.repository?.full_name),
    conclusion: sanitizeField(job?.conclusion) ?? null,
    htmlUrl: sanitizeField(job?.html_url),
    updatedAt: new Date().toISOString(),
    steps,
    hasFailedStep
  };
}

export function categorizeJob(
  hasFailedStep: boolean,
  steps: JobStepSummary[]
): JobCategory {
  if (hasFailedStep) return "failure";
  if (steps.some((step) => step.trend?.isRegression)) return "regression";
  return "success";
}

// ── D1: durable step-duration history for trend/regression comparisons ──
//
// The ChatAgent DO state only keeps the last 20 runs / 30 jobs for quick
// lookup and gets evicted/recreated over time — it's not a trend store.
// Every completed step gets one row here, keyed by (repo, workflow, step),
// so a step's current duration can be compared against its own history
// instead of only ever getting a one-off "this took N seconds" answer.

export async function insertStepRuns(
  db: D1Database,
  repo: string | undefined,
  workflow: string | undefined,
  runId: number | undefined,
  jobId: number | undefined,
  steps: JobStepSummary[],
  timestamp: string
): Promise<void> {
  if (!repo || !workflow || runId === undefined) return;
  const rows = steps.filter((step) => step.name);
  if (rows.length === 0) return;
  const stmt = db.prepare(
    `INSERT INTO step_runs (repo, workflow, run_id, job_id, step, duration_seconds, status, timestamp)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  );
  await db.batch(
    rows.map((step) =>
      stmt.bind(
        repo,
        workflow,
        runId,
        jobId ?? null,
        step.name,
        step.durationSeconds ?? null,
        step.conclusion ?? step.status ?? null,
        timestamp
      )
    )
  );
}

const TREND_SAMPLE_SIZE = 10;
// How much slower than its own baseline a step has to be before we call it
// a regression rather than normal run-to-run variance.
const REGRESSION_FACTOR = 1.5;
// Below this many prior samples, an "average" is too noisy to act on.
const MIN_SAMPLES_FOR_TREND = 2;

export async function getStepTrend(
  db: D1Database,
  repo: string,
  workflow: string,
  step: string,
  runId: number | undefined,
  currentDurationSeconds: number | undefined
): Promise<StepTrend | null> {
  if (currentDurationSeconds === undefined) return null;
  const result = await db
    .prepare(
      `SELECT duration_seconds FROM step_runs
       WHERE repo = ? AND workflow = ? AND step = ?
         AND duration_seconds IS NOT NULL
         AND status = 'success'
         AND run_id != ?
       ORDER BY timestamp DESC
       LIMIT ?`
    )
    .bind(repo, workflow, step, runId ?? -1, TREND_SAMPLE_SIZE)
    .all<{ duration_seconds: number }>();
  const samples = result.results ?? [];
  if (samples.length < MIN_SAMPLES_FOR_TREND) return null;
  const average =
    samples.reduce((sum, row) => sum + row.duration_seconds, 0) /
    samples.length;
  return {
    averageDurationSeconds: Math.round(average),
    sampleSize: samples.length,
    isRegression: currentDurationSeconds > average * REGRESSION_FACTOR
  };
}

// ── Trend view (chat UI) ──
//
// A compact overview across the most recently active (repo, workflow, step)
// combinations, for the chat UI's Trends panel — distinct from getStepTrend
// above, which answers "is this one specific run's duration a regression."
// Includes every recorded status (not just successes), since this is a
// general activity view rather than a regression baseline.

export interface StepTrendSeries {
  repo: string;
  workflow: string;
  step: string;
  // Oldest → newest; the last entry is the most recent run.
  durations: number[];
  latestDurationSeconds: number;
  averageDurationSeconds: number;
  isRegression: boolean;
  sampleSize: number;
  latestStatus: string | null;
  latestTimestamp: string;
}

const TREND_PANEL_STEP_LIMIT = 8;
const TREND_PANEL_SAMPLE_LIMIT = 10;

export async function getRecentStepTrends(
  db: D1Database
): Promise<StepTrendSeries[]> {
  const groups = await db
    .prepare(
      `SELECT repo, workflow, step, MAX(timestamp) AS latest_timestamp, COUNT(*) AS sample_size
       FROM step_runs
       WHERE duration_seconds IS NOT NULL
       GROUP BY repo, workflow, step
       ORDER BY latest_timestamp DESC
       LIMIT ?`
    )
    .bind(TREND_PANEL_STEP_LIMIT)
    .all<{
      repo: string;
      workflow: string;
      step: string;
      latest_timestamp: string;
      sample_size: number;
    }>();

  return Promise.all(
    (groups.results ?? []).map(async (group) => {
      const rows = await db
        .prepare(
          `SELECT duration_seconds, status, timestamp FROM step_runs
           WHERE repo = ? AND workflow = ? AND step = ? AND duration_seconds IS NOT NULL
           ORDER BY timestamp DESC
           LIMIT ?`
        )
        .bind(group.repo, group.workflow, group.step, TREND_PANEL_SAMPLE_LIMIT)
        .all<{
          duration_seconds: number;
          status: string | null;
          timestamp: string;
        }>();
      // Query is newest-first (for the LIMIT to keep the *recent* N); the
      // panel reads left-to-right as oldest-to-newest, so reverse it here.
      const recent = (rows.results ?? []).slice().reverse();
      const durations = recent.map((row) => row.duration_seconds);
      const latest = recent[recent.length - 1];
      const priorDurations = durations.slice(0, -1);
      const average =
        durations.reduce((sum, d) => sum + d, 0) / (durations.length || 1);
      const priorAverage =
        priorDurations.length > 0
          ? priorDurations.reduce((sum, d) => sum + d, 0) /
            priorDurations.length
          : average;

      return {
        repo: group.repo,
        workflow: group.workflow,
        step: group.step,
        durations,
        latestDurationSeconds: latest?.duration_seconds ?? 0,
        averageDurationSeconds: Math.round(average),
        // Same MIN_SAMPLES_FOR_TREND/REGRESSION_FACTOR rule as getStepTrend,
        // so the panel and the chat tool never disagree about what counts
        // as a regression.
        isRegression:
          priorDurations.length >= MIN_SAMPLES_FOR_TREND &&
          (latest?.duration_seconds ?? 0) > priorAverage * REGRESSION_FACTOR,
        sampleSize: group.sample_size,
        latestStatus: latest?.status ?? null,
        latestTimestamp: latest?.timestamp ?? group.latest_timestamp
      };
    })
  );
}
