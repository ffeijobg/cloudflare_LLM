# GitHub Actions Assistant

A GitHub Actions CI/CD monitoring chatbot built on Cloudflare (Workers, Workflows,
Workers AI, D1, Durable Objects). It listens to GitHub Actions webhook deliveries,
parses and categorizes each job, generates an automated diagnosis for anything
that isn't a plain success, and lets you ask about it in a chat UI backed by
real historical trend data.

> **This project was built with AI-assisted coding (Claude Code).** The full
> prompt history — every request made throughout development, in order — is
> kept at [`../prompts.md`](../prompts.md). The scope brief and the
> requirements analysis this app was built against are at
> [`../scope.md`](../scope.md) and [`../review.md`](../review.md).

## Scope

The brief ([`../scope.md`](../scope.md)) asks for four things, and
[`../review.md`](../review.md) maps them to specific Cloudflare products. This
app's mapping:

| Requirement                 | What it means                                                                    | Used here                                                                                                                                                                                           |
| --------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM**                     | The decision engine — reasons about input and picks tool calls                   | Workers AI, **Llama 3.3** (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`), used both in the interactive chat and in the automated diagnosis step below                                                 |
| **Workflow / coordination** | Sequences multi-step logic, retries on failure, survives longer than one request | Both layers, on purpose: the **Agents SDK** (a Durable Object) drives the non-deterministic chat loop, and a real **Cloudflare Workflow** drives the deterministic per-webhook pipeline (see below) |
| **User input**              | A live channel for a human to send messages and get responses back               | A chat UI (Pages-style, `app.tsx`) over WebSocket — streaming, not request/response                                                                                                                 |
| **Memory / state**          | Persistence across turns and restarts                                            | Durable Object SQLite state (fast, recent-only) **plus** D1 (durable, queryable trend history) — see [D1 is a remote resource](#d1-is-a-remote-resource) below                                      |

This is deliberately narrower than a general-purpose chat agent: the system
prompt and a pre-model guardrail both restrict it to GitHub Actions questions
for the one connected repo. It has no weather/calculator/general-knowledge
capability, and doesn't support other CI/CD platforms (GitLab, Jenkins,
CircleCI, ...) — asking about those gets refused before the model is even
called.

## How it works

**Webhook → Workflow (automated path).** GitHub POSTs to `/webhook` on every
`workflow_run` and `workflow_job` event. The Worker's only job there is fast:
verify the HMAC-SHA256 signature (`GITHUB_WEBHOOK_SECRET`), do the minimum
parsing needed to route the event, and hand off to a Cloudflare Workflow
(`GithubRunWorkflow`) — then return `200` immediately. Everything else runs
inside the Workflow, as independently-retried, checkpointed steps:

1. **parse** — turn the raw GitHub payload into a structured summary
   (`src/github.ts`).
2. **fetch** — for a job with a failed or timed-out step, fetch the raw job
   log via the GitHub API (`GITHUB_PAT`) and keep the tail. Only runs for
   jobs that actually failed; a plain success costs nothing here.
3. **categorize** — compare each step's duration against its own D1 history
   and label the job `"success"`, `"failure"`, or `"regression"`.
4. **diagnose** — for anything other than a plain success, one Workers AI
   call produces a 2-3 sentence diagnosis and next steps. Skipped entirely
   for successes, so normal jobs don't spend any inference budget.
5. **store** — write to the Durable Object's state (fast, capped recent-N
   view, broadcasts live to any open chat session) and to D1 (durable,
   unbounded trend history).

A GitHub API hiccup during step 2 retries automatically instead of losing the
log, and if it still fails after retrying, the job still gets recorded — the
log excerpt is just missing, not a reason to drop the whole delivery.

**Chat (interactive path).** The chat UI talks to `ChatAgent`, an Agents SDK
Durable Object running Llama 3.3 in an agentic tool-calling loop — the model
decides whether and which tool to call, not a fixed script. Two tools read
back what the Workflow already computed and stored:

- `getGithubWorkflowRuns` — recent run status/conclusion.
- `getGithubJobSteps` — per-step durations, the stored `category`/`diagnosis`,
  trend data, and (for failures) the log excerpt. Optionally filtered to one
  run via `runId`.

The chat header also has a **Trends** panel — a D1-backed sparkline view of
the most recently active steps, independent of asking the model anything
(`agent.stub.getStepTrends()`, a `@callable()` RPC method, no separate HTTP
endpoint).

**Guardrails.** Before the model is ever invoked: message length, profanity,
prompt-injection phrasing, an off-topic deny-list (weather, trivia, other
CI/CD platforms), and a sliding-window rate limit — all rejected at zero
inference cost. Llama 3.3 (fp8-fast) intermittently leaks a tool call as raw
text instead of calling it structurally; `onFinish` detects and salvages that
automatically, and the resulting correction round-trip is hidden from the
rendered chat transcript.

## Required secrets

Nothing GitHub-related works without these two Worker secrets:

```bash
npx wrangler secret put GITHUB_WEBHOOK_SECRET
npx wrangler secret put GITHUB_PAT
```

- **`GITHUB_WEBHOOK_SECRET`** — the signing secret configured on the GitHub
  webhook itself. Every delivery is HMAC-SHA256 verified against this before
  anything else happens; without a match, the request is rejected (`401`)
  before it's even parsed.
- **`GITHUB_PAT`** — a personal access token with `actions:read` (or `repo`
  scope for a classic token). Used only to fetch job logs for a failed step
  via the GitHub REST API — nothing else calls out to GitHub with it.

### Configuring the webhook on GitHub

In the target repo: **Settings → Webhooks → Add webhook**

- **Payload URL**: `https://<your-worker>.<subdomain>.workers.dev/webhook`
- **Content type**: `application/json` (form-encoded is also supported, but
  JSON is simpler)
- **Secret**: the same value as `GITHUB_WEBHOOK_SECRET`
- **Events**: select "Workflow runs" and "Workflow jobs" (or "Send me
  everything" — other event types are safely ignored and return `200` with
  no processing)

## D1 is a remote resource

The D1 database (`step_runs` table — trend/regression history) is a real,
hosted database in your Cloudflare account, **not a local file**. Provisioning
it and applying the schema are one-time setup steps against your account:

```bash
npx wrangler d1 create github-action-assistant-db
# paste the printed database_id into wrangler.jsonc's d1_databases[0].database_id
npx wrangler d1 migrations apply github-action-assistant-db --remote
```

`npm run dev` (via the Cloudflare Vite plugin) uses a **separate, local** D1
instance for fast iteration — apply the same migration with `--local` to set
that up too:

```bash
npx wrangler d1 migrations apply github-action-assistant-db --local
```

Local and remote D1 do not sync. Trend data you generate while running
`npm run dev` locally never appears in the deployed app, and vice versa —
each environment builds its own history independently.

## Setup

```bash
npm install
npx wrangler login                 # Workers AI has no local simulator
cp .dev.vars.example .dev.vars     # local secrets for npm run dev
npx wrangler secret put GITHUB_WEBHOOK_SECRET
npx wrangler secret put GITHUB_PAT
npx wrangler d1 create github-action-assistant-db   # then update wrangler.jsonc
npx wrangler d1 migrations apply github-action-assistant-db --remote
npx wrangler d1 migrations apply github-action-assistant-db --local
npm run deploy
```

Then add the webhook on GitHub as described above, pointing at your deployed
Worker's `/webhook` URL.

## Local development & testing

```bash
npm run dev   # http://localhost:5173
```

`fixtures/` has real-shaped GitHub webhook payloads — `workflow_run` and
`workflow_job` across their full lifecycle (queued/in_progress/completed) and
conclusions (success/failure/timed_out/cancelled/skipped), plus edge cases
(malformed JSON, an oversized body, form-urlencoded content type, ignored
event types). `scripts/` replays them with a correctly computed HMAC
signature, so they exercise the real verification path instead of bypassing
it:

```bash
# Send one fixture as a specific event
scripts/send-webhook-fixture.sh workflow_job fixtures/workflow_job-completed-failure.json

# Run every transport/security edge case and check the expected HTTP status
scripts/send-webhook-edge-cases.sh

# See what actually landed, bypassing the LLM entirely
curl -s http://localhost:5173/webhook/status | jq
npx wrangler d1 execute github-action-assistant-db --local \
  --command "SELECT * FROM step_runs ORDER BY id DESC LIMIT 20"
```

## Project structure

```
src/
  server.ts           # ChatAgent (Durable Object): chat loop, tools,
                      # guardrails, webhook entry point (verify + hand off
                      # to the Workflow)
  github.ts           # Pure logic: payload parsing, sanitization, D1
                      # queries — shared by server.ts and
                      # github-workflow.ts, no Workers-only imports
  github-workflow.ts  # GithubRunWorkflow: the fetch/parse/categorize/
                      # diagnose/store pipeline
  shared.ts           # The one constant shared with the browser bundle
  app.tsx             # Chat UI (Kumo components): messages, Trends panel,
                      # MCP panel
  client.tsx          # React entry point
migrations/           # D1 schema
fixtures/             # Sample webhook payloads for local testing
scripts/              # Fixture-replay and edge-case test runners
```

## Known limitations

- **No output delivery back to GitHub yet.** `GITHUB_PAT` is read-only in
  this app — nothing posts a PR/commit comment with the diagnosis. The
  diagnosis currently only surfaces in the chat UI.
- **No proactive notification.** A completed job broadcasts to any open chat
  session, but the client doesn't yet turn that into a toast/alert — you have
  to ask, or open the Trends panel.
- `pre-log.json` / `post-log.json` at the repo root are empty legacy
  placeholders, superseded by `fixtures/`.

## License

MIT
