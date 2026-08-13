import {
  WorkflowEntrypoint,
  type WorkflowEvent,
  type WorkflowStep
} from "cloudflare:workers";
import { getAgentByName } from "agents";
import { createWorkersAI } from "workers-ai-provider";
import { generateText } from "ai";
import {
  GITHUB_MONITOR_AGENT_NAME,
  categorizeJob,
  fetchJobLogExcerpt,
  getStepTrend,
  insertStepRuns,
  parseWorkflowJobPayload,
  parseWorkflowRunPayload,
  type JobCategory,
  type JobStepSummary
} from "./github";

export interface GithubRunWorkflowParams {
  event: "workflow_run" | "workflow_job";
  payload: unknown;
}

// Runs the fetch (GitHub log API) → parse (payload → structured summary) →
// categorize (failure/regression/success) → diagnose (LLM) → store
// (Durable Object state + D1) pipeline for one webhook delivery. Invoked
// from handleGitHubWebhook via env.GITHUB_RUN_WORKFLOW.create(), which just
// enqueues the instance and returns — keeping the webhook handler itself
// fast regardless of how long the GitHub API or diagnosis LLM call takes.
// Each step.do() call below is checkpointed and independently retryable,
// instead of the webhook handler being an all-or-nothing synchronous chain.
export class GithubRunWorkflow extends WorkflowEntrypoint<
  Env,
  GithubRunWorkflowParams
> {
  async run(
    event: Readonly<WorkflowEvent<GithubRunWorkflowParams>>,
    step: WorkflowStep
  ) {
    const { event: eventType, payload } = event.payload;

    if (eventType === "workflow_run") {
      const summary = await step.do("parse-workflow-run", async () =>
        parseWorkflowRunPayload(payload)
      );
      await step.do("store-workflow-run", async () => {
        const agent = await getAgentByName(
          this.env.ChatAgent,
          GITHUB_MONITOR_AGENT_NAME
        );
        await agent.recordWorkflowRun(summary);
      });
      return;
    }

    const parsed = await step.do("parse-workflow-job", async () =>
      parseWorkflowJobPayload(payload)
    );
    // Not a "completed" delivery (queued/in_progress) — nothing to process
    // yet; a later "completed" delivery for the same job will re-invoke us.
    if (!parsed) return;

    const { hasFailedStep, steps, ...job } = parsed;

    // fetch: only for a job that actually failed/timed out, and only the
    // raw log tail. Retries are declared here instead of hand-rolled
    // try/catch, since that's the whole point of moving this into a
    // Workflow — a transient GitHub API hiccup shouldn't lose the log.
    const failureExcerpt = hasFailedStep
      ? await step.do(
          "fetch-failure-log",
          {
            retries: { limit: 3, delay: "10 seconds", backoff: "exponential" }
          },
          () => fetchJobLogExcerpt(this.env.GITHUB_PAT, job.repo, job.jobId)
        )
      : undefined;

    // categorize: attach each step's historical trend so categorization
    // (and the diagnosis prompt) can tell a regression from a one-off.
    const enrichedSteps = await step.do("categorize", async () => {
      if (!job.repo || !job.workflowName) return steps;
      const repo = job.repo;
      const workflowName = job.workflowName;
      return Promise.all(
        steps.map(async (s) => {
          if (!s.name) return s;
          const trend = await getStepTrend(
            this.env.DB,
            repo,
            workflowName,
            s.name,
            job.runId,
            s.durationSeconds
          );
          return trend ? { ...s, trend } : s;
        })
      );
    });
    const category: JobCategory = categorizeJob(hasFailedStep, enrichedSteps);

    // diagnose: skip the LLM call entirely for a plain success — nothing to
    // explain, and no reason to spend inference budget on it.
    const diagnosis =
      category === "success"
        ? undefined
        : await step.do("diagnose", () =>
            this.diagnose(job, enrichedSteps, category, failureExcerpt)
          );

    await step.do("store-workflow-job", async () => {
      const agent = await getAgentByName(
        this.env.ChatAgent,
        GITHUB_MONITOR_AGENT_NAME
      );
      await agent.recordWorkflowJob({
        ...job,
        steps: enrichedSteps,
        failureExcerpt,
        category,
        diagnosis
      });
    });

    // Durable trend history, independent of the DO's capped in-memory list.
    // Uses the un-enriched steps (no need to persist the trend snapshot
    // itself — it's recomputed fresh from this table on the next run).
    await step.do("store-step-history", async () => {
      await insertStepRuns(
        this.env.DB,
        job.repo,
        job.workflowName,
        job.runId,
        job.jobId,
        steps,
        job.updatedAt
      );
    });
  }

  private async diagnose(
    job: { jobName?: string; workflowName?: string; repo?: string },
    steps: JobStepSummary[],
    category: JobCategory,
    failureExcerpt: string | undefined
  ): Promise<string> {
    const workersai = createWorkersAI({ binding: this.env.AI });
    const stepLines = steps
      .map((s) => {
        const bits = [
          s.name ?? "unknown step",
          s.conclusion ?? s.status ?? "unknown"
        ];
        if (s.durationSeconds !== undefined) bits.push(`${s.durationSeconds}s`);
        if (s.trend) {
          bits.push(
            `avg ${s.trend.averageDurationSeconds}s over last ${s.trend.sampleSize} runs${
              s.trend.isRegression ? ", REGRESSION" : ""
            }`
          );
        }
        return `- ${bits.join(", ")}`;
      })
      .join("\n");

    const { text } = await generateText({
      model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
      system:
        "You are a CI/CD diagnostics assistant. Given step-level results for one GitHub Actions job, write a 2-3 sentence plain-language diagnosis plus 1-2 concrete next steps. Be specific and concise — no filler, no restating the input verbatim, no markdown headers.",
      prompt: `Job "${job.jobName ?? "unknown"}" in workflow "${job.workflowName ?? "unknown"}" (${job.repo ?? "unknown repo"}) is categorized as: ${category}.

Steps:
${stepLines}${failureExcerpt ? `\n\nFailure log excerpt (tail):\n${failureExcerpt}` : ""}`
    });
    return text;
  }
}
