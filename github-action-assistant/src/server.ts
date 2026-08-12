import { createWorkersAI } from "workers-ai-provider";
import {
  callable,
  getAgentByName,
  routeAgentRequest,
  type Schedule
} from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool
} from "ai";
import { z } from "zod";

interface WorkflowRunSummary {
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

interface ChatAgentState {
  workflowRuns: WorkflowRunSummary[];
}

export class ChatAgent extends AIChatAgent<Env, ChatAgentState> {
  maxPersistedMessages = 100;
  chatRecovery = true;
  // Wait for MCP connections to be re-established after hibernation before
  // processing a message, so MCP tools aren't intermittently missing.
  waitForMcpConnections = true;

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    const result = streamText({
      model: workersai("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {
        sessionAffinity: this.sessionAffinity
      }),
      system: `You are a helpful assistant that can understand images. You can check the weather, get the user's timezone, run calculations, and schedule tasks. When users share images, describe what you see and answer questions about them.

${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the schedule tool to schedule the task.`,
      // Prune old tool calls and reasoning to save tokens on long conversations
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        // Server-side tool: runs automatically on the server
        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            // Replace with a real weather API in production
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius"
            };
          }
        }),

        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(20),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async recordWorkflowRun(run: WorkflowRunSummary) {
    const existing = this.state?.workflowRuns ?? [];
    this.setState({ workflowRuns: [run, ...existing].slice(0, 20) });
    this.broadcast(JSON.stringify({ type: "workflow-run", run }));
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

const GITHUB_WEBHOOK_PATH = "/webhook";

async function verifyGitHubSignature(
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

function parseGitHubPayload(rawBody: string, contentType: string) {
  if (contentType.includes("application/json")) {
    return JSON.parse(rawBody);
  }
  // GitHub webhook configured with content type "form" sends the JSON
  // payload URL-encoded under a single "payload" field.
  const encoded = new URLSearchParams(rawBody).get("payload");
  if (!encoded) throw new Error("Missing payload field in form body");
  return JSON.parse(encoded);
}

async function handleGitHubWebhook(
  request: Request,
  env: Env
): Promise<Response> {
  const rawBody = await request.text();
  const valid = await verifyGitHubSignature(
    env.GITHUB_WEBHOOK_SECRET,
    rawBody,
    request.headers.get("X-Hub-Signature-256")
  );
  if (!valid) {
    console.error("GitHub webhook: signature verification failed");
    return new Response("Invalid signature", { status: 401 });
  }

  const event = request.headers.get("X-Github-Event");
  console.log(`GitHub webhook: received event=${event}`);

  if (event === "ping") {
    return new Response("pong", { status: 200 });
  }

  if (event === "workflow_run") {
    let payload: {
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
    };
    try {
      payload = parseGitHubPayload(
        rawBody,
        request.headers.get("content-type") ?? ""
      );
    } catch (error) {
      console.error("GitHub webhook: failed to parse payload", error);
      return new Response("Bad payload", { status: 400 });
    }

    const run = payload.workflow_run;
    const agent = await getAgentByName(env.ChatAgent, "github-monitor");
    await agent.recordWorkflowRun({
      action: payload.action,
      repo: payload.repository?.full_name,
      workflowName: run?.name,
      runId: run?.id,
      runNumber: run?.run_number,
      status: run?.status,
      conclusion: run?.conclusion,
      htmlUrl: run?.html_url,
      headSha: run?.head_sha,
      updatedAt: run?.updated_at
    });
    console.log(
      `GitHub webhook: recorded workflow_run ${run?.id} (${run?.status}/${run?.conclusion})`
    );
  }

  return new Response("ok", { status: 200 });
}

export default {
  async fetch(request: Request, env: Env) {
    const url = new URL(request.url);
    if (url.pathname === GITHUB_WEBHOOK_PATH && request.method === "POST") {
      return handleGitHubWebhook(request, env);
    }

    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
