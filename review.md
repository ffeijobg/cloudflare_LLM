# Review

## Source
- Project brief: https://developers.cloudflare.com/agents/
- Task description: build an AI-powered application with an LLM, workflow/coordination, user input, and memory/state.

## Requirements summary

The brief lists four required components. Cloudflare's own docs frame these as the three functional pieces every agent needs (decision engine, tools, memory) plus a delivery channel to reach a user — the four line up like this:

| # | Requirement | What it actually means | Recommended CF product | Alternative |
|---|---|---|---|---|
| 1 | **LLM** | The decision-making engine that reasons about input and picks actions/tool calls | Workers AI running **Llama 3.3** (`@cf/meta/llama-3.3-70b-instruct-fp8-fast` or similar) | Any external LLM API (OpenAI, Anthropic, Gemini) called from a Worker |
| 2 | **Workflow / coordination** | Something that sequences multi-step logic, retries on failure, and survives longer than one request | **Workflows** (durable, retryable, can pause for webhooks/approvals) | Plain **Workers** for simple single-request logic, or **Durable Objects** if you need a stateful coordinator with its own identity |
| 3 | **User input (chat or voice)** | A channel for a human to send messages/audio in and get responses out | **Pages** (chat UI) | **Realtime** (RealtimeKit/SFU/TURN) for voice/video input |
| 4 | **Memory or state** | Persistence of conversation history / agent context across turns and restarts | Agents SDK's built-in **SQLite-per-instance** state (via a Durable Object), accessed with `this.setState()` | KV, D1, R2, or Vectorize if you need shared/queryable/large-scale storage instead of per-agent state |

### Non-obvious/comprehensive requirements the one-line brief doesn't spell out

- **The LLM alone is not an agent.** Cloudflare's docs define an agent as needing all three of: a decision engine (LLM), tool integration (APIs/functions it can call), and a memory system. A component that only calls an LLM and returns the answer (no tools, no state) does not satisfy "agent" — it's a chatbot.
- **Agents vs. Workflows is a real architectural choice, not just two options for the same box.** Workflows are *linear and deterministic* (fixed step sequence with retries); Agents are *non-linear and non-deterministic* (the LLM decides what happens next). If the "coordination" layer must let the LLM's output influence what step runs next, a plain Workflow isn't enough — you need the Agents SDK (built on Durable Objects) driving the loop, with Workflows invoked as a durable sub-step when you need guaranteed multi-step execution (e.g., a long-running task or a step needing automatic retries).
- **State must be JSON-serializable** and is broadcast to every connected client on each change — large or historical data belongs in SQL tables (queried via the Agent's embedded SQLite), not in `state` itself.
- **Real-time sync is implied, not just storage.** "Memory or state" in the Agents SDK isn't just a database write — `setState()` auto-syncs to all connected WebSocket clients and triggers `onStateChanged()`, which is what makes a chat UI feel live. A pure database (D1/KV) with no sync mechanism technically stores state but won't give you that behavior for free.
- **"User input via chat or voice" implies a live connection, not just an HTTP form.** The recommended tools (Pages + Realtime) both assume WebSocket or WebRTC-based interaction so responses can stream back, not just request/response.
- **No API keys are required to get started** if you stay on Workers AI — this is worth knowing before assuming you need to provision external LLM credentials.

## Short guide: covering every requirement comprehensively

1. **Scaffold the project**
   ```sh
   npx create-cloudflare@latest --template cloudflare/agents-starter
   ```
   This starter already wires an Agent (Durable Object), a Worker entrypoint, and a chat UI — i.e., requirements 2–4 out of the box.

2. **LLM (requirement 1)**
   - Default: call Workers AI's Llama 3.3 model from inside your Agent's request handler — no separate account/API key needed.
   - If using an external LLM instead, store its API key as a Worker secret (`wrangler secret put`) and call it from the same handler; keep the call behind an abstraction so you can swap providers later (the Agents SDK supports Workers AI, OpenAI, Anthropic, and Gemini interchangeably).

3. **Workflow / coordination (requirement 2)**
   - For the core "loop" (read input → call LLM → decide tool calls → respond), let the **Agents SDK** class (a Durable Object) own that loop — this is what makes it non-deterministic/agentic rather than a fixed pipeline.
   - For any sub-task that must survive minutes-to-weeks, retry automatically, or wait on an external event/approval (e.g., a long-running data fetch, a human-in-the-loop approval step), delegate that specific step to a **Workflow** invoked from the Agent, rather than modeling the whole agent as one Workflow.

4. **User input — chat or voice (requirement 3)**
   - Chat: deploy a **Pages** frontend that opens a WebSocket to the Agent (via `useAgent` React hook or `AgentClient` in vanilla JS) so messages and state updates stream in both directions.
   - Voice: use **Realtime** (RealtimeKit for a quick pre-built UI, or the SFU/TURN primitives for custom WebRTC) to capture/stream audio into the same Agent backend.

5. **Memory / state (requirement 4)**
   - Use `this.setState()` inside the Agent for conversation/session state — it persists to the Agent's embedded SQLite and auto-broadcasts to connected clients.
   - Override `validateStateChange()` if you need to reject bad updates, and `onStateChanged()` for side effects (e.g., logging, triggering a Workflow).
   - For anything large, historical, or queryable (past conversations, analytics, documents), use the Agent's SQL storage or an external store (D1/KV/R2/Vectorize) rather than stuffing it into `state`.

6. **Verify comprehensiveness before calling it done**
   - Decision engine: does the LLM call have access to tool definitions, not just a single prompt/response? (Otherwise it's a chatbot, not an agent.)
   - Coordination: can the next step change based on the LLM's output (agentic), and does any long-running step survive a restart (Workflow-backed)?
   - Input channel: does it stream (WebSocket/WebRTC), not just poll/HTTP-post?
   - Memory: does state survive a reload/reconnect, and does it sync live to the client?
