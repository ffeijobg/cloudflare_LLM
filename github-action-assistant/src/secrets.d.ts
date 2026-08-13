// Worker secrets (set via `wrangler secret put`) aren't declared in
// wrangler.jsonc, so `wrangler types` can't see them — everything else on
// Env (DB, AI, ChatAgent, GITHUB_RUN_WORKFLOW, ...) comes from the
// generated env.d.ts and shouldn't be duplicated here.
interface Env {
  GITHUB_WEBHOOK_SECRET: string;
  GITHUB_PAT: string;
}
