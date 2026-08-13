// Shared between server.ts (Worker/Durable Object) and app.tsx (browser
// bundle) — keep this free of any Cloudflare-Workers-only imports so it
// bundles cleanly into the client build too.

// Marks a synthetic, server-injected correction message (see
// buildCorrectionMessage in server.ts) so the client can hide it from the
// rendered transcript — it's internal plumbing (a fake "user" turn carrying
// a raw tool-result dump), not something the real user said.
export const CORRECTION_MARKER = "[Automated correction:";
