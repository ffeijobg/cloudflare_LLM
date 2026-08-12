# Prompts

## 2026-08-12
1. for @cloudflare_LLM create two MD files, one called prompts and one called review
2. save every prompt query at @cloudflare_LLM/prompts.md
3. For the project from https://developers.cloudflare.com/agents/ and below description create a summary if the requirements and a short guide to be more comprehensive on what are the actual requirements: An AI-powered application should include the following components: LLM (recommend using Llama 3.3 on Workers AI), or an external LLM of your choice; Workflow / coordination (recommend using Workflows, Workers or Durable Objects); User input via chat or voice (recommend using Pages or Realtime); Memory or state
4. read @cloudflare_LLM/scope.md and suggest stepped changes to connect to my https://github.com/ffeijobg to github actions review use the default Llama 3.3 with wrangler to allow github actions status/log upload
5. connection should be with PAT secret
6. secrets were uploaded to wrangler and project was deployed, help me check why logs are not streaming, i see no errors with PAT or webhook (pasted a GitHub `ping` webhook delivery to https://github-action-assistant.ffeijobg.workers.dev/webhook, hook events: ["push"], response headers only, Content-Length: 0)
7. fix 1, 2 and 4 in the code. 3 is already fixed in github i will run 5 manually
