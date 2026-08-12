# Prompts

## 2026-08-12
1. for @cloudflare_LLM create two MD files, one called prompts and one called review
2. save every prompt query at @cloudflare_LLM/prompts.md
3. For the project from https://developers.cloudflare.com/agents/ and below description create a summary if the requirements and a short guide to be more comprehensive on what are the actual requirements: An AI-powered application should include the following components: LLM (recommend using Llama 3.3 on Workers AI), or an external LLM of your choice; Workflow / coordination (recommend using Workflows, Workers or Durable Objects); User input via chat or voice (recommend using Pages or Realtime); Memory or state
4. read @cloudflare_LLM/scope.md and suggest stepped changes to connect to my https://github.com/ffeijobg to github actions review use the default Llama 3.3 with wrangler to allow github actions status/log upload
5. connection should be with PAT secret
6. secrets were uploaded to wrangler and project was deployed, help me check why logs are not streaming, i see no errors with PAT or webhook (pasted a GitHub `ping` webhook delivery to https://github-action-assistant.ffeijobg.workers.dev/webhook, hook events: ["push"], response headers only, Content-Length: 0)
7. fix 1, 2 and 4 in the code. 3 is already fixed in github i will run 5 manually
8. I can see webhook is working, add a memory to allow logs to be used in a conversation or analysis
   (implemented: getWorkflowRuns() on ChatAgent + getGithubWorkflowRuns chat tool reading from the shared "github-monitor" agent instance)
9. Agent is returning "Your input is lacking necessary details. Please provide more information or specify the task you need help with." but i can see on webhooks logs aer being streamed, help me check the memory and model data ingestion status
10. add guard rails for profanity, and attacks like Prompt Injection, Jailbreaking, Data Poisoning, Data Exfiltration and Memory Attacks, Denial of Service (DoS / Denial of Wallet), make the user always clear the context using the gui if ignore and start over is needed.
11. prompts are still not returning results as expected, reevaluate memory function and make it persistent, as well as valid for multiple chats. meaning webhook events should be maintained and available even if agent is down, crashed or temporarily inaccessible (pasted 3 more chat log lines, including an "ignore previous prompts" injection attempt that leaked a raw getWeather tool-call JSON as text)
    - findings: memory already durable (proved via live /webhook/status call, 11 real runs) and already shared across chats; real bug is Llama 3.3 fp8-fast leaking tool calls as text under toolChoice "auto"
    - user chose "Auto-repair leaked calls" option when asked how to proceed
    - implemented: parseLeakedToolCall + buildSalvageExecutors + onFinish hook that executes the leaked call server-side and triggers one automatic corrective turn via saveMessages(); also fixed the injection-pattern gap that let "ignore previous prompts" through
12. memory data still does not stream clearly, the last prompt injection was to break the function and ultimately it returned the "greeting" and "success" but on the original prompts it had failed, check logs and help me correct this (pasted 4-message log: 3 direct CI-status questions all got vague refusals with zero tool attempt; the 4th, phrased as "ignore last prompts...", leaked a tool call as text and the auto-repair from the previous turn correctly salvaged it)
    - findings: auto-repair worked correctly (proves that fix works); real remaining bug is a second, separate failure mode — direct/legitimate questions get zero tool-call attempt at all (not even a leak), so the salvage mechanism never triggers for them; also "ignore last prompts" bypassed the injection guardrail (only matched previous/prior/above/all/any/the, not "last")
    - implemented: prepareStep forces toolChoice to getGithubWorkflowRuns on step 0 when the message matches CI-shaped intent (workflow/pipeline/build status/deploy/latest run etc.), falling back to auto after; widened INJECTION_PATTERNS to also match last/earlier/recent
