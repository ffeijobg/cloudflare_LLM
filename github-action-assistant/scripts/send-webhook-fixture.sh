#!/usr/bin/env bash
# Sends a fixture from fixtures/ to the webhook endpoint with a correctly
# computed HMAC-SHA256 signature, matching how GitHub actually signs
# deliveries — so it exercises verifyGitHubSignature() for real instead of
# bypassing it.
#
# Usage:
#   scripts/send-webhook-fixture.sh <event> <fixture-file> [url] [secret]
#
# <event>         value for the X-GitHub-Event header (e.g. workflow_job)
# <fixture-file>  path to a JSON body, e.g. fixtures/workflow_job-completed-success.json
# [url]           defaults to http://localhost:5173/webhook (npm run dev)
# [secret]        defaults to $GITHUB_WEBHOOK_SECRET, which .dev.vars supplies
#                 automatically to `npm run dev` — only pass this explicitly
#                 when posting straight at a deployed Worker.
#
# Examples:
#   scripts/send-webhook-fixture.sh workflow_job fixtures/workflow_job-completed-failure.json
#   scripts/send-webhook-fixture.sh workflow_run fixtures/workflow_run-completed-success.json \
#     https://github-action-assistant.<subdomain>.workers.dev/webhook "$PROD_WEBHOOK_SECRET"

set -euo pipefail

event="${1:?Usage: send-webhook-fixture.sh <event> <fixture-file> [url] [secret]}"
fixture="${2:?Usage: send-webhook-fixture.sh <event> <fixture-file> [url] [secret]}"
url="${3:-http://localhost:5173/webhook}"
secret="${4:-${GITHUB_WEBHOOK_SECRET:-}}"

if [ -z "$secret" ]; then
  echo "No webhook secret available. Set GITHUB_WEBHOOK_SECRET, put it in .dev.vars, or pass it as the 4th argument." >&2
  exit 1
fi

if [ ! -f "$fixture" ]; then
  echo "Fixture file not found: $fixture" >&2
  exit 1
fi

# Signs the exact bytes curl will send (--data-binary @file below), so this
# always matches regardless of trailing newlines or formatting.
signature="sha256=$(openssl dgst -sha256 -hmac "$secret" "$fixture" | awk '{print $NF}')"

echo "POST $url  (event: $event, fixture: $fixture)"
curl -sS -i -X POST "$url" \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: $event" \
  -H "X-Hub-Signature-256: $signature" \
  --data-binary @"$fixture"
echo
