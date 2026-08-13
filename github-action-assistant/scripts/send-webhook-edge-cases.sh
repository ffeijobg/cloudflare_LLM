#!/usr/bin/env bash
# Runs every transport/security edge case against the webhook endpoint and
# checks each one got the expected HTTP status — a repeatable regression
# check instead of five one-off curl commands to copy/paste and eyeball.
#
# Usage:
#   scripts/send-webhook-edge-cases.sh [url] [secret]
#
# [url]    defaults to http://localhost:5173/webhook (npm run dev)
# [secret] defaults to $GITHUB_WEBHOOK_SECRET, which .dev.vars supplies
#          automatically to `npm run dev`
#
# Exits non-zero if any check fails, so this can gate a deploy later.

set -uo pipefail

url="${1:-http://localhost:5173/webhook}"
secret="${2:-${GITHUB_WEBHOOK_SECRET:-}}"
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixtures="$dir/fixtures"

if [ -z "$secret" ]; then
  echo "No webhook secret available. Set GITHUB_WEBHOOK_SECRET, put it in .dev.vars, or pass it as the 2nd argument." >&2
  exit 1
fi

pass=0
fail=0

check() {
  local name="$1" expected="$2" actual="$3"
  if [ "$actual" = "$expected" ]; then
    echo "PASS  $name (got $actual)"
    pass=$((pass + 1))
  else
    echo "FAIL  $name (expected $expected, got $actual)"
    fail=$((fail + 1))
  fi
}

# Signs $2 (a file) with secret $1 — the exact bytes curl sends below, via
# --data-binary @file, so signature and body always match.
sign_with() {
  openssl dgst -sha256 -hmac "$1" "$2" | awk '{print $NF}'
}

# 1. Invalid signature — correctly-shaped header, wrong secret.
body="$fixtures/workflow_job-completed-success.json"
bad_sig="sha256=$(sign_with "${secret}-wrong" "$body")"
status=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$url" \
  -H "Content-Type: application/json" -H "X-GitHub-Event: workflow_job" \
  -H "X-Hub-Signature-256: $bad_sig" --data-binary @"$body")
check "invalid signature" 401 "$status"

# 2. Missing signature header entirely.
status=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$url" \
  -H "Content-Type: application/json" -H "X-GitHub-Event: workflow_job" \
  --data-binary @"$body")
check "missing signature header" 401 "$status"

# 3. Oversized body (>256KB) — real Content-Length, not a spoofed header.
body="$fixtures/workflow_job-oversized.json"
sig="sha256=$(sign_with "$secret" "$body")"
status=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$url" \
  -H "Content-Type: application/json" -H "X-GitHub-Event: workflow_job" \
  -H "X-Hub-Signature-256: $sig" --data-binary @"$body")
check "oversized body (>256KB)" 413 "$status"

# 4. Malformed JSON body — valid signature, broken content, so it actually
# reaches parseGitHubPayload instead of failing at signature verification.
body="$fixtures/malformed-payload.json"
sig="sha256=$(sign_with "$secret" "$body")"
status=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$url" \
  -H "Content-Type: application/json" -H "X-GitHub-Event: workflow_job" \
  -H "X-Hub-Signature-256: $sig" --data-binary @"$body")
check "malformed JSON body" 400 "$status"

# 5. Form-urlencoded content type — GitHub's alternate delivery format
# (payload=<url-encoded-json>), exercises the other branch of
# parseGitHubPayload.
body="$fixtures/workflow_job-completed-success-form-encoded.txt"
sig="sha256=$(sign_with "$secret" "$body")"
status=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$url" \
  -H "Content-Type: application/x-www-form-urlencoded" -H "X-GitHub-Event: workflow_job" \
  -H "X-Hub-Signature-256: $sig" --data-binary @"$body")
check "form-urlencoded content type" 200 "$status"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
