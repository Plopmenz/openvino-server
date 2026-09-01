#!/usr/bin/env bash
# Smoke test for the OpenAI-compatible /v1/images/generations endpoint.
# Assumes the server is already running on http://localhost:8080 with
# a loaded Qwen-Image model (the only model loads automatically).
#
#   tools/smoke_test.sh [prompt]

set -uo pipefail

BASE="${BASE:-http://localhost:8080}"
PROMPT="${1:-a red fox in a snowy forest, photorealistic}"

echo "== GET $BASE/v1/models"
curl -sf "$BASE/v1/models" | python3 -m json.tool
echo

echo "== POST $BASE/v1/images/generations"
RESP="$(curl -sf -X POST "$BASE/v1/images/generations" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen-image\",\"prompt\":\"$PROMPT\",\"size\":\"512x512\",\"negative_prompt\":\"blurry, bad quality\"}")" \
    || { echo "curl failed"; exit 1; }

echo "$RESP" | python3 - <<'EOF'
import json, sys, base64
data = json.load(sys.stdin)
imgs = data.get("data", [])
print(f"created={data.get('created')} images={len(imgs)}")
for i, img in enumerate(imgs):
    b64 = img.get("b64_json") or ""
    out = f"/tmp/ovserver_smoke_{i}.png"
    with open(out, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"  saved {out} ({len(b64)} b64 chars)")
EOF