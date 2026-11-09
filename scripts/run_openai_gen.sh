#!/bin/bash
# Local (Windows/Git Bash) run of a commercial generator through its OpenAI-compatible chat API.
# usage: [BACKEND=openai|anthropic|gemini] [MODEL=..] [TAG=..] [LIMIT=..] ./scripts/run_openai_gen.sh [gen|audit]
# Keys are read from C:\CODICI\IEEEACCESSPAPER\.env. Resumable (appends to the output parquet).
cd "$(dirname "$0")/.."
for k in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY; do
  export $k=$(grep "^$k=" ../.env | cut -d= -f2- | tr -d '\r')
done
BACKEND=${BACKEND:-openai}
MODEL=${MODEL:-gpt-5.4-mini}
TAG=${TAG:-gpt-5.4-mini}
LIMIT=${LIMIT:-400}
if [ "${1:-gen}" = audit ]; then   # S1-S3 audit outputs on the audit subsample (300 seeds per language)
  python src/civic/generate.py run --backend "$BACKEND" --model "$MODEL" --tag "$TAG" \
    --sample data/audit/audit_sample.parquet --prompts configs/audit_prompts.yaml \
    --modes s1_improve s2_notes s3_persuasive --out "data/audit/audit_$TAG.parquet" --max-tokens 1200
else
  python src/civic/generate.py run --backend "$BACKEND" --model "$MODEL" --tag "$TAG" \
    --sample data/generated/human_sample.parquet --prompts configs/prompts.yaml \
    --out "data/generated/gen_$TAG.parquet" --limit-per-lang "$LIMIT" --max-tokens 900
fi
