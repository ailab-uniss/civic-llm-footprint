#!/bin/bash
# Audit stage: generate S1-S3 outputs for a per-language subsample with each model (H0: no tools),
# build (original, output) pairs and compute L1-L7 metrics.
# usage: ./scripts/run_audit.sh subsample | generate <model> <tag> | metrics
set -o pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/civic/bin/activate"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
N_AUDIT=${N_AUDIT:-300}
stage=${1:-metrics}

case $stage in
subsample)
  python - <<PY
import pandas as pd
S = pd.read_parquet("data/generated/human_sample.parquet")
sub = pd.concat([g.sample(min(len(g), $N_AUDIT), random_state=1) for _, g in S.groupby("lang")])
sub.to_parquet("data/audit/audit_sample.parquet", index=False)
print(sub.groupby("lang").size().to_string())
PY
  ;;
generate)
  model=$2; tag=$3; shift 3   # remaining args: model-specific vLLM options (quantization, gpu-util, eager, seqs, limit)
  [ -f "data/audit/audit_$tag.done" ] && { echo "skip $tag"; exit 0; }
  python src/civic/generate.py run --model "$model" --tag "$tag" --sample data/audit/audit_sample.parquet \
      --prompts configs/audit_prompts.yaml --modes s1_improve s2_notes s3_persuasive \
      --out "data/audit/audit_$tag.parquet" --max-tokens 1200 "$@" 2>&1 | grep -vE "^INFO|^WARNING|^\(EngineCore" | tail -20 \
      && touch "data/audit/audit_$tag.done"
  ;;
metrics)
  python - <<'PY'
import pandas as pd, glob, yaml
S = pd.read_parquet("data/audit/audit_sample.parquet")[["doc_id","text","instance","lang"]].rename(columns={"text":"orig"})
cfg = yaml.safe_load(open("configs/instances.yaml"))["instances"]
S["country"] = S.instance.map(lambda i: cfg.get(i, {}).get("country", "INT"))
G = pd.concat([pd.read_parquet(p) for p in glob.glob("data/audit/audit_*.parquet") if "sample" not in p], ignore_index=True)
P = G.merge(S[["doc_id","orig","country"]], on="doc_id")
P = P[P.output.fillna("").str.split().str.len() >= 10]
P.to_parquet("data/audit/pairs.parquet", index=False)
print(P.groupby(["lang","model","mode"]).size().unstack().to_string())
PY
  python src/civic/audit.py --pairs data/audit/pairs.parquet --out results/audit_metrics.parquet 2>&1 | tail -60
  ;;
esac
echo "AUDIT $stage DONE $(date +%H:%M)"
