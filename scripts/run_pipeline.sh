#!/bin/bash
# Master pipeline on the GPU server. Resumable: each stage skips when its output exists.
# usage: ./scripts/run_pipeline.sh <stage> [args]   stages: preprocess sample generate detectors benchmark campaigns analysis all
set -o pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/civic/bin/activate"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
LANGS="ca es fr it fi en"
N_SAMPLE=${N_SAMPLE:-1500}
stage=${1:-all}

stage_preprocess() {
  mkdir -p data/processed
  for f in data/interim/*.parquet; do
    b=$(basename "$f" .parquet)
    [ -s "data/processed/$b.parquet" ] && continue
    echo "=== preprocess $b $(date +%H:%M)"
    python src/civic/preprocess.py --inp "$f" --out "data/processed/$b.parquet" --n-process 6 2>&1 | tail -30
  done
}

stage_sample() {
  [ -s data/generated/human_sample.parquet ] && return
  python src/civic/generate.py sample --processed data/processed --out data/generated --n $N_SAMPLE --langs $LANGS
}

# generate <model_path_or_id> <tag> [extra args]
stage_generate() {
  local model=$1 tag=$2; shift 2
  [ -f "data/generated/gen_$tag.done" ] && { echo "skip $tag"; return; }
  echo "=== generate $tag $(date +%H:%M)"
  python src/civic/generate.py run --model "$model" --tag "$tag" --sample data/generated/human_sample.parquet \
      --prompts configs/prompts.yaml --out "data/generated/gen_$tag.parquet" "$@" 2>&1 | grep -vE "^INFO|^WARNING|^\(EngineCore" | tail -40 \
      && touch "data/generated/gen_$tag.done"
}

stage_postgen() {   # language-check + lemmatise generated texts into one parquet
  while pgrep -f "relemmatise.py|install_spacy_models.sh" >/dev/null; do sleep 60; done   # spaCy models / re-lemmatisation in progress
  python - <<'PY'
import pandas as pd, glob, sys
sys.path.insert(0, "src/civic")
from preprocess import normalise, n_words, detect_languages, Lemmatiser
frames = [pd.read_parquet(p) for p in sorted(glob.glob("data/generated/gen_*.parquet"))]
G = pd.concat(frames, ignore_index=True)
G["text"] = G.output.fillna("").map(lambda s: normalise(s.strip().strip('"')))
G["n_words"] = G.text.map(n_words)
G = G[G.n_words >= 30].copy()
langs, confs = detect_languages(G.text.tolist())
G["lang_det"], G["lang_conf"] = langs, confs
ok = (G.lang_det == G.lang)
print("language mismatch rate by model:\n", (~ok).groupby(G.model).mean().round(3).to_string())
G = G[ok].copy()
lem = Lemmatiser()
toks = pd.Series([None]*len(G), index=G.index, dtype=object)
for lang, g in G.groupby("lang"):
    toks.loc[g.index] = pd.Series(lem.tokens(g.text.tolist(), lang, n_process=6), index=g.index, dtype=object)
G["tokens"] = toks
G.to_parquet("data/generated/generated_processed.parquet", index=False)
print(G.groupby(["lang","model","mode"]).size().unstack().to_string())
S = pd.read_parquet("data/generated/human_sample.parquet")
st = G.groupby(["model","lang"]).size().rename("n").reset_index()
st["n_seeds"] = st.lang.map(S.groupby("lang").size())
st.to_csv("results/generation_stats.csv", index=False)
PY
}

stage_detectors() {
  mkdir -p data/scores
  python - <<'PY'
import pandas as pd, glob
H = pd.concat([pd.read_parquet(p, columns=["doc_id","lang","text","period","is_dup_exact","n_words"]) for p in glob.glob("data/processed/*.parquet")])
H = H[(H.period=="pre") & (~H.is_dup_exact)]
S = pd.read_parquet("data/generated/human_sample.parquet", columns=["doc_id"])
# score: all human sample docs + a further random 3000 human pre docs per language + all generated docs
extra = pd.concat([g.sample(min(len(g), 3000), random_state=0) for _, g in H[~H.doc_id.isin(S.doc_id)].groupby("lang")])
G = pd.read_parquet("data/generated/generated_processed.parquet", columns=["doc_id","lang","text","model","mode","variant"])
G["doc_id"] = G.doc_id + "|" + G.model + "|" + G["mode"] + "|" + G.variant.astype(str)
T = pd.concat([H[H.doc_id.isin(S.doc_id)][["doc_id","lang","text"]], extra[["doc_id","lang","text"]], G[["doc_id","lang","text"]]], ignore_index=True)
T.to_parquet("data/scores/to_score.parquet", index=False)
print(len(T), "texts to score")
PY
  for d in binoculars fastdetect pangram; do   # resumable: detectors.py scores only unseen doc_ids
    if [ $d = pangram ] && ! { ls "$HOME"/.cache/huggingface/hub/models--pangram--editlens_Llama-3.2-3B/snapshots/*/adapter_model.safetensors >/dev/null 2>&1                                && ls "$HOME"/.cache/huggingface/hub/models--unsloth--Llama-3.2-3B/snapshots/*/*.safetensors >/dev/null 2>&1; }; then
      echo "pangram adapter or Llama-3.2-3B base not downloaded yet; skipped"; continue
    fi
    python src/civic/detectors.py --detector $d --inp data/scores/to_score.parquet --out data/scores/scores_$d.parquet --batch $([ $d = pangram ] && echo 8 || echo 4) 2>&1 | tail -3
  done
}

stage_benchmark() {
  mkdir -p results
  python - <<'PY'
# give generated docs the same composite doc_id used for scoring
import pandas as pd
G = pd.read_parquet("data/generated/generated_processed.parquet")
G["src_doc_id"] = G.doc_id
G["doc_id"] = G.doc_id + "|" + G.model + "|" + G["mode"] + "|" + G.variant.astype(str)
G.to_parquet("data/generated/generated_bench.parquet", index=False)
PY
  python src/civic/benchmark.py --human data/processed/*.parquet --gen data/generated/generated_bench.parquet \
      --scores data/scores/scores_*.parquet --out results/benchmark.parquet --reps 10 --n-boot 200 --jobs 9 \
      --heldout-only gpt-4o-mini 2>&1 | tail -60
}

stage_campaigns() {
  [ -s results/campaigns.parquet ] && return
  python - <<'PY'
import pandas as pd, glob
H = pd.concat([pd.read_parquet(p) for p in glob.glob("data/processed/*.parquet")], ignore_index=True)
H[["doc_id","instance","lang","type","author_hash","created_at","text"]].to_parquet("data/processed_all_text.parquet", index=False)
print(len(H))
PY
  python src/civic/campaigns.py --inp data/processed_all_text.parquet --out results/campaigns.parquet --emb-out results/embeddings.parquet 2>&1 | tail -40
}

stage_analysis() {
  local extra=""
  [ -s results/campaigns.parquet ] && extra="--campaigns results/campaigns.parquet"
  [ -s results/embeddings.parquet ] && extra="$extra --emb results/embeddings.parquet"
  # main results: length-calibrated likelihood, market-share-weighted commercial reference (configs/reference_weights.yaml)
  python src/civic/analysis.py --processed data/processed/*.parquet --gen data/generated/generated_processed.parquet \
      $extra --likelihood length --weights configs/reference_weights.yaml --out results 2>&1 | tail -80
  # consultation portals: estimates per consultation (publication / docket) and medians across consultations
  python src/civic/consultations.py --processed data/processed/*.parquet --gen data/generated/generated_processed.parquet \
      --likelihood length --weights configs/reference_weights.yaml --out results 2>&1 | tail -5
  # sensitivity: equal-weight mixture of all generators (legacy held-out-only generators excluded)
  python src/civic/analysis.py --processed data/processed/*.parquet --gen data/generated/generated_processed.parquet \
      $extra --likelihood length --exclude-models gpt-4o-mini --out results/analysis_equal 2>&1 | tail -20
}
stage_excess() {   # model-free lower bound by excess vocabulary (Kobak et al.), per instance-language-year
  python src/civic/excess.py --processed data/processed/*.parquet --out results/excess 2>&1 | tail -40
}

case $stage in
  preprocess) stage_preprocess ;;
  sample) stage_sample ;;
  generate) shift; stage_generate "$@" ;;
  postgen) stage_postgen ;;
  detectors) stage_detectors ;;
  benchmark) stage_benchmark ;;
  campaigns) stage_campaigns ;;
  analysis) stage_analysis ;;
  excess) stage_excess ;;
  all) stage_preprocess; stage_sample ;;
esac
echo "STAGE $stage DONE $(date +%H:%M)"
