#!/bin/bash
# Install the spaCy pipelines used by preprocess.py into the civic venv (no pip in the venv: uv is used with the
# direct wheel URLs of the spacy-models releases matching the installed spaCy minor version).
# usage: ./scripts/install_spacy_models.sh   (then run scripts/relemmatise.py)
cd "$(dirname "$0")/.."
PY=$HOME/venvs/civic/bin/python
UV=$HOME/paper_federated_fs/.venv/bin/uv
VER=$($PY -c "import spacy; v=spacy.__version__.split('.'); print(v[0]+'.'+v[1]+'.0')")
echo "spaCy model version $VER"
export HF_HUB_DISABLE_XET=1
for m in ca_core_news_md es_core_news_md fr_core_news_md it_core_news_md fi_core_news_md en_core_web_md \
         sv_core_news_md de_core_news_md pt_core_news_md nl_core_news_md pl_core_news_md da_core_news_md \
         el_core_news_md ro_core_news_md; do
  $PY -c "import spacy; spacy.load('$m')" 2>/dev/null && { echo "$m present"; continue; }
  for attempt in 1 2 3 4; do
    echo "=== $m attempt $attempt $(date +%H:%M)"
    $UV pip install --python $PY "https://github.com/explosion/spacy-models/releases/download/$m-$VER/$m-$VER-py3-none-any.whl" > /tmp/spacy_dl.log 2>&1 && break
    tail -n 2 /tmp/spacy_dl.log; sleep 60
  done
done
$PY - <<'EOF'
import spacy
for m in ["ca_core_news_md","es_core_news_md","fr_core_news_md","it_core_news_md","fi_core_news_md","en_core_web_md","sv_core_news_md","de_core_news_md","pt_core_news_md"]:
    try:
        nlp = spacy.load(m, disable=["ner","parser"]); print(m, "OK", [ (t.lemma_, t.pos_) for t in nlp("Les propostes dels veïns són importants.")][:3])
    except Exception as e:
        print(m, "MISSING", e)
EOF
echo SPACY_DONE
