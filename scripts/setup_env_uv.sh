#!/bin/bash
# Create the project venv with uv (fast resolver), Python 3.11, on the ext4 root disk.
set -o pipefail
cd "$(dirname "$0")/.."
rm -rf "$HOME/venvs/civic"
"$HOME/miniconda3/bin/python" -m pip install -q uv || exit 1
export PATH="$HOME/miniconda3/bin:$PATH"
uv venv "$HOME/venvs/civic" --python 3.11 || exit 1
source "$HOME/venvs/civic/bin/activate"
uv pip install vllm pandas pyarrow polars duckdb datasketch sentence-transformers faiss-cpu hdbscan statsmodels scipy matplotlib seaborn spacy lingua-language-detector stanza tqdm scikit-learn pyyaml requests || exit 1
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
for m in ca_core_news_md es_core_news_md fr_core_news_md it_core_news_md fi_core_news_md en_core_web_md sv_core_news_md de_core_news_md nl_core_news_md pt_core_news_md; do
  python -m spacy download -q $m || echo "spacy model $m failed"
done
echo ENV_DONE
