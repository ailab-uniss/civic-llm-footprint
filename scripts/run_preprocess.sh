#!/bin/bash
# Pre-process every ingested instance (language ID, lemmatisation) with the project venv.
cd "$(dirname "$0")/.."
source "$HOME/venvs/civic/bin/activate"
mkdir -p data/processed
for f in data/interim/*.parquet; do
  b=$(basename "$f" .parquet)
  [ -s "data/processed/$b.parquet" ] && { echo "skip $b"; continue; }
  echo "=== $b"
  python src/civic/preprocess.py --inp "$f" --out "data/processed/$b.parquet" --n-process 6 2>&1 | tail -25
done
echo PREPROCESS_DONE
