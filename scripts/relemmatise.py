"""Recompute the lemma|POS tokens of every processed parquet (and of the generated texts, if present) with the
installed spaCy pipelines. Needed because the first preprocessing run happened before the spaCy models were installed
and silently fell back to raw words ('word|X'). Files whose tokens already carry POS tags are skipped.

  python scripts/relemmatise.py --n-process 6
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, "src/civic")
from preprocess import Lemmatiser, SPACY_MODELS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-process", type=int, default=6)
ap.add_argument("--files", nargs="*", default=None)
a = ap.parse_args()

files = a.files or sorted(glob.glob("data/processed/*.parquet")) + ["data/generated/generated_processed.parquet"]
lem = Lemmatiser()
for f in files:
    p = Path(f)
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    if len(df) == 0 or "tokens" not in df.columns:
        continue
    sample = [t for t in df.tokens.head(200) if len(t)]
    if sample and all(any(not w.endswith("|X") for w in t) for t in sample):
        print(f"{p.name}: already lemmatised, skipped", flush=True)
        continue
    t0 = time.time()
    toks = pd.Series([None] * len(df), index=df.index, dtype=object)
    for lang, g in df.groupby("lang"):
        if lang not in SPACY_MODELS or lem.get(lang) is None:
            toks.loc[g.index] = g.tokens  # no pipeline for this language: keep the raw-word fallback
            continue
        toks.loc[g.index] = pd.Series(lem.tokens(g.text.tolist(), lang, n_process=a.n_process), index=g.index, dtype=object)
    df["tokens"] = toks
    tmp = p.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)
    print(f"{p.name}: {len(df)} docs re-lemmatised in {time.time() - t0:.0f}s", flush=True)
print("RELEM_DONE")
