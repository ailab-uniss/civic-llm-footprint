"""Add MULTITuDE v1 (Macko et al., EMNLP 2023: GPT-4, GPT-3.5-turbo, text-davinci-003 and 2023 open models, 11 languages)
to the texts prepared from v3 by multitude_prepare.py.

Input : data/external/multitude_v1/multitude.csv (text, label, multi_label, split, language, length, source)
        data/external/multitude/mt_texts.parquet (v3: human + 2023-24 generators)
Output: data/external/multitude/mt_texts_all.parquet  v3 rows + v1 rows (model 'mt1_<generator>' for machine texts,
        'human' for human news not already present in v3, matched on normalised text)
The v1 generators are only ever held out in the benchmark; the references stay those generated from the v3 seeds.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def key(t: str) -> str:
    return re.sub(r"\W+", " ", str(t).lower()).strip()[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="data/external/multitude_v1/multitude.csv")
    ap.add_argument("--v3", default="data/external/multitude/mt_texts.parquet")
    ap.add_argument("--out", default="data/external/multitude/mt_texts_all.parquet")
    ap.add_argument("--langs", nargs="+", default=["en", "es", "de", "ca", "pt"])
    ap.add_argument("--min-words", type=int, default=30)
    a = ap.parse_args()
    T3 = pd.read_parquet(a.v3)
    d = pd.read_csv(a.v1)
    d = d[d.language.isin(a.langs)].copy()
    d["text"] = d.text.astype(str).str.strip()
    d["n_words"] = d.text.str.split().str.len()
    d = d[d.n_words >= a.min_words].copy()
    d["lang"] = d.language
    d["row"] = d.index
    human = d.multi_label == "human"
    k3 = set(T3[T3.model == "human"].text.map(key))
    dup = human & d.text.map(key).isin(k3)
    print("v1 human texts already in v3:", int(dup.sum()), "of", int(human.sum()))
    d = d[~dup].copy()
    d["model"] = d.multi_label.where(human, "mt1_" + d.multi_label)
    d["doc_id"] = "multitude1:" + d.multi_label + ":" + d.row.astype(str) + ":" + d.lang
    T1 = d[["doc_id", "lang", "model", "split", "source", "text", "n_words"]]
    T = pd.concat([T3, T1], ignore_index=True).drop_duplicates("doc_id")
    T.to_parquet(a.out, index=False)
    print(pd.crosstab(T.lang, T.model).T.to_string())


if __name__ == "__main__":
    main()
