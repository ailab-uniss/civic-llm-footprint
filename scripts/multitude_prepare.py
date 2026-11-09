"""Prepare the MULTITuDE v3 texts (Macko et al., 2023/2024) for the temporal-transfer test.

Input : data/external/multitude/multitude_v3_clean.csv.gz  (text, label, multi_label, split, language, length, source)
Output: data/external/multitude/mt_texts.parquet        all human and machine texts of the chosen languages (doc_id, lang, model, text)
        data/external/multitude/mt_seed_sample.parquet  human seeds for generating reference texts with today's models
                                                        (same columns as data/generated/human_sample.parquet)
The 2023-era generators of MULTITuDE (gpt-3.5-turbo, Llama-2-70b-chat, Mistral-7B-v0.2, vicuna-13b, aya-101, ...) are
only ever held out in the benchmark; references are built from texts generated now, from the same human seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LANGS = ["en", "es", "de", "ca", "pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/external/multitude/multitude_v3_clean.csv.gz")
    ap.add_argument("--out", default="data/external/multitude")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--n-seeds", type=int, default=700, help="human seeds per language for reference generation")
    ap.add_argument("--min-words", type=int, default=30)
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    d = pd.read_csv(a.inp)
    d = d[d.language.isin(a.langs)].copy()
    d["text"] = d.text.astype(str).str.strip()
    d["n_words"] = d.text.str.split().str.len()
    d = d[d.n_words >= a.min_words].copy()
    d["lang"] = d.language
    d["model"] = d.multi_label
    d["row"] = d["Unnamed: 0"] if "Unnamed: 0" in d.columns else d.index
    d["doc_id"] = "multitude:" + d.model + ":" + d.row.astype(str) + ":" + d.lang
    T = d[["doc_id", "lang", "model", "split", "source", "text", "n_words"]].reset_index(drop=True)
    T.to_parquet(out / "mt_texts.parquet", index=False)
    print("texts per language x model:\n", pd.crosstab(T.lang, T.model).to_string())
    H = T[(T.model == "human") & (T.n_words <= a.max_words)]
    S = pd.concat([g.sample(min(len(g), a.n_seeds), random_state=a.seed) for _, g in H.groupby("lang")], ignore_index=True)
    S["instance"] = "multitude_news"
    S["type"] = "article"
    S["id"] = S.doc_id.str.split(":").str[2]
    S["title"] = None
    S["created_at"] = pd.Timestamp("2020-01-01", tz="UTC")
    S["is_dup_exact"] = False
    S["period"] = "pre"
    S = S[["instance", "lang", "type", "id", "created_at", "text", "title", "n_words", "is_dup_exact", "period", "doc_id"]]
    S.to_parquet(out / "mt_seed_sample.parquet", index=False)
    print("seeds per language:", S.groupby("lang").size().to_dict())


if __name__ == "__main__":
    main()
