"""Lemmatise the MULTITuDE texts and the reference texts generated from their human seeds (server; spaCy models).

Reads  data/external/multitude/mt_texts.parquet (human + 2023-era machine texts) and gen_mt_*.parquet (today's generators)
Writes data/external/multitude/mt_human_processed.parquet     (benchmark --human: period=pre, tokens, doc_id, lang)
       data/external/multitude/mt_generated_processed.parquet (benchmark --gen: model, mode, doc_id, src_doc_id, lang, tokens)
The MULTITuDE machine texts get model = 'mt_<generator>' and mode = 'generate' (they were written from a headline).
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, "src/civic")
from preprocess import Lemmatiser, detect_languages, n_words, normalise  # noqa: E402

D = "data/external/multitude"
TEXTS = os.environ.get("MT_TEXTS", "mt_texts.parquet")     # v3 only, or mt_texts_all.parquet (v3 + v1)
SUF = os.environ.get("MT_SUFFIX", "")                      # suffix of the processed outputs


def clean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df[col].fillna("").map(lambda s: normalise(s.strip().strip('"')))
    df["n_words"] = df.text.map(n_words)
    df = df[df.n_words >= 30].copy()
    langs, confs = detect_languages(df.text.tolist())
    df["lang_det"], df["lang_conf"] = langs, confs
    return df


T = clean(pd.read_parquet(f"{D}/{TEXTS}"), "text")
print("MULTITuDE language mismatch rate by model:\n", (T.lang_det != T.lang).groupby(T.model).mean().round(3).to_string())
T = T[T.lang_det == T.lang].copy()

frames = [pd.read_parquet(p) for p in sorted(glob.glob(f"{D}/gen_mt_*.parquet"))]
G = clean(pd.concat(frames, ignore_index=True), "output")
G = G[~G.finish.fillna("").str.startswith("error")]
print("generated language mismatch rate by model:\n", (G.lang_det != G.lang).groupby(G.model).mean().round(3).to_string())
G = G[G.lang_det == G.lang].copy()

lem = Lemmatiser()
for df in (T, G):
    toks = pd.Series([None] * len(df), index=df.index, dtype=object)
    for lang, g in df.groupby("lang"):
        toks.loc[g.index] = pd.Series(lem.tokens(g.text.tolist(), lang, n_process=6), index=g.index, dtype=object)
    df["tokens"] = toks

H = T[T.model == "human"].copy()
H["instance"], H["type"], H["id"] = "multitude_news", "article", H.doc_id.str.split(":").str[2]
H["is_dup_exact"], H["period"] = False, "pre"
H[["instance", "type", "id", "lang", "text", "n_words", "is_dup_exact", "period", "tokens", "doc_id"]].to_parquet(
    f"{D}/mt_human_processed{SUF}.parquet", index=False)

M = T[T.model != "human"].copy()
M["model"] = M.model.where(M.model.str.startswith("mt1_"), "mt_" + M.model)   # v1 rows already carry mt1_
M["mode"], M["variant"], M["src_doc_id"] = "generate", 0, M.doc_id
cols = ["model", "mode", "variant", "doc_id", "src_doc_id", "lang", "text", "n_words", "tokens"]
pd.concat([M[cols], G[cols]], ignore_index=True).to_parquet(f"{D}/mt_generated_processed{SUF}.parquet", index=False)
print("human:", H.groupby("lang").size().to_dict())
print("generated:\n", pd.concat([M, G]).groupby(["lang", "model"]).size().unstack(0).to_string())
