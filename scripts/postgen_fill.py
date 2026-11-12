"""Append fill texts of generators already present to data/generated/generated_processed.parquet (same steps as
scripts/postgen_add.py, which replaces a generator instead): normalise, drop outputs under 30 words or in the wrong
language, lemmatise. The previous file is kept once as generated_processed.prefill.parquet.

  python scripts/postgen_fill.py data/generated/gen_gpt-4o_instfill.parquet data/generated/gen_gpt-4o-mini_instfill.parquet
"""
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, "src/civic")
from preprocess import Lemmatiser, detect_languages, n_words, normalise  # noqa: E402

P = "data/generated/generated_processed.parquet"
BAK = P.replace(".parquet", ".prefill.parquet")
if not os.path.exists(BAK):
    shutil.copy(P, BAK)
old = pd.read_parquet(BAK)
G = pd.concat([pd.read_parquet(p) for p in sys.argv[1:]], ignore_index=True)
G = G[~G.finish.astype(str).str.startswith("error")].copy()
G["text"] = G.output.fillna("").map(lambda s: normalise(s.strip().strip('"')))
G["n_words"] = G.text.map(n_words)
G = G[G.n_words >= 30].copy()
G["lang_det"], G["lang_conf"] = detect_languages(G.text.tolist())
print("language mismatch rate:", round(float((G.lang_det != G.lang).mean()), 3))
G = G[G.lang_det == G.lang].copy()
lem = Lemmatiser()
toks = pd.Series([None] * len(G), index=G.index, dtype=object)
for lang, g in G.groupby("lang"):
    toks.loc[g.index] = pd.Series(lem.tokens(g.text.tolist(), lang, n_process=6), index=g.index, dtype=object)
G["tokens"] = toks
missing = [c for c in old.columns if c not in G.columns]
assert not missing, missing
new = pd.concat([old, G[old.columns]], ignore_index=True).drop_duplicates(["doc_id", "model", "mode", "variant"], keep="first")
new.to_parquet(P, index=False)
print("added", len(new) - len(old), "texts;", len(new), "in total")
print(G.groupby(["model", "lang", "mode"]).size().unstack().to_string())
