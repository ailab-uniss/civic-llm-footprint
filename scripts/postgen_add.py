"""Add one generator's texts to data/generated/generated_processed.parquet (same steps as the postgen stage):
normalise, drop outputs under 30 words or in the wrong language, lemmatise.

  python scripts/postgen_add.py data/generated/gen_gpt-4o.parquet
"""
import sys

import pandas as pd

sys.path.insert(0, "src/civic")
from preprocess import Lemmatiser, detect_languages, n_words, normalise  # noqa: E402

P = "data/generated/generated_processed.parquet"
G = pd.read_parquet(sys.argv[1])
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
old = pd.read_parquet(P)
old = old[~old.model.isin(G.model.unique())]
pd.concat([old, G[old.columns]], ignore_index=True).to_parquet(P, index=False)
print(G.groupby(["lang", "mode"]).size().unstack().to_string())
