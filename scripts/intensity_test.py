"""Identifiability test: prevalence x editing intensity (server, CPU only).

The mixture model reads alpha as a share of documents, but a corpus where many citizens make light edits may give the same
lexical signal as one where fewer citizens rewrite completely. This script splits the existing rewrite outputs into
intensity tiers by lexical overlap with their seed (Jaccard on content lemmas), relabels their mode
(rewrite_light / rewrite_mid / rewrite_heavy), and runs the benchmark with a FIXED reference (all modes) so that only the
test component changes: the recovered fraction alpha_hat / alpha per tier measures how a lightly edited document counts.

Usage (server): python scripts/intensity_test.py && see scripts/run_pipeline.sh for the benchmark call
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def content(tokens) -> set:
    return {t for t in (tokens if tokens is not None else []) if "|" in t and t.rsplit("|", 1)[1] in CONTENT}


G = pd.read_parquet("data/generated/generated_processed.parquet")
seeds = set(G.doc_id)
H = pd.concat([pd.read_parquet(p, columns=["doc_id", "tokens"]) for p in sorted(glob.glob("data/processed/*.parquet"))], ignore_index=True)
H = H[H.doc_id.isin(seeds)].drop_duplicates("doc_id").set_index("doc_id")
hs = H.tokens.map(content)
rw = G["mode"] == "rewrite"
sim = []
for d, t in zip(G.loc[rw, "doc_id"], G.loc[rw, "tokens"]):
    a, b = content(t), hs.get(d, set())
    sim.append(len(a & b) / len(a | b) if (a or b) else np.nan)
G.loc[rw, "jaccard"] = sim
G["mode_int"] = G["mode"]
for lang, g in G[rw].groupby("lang"):
    q1, q2 = g.jaccard.quantile([1 / 3, 2 / 3])
    tier = np.where(g.jaccard >= q2, "rewrite_light", np.where(g.jaccard < q1, "rewrite_heavy", "rewrite_mid"))
    G.loc[g.index, "mode_int"] = tier
    print(f"{lang}: jaccard terciles {q1:.2f} / {q2:.2f}; light median {g[g.jaccard >= q2].jaccard.median():.2f}, heavy median {g[g.jaccard < q1].jaccard.median():.2f}")
G["mode_orig"] = G["mode"]
G["mode"] = G["mode_int"]
G.drop(columns=["mode_int"]).to_parquet("data/generated/generated_intensity.parquet", index=False)
print(G.groupby(["lang", "mode"]).size().unstack().to_string())
