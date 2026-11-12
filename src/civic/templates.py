"""Templated campaigns: contributions assembled from a shared pool of sentences.

The near-duplicate detector (campaigns.py) compares whole texts, so it misses the letters that a campaign tool assembles
by recombining and reordering a fixed set of sentences: each letter is different, yet most of its sentences occur verbatim
in many others. For every contribution this script computes the share of its sentences (at least MIN_WORDS words, case and
punctuation removed) that occur in at least MIN_DOCS other contributions of the same instance and language, and flags the
contribution as templated when that share is at least MIN_SHARE and at least two of its sentences are shared.

Such letters are not independent texts: a classifier applied to them measures the style of a few dozen sentences written
once, so the study reports the estimates with and without them.

  python src/civic/templates.py --processed data/processed/*.parquet --out results/templates.parquet
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MIN_WORDS, MIN_DOCS, MIN_SHARE = 6, 5, 0.5
SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+")
NORM = re.compile(r"[\W_]+")


def sentences(text: str) -> list:
    out = []
    for s in SPLIT.split(text):
        if len(s.split()) >= MIN_WORDS:
            out.append(NORM.sub(" ", s.lower()).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = ["doc_id", "instance", "lang", "text", "is_dup_exact", "period"]
    A = pd.concat([pd.read_parquet(p, columns=[c for c in cols if c in pq.read_schema(p).names]) for p in a.processed],
                  ignore_index=True)
    A = A[(~A.is_dup_exact) & A.text.notna()].drop_duplicates("doc_id")
    rows = []
    for (inst, lang), x in A.groupby(["instance", "lang"]):
        ss = [sentences(t) for t in x.text]
        c = Counter(s for d in ss for s in set(d))
        n_sent = np.array([len(d) for d in ss])
        n_shared = np.array([sum(c[s] > MIN_DOCS for s in d) for d in ss])
        share = np.divide(n_shared, n_sent, out=np.zeros(len(ss)), where=n_sent > 0)
        rows.append(pd.DataFrame({"doc_id": x.doc_id.values, "n_sent": n_sent, "n_shared": n_shared, "reuse_share": share,
                                  "is_template": (share >= MIN_SHARE) & (n_shared >= 2)}))
    R = pd.concat(rows, ignore_index=True)
    R.to_parquet(a.out, index=False)
    A = A.merge(R, on="doc_id")
    print(A.groupby(["instance", "period"]).is_template.mean().unstack().round(3).to_string())
    print("templated:", int(R.is_template.sum()), "of", len(R))


if __name__ == "__main__":
    main()
