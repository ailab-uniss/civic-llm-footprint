"""Data corrections of 10 November 2026 (run once on the server, after scripts/lang_check.py).

1. Wrong-language texts. preprocess.py identified the language among 19 candidates; texts in other languages (Czech,
   Slovak, Hungarian, Slovenian ... on the EU portal) were assigned to the nearest candidate (mostly Swedish, Spanish,
   Portuguese, Polish) with full confidence. Every processed text whose language differs under the detector with all
   languages (results/lang_check.parquet) is removed; generated texts are removed when their own language differs.
2. Seed identifiers. The identifier of a processed text ends with the language declared by the platform, that of a
   generation seed with the detected language, so that the generated texts of a seed whose two languages differ (or whose
   declared language is empty) did not match their seed: they did not inherit its fold, year and author type. Generated
   texts, intensity tiers and the seed sample are re-keyed to the identifier of the processed text (same instance, type,
   id and detected language); those without a valid seed are removed.

The original files are kept in data/processed_langraw/ and as *.langraw.parquet.
"""
import glob
import os
import shutil

import pandas as pd

LC = pd.read_parquet("results/lang_check.parquet")
P = LC[LC.set == "processed"]
if not os.path.isdir("data/processed_langraw"):
    shutil.copytree("data/processed", "data/processed_langraw")
key = {}
for p in sorted(glob.glob("data/processed_langraw/*.parquet")):
    q = p.replace("processed_langraw", "processed")
    d = pd.read_parquet(p)
    ok = P[P.file == q].ok.values
    assert len(ok) == len(d), (p, len(ok), len(d))
    c = d[ok]
    c.to_parquet(q, index=False)
    for i, l in zip(c.doc_id, c.lang):
        key.setdefault((i.rsplit(":", 1)[0], l), i)
    if len(c) < len(d):
        print(f"{q}: {len(d)} -> {len(c)}", flush=True)


def rekey(ids):
    return [key.get((i.rsplit(":", 1)[0], i.rsplit(":", 1)[1])) if isinstance(i, str) else None for i in ids]


for path, name in (("data/generated/generated_processed.parquet", "generated"), ("data/generated/generated_intensity.parquet", "intensity")):
    raw = path.replace(".parquet", ".langraw.parquet")
    if not os.path.exists(raw):
        shutil.copy(path, raw)
    g = pd.read_parquet(raw)
    ok = LC[LC.set == name].sort_values("row").ok.values
    assert len(ok) == len(g), (path, len(ok), len(g))
    new = rekey(g.doc_id)
    changed = sum(1 for a, b in zip(g.doc_id, new) if b is not None and a != b)
    g["doc_id"] = new
    keep = ok & g.doc_id.notna().values
    if "src_doc_id" in g:
        src = rekey(g.src_doc_id)
        g["src_doc_id"] = src
        keep &= g.src_doc_id.notna().values
    print(f"{path}: {len(g)} -> {int(keep.sum())} (wrong output language {int((~ok).sum())}, no valid seed {int(g.doc_id.isna().sum())}, re-keyed {changed})", flush=True)
    g = g[keep]
    g.to_parquet(path, index=False)
    if name == "generated":
        print(g.groupby("lang").size().to_string())
        print(g.groupby("model").size().to_string())
for path in ("data/generated/human_sample.parquet",):
    raw = path.replace(".parquet", ".langraw.parquet")
    if not os.path.exists(raw):
        shutil.copy(path, raw)
    s = pd.read_parquet(raw)
    s["doc_id"] = rekey(s.doc_id)
    print(f"{path}: {len(s)} -> {int(s.doc_id.notna().sum())}")
    s[s.doc_id.notna()].to_parquet(path, index=False)
A = pd.concat([pd.read_parquet(p, columns=["doc_id", "instance", "lang", "period", "is_dup_exact"]) for p in sorted(glob.glob("data/processed/*.parquet"))])
print(A[A.instance == "eu_hys"].groupby(["lang", "period"]).size().unstack().to_string())
print("FIX_DONE")
