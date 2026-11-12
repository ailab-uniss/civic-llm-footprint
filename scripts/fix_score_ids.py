"""Re-key the detector scores of generated texts (data/scores/scores_*.parquet, id = seed|model|mode|variant) to the seed
identifiers of the processed texts, as scripts/fix_data.py did for the generated texts; rows without a valid seed are dropped
together with the scores of the removed wrong-language human texts. Originals are kept as *.langraw.parquet."""
import glob
import os
import shutil

import pandas as pd

A = pd.concat([pd.read_parquet(p, columns=["doc_id", "lang"]) for p in sorted(glob.glob("data/processed/*.parquet"))])
key = {}
for i, l in zip(A.doc_id, A.lang):
    key.setdefault((i.rsplit(":", 1)[0], l), i)
valid = set(A.doc_id)
for p in sorted(glob.glob("data/scores/scores_*.parquet")):
    if p.endswith(".langraw.parquet"):
        continue
    raw = p.replace(".parquet", ".langraw.parquet")
    if not os.path.exists(raw):
        shutil.copy(p, raw)
    s = pd.read_parquet(raw)
    new = []
    for i in s.doc_id:
        if "|" in i:
            seed, rest = i.split("|", 1)
            k = key.get((seed.rsplit(":", 1)[0], seed.rsplit(":", 1)[1]))
            new.append(None if k is None else k + "|" + rest)
        else:
            new.append(i if i in valid else None)
    s["doc_id"] = new
    print(p, len(s), "->", int(s.doc_id.notna().sum()), flush=True)
    s[s.doc_id.notna()].drop_duplicates("doc_id").to_parquet(p, index=False)
print("SCOREIDS_DONE")
