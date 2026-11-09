"""Extend the human seed sample without touching existing seeds, so that generation runs resume and only add
the new prompts. Two uses:
  python scripts/extend_sample.py --langs de sv pt --n 1500          # languages not yet in the sample (EU Have Your Say)
  python scripts/extend_sample.py --instance decide_madrid --langs es --n 500   # seeds from one more platform
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src/civic")
from generate import make_sample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--langs", nargs="+", default=["de", "sv", "pt"])
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--instance", default=None, help="draw seeds from this processed file only (added even if the language exists)")
ap.add_argument("--sample", default="data/generated/human_sample.parquet")
ap.add_argument("--add", action="store_true", help="add seeds for an instance/language already present (existing seeds are excluded)")
a = ap.parse_args()

S = pd.read_parquet(a.sample)
if a.instance:
    p = Path("data/processed") / f"{a.instance}.parquet"
    if not p.exists():
        print(f"{p} missing"); sys.exit(0)
    if ((S.instance == a.instance) & S.lang.isin(a.langs)).any() and not a.add:
        print("instance already in the sample for these languages (use --add to top it up)"); sys.exit(0)
    df = pd.read_parquet(p, columns=["instance", "lang", "type", "id", "created_at", "text", "title", "n_words",
                                     "is_dup_exact", "period"])
    df = df[(df.period == "pre") & (~df.is_dup_exact) & (df.n_words >= 30) & (df.n_words <= 400) & df.lang.isin(a.langs)]
    ids = df["instance"] + ":" + df["type"] + ":" + df["id"].astype(str) + ":" + df["lang"]
    df = df[~ids.isin(set(S.doc_id))]  # never redraw an existing seed
    rng = random.Random(0)
    parts = []
    for lang, g in df.groupby("lang"):
        idx = list(g.index); rng.shuffle(idx)
        parts.append(g.loc[idx[: a.n]])
    N = pd.concat(parts, ignore_index=True)
    N["doc_id"] = N["instance"] + ":" + N["type"] + ":" + N["id"].astype(str) + ":" + N["lang"]
else:
    new_langs = [l for l in a.langs if l not in set(S.lang)]
    if not new_langs:
        print("nothing to add"); sys.exit(0)
    with tempfile.TemporaryDirectory() as td:
        N = make_sample(Path("data/processed"), Path(td), a.n, new_langs)
N = N[[c for c in S.columns if c in N.columns]]
out = pd.concat([S, N], ignore_index=True).drop_duplicates("doc_id")
out.to_parquet(a.sample, index=False)
print(out.groupby("lang").size().to_string())
