"""Second language identification with the full set of languages.

preprocess.py identified the language among 19 candidates. A text in a language outside the candidates (Czech, Slovak,
Hungarian, Slovenian, Croatian, Lithuanian, Latvian, Estonian, Bulgarian ... on the EU portal) was then assigned to the
nearest candidate with full confidence (Czech and Slovak mostly to Swedish, Hungarian to Spanish) and kept. This script
detects the language of every processed and generated text among all the languages of the detector and marks a text as
valid when the result agrees with the language assigned by preprocess.py.

  python scripts/lang_check.py -> results/lang_check.parquet (set, key columns, lang, lang_full, ok)
"""
import glob
import sys

import pandas as pd
import pyarrow.parquet as pq
from lingua import LanguageDetectorBuilder

det = LanguageDetectorBuilder.from_all_languages().build()


def full(texts):
    r = det.detect_languages_in_parallel_of([str(t)[:2000] for t in texts])
    return [x.iso_code_639_1.name.lower() if x is not None else "" for x in r]


rows = []
for p in sorted(glob.glob("data/processed/*.parquet")):
    names = pq.read_schema(p).names
    d = pd.read_parquet(p, columns=[c for c in ["doc_id", "instance", "lang", "lang_declared", "period", "text"] if c in names])
    d["lang_full"] = full(d.text)
    rows.append(d.drop(columns="text").assign(set="processed", file=p))
    print(p, len(d), round((d.lang_full != d.lang).mean(), 4), flush=True)
for p, name in [("data/generated/generated_processed.parquet", "generated"), ("data/generated/generated_intensity.parquet", "intensity")] + \
        [(q, "audit") for q in sorted(glob.glob("data/audit/*.parquet"))] + \
        [(q, "multitude") for q in sorted(glob.glob("data/external/multitude/*processed_all.parquet"))]:
    try:
        names = pq.read_schema(p).names
        tcol = "text" if "text" in names else "output" if "output" in names else None
        if tcol is None or "lang" not in names:
            print("skip", p, names, flush=True)
            continue
        d = pd.read_parquet(p, columns=[c for c in ["doc_id", "src_doc_id", "lang", "model", "mode", "variant", tcol] if c in names])
        d["lang_full"] = full(d[tcol])
        rows.append(d.drop(columns=tcol).assign(set=name, file=p, row=range(len(d))))
        print(p, len(d), round((d.lang_full != d.lang).mean(), 4), flush=True)
    except Exception as e:  # noqa: BLE001
        print("failed", p, e, flush=True)
R = pd.concat(rows, ignore_index=True)
R["ok"] = R.lang_full == R.lang
for c in ("variant",):
    if c in R:
        R[c] = R[c].astype(str)
R.to_parquet("results/lang_check.parquet", index=False)
P = R[R.set == "processed"]
print(pd.crosstab(P.lang, P.ok).to_string())
print(P[~P.ok].groupby(["instance", "lang", "lang_full"]).size().sort_values(ascending=False).head(60).to_string())
G = R[R.set == "generated"]
print(pd.crosstab(G.lang, G.ok).to_string())
print(G[~G.ok].groupby(["lang", "lang_full"]).size().sort_values(ascending=False).head(30).to_string())
print("LANGCHECK_DONE")
