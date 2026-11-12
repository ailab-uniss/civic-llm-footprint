"""L7 language quality: share of audit outputs not written in the requested language (lingua, as in pre-processing).

  python scripts/audit_l7.py --pairs data/audit/pairs.parquet --out results/audit_l7.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "civic"))
from preprocess import detect_languages  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="data/audit/pairs.parquet")
ap.add_argument("--out", default="results/audit_l7.csv")
a = ap.parse_args()
P = pd.read_parquet(a.pairs, columns=["model", "lang", "mode", "doc_id", "output"])
P = P[P.output.notna() & (P.output.str.strip() != "")].copy()
langs, conf = detect_languages(P.output.astype(str).tolist())
P["lang_out"], P["lang_conf"] = langs, conf
P["l7_wrong_lang"] = (P.lang_out != P.lang).astype(int)
P[["doc_id", "model", "lang", "mode", "lang_out", "lang_conf", "l7_wrong_lang"]].to_csv(a.out, index=False)
print(P.groupby("mode").l7_wrong_lang.mean().round(4).to_string())
print(P.groupby(["mode", "lang"]).l7_wrong_lang.mean().unstack().round(3).to_string())
print(P.groupby("model").l7_wrong_lang.mean().round(4).to_string())
print(P[P.l7_wrong_lang == 1].groupby(["lang", "lang_out"]).size().sort_values(ascending=False).head(15).to_string())
