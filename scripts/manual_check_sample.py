"""Draw the manual-annotation sample for the audit (L1 legal references, L5 intent preservation).
Run on the server after `run_audit.sh metrics`:  python scripts/manual_check_sample.py --n 50
Writes audit/manual_check_sample.csv with columns for the annotator (l1_manual, l5_manual, notes)."""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--langs", nargs="+", default=["it", "es", "ca", "fr", "en"])
a = ap.parse_args()
P = pd.read_parquet("data/audit/pairs.parquet")
M = pd.read_parquet("results/audit_metrics.parquet")
D = P.merge(M, on=["doc_id", "lang", "model", "mode"])
rows = []
for lang in a.langs:
    g = D[D.lang == lang]
    if g.empty:
        continue
    rows.append(g.sample(min(a.n, len(g)), random_state=7))
S = pd.concat(rows)
cols = ["doc_id", "lang", "model", "mode", "orig", "output", "l1_new_refs", "l5_claim_coverage"]
S = S[[c for c in cols if c in S.columns]].copy()
S["l1_manual_new_refs"] = ""
S["l1_manual_unverifiable"] = ""
S["l5_manual_intent_preserved"] = ""
S["notes"] = ""
S.to_csv("audit/manual_check_sample.csv", index=False)
print(S.groupby("lang").size().to_string())
