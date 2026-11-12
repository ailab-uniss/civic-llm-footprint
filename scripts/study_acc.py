"""Robustness of the yearly estimates to the quantifier: adjusted count at 2% false positives on the same cross-fitted
scores and calibration texts as the HDy estimate (instances calibrated on their own pre-ChatGPT texts; the EU portal and
Regulations.gov, which are split by author type and agency in the main analysis, are not included).

  python scripts/study_acc.py results/supervised/study_scores_aug.parquet results/analysis_cls5/alpha_acc_instance_year.csv
"""
import glob
import sys

import numpy as np
import pandas as pd
import yaml

MIN_REF, MIN_DOCS, MIN_G = 300, 150, 60
S = pd.read_parquet(sys.argv[1])
W = {str(k): float(v) for k, v in yaml.safe_load(open("configs/reference_weights_vendor.yaml", encoding="utf-8"))["weights"].items()}
P = pd.concat([pd.read_parquet(p, columns=["doc_id", "instance", "lang", "year", "period", "is_dup_exact"]) for p in sorted(glob.glob("data/processed/*.parquet"))])
P = P[~P.is_dup_exact & ~P.instance.str.startswith(("eu_hys", "regulations_gov"))].drop_duplicates("doc_id")
H = P.merge(S[S.set == "civic_h"][["doc_id", "score"]].drop_duplicates("doc_id"), on="doc_id")
G = S[(S.set == "civic_g") & S["mode"].isin(["rewrite", "generate"]) & S.model.isin(W)].copy()
G["instance"] = G.doc_id.str.split(":").str[0]
rows = []
for (inst, lang), h in H.groupby(["instance", "lang"]):
    pre = h[h.period == "pre"]
    if len(pre) < MIN_REF:
        continue
    t = np.quantile(pre.score.values, 0.98)
    fpr = float((pre.score.values > t).mean())
    tpr, wsum = 0.0, 0.0
    for m, w in W.items():
        g = G[(G.model == m) & (G.lang == lang) & (G.instance == inst)]
        if len(g) < MIN_G:
            g = G[(G.model == m) & (G.lang == lang)]
        if len(g) >= MIN_G:
            tpr += w * float((g.score.values > t).mean())
            wsum += w
    if wsum == 0:
        continue
    tpr /= wsum
    for y, c in h.groupby("year"):
        if len(c) < MIN_DOCS:
            continue
        p = float((c.score.values > t).mean())
        rows.append(dict(instance=inst, lang=lang, year=int(y), n_docs=len(c), alpha_acc=float(np.clip((p - fpr) / (tpr - fpr), 0, 1)), tpr=tpr, fpr=fpr))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print(pd.DataFrame(rows)[lambda d: d.year >= 2023].pivot_table(index=["instance", "lang"], columns="year", values="alpha_acc").round(3).to_string())
