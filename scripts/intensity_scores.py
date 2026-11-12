"""Identifiability test for the classifier estimator: how a document counts as a function of the editing intensity.

Same design as scripts/intensity_test.py (rewrites split in terciles of lexical overlap with their seed), with the
cross-fitted classifier scores: the reference is fixed (human texts and rewrite + generate texts of the reference
generators, from one hash half of the seeds) and only the test component changes (texts of one tier from the other half,
same generators). Recovered fraction = estimate / true share, for the point estimate and for the lower bound.

  python scripts/intensity_scores.py results/supervised/study_scores_aug.parquet -> results/intensity/cls_intensity.csv
"""
import glob
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, "src/civic")
from analysis import fold_of  # noqa: E402
from score_estimator import HDy  # noqa: E402

W = {str(k): float(v) for k, v in yaml.safe_load(open("configs/reference_weights_vendor.yaml", encoding="utf-8"))["weights"].items()}
S = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "results/supervised/study_scores_aug.parquet")
I = pd.read_parquet("data/generated/generated_intensity.parquet", columns=["doc_id", "model", "lang", "mode", "mode_orig", "variant", "jaccard"])
g = S[S.set == "civic_g"][["doc_id", "model", "mode", "variant", "score"]].rename(columns={"mode": "mode_orig"})
I = I.merge(g.drop_duplicates(["doc_id", "model", "mode_orig", "variant"]), on=["doc_id", "model", "mode_orig", "variant"])
I = I[I.model.isin(W)]
I["half"] = fold_of(I.doc_id)
cols = ["doc_id", "lang", "period", "is_dup_exact", "created_at"]
A = pd.concat([pd.read_parquet(p, columns=[c for c in cols if c in pq.read_schema(p).names]) for p in sorted(glob.glob("data/processed/*.parquet"))])
A = A[(A.period == "pre") & ~A.is_dup_exact].drop_duplicates("doc_id")
H = A.merge(S[S.set == "civic_h"][["doc_id", "score"]].drop_duplicates("doc_id"), on="doc_id")
H["half"] = fold_of(H.doc_id); H["year"] = H.created_at.dt.year
rng = np.random.default_rng(0)
rows = []
for lang in ["en", "de", "fr", "it", "fi"]:
    h, gl = H[H.lang == lang], I[I.lang == lang]
    ref = gl[(gl.half == 0) & gl.mode_orig.isin(["rewrite", "generate"])]
    e = HDy(h[h.half == 0].score.values, {m: x.score.values for m, x in ref.groupby("model")}, W, groups=h[h.half == 0].year.values)
    ht = h[h.half == 1].score.values
    for tier, t in gl[gl.half == 1].groupby("mode"):
        pg = t.score.values
        if len(pg) < 100:
            continue
        for alpha in (0.1, 0.2, 0.3):
            for r in range(10):
                n = min(1000, len(ht), int(len(pg) / alpha)); k = int(round(alpha * n))
                s = np.r_[ht[rng.choice(len(ht), n - k, replace=False)], pg[rng.choice(len(pg), k, replace=False)]]
                o = e.estimate(s)
                rows.append(dict(lang=lang, tier=tier, alpha=alpha, jaccard=t.jaccard.median(), alpha_hat=o["alpha"], lb=o["lb"]))
R = pd.DataFrame(rows)
R.to_csv("results/intensity/cls_intensity.csv", index=False)
R["rec"], R["rec_lb"] = R.alpha_hat / R.alpha, R.lb / R.alpha
print(R.groupby("tier").agg(jaccard=("jaccard", "median"), recovered=("rec", "mean"), recovered_lb=("rec_lb", "mean")).round(2).to_string())
print(R.groupby(["tier", "lang"]).rec.mean().round(2).unstack().to_string())
