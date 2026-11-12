"""Out-of-domain test of the generator-free lower bound on the assistants of 2023 (RAID, English Reddit posts and reviews).

The texts are scored by the final study classifier (models/xlmr_civic), which never saw RAID, Reddit, reviews or these
models. Random halves of the human texts of a domain give the threshold (their 98th percentile); mixtures of the other half with the
texts of one model and decoding at known shares are then bounded with score_estimator.lower_bound.

  python scripts/raid_lb.py -> results/raid_lb.csv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src/civic")
from score_estimator import LB_Q, lower_bound  # noqa: E402

S = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "data/external/raid/raid_texts.scores.parquet")
rng = np.random.default_rng(0)
rows = []
for dom, d in S.groupby("domain"):
    h = d[d.model == "human"].score.values
    for split in range(20):          # random halves of the human texts: threshold from one, mixtures from the other
        idx = rng.permutation(len(h))
        cal, test = h[idx[:len(h) // 2]], h[idx[len(h) // 2:]]
        t = np.quantile(cal, LB_Q)
        for (m, dec), g in d[d.model != "human"].groupby(["model", "decoding"]):
            pg = g.score.values
            for alpha in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
                n = len(test); k = int(round(alpha * n))
                s = np.r_[test[rng.choice(len(test), n - k, replace=False)], pg[rng.choice(len(pg), k, replace=False)]]
                rows.append(dict(domain=dom, model=m, decoding=dec, alpha=alpha, split=split, n=n, lb=lower_bound(s, cal),
                                 human_above=float((test > t).mean()), tpr=float((pg > t).mean())))
R = pd.DataFrame(rows)
R.to_csv(sys.argv[2] if len(sys.argv) > 2 else "results/raid_lb.csv", index=False)
P = R[R.alpha > 0].assign(rec=lambda x: x.lb / x.alpha, over=lambda x: x.lb > x.alpha + 0.01)
print(P.groupby(["domain", "model", "decoding"]).agg(recovered=("rec", "mean"), above_truth_1pp=("over", "mean"), tpr=("tpr", "mean")).round(3).to_string())
print(R[R.alpha == 0].groupby("domain").agg(lb_at_zero=("lb", "mean"), lb_max=("lb", "max"), human_above=("human_above", "mean")).round(4).to_string())
