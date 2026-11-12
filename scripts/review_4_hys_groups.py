"""Referee analysis 4: Have Your Say, post-ChatGPT, citizens versus organisations without the two campaign consultations
(publication_id 20100 and 20315), and also without templated letters. -> results/review/review_4_hys_groups.csv"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from review_common import OUT, ScoreReferences, load  # noqa: E402

H, G, W = load()
refs = ScoreReferences(H, G, W, 300)
P = H[H.instance.isin(["eu_hys_citizens", "eu_hys_organisations"]) & (H.period == "post")].copy()
P["pubn"] = pd.to_numeric(P.publication_id, errors="coerce")
camp = P.pubn.isin([20100, 20315])
print("post texts:", P.groupby("instance").size().to_dict(), "| in 20100/20315:", P[camp].groupby(["instance", "pubn"]).size().to_dict(),
      "| templated:", P[P.is_template].groupby("instance").size().to_dict(), flush=True)
variants = [("all post-ChatGPT texts", P), ("without publications 20100 and 20315", P[~camp]),
            ("without publications 20100 and 20315 and without templated letters", P[~camp & ~P.is_template]),
            ("without templated letters only", P[~P.is_template])]
rows = []
for name, X in variants:
    for (inst, lang), x in X.groupby(["instance", "lang"]):
        est = refs.get(inst, lang)
        if est is None or len(x) < 150:
            continue
        r = est.estimate(x.score.values, n_boot=200)
        rows.append(dict(variant=name, instance=inst, lang=lang, n_docs=len(x), alpha=r["alpha"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                         lb=r["lb"], lb_lo=r["lb_lo"], pooled_ref=(inst, lang) in refs.pooled))
R = pd.DataFrame(rows)
pool = []
for (name, inst), g in R.groupby(["variant", "instance"], sort=False):
    w = g.n_docs
    pool.append(dict(variant=name, instance=inst, lang="ALL (n_docs-weighted mean)", n_docs=int(w.sum()), n_langs=len(g),
                     **{k: float(np.average(g[k], weights=w)) for k in ("alpha", "ci_lo", "ci_hi", "lb", "lb_lo")}))
# languages present in both groups (same language set for the comparison)
for name, g in R.groupby("variant", sort=False):
    common = set(g[g.instance == "eu_hys_citizens"].lang) & set(g[g.instance == "eu_hys_organisations"].lang)
    for inst, gi in g[g.lang.isin(common)].groupby("instance"):
        pool.append(dict(variant=name, instance=inst, lang="COMMON languages (n_docs-weighted mean)", n_docs=int(gi.n_docs.sum()), n_langs=len(gi),
                         **{k: float(np.average(gi[k], weights=gi.n_docs)) for k in ("alpha", "ci_lo", "ci_hi", "lb", "lb_lo")}))
R = pd.concat([R, pd.DataFrame(pool)], ignore_index=True)
R.to_csv(OUT / "review_4_hys_groups.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 80)
print(R.round(4).to_string())
print("REVIEW4_DONE")
