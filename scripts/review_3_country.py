"""Referee analysis 3: native-English countries versus the other countries within the same Have Your Say consultations
(citizens, English, 2025-26, templated letters excluded). -> results/review/review_3_country_within.csv"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from review_common import OUT, ScoreReferences, load  # noqa: E402

NATIVE = {"IRL", "GBR", "USA"}
H, G, W = load()
refs = ScoreReferences(H, G, W, 300)
est = refs.get("eu_hys_citizens", "en")
C = H[(H.instance == "eu_hys_citizens") & ~H.is_template & H.country.notna() & (H.lang == "en") & (H.year >= 2025)].copy()
C["native"] = C.country.isin(NATIVE)
C["pub"] = C.publication_id.astype(str)
rows = []


def add(label, x, **kw):
    r = est.estimate(x.score.values, n_boot=200)
    rows.append(dict(estimate=label, n_docs=len(x), n_publications=x["pub"].nunique(), alpha=r["alpha"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                     lb=r["lb"], lb_lo=r["lb_lo"], top_publication_share=x["pub"].value_counts(normalize=True).iloc[0], **kw))


add("NATIVE, all publications", C[C.native])
add("OTHER, all publications", C[~C.native])
nn = C[C.native].groupby("pub").size()
keep = set(nn[nn >= 5].index)
R = C[C["pub"].isin(keep)]
add("(a) NATIVE, publications with >= 5 NATIVE texts", R[R.native])
add("(a) OTHER, publications with >= 5 NATIVE texts", R[~R.native])
for c in sorted(NATIVE):
    x = R[R.country == c]
    if len(x) >= 150:
        add(f"(a) {c}, publications with >= 5 NATIVE texts", x)

rng = np.random.default_rng(0)
for min_other in (1, 5):
    no = R[~R.native].groupby("pub").size()
    pubs = sorted(p for p in keep if no.get(p, 0) >= min_other)
    M = R[R["pub"].isin(pubs)]
    nat, oth = M[M.native], M[~M.native]
    add(f"(b) NATIVE, publications with >= 5 NATIVE and >= {min_other} OTHER texts", nat)
    add(f"(b) OTHER unweighted, publications with >= 5 NATIVE and >= {min_other} OTHER texts", oth)
    share = nat.groupby("pub").size().reindex(pubs) / len(nat)
    idx = {p: v.values for p, v in oth.groupby("pub").score}
    n = len(oth)
    al, lb = [], []
    for _ in range(20):
        draw = rng.choice(len(pubs), size=n, p=share.values)
        s = np.concatenate([rng.choice(idx[pubs[k]], size=c, replace=True) for k, c in zip(*np.unique(draw, return_counts=True))])
        r = est.estimate(s, n_boot=0)
        al.append(r["alpha"]); lb.append(r["lb"])
    # effective sample size of the re-weighting (weights = NATIVE share / OTHER share of the publication)
    w = (share / (oth.groupby("pub").size().reindex(pubs) / len(oth))).reindex(oth["pub"]).values
    rows.append(dict(estimate=f"(b) OTHER re-sampled to the NATIVE distribution over publications (20 repetitions), >= 5 NATIVE and >= {min_other} OTHER",
                     n_docs=n, n_publications=len(pubs), alpha=float(np.mean(al)), alpha_sd=float(np.std(al, ddof=1)), alpha_min=float(np.min(al)),
                     alpha_max=float(np.max(al)), lb=float(np.mean(lb)), lb_sd=float(np.std(lb, ddof=1)), n_effective=float(w.sum() ** 2 / (w ** 2).sum()),
                     native_texts_covered=len(nat) / len(R[R.native])))
T = pd.DataFrame(rows)
T.to_csv(OUT / "review_3_country_within.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 110)
print(T.round(4).to_string())
print("publications 2025-26 (en, citizens, no templates):", C["pub"].nunique(), "| with >= 5 NATIVE:", len(keep))
print("countries NATIVE:", C[C.native].country.value_counts().to_dict())
print("REVIEW3_DONE")
