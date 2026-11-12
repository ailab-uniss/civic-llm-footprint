"""Referee analysis 1: false-positive floor on real pre-ChatGPT texts as a function of the gap between the calibration
cut-off and the tested year. -> results/review/review_1_drift_gap.csv (+ summary by gap)."""
import sys

import pandas as pd

sys.path.insert(0, "scripts")
from review_common import OUT, HDy, ScoreReferences, by_model, gen_setup, load  # noqa: E402

H, G, W = load()
refs = ScoreReferences(H, G, W, 300)
Gf, region = gen_setup(G, W)
pre = H[H.period == "pre"]
rows = []
for (inst, lang), hi in pre.groupby(["instance", "lang"]):
    if (inst, lang) not in refs.inst or (inst, lang) in refs.pooled:
        continue
    cnt = hi.groupby("year").size().sort_index()
    if len(cnt) < 5:
        continue
    cum = cnt.cumsum()
    ok = cum[cum >= 300]
    if ok.empty:
        continue
    cutoff = 2018 if cum[cum.index <= 2018].max() >= 300 else int(ok.index[0])
    if pd.isna(cutoff):
        cutoff = int(ok.index[0])
    cal = hi[hi.year <= cutoff]
    est = HDy(cal.score.values, by_model(Gf, inst, lang), W, groups=cal.year.values, region=region)
    for y, hy in hi[hi.year > cutoff].groupby("year"):
        if len(hy) < 150:
            continue
        r = est.estimate(hy.score.values, n_boot=200)
        rows.append(dict(instance=inst, lang=lang, cutoff=cutoff, test_year=int(y), gap_years=int(y) - cutoff, n_docs=len(hy),
                         n_cal=len(cal), alpha=r["alpha"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], lb=r["lb"], lb_lo=r["lb_lo"]))
    print(inst, lang, cutoff, len(cal), flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT / "review_1_drift_gap.csv", index=False)
R["gap"] = R.gap_years.clip(upper=4).map(lambda g: "4+" if g >= 4 else str(g))
S = R.groupby("gap").agg(n_cells=("alpha", "size"), n_pairs=("instance", lambda s: len(set(zip(s, R.loc[s.index, "lang"])))),
                         n_docs=("n_docs", "sum"), alpha_mean=("alpha", "mean"), alpha_median=("alpha", "median"),
                         alpha_max=("alpha", "max"), alpha_wmean=("alpha", lambda s: (s * R.loc[s.index, "n_docs"]).sum() / R.loc[s.index, "n_docs"].sum()),
                         n_above_5=("alpha", lambda s: int((s > 0.05).sum())), lb_lo_mean=("lb_lo", "mean"), lb_lo_max=("lb_lo", "max"),
                         n_lb_lo_pos=("lb_lo", lambda s: int((s > 0).sum()))).reset_index()
S.to_csv(OUT / "review_1_drift_gap_summary.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(4).to_string())
print(S.round(4).to_string())
print("pairs:", R.groupby(["instance", "lang"]).ngroups, "cutoffs:", R.drop_duplicates(["instance", "lang"]).cutoff.value_counts().to_dict())
print("REVIEW1_DONE")
