"""Referee analysis 2: out-of-sample floor per pre-ChatGPT process: the calibration of the instance-language is rebuilt
without the texts of the tested process. -> results/review/review_2_process_oos.csv"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from review_common import OUT, HDy, ScoreReferences, by_model, gen_setup, load  # noqa: E402
from consultations import consultation_id  # noqa: E402

H, G, W = load()
refs = ScoreReferences(H, G, W, 300)
Gf, region = gen_setup(G, W)
H["consultation"] = consultation_id(H)
pre = H[H.period == "pre"]
n_all = H[H.consultation.notna()].groupby(["instance", "lang", "consultation"]).size()
C = pd.read_csv("results/analysis_cls5/alpha_cells_consultation.csv", dtype={"consultation": str})
cin = {(i, l, c): (n, a) for i, l, c, n, a in zip(C.instance, C.lang, C.consultation, C.n_docs, C.alpha)}
rows = []
for (inst, lang), hi in pre.groupby(["instance", "lang"]):
    if refs.get(inst, lang) is None:
        continue
    pooled = (inst, lang) not in refs.inst or (inst, lang) in refs.pooled
    base = pre[pre.lang == lang] if pooled else hi          # calibration texts of the study for this pair
    by = by_model(Gf, inst, lang) if (inst, lang) in refs.inst else {m: x.score.values for m, x in Gf[Gf.lang == lang].groupby("model")}
    full = refs.get(inst, lang)
    for cons, x in hi[hi.consultation.notna()].groupby("consultation"):
        if len(x) < 150:
            continue
        cal = base[~base.doc_id.isin(set(x.doc_id))]
        if len(cal) < 300:
            rows.append(dict(instance=inst, lang=lang, process=cons, n_docs=len(x), n_cal_out=len(cal), skipped=True, pooled_ref=pooled))
            continue
        r = HDy(cal.score.values, by, W, groups=cal.year.values, region=region).estimate(x.score.values, n_boot=100)
        ri = full.estimate(x.score.values, n_boot=0)
        k = cin.get((inst, lang, str(cons)))
        rows.append(dict(instance=inst, lang=lang, process=cons, n_docs=len(x), n_docs_all_periods=int(n_all.get((inst, lang, cons), len(x))),
                         n_cal_out=len(cal), share_of_calibration=len(x) / len(base), pooled_ref=pooled, skipped=False,
                         alpha_in_sample=(k[1] if k is not None and k[0] == len(x) else np.nan), alpha_in_recomputed=ri["alpha"],
                         lb_in_recomputed=ri["lb"], alpha_out=r["alpha"], ci_lo_out=r["ci_lo"], ci_hi_out=r["ci_hi"], lb_out=r["lb"], lb_lo_out=r["lb_lo"]))
    print(inst, lang, len(rows), flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT / "review_2_process_oos.csv", index=False)
pd.set_option("display.width", 250)
D = R[~R.skipped]
print(D.sort_values("alpha_out", ascending=False).head(40).round(4).to_string())


def summ(d, name):
    return dict(subset=name, n=len(d), mean_out=d.alpha_out.mean(), median_out=d.alpha_out.median(), max_out=d.alpha_out.max(),
                n_above_5=int((d.alpha_out > 0.05).sum()), n_above_10=int((d.alpha_out > 0.10).sum()),
                mean_in=d.alpha_in_recomputed.mean(), max_in=d.alpha_in_recomputed.max(),
                n_in_above_5=int((d.alpha_in_recomputed > 0.05).sum()), n_in_above_10=int((d.alpha_in_recomputed > 0.10).sum()),
                mean_lb_lo_out=d.lb_lo_out.mean(), max_lb_lo_out=d.lb_lo_out.max(), n_lb_lo_pos=int((d.lb_lo_out > 0).sum()))


S = [summ(D, "all"), summ(D[~D.pooled_ref.astype(bool)], "own calibration"), summ(D[D.pooled_ref.astype(bool)], "language-pool calibration")]
S += [summ(d, "group=" + g) for g, d in D.groupby(D.instance.map(lambda s: "eu_hys" if s.startswith("eu_hys") else "regulations_gov" if s.startswith("regulations_gov") else "decidim"))]
S = pd.DataFrame(S)
S.to_csv(OUT / "review_2_process_oos_summary.csv", index=False)
print(S.round(4).to_string())
print("skipped (calibration < 300 without the process):", int(R.skipped.sum()), "| with matching in-sample csv value:", int(D.alpha_in_sample.notna().sum()))
print("REVIEW2_DONE")
