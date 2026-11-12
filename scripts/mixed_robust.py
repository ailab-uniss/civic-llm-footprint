"""Robustness of the cell-level models (review M6): precision weighting and a ramp specification.

  A  period cells (instance x language x type x period): WLS with weights 1/(se^2 + tau^2), se from the bootstrap
     interval, tau^2 the between-cell residual variance (method of moments); standard errors clustered by instance.
  B  half-year cells: ramp model alpha ~ language + post + years_since_release*post, mixed model with instance random
     intercepts (unweighted) and the same WLS with clustered errors (weighted).
  C  interrupted time series per instance-language pair with Newey-West (HAC, 1 lag) errors instead of OLS errors.

  python scripts/mixed_robust.py   -> results/mixed_robust.csv, results/its_hac.csv
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
R = "results/"
rows = []


def wls_cluster(f, d):
    d = d.copy()
    d["se"] = ((d.ci_hi - d.ci_lo) / 3.92).clip(lower=0.005)
    ols = smf.ols(f, d).fit()
    tau2 = max(0.0, float(np.var(ols.resid)) - float((d.se ** 2).mean()))
    d["wt"] = 1.0 / (d.se ** 2 + tau2)
    return smf.wls(f, d, weights=d.wt).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.instance)[0]}), tau2


P = pd.read_csv(R + "alpha_cells_period.csv").dropna(subset=["alpha", "ci_lo", "ci_hi"])
P["post"] = (P.period == "post").astype(int)
fA = "alpha ~ C(lang) + C(type) + post"
m, tau2 = wls_cluster(fA, P)
rows.append(dict(model="A period cells, WLS 1/(se2+tau2), cluster(instance)", term="post", coef=m.params["post"], se=m.bse["post"],
                 lo=m.conf_int().loc["post", 0], hi=m.conf_int().loc["post", 1], p=m.pvalues["post"], n=len(P), tau=np.sqrt(tau2)))
mu = smf.mixedlm(fA, P, groups=P["instance"]).fit(reml=True)
rows.append(dict(model="A period cells, mixed (unweighted, as Table)", term="post", coef=mu.params["post"], se=mu.bse["post"],
                 lo=mu.conf_int().loc["post", 0], hi=mu.conf_int().loc["post", 1], p=mu.pvalues["post"], n=len(P)))

Hc = pd.read_csv(R + "alpha_cells_half.csv").dropna(subset=["alpha", "ci_lo", "ci_hi"])
Hc["post"] = (Hc.t > 0).astype(int)
Hc["ramp"] = Hc.t.clip(lower=0)
fB = "alpha ~ C(lang) + post + ramp"
mb = smf.mixedlm(fB, Hc, groups=Hc["instance"]).fit(reml=True)
for term in ["post", "ramp"]:
    rows.append(dict(model="B half-year ramp, mixed", term=term, coef=mb.params[term], se=mb.bse[term],
                     lo=mb.conf_int().loc[term, 0], hi=mb.conf_int().loc[term, 1], p=mb.pvalues[term], n=len(Hc)))
mw, tau2b = wls_cluster(fB, Hc)
for term in ["post", "ramp"]:
    rows.append(dict(model="B half-year ramp, WLS cluster(instance)", term=term, coef=mw.params[term], se=mw.bse[term],
                     lo=mw.conf_int().loc[term, 0], hi=mw.conf_int().loc[term, 1], p=mw.pvalues[term], n=len(Hc), tau=np.sqrt(tau2b)))
out = pd.DataFrame(rows)
out.to_csv(R + "mixed_robust.csv", index=False)
print(out.round(4).to_string())

# ---- C: ITS per pair with HAC errors (same specification as analysis.py: level shift at release and slope change)
its = []
for (inst, lang), g in Hc.groupby(["instance", "lang"]):
    if (g.t <= 0).sum() < 3 or (g.t > 0).sum() < 3:
        continue
    r = smf.ols("alpha ~ t + post + ramp", g).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    its.append(dict(instance=inst, lang=lang, n=len(g), level=r.params["post"], p_level=r.pvalues["post"],
                    slope=r.params["ramp"], p_slope=r.pvalues["ramp"]))
I = pd.DataFrame(its)
I.to_csv(R + "its_hac.csv", index=False)
print(len(I), "pairs; level significant:", int((I.p_level < 0.05).sum()), "slope significant:", int((I.p_slope < 0.05).sum()),
      "positive level & significant:", int(((I.p_level < 0.05) & (I.level > 0)).sum()))
