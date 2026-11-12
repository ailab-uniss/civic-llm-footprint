"""Audit analysis: per metric, a mixed-effects model with scenario, language
and generator as fixed effects and the seed document as random intercept; Benjamini-Hochberg correction across all
scenario contrasts and language tests.

  python scripts/audit_mixed.py --metrics results/audit_metrics.parquet --l7 results/audit_l7.csv --out results/audit_mixed.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--metrics", default="results/audit_metrics.parquet")
ap.add_argument("--l7", default="results/audit_l7.csv")
ap.add_argument("--out", default="results/audit_mixed.csv")
a = ap.parse_args()
M = pd.read_parquet(a.metrics)
L7 = pd.read_csv(a.l7)[["doc_id", "model", "mode", "l7_wrong_lang"]]
M = M.merge(L7, on=["doc_id", "model", "mode"], how="left")
METRICS = ["l1_any_new", "l6_any_foreign", "l2_ents_kept", "l2_nums_kept", "l4_len_ratio", "l5_claim_coverage",
           "l5_output_grounded", "l7_wrong_lang"]
rows = []
for m in METRICS:
    d = M[["doc_id", "lang", "model", "mode", m]].dropna().rename(columns={m: "y"})
    if d.y.nunique() < 2:
        continue
    f = "y ~ C(mode, Treatment('s1_improve')) + C(lang, Treatment('en')) + C(model)"
    try:
        r = smf.mixedlm(f, d, groups=d["doc_id"]).fit(reml=False, method="lbfgs")
        # Wald test of the language coefficients (language is nested in the seed document, so a LR test against a
        # model without language is ill-conditioned)
        # language tested on the OLS fit with document-clustered errors (the mixed-model covariance of the language
        # terms is ill-conditioned because language is constant within each seed document)
        ro = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.doc_id)[0]})
        names = [c for c in ro.params.index if c.startswith("C(lang")]
        Rm = np.zeros((len(names), len(ro.params))); 
        for i, c in enumerate(names):
            Rm[i, list(ro.params.index).index(c)] = 1
        p_lang = float(ro.wald_test(Rm, scalar=True).pvalue)
        kind = "mixed"
    except Exception as e:  # fall back to OLS with document-clustered errors
        r = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.doc_id)[0]})
        p_lang = float(r.f_test(" = ".join([]) or [c for c in r.params.index if "lang" in c and False] or np.eye(1))) if False else np.nan
        kind = f"ols ({type(e).__name__})"
    for sc in ["s2_notes", "s3_persuasive"]:
        k = f"C(mode, Treatment('s1_improve'))[T.{sc}]"
        rows.append(dict(metric=m, term=f"{sc} vs s1", coef=r.params[k], se=r.bse[k], p=r.pvalues[k], model=kind, n=len(d)))
    rows.append(dict(metric=m, term="language (Wald test)", coef=np.nan, se=np.nan, p=p_lang, model=kind, n=len(d)))
    print(m, kind, flush=True)
R = pd.DataFrame(rows)
ok = R.p.notna()
R.loc[ok, "p_bh"] = multipletests(R.loc[ok, "p"], method="fdr_bh")[1]
R.to_csv(a.out, index=False)
pd.set_option("display.width", 200)
print(R.round(4).to_string())
