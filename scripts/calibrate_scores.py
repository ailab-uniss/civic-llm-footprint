"""Calibrated intervals of the classifier estimator from the forward-validation benchmark.

The bootstrap interval of the point estimate covers the sampling of the slice and of the calibration texts, not the
difference between the reference generators and the generator in use nor the drift of the human texts after the
calibration period; its coverage in the benchmark is below the nominal level. The interval reported in the study is
therefore the error band of the forward-validation benchmark (calibration on texts of 2020 and earlier, test on 2021-22,
generator held out): the 2.5th and 97.5th percentiles of the error e = a_hat - a are modelled as linear functions of a_hat
by quantile regression, and the interval is [a_hat - q_hi(a_hat), a_hat - q_lo(a_hat)], clipped to [0, 1] (a split-
conformal interval with a level-dependent band). Coverage is evaluated leaving one generator out of the fit. Separate
bands are fitted for the instances calibrated on their own pre-ChatGPT texts and for those using the language pool.
The band is fitted without the generator in OUTSIDE (Salamandra), whose texts the classifier separates poorly from human
texts: its coverage is reported apart, as the case of a generator outside the range the band was calibrated on.

  python scripts/calibrate_scores.py [quant parquet] [method] -> results/supervised/calibration_{coverage,coefs}.csv
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

src = sys.argv[1] if len(sys.argv) > 1 else "results/supervised/quant_aug.parquet"
R = pd.read_parquet(src)
METHOD = sys.argv[2] if len(sys.argv) > 2 else "hdyv_inst"   # estimator of the study (vendor-weighted reference)
d = R[(R.method == METHOD) & ~R.heldout_model.str.startswith("group:")].copy()
d["e"] = d.alpha_hat - d.alpha
d["own"] = d.own.astype(bool)
OUTSIDE = ["salamandra-7b"]


def fit(g):
    lo = smf.quantreg("e ~ alpha_hat", g).fit(q=0.025).params
    hi = smf.quantreg("e ~ alpha_hat", g).fit(q=0.975).params
    return dict(lo0=lo["Intercept"], lo1=lo["alpha_hat"], hi0=hi["Intercept"], hi1=hi["alpha_hat"])


def band(a_hat, c):
    return (np.clip(a_hat - np.maximum(c["hi0"] + c["hi1"] * a_hat, 0), 0, 1),
            np.clip(a_hat - np.minimum(c["lo0"] + c["lo1"] * a_hat, 0), 0, 1))


rows, coefs = [], []
for own in (True, False):
    f = d[(d.exp == "period") & (d.own == own) & ~d.heldout_model.isin(OUTSIDE)]
    c = fit(f)
    coefs.append(dict(calibration="own" if own else "pooled", **c))
    for exp in ("period", "instance"):
        t = d[(d.exp == exp) & (d.own == own)]
        for m, gm in t.groupby("heldout_model"):
            lo, hi = band(gm.alpha_hat.values, fit(f[f.heldout_model != m]))
            b = gm[gm.ci_lo.notna()]
            rows.append(dict(exp=exp, calibration="own" if own else "pooled", heldout_model=m,
                             coverage_boot=((b.ci_lo <= b.alpha + 1e-9) & (b.alpha <= b.ci_hi + 1e-9)).mean(),
                             coverage_cal=((lo <= gm.alpha + 1e-9) & (gm.alpha <= hi + 1e-9)).mean(), width_cal=(hi - lo).mean()))
C, K = pd.DataFrame(rows), pd.DataFrame(coefs)
C.to_csv("results/supervised/calibration_coverage.csv", index=False)
K.to_csv("results/supervised/calibration_coefs.csv", index=False)
print(C[~C.heldout_model.isin(OUTSIDE)].groupby(["calibration", "exp"])[["coverage_boot", "coverage_cal", "width_cal"]].mean().round(3).to_string())
print(C.pivot_table(index="heldout_model", columns=["calibration", "exp"], values="coverage_cal").round(2).to_string())
print(K.round(4).to_string())
for _, k in K.iterrows():
    print(k.calibration, {a: tuple(np.round(band(np.array([a]), k), 3).ravel()) for a in (0.0, 0.05, 0.1, 0.2, 0.4)})
