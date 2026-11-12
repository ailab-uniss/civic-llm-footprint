"""Interval recalibration (review M5): add the between-generator error variance to the document-bootstrap variance.

For each language, the squared error of the estimator on the benchmark is modelled as e^2 = c0 + c1 * a_hat^2 (a_hat the
estimate), fitted on the held-out generators OTHER than the one being evaluated (leave-one-generator-out), and the
interval becomes a_hat +- sqrt(hw_boot^2 + z^2 * max(0, c0 + c1 a_hat^2)), clipped to [0, 1]. Coverage is evaluated on
the held-out generator. The final per-language coefficients (fitted on all generators) are written for use on the study.

  python scripts/calibrate_intervals.py  -> results/calibration_coverage.csv, results/calibration_coefs.csv
"""
import numpy as np
import pandas as pd

Z = 1.96
b = pd.read_parquet("results/benchmark.parquet")
best_sel, best_lik = open("results/best_config.txt").read().split()
d = b[(b.method == "distributional") & (b.selection == best_sel) & (b.likelihood == best_lik) & (b.scenario == "all")].copy()
d = d.dropna(subset=["ci_lo", "ci_hi"])
d["err2"] = (d.alpha_hat - d.alpha) ** 2
d["hw"] = (d.ci_hi - d.ci_lo) / 2


def fit(g):
    X = np.c_[np.ones(len(g)), g.alpha_hat.values ** 2]
    y = np.maximum(g.err2.values - (g.hw.values / Z) ** 2, 0)   # between-generator part beyond the bootstrap variance
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c


rows, coefs = [], []
for lang, g in d.groupby("lang"):
    for m, gm in g.groupby("heldout_model"):
        c = fit(g[g.heldout_model != m])
        tau2 = np.maximum(c[0] + c[1] * gm.alpha_hat.values ** 2, 0)
        hw = np.sqrt(gm.hw.values ** 2 + Z ** 2 * tau2)
        lo, hi = np.clip(gm.alpha_hat - hw, 0, 1), np.clip(gm.alpha_hat + hw, 0, 1)
        cov_new = ((lo <= gm.alpha) & (gm.alpha <= hi)).mean()
        cov_old = ((gm.ci_lo <= gm.alpha) & (gm.alpha <= gm.ci_hi)).mean()
        rows.append(dict(lang=lang, heldout_model=m, coverage_boot=cov_old, coverage_cal=cov_new, hw_boot=gm.hw.mean(), hw_cal=hw.mean()))
    c = fit(g)
    coefs.append(dict(lang=lang, c0=c[0], c1=c[1], added_hw_at_0=Z * np.sqrt(max(c[0], 0)),
                      added_hw_at_20=Z * np.sqrt(max(c[0] + c[1] * 0.04, 0))))
R = pd.DataFrame(rows); C = pd.DataFrame(coefs)
R.to_csv("results/calibration_coverage.csv", index=False); C.to_csv("results/calibration_coefs.csv", index=False)
print(R.groupby("lang")[["coverage_boot", "coverage_cal", "hw_boot", "hw_cal"]].mean().round(3).to_string())
print("overall", R[["coverage_boot", "coverage_cal", "hw_boot", "hw_cal"]].mean().round(3).to_dict())
print(C.round(4).to_string())
