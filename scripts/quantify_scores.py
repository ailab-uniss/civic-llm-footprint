"""Quantifiers on saved classifier scores (src/civic/supervised_scores.py), under the benchmark protocols.

Calibration distributions come from the calibration part of the fit set, never used for training: at language level
(as a language-wide reference) or at instance level (the instance's own pre-ChatGPT texts and the generated texts seeded
from them, falling back to the language when fewer than MIN_H / MIN_G exist), as available in the study.

  cc          share of documents with score > 0
  acc         adjusted count at threshold 0 (Forman 2008)
  acc_sp      adjusted count at the threshold giving 2% false positives on the calibration human texts
  ms          median sweep: median of adjusted counts over thresholds with tpr - fpr >= 0.25
  hdy         Hellinger-distance distribution matching on 16-bin score histograms (Gonzalez-Castro et al. 2013)
  hdyx/w/v/c  the same with the estimator of the study (score_estimator.HDy): equal generator weights, market shares on the
              current models, vendor shares split over the models of the vendor, equal weights over the commercial models
  em          EM prior adjustment (Saerens et al. 2002) of calibrated posteriors (Platt scaling on calibration texts)
Suffix _lang / _inst: calibration level.

  python scripts/quantify_scores.py results/supervised/scores.parquet -> results/supervised/quant.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "civic"))
from score_estimator import HDy, REGION, lower_bound_bbe  # noqa: E402

ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
MIN_H, MIN_G, MIN_CAL, REPS, NMAX, N_BOOT = 150, 60, 100, 10, 500, 100
W = {str(k): float(v) for k, v in yaml.safe_load(open("configs/reference_weights.yaml", encoding="utf-8"))["weights"].items()}
WV = {str(k): float(v) for k, v in yaml.safe_load(open("configs/reference_weights_vendor.yaml", encoding="utf-8"))["weights"].items()}
GRID = np.linspace(0, 1, 401)


class Cal:
    def __init__(self, sh, sg):
        self.sh, self.sg = np.sort(sh), np.sort(sg)
        self.thr = np.quantile(np.concatenate([sh, sg]), np.linspace(0.02, 0.98, 49))
        self.fpr = 1 - np.searchsorted(self.sh, self.thr, side="right") / len(self.sh)
        self.tpr = 1 - np.searchsorted(self.sg, self.thr, side="right") / len(self.sg)
        self.edges = np.quantile(np.concatenate([sh, sg]), np.linspace(0, 1, 17))[1:-1]
        self.hh = np.bincount(np.searchsorted(self.edges, sh), minlength=16) / len(sh)
        self.hg = np.bincount(np.searchsorted(self.edges, sg), minlength=16) / len(sg)
        self.t_sp = np.quantile(sh, 0.98)
        # Platt scaling on balanced calibration texts
        from sklearn.linear_model import LogisticRegression
        x = np.concatenate([sh, sg]); y = np.r_[np.zeros(len(sh)), np.ones(len(sg))]
        w = np.r_[np.full(len(sh), 0.5 / len(sh)), np.full(len(sg), 0.5 / len(sg))]
        self.lr = LogisticRegression().fit(x[:, None], y, sample_weight=w * len(x))

    @staticmethod
    def _acc(rate, tpr, fpr):
        return (rate - fpr) / (tpr - fpr) if tpr - fpr > 0.05 else rate

    def all(self, s):
        out = {"cc": float((s > 0).mean())}
        f0 = (self.sh > 0).mean(); t0 = (self.sg > 0).mean()
        out["acc"] = self._acc(out["cc"], t0, f0)
        out["acc_sp"] = self._acc(float((s > self.t_sp).mean()), float((self.sg > self.t_sp).mean()), 0.02)
        rate = (s[None, :] > self.thr[:, None]).mean(1)
        ok = (self.tpr - self.fpr) >= 0.25
        out["ms"] = float(np.median((rate[ok] - self.fpr[ok]) / (self.tpr[ok] - self.fpr[ok]))) if ok.any() else out["acc"]
        ht = np.bincount(np.searchsorted(self.edges, s), minlength=16) / len(s)
        mix = (1 - GRID)[:, None] * self.hh[None, :] + GRID[:, None] * self.hg[None, :]
        out["hdy"] = float(GRID[np.argmin(((np.sqrt(mix) - np.sqrt(ht)[None, :]) ** 2).sum(1))])
        p = self.lr.predict_proba(s[:, None])[:, 1]; q = 0.5
        for _ in range(300):
            w1, w0 = q / 0.5, (1 - q) / 0.5
            qn = float((p * w1 / (p * w1 + (1 - p) * w0)).mean())
            if abs(qn - q) < 1e-7:
                break
            q = qn
        out["em"] = q
        return {k: float(min(1, max(0, v))) for k, v in out.items()}


def case(arg):
    """One protocol and held-out generator (run in parallel)."""
    import zlib
    (proto, m), d = arg
    rng = np.random.default_rng(zlib.crc32(f"{proto}|{m}".encode()))
    rows = []
    if True:
        ch, cg, th, tg = (d[d.role == r] for r in ("cal_h", "cal_g", "test_h", "test_g"))
        for lang in sorted(set(th.lang) & set(tg.lang)):
            chl, cgl = ch[ch.lang == lang], cg[(cg.lang == lang)]
            if len(chl) < MIN_H or len(cgl) < MIN_G:
                continue
            cal_lang = Cal(chl.score.values, cgl.score.values)
            thl, tgl = th[th.lang == lang], tg[tg.lang == lang]
            cases = []
            if proto == "hash":
                cases.append(("main", None, thl.score.values, tgl.score.values, cal_lang, cal_lang))
            def hdy_objs(chd, cgs):
                by_lang = {g: x.score.values for g, x in cgl.groupby("model")}
                by = {g: (cgs[cgs.model == g].score.values if (cgs.model == g).sum() >= MIN_G else v) for g, v in by_lang.items()}
                wm = {g: W[g] for g in by if g in W}
                kw = dict(groups=chd.year.values, region=[g for g in REGION if g in by])
                def mk(wd):
                    wd = {g: v for g, v in wd.items() if g in by}
                    return HDy(chd.score.values, by, wd, **kw) if wd else None
                # equal weights (all generators), market shares on the current models, vendor shares split over the models
                # of the vendor, equal weights over the commercial models
                return HDy(chd.score.values, by, None, **kw), mk(W), mk(WV), mk({g: 1.0 for g in REGION})
            hl = hdy_objs(chl, cgl.iloc[:0])
            if proto == "hash":
                cases[-1] = cases[-1] + (hl, hl)
            for inst, hi in thl.groupby("instance"):
                if len(hi) < MIN_H:
                    continue
                gi = tgl[tgl.instance == inst]
                pg = gi.score.values if len(gi) >= MIN_G else tgl.score.values
                chd, cgi = chl[chl.instance == inst], cgl[cgl.instance == inst]
                chi = chd.score.values
                pooled = len(chi) < MIN_CAL
                cal_inst = cal_lang if pooled else Cal(chi, cgi.score.values if len(cgi) >= MIN_G else cgl.score.values)
                hi_obj = hl if pooled else hdy_objs(chd, cgi)
                cases.append(("instance" if proto == "hash" else "period", inst, hi.score.values, pg, cal_lang, cal_inst, hl, hi_obj))
            for exp, inst, ph, pg, cl, ci, hdl, hdi in cases:
                if len(pg) < 30:
                    continue
                for alpha in ALPHAS:
                    for r in range(REPS):
                        n = min(NMAX if exp != "main" else 1000, len(ph), int(len(pg) / max(alpha, 0.5)))
                        k = int(round(alpha * n))
                        s = np.concatenate([ph[rng.choice(len(ph), n - k, replace=False)],
                                            pg[rng.choice(len(pg), k, replace=False)] if k else np.array([])])
                        base = dict(exp=exp, lang=lang, heldout_model=m, instance=inst, alpha=alpha, rep=r, n=n)
                        for lev, c, hd in (("lang", cl, hdl), ("inst", ci, hdi)):
                            if exp == "main" and lev == "inst":
                                continue
                            for meth, v in c.all(s).items():
                                rows.append(dict(base, method=f"{meth}_{lev}", alpha_hat=v))
                            for tag, h in zip(("hdyx", "hdyw", "hdyv", "hdyc"), hd):
                                if h is None:
                                    continue
                                o = h.estimate(s, n_boot=N_BOOT if r < 2 else 0)
                                rows.append(dict(base, method=f"{tag}_{lev}", alpha_hat=o["alpha"], ci_lo=o.get("ci_lo"), ci_hi=o.get("ci_hi"),
                                                 own=all(h is not x for x in hdl), gmin=o.get("alpha_gmin"), gmax=o.get("alpha_gmax"),
                                                 gmax_hi=o.get("gmax_hi"), lb=o["lb"], lb_lo=o.get("lb_lo")))
                                if tag == "hdyx":   # generator-free lower bounds (human calibration scores only)
                                    rows.append(dict(base, method=f"lb_{lev}", alpha_hat=o["lb"], ci_lo=o.get("lb_lo")))
                                    rows.append(dict(base, method=f"bbe_{lev}", alpha_hat=lower_bound_bbe(s, h.sh)))
        print(proto, m, len(rows), flush=True)
    return rows


def main():
    from multiprocessing import Pool
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/supervised/scores.parquet")
    D = pd.read_parquet(src)
    D = D[D.protocol.isin(["hash", "period"])]
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 6) as pool:
        rows = [r for rs in pool.imap_unordered(case, list(D.groupby(["protocol", "heldout_model"]))) for r in rs]
    R = pd.DataFrame(rows)
    out = src.with_name("quant.parquet" if src.stem == "scores" else f"quant_{src.stem.replace('scores_', '')}.parquet")
    R.to_parquet(out, index=False)
    R["ae"] = (R.alpha_hat - R.alpha).abs() * 100; R["b"] = (R.alpha_hat - R.alpha) * 100
    print(R.groupby(["exp", "method"]).agg(mae=("ae", "mean"), bias=("b", "mean")).round(2).unstack(0).to_string())


if __name__ == "__main__":
    main()


def summary(path="results/supervised/quant.parquet"):
    R = pd.read_parquet(path)
    R["ae"] = (R.alpha_hat - R.alpha).abs() * 100
    R["cov"] = np.where(R.ci_lo.notna(), (R.ci_lo <= R.alpha + 1e-9) & (R.alpha - 1e-9 <= R.ci_hi), np.nan) if "ci_lo" in R else np.nan
    return R.groupby(["exp", "method"]).agg(mae=("ae", "mean"), cov=("cov", "mean")).round(3).unstack(0)
