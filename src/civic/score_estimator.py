"""Share of LLM-generated text from classifier scores: distribution matching calibrated per instance.

The fine-tuned classifier (study_scores.py) gives every document an out-of-sample score s. For a corpus slice of an
instance, the score distribution is modelled as the mixture (1 - alpha) F_H + alpha F_M, where F_H is the distribution of
the scores of the instance's own pre-ChatGPT texts and F_M that of the generated texts seeded from the instance (per
generator, mixed with the reference weights; the language-level texts of a generator are used when the instance has
fewer than MIN_G of them). Three quantities are returned for a slice:

  alpha         point estimate: the mixture weight minimising the Hellinger distance between the histograms (HDy,
                Gonzalez-Castro et al. 2013) on 16 bins with edges at the quantiles of the pooled calibration scores,
                with F_M the weighted mixture of the reference generators;
  alpha_gmin,   reference-sensitivity range: the smallest and largest estimate obtained when F_M is one generator of
  alpha_gmax    the region set at a time (the generators of the assistants plausibly in use);
  lb            generator-free lower bound (partial identification under contaminated sampling, Horowitz and Manski
                1995): it uses F_H only and holds whatever produced the other texts.

The bootstrap resamples the slice and the calibration texts, so the intervals include the uncertainty of the calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRID = np.linspace(0, 1, 1001)
N_BINS = 16
MIN_G = 60
LB_Q = 0.98     # human quantile of the lower bound
MIN_YEAR = 100  # pre-ChatGPT texts a year needs to enter the drift allowance of the lower bound
REGION = ["gpt-5.4-mini", "gemini-3.6-flash", "claude-haiku-5-5", "gpt-4o", "gpt-4o-mini"]


def lower_bound(s, sh, groups=None, q: float = LB_Q) -> float:
    """Generator-free lower bound on the share of generated texts. For a threshold t the mixture gives
    P(s > t) = (1 - alpha) FPR + alpha TPR <= (1 - alpha) FPR + alpha, hence alpha >= (P(s > t) - FPR) / (1 - FPR):
    only the human score distribution is needed (t = its q-quantile), and the bound holds for every generator, being
    tight for those whose texts score above t. The bound assumes that the human distribution of the slice is that of the
    calibration texts; to allow for its drift, FPR is the largest exceedance rate observed in any single calibration
    group (the pre-ChatGPT years of the instance with at least MIN_YEAR texts), not the pooled one."""
    t = np.quantile(sh, q)
    fpr = float((sh > t).mean())
    if groups is not None:
        g, n = np.unique(groups, return_counts=True)
        fpr = max([fpr] + [float((sh[groups == y] > t).mean()) for y, k in zip(g, n) if k >= MIN_YEAR])
    return float(max(0.0, ((s > t).mean() - fpr) / (1 - fpr))) if fpr < 1 else 0.0


QP = np.arange(0.10, 0.995, 0.01)


def lower_bound_bbe(s, sh, delta: float = 0.1, gamma: float = 0.01) -> float:
    """Best bin estimation (Garg et al. 2021), the mixture-proportion estimator of positive-unlabelled learning, kept
    for comparison: the human share is bounded by q_u(c) / q_p(c) (shares of slice and human texts with score <= c) at
    the threshold minimising its upper confidence bound. It presumes that the human sample is drawn from the human
    component of the slice; under the temporal drift of civic texts it is not a bound (benchmark: it exceeds the true
    share in most forward-validation mixtures), whereas the upper-tail bound of lower_bound() is."""
    s, sh = np.asarray(s, float), np.sort(np.asarray(sh, float))
    c = np.quantile(sh, QP)
    q_p = np.searchsorted(sh, c, side="right") / len(sh)
    q_u = np.searchsorted(np.sort(s), c, side="right") / len(s)
    pen = (1 + gamma) / q_p * (np.sqrt(np.log(4 / delta) / (2 * len(s))) + np.sqrt(np.log(4 / delta) / (2 * len(sh))))
    i = int(np.argmin(q_u / q_p + pen))
    return float(min(1.0, max(0.0, 1 - q_u[i] / q_p[i])))


class HDy:
    uses_scores = True

    def __init__(self, sh, sg_by_model: dict, weights: dict | None = None, seed: int = 0, groups=None,
                 region: list | None = None):
        """sh: scores of the human calibration texts; groups: their calibration group (year), for the drift allowance of
        the lower bound; sg_by_model: scores of the generated calibration texts per generator; weights: generator
        weights of the point estimate (None: equal); region: generators of the sensitivity range (None: all)."""
        self.sh = np.asarray(sh, float)
        self.groups = None if groups is None else np.asarray(groups)
        self.sg = {m: np.asarray(v, float) for m, v in sg_by_model.items() if len(v)}
        w = {m: (weights.get(m, 0.0) if weights else 1.0) for m in self.sg}
        if sum(w.values()) <= 0:
            w = {m: 1.0 for m in self.sg}
        tot = sum(w.values())
        self.w = {m: v / tot for m, v in w.items() if v > 0}
        self.models = sorted(self.w)
        self.region = sorted(m for m in self.sg if region is None or m in region)
        self.rng = np.random.default_rng(seed)
        pooled = np.concatenate([self.sh] + [self.sg[m] for m in self.models])
        self.edges = np.quantile(pooled, np.linspace(0, 1, N_BINS + 1))[1:-1]

    def _hist(self, s):
        return np.bincount(np.searchsorted(self.edges, s), minlength=N_BINS) / max(len(s), 1)

    def _alpha(self, ht, hh, hg):
        mix = (1 - GRID)[:, None] * hh[None, :] + GRID[:, None] * hg[None, :]
        return float(GRID[np.argmin(((np.sqrt(mix) - np.sqrt(ht)[None, :]) ** 2).sum(1))])

    def _all(self, s, sh, groups, sg: dict, w: dict, region: bool):
        ht, hh = self._hist(s), self._hist(sh)
        hgm = {m: self._hist(sg[m]) for m in set(w) | (set(self.region) if region else set())}
        out = {"alpha": self._alpha(ht, hh, sum(w[m] * hgm[m] for m in w)), "lb": lower_bound(s, sh, groups)}
        if region and self.region:
            per = [self._alpha(ht, hh, hgm[m]) for m in self.region]
            out["alpha_gmin"], out["alpha_gmax"] = min(per), max(per)
        return out

    def estimate(self, s, n_boot: int = 0, ci: float = 0.95, model: str | None = None) -> dict:
        s = np.asarray(s, float)
        w = {model: 1.0} if model else self.w
        region = model is None
        out = self._all(s, self.sh, self.groups, self.sg, w, region)
        if n_boot:
            bs = []
            need = set(w) | (set(self.region) if region else set())
            for _ in range(n_boot):
                i = self.rng.integers(0, len(s), len(s))
                j = self.rng.integers(0, len(self.sh), len(self.sh))
                sg = {m: self.sg[m][self.rng.integers(0, len(self.sg[m]), len(self.sg[m]))] for m in need}
                bs.append(self._all(s[i], self.sh[j], None if self.groups is None else self.groups[j], sg, w, region))
            b = pd.DataFrame(bs)
            out["ci_lo"], out["ci_hi"] = float(b.alpha.quantile((1 - ci) / 2)), float(b.alpha.quantile(1 - (1 - ci) / 2))
            out["lb_lo"] = float(b.lb.quantile(1 - ci))            # one-sided: at least lb_lo with confidence ci
            if "alpha_gmax" in b:
                out["gmax_hi"] = float(b.alpha_gmax.quantile(1 - (1 - ci) / 2))
        return out


class ScoreReferences:
    """Per instance-language calibration sets (the score analogue of analysis.References).

    H: human texts with columns instance, lang, period, created_at, score; G: generated texts with model, inst, lang,
    mode, score. Instances with fewer than min_ref pre-ChatGPT texts use the language's pre-ChatGPT texts ("pooled").
    With weights, the point estimate uses the weighted generators and the sensitivity range the REGION generators;
    without, both use all generators."""

    def __init__(self, H: pd.DataFrame, G: pd.DataFrame, weights: dict | None, min_ref: int = 300):
        self.base, self.inst, self.pooled = {}, {}, set()
        pre = H[H.period == "pre"].copy()
        pre["year"] = pre.created_at.dt.year
        region = None
        if weights:
            region = [m for m in REGION if m in set(G.model)]
            G = G[G.model.isin(set(weights) | set(region))]
        G = G[G["mode"].isin(["rewrite", "generate"])]
        for lang in sorted(set(pre.lang) & set(G.lang)):
            h, gl = pre[pre.lang == lang], G[G.lang == lang]
            if len(h) < min_ref or gl.empty:
                continue
            by_lang = {m: x.score.values for m, x in gl.groupby("model")}
            self.base[lang] = HDy(h.score.values, by_lang, weights, groups=h.year.values, region=region)
            for inst, hi in h.groupby("instance"):
                gi = gl[gl.inst == inst]
                by = {m: (gi[gi.model == m].score.values if (gi.model == m).sum() >= MIN_G else v) for m, v in by_lang.items()}
                own = hi if len(hi) >= min_ref else h
                self.inst[(inst, lang)] = HDy(own.score.values, by, weights, groups=own.year.values, region=region)
                if len(hi) < min_ref:
                    self.pooled.add((inst, lang))
            print(f"[{lang}] score calibration: {len(h)} human, {len(gl)} generated, "
                  f"{sum(1 for k in self.inst if k[1] == lang)} instances", flush=True)

    def get(self, inst: str, lang: str):
        return self.inst.get((inst, lang), self.base.get(lang))


def null_calibration_scores(H: pd.DataFrame, G: pd.DataFrame, weights: dict | None, min_docs: int = 150,
                            min_ref: int = 300) -> pd.DataFrame:
    """False-positive floors on pre-ChatGPT texts, with calibration sets that exclude the tested texts:
    fold1 (calibration on the other hash half), year=y (calibration on the other pre-ChatGPT years: leave one year out)
    and 2021-22 (calibration on 2020 and earlier: forward validation, the situation of the post-ChatGPT texts)."""
    from analysis import fold_of
    pre = H[H.period == "pre"].copy()
    pre["fold2"] = fold_of(pre.doc_id)
    pre["year"] = pre.created_at.dt.year
    region = None
    if weights:
        region = [m for m in REGION if m in set(G.model)]
        G = G[G.model.isin(set(weights) | set(region))]
    G = G[G["mode"].isin(["rewrite", "generate"])]
    rows = []
    for (inst, lang), hi in pre.groupby(["instance", "lang"]):
        gl = G[G.lang == lang]
        if gl.empty or len(hi) < min_docs:
            continue
        gi = gl[gl.inst == inst]
        by = {m: (gi[gi.model == m].score.values if (gi.model == m).sum() >= MIN_G else gl[gl.model == m].score.values)
              for m in gl.model.unique()}
        slices = [("fold1", hi[hi.fold2 == 0], hi[hi.fold2 == 1])]
        slices += [(f"year={y}", hi[hi.year != y], hy) for y, hy in hi.groupby("year")]
        slices.append(("2021-22", hi[hi.year <= 2020], hi[hi.year >= 2021]))
        for name, cal, test in slices:
            if len(test) < min_docs or len(cal) < min_ref // 2:
                continue
            r = HDy(cal.score.values, by, weights, groups=cal.year.values, region=region).estimate(test.score.values, n_boot=100)
            rows.append(dict(lang=lang, instance=inst, slice=name, reference="adapted", n_docs=len(test), n_cal=len(cal),
                             **{k: r.get(k) for k in ("alpha", "ci_hi", "lb", "lb_lo", "alpha_gmax")}))
    return pd.DataFrame(rows)
