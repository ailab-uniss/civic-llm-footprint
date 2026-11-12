"""Corpus-level estimation of the share of LLM-generated/modified texts (alpha).

Three estimators are implemented:

1. DistributionalEstimator (main method). A corpus C is modelled as a mixture
       P_C(w) = (1 - alpha) P_H(w) + alpha P_M(w)
   where P_H(w) / P_M(w) are document-level occurrence probabilities of lemma w in
   human reference texts / LLM reference texts. alpha is estimated by maximum likelihood
   over a selected vocabulary, assuming word-wise independence (Liang et al., 2024).
   Extensions: per-language vocabularies on lemmas; multi-generator reference mixture;
   vocabulary selection strategies; document bootstrap confidence intervals; and three
   likelihoods:
     'occurrence' - document-level occurrence indicators with length-free probabilities (Liang et al.);
     'length'     - occurrence indicators whose probabilities depend on document length through a
                    Poisson occurrence model p(w | L) = 1 - exp(-lambda_w L), with lambda_w calibrated so
                    that the model reproduces the reference occurrence rate at the reference length
                    distribution (robust to a length mismatch between reference and target corpus);
     'frequency'  - token-level unigram mixture over the selected vocabulary plus an "other" bucket
                    (length-invariant by construction, but sensitive to burstiness).

2. rogan_gladen(): detector + misclassification correction (sensitivity/specificity).

3. bayes_prevalence(): detector + Bayesian misclassification model with uncertainty on
   sensitivity/specificity propagated from validation counts (Gelman & Carpenter, 2020).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist

EPS = 1e-9
LIKELIHOODS = ("occurrence", "length", "frequency")
# fixed length bins (content tokens per document) used by the 'length' likelihood
LENGTH_EDGES = np.array([0, 10, 15, 20, 30, 40, 55, 75, 100, 140, 200, 300, 10**9])


def doc_frequency(docs) -> Counter:
    c = Counter()
    for d in docs:
        c.update(set(d))
    return c


def token_counts(docs) -> tuple[Counter, int]:
    c = Counter()
    n = 0
    for d in docs:
        c.update(d)
        n += len(d)
    return c, n


def calibrate_rates(p_occ: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Find lambda_w such that mean_d [1 - exp(-lambda_w L_d)] = p_occ[w] over the reference lengths
    (vectorised bisection in log space)."""
    L, cnt = np.unique(lengths, return_counts=True)
    w = cnt / cnt.sum()
    lo = np.full(len(p_occ), -20.0)
    hi = np.full(len(p_occ), 5.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        f = (w[None, :] * (1 - np.exp(-np.exp(mid)[:, None] * L[None, :]))).sum(1) - p_occ
        hi = np.where(f > 0, mid, hi)
        lo = np.where(f > 0, lo, mid)
    return np.exp((lo + hi) / 2)


@dataclass
class ReferenceDistribution:
    vocab: list[str]
    p_h: np.ndarray   # occurrence prob in human docs
    p_m: np.ndarray   # occurrence prob in LLM docs (ensemble)
    n_h: int
    n_m: int
    selection: str
    per_model_p_m: dict[str, np.ndarray] = field(default_factory=dict)
    # token rates (frequency likelihood) and calibrated Poisson rates (length likelihood)
    f_h: np.ndarray | None = None
    f_m: dict[str, np.ndarray] = field(default_factory=dict)
    lam_h: np.ndarray | None = None
    lam_m: dict[str, np.ndarray] = field(default_factory=dict)
    # content lengths of the reference documents (kept for re-calibration after adaptation)
    len_h: np.ndarray | None = None
    len_m: dict[str, np.ndarray] = field(default_factory=dict)


class DistributionalEstimator:
    """Maximum-likelihood mixture estimator over document-level lemma occurrences (or token frequencies)."""

    def __init__(self, selection: str = "ratio", top_k: int = 1000, min_df: float = 0.005,
                 smoothing: float = 0.5, pos_filter: set[str] | None = None, seed: int = 0,
                 likelihood: str = "occurrence", split_selection: bool = True, model_weights: dict | None = None,
                 stable_q: float = 0.1, min_group: int = 40):
        """
        selection: 'ratio'  -> top_k lemmas by |log(p_m/p_h)| (symmetric; excess + deficit)
                   'ratio_pos' -> as 'ratio', restricted to lemmas whose POS tag is in pos_filter
                   'excess' -> top_k lemmas with largest log(p_m/p_h) (LLM-overused only)
                   'all'    -> all lemmas with df >= min_df
                   'pos'    -> all lemmas with df >= min_df whose POS tag is in pos_filter
                               (requires tokens formatted 'lemma|POS')
        min_df:   minimum document frequency (fraction) in the union of reference docs
        smoothing: additive pseudo-count for occurrence probabilities
        likelihood: 'occurrence' (default), 'length' or 'frequency' (see module docstring)
        split_selection: for 'ratio'/'excess', rank lemmas on a random half of the reference documents and
                  estimate the probabilities on the other half (no effect when selection_docs is given)
        """
        assert likelihood in LIKELIHOODS, likelihood
        self.selection = selection
        self.top_k = top_k
        self.min_df = min_df
        self.smoothing = smoothing
        self.pos_filter = pos_filter or {"ADJ", "ADV", "VERB", "NOUN"}
        self.likelihood = likelihood
        self.split_selection = split_selection
        self.model_weights = dict(model_weights) if model_weights else None   # generator -> weight in P_M (None: equal)
        # 'stable' selection: |log(p_m/p_h,g)| must hold with the same sign in every human group g (e.g. instance x year);
        # lemmas are ranked by the stable_q quantile over groups of the signed log-ratio (0 when the sign changes)
        self.stable_q, self.min_group = stable_q, min_group
        self.rng = np.random.default_rng(seed)
        self.ref: ReferenceDistribution | None = None

    def _mix(self, per_model: dict) -> np.ndarray:
        """P_M as a mixture of the per-generator distributions: weighted by self.model_weights (renormalised over the
        generators present; generators without a weight get 0) or, without weights, the equal-weight mean."""
        keys = list(per_model)
        w = np.array([self.model_weights.get(m, 0.0) for m in keys]) if self.model_weights else np.ones(len(keys))
        if w.sum() <= 0:
            w = np.ones(len(keys))
        w = w / w.sum()
        return np.tensordot(w, np.stack([per_model[m] for m in keys]), axes=1)

    # ------------------------------------------------------------------ reference
    def fit_reference(self, human_docs, llm_docs_by_model: dict, selection_docs=None, human_groups=None):
        """Build P_H and P_M from token lists (or sets) per document. If selection_docs is given,
        vocabulary selection uses that (held-out) split and probabilities are estimated on
        human_docs/llm_docs_by_model, avoiding selection bias."""
        human_docs = [list(d) for d in human_docs]
        llm_docs_by_model = {m: [list(d) for d in docs] for m, docs in llm_docs_by_model.items()}
        sel_groups = None
        if self.selection == "stable":
            assert human_groups is not None and len(human_groups) == len(human_docs), "stable selection needs human_groups"
            human_groups = list(human_groups)
        if selection_docs is None and self.selection in ("ratio", "excess", "ratio_pos", "stable") and self.split_selection:
            # avoid selection bias: rank lemmas on one random half of the reference, estimate probabilities on the other
            def halves(docs):
                idx = self.rng.permutation(len(docs))
                return [docs[i] for i in idx[: len(docs) // 2]], [docs[i] for i in idx[len(docs) // 2:]]
            if human_groups is not None:
                idx = self.rng.permutation(len(human_docs)); half = idx[: len(idx) // 2]
                sel_groups = [human_groups[i] for i in half]
                s_h = [human_docs[i] for i in half]; human_docs = [human_docs[i] for i in idx[len(idx) // 2:]]
            else:
                s_h, human_docs = halves(human_docs)
            s_m, rest = {}, {}
            for m, docs in llm_docs_by_model.items():
                s_m[m], rest[m] = halves(docs)
            llm_docs_by_model = rest
            selection_docs = (s_h, s_m)
        n_h = len(human_docs)
        df_h = doc_frequency(human_docs)
        df_m_models = {m: doc_frequency(d) for m, d in llm_docs_by_model.items()}
        n_m_models = {m: len(d) for m, d in llm_docs_by_model.items()}

        if selection_docs is not None:
            s_h, s_m = selection_docs
            sel_df_h, sel_n_h = doc_frequency(s_h), len(s_h)
            sel_df_m = Counter()
            sel_n_m = 0
            for d in s_m.values():
                sel_df_m.update(doc_frequency(d))
                sel_n_m += len(d)
        else:
            sel_df_h, sel_n_h = df_h, n_h
            sel_df_m = Counter()
            sel_n_m = 0
            for m, c in df_m_models.items():
                sel_df_m.update(c)
                sel_n_m += n_m_models[m]

        cand = set(sel_df_h) | set(sel_df_m)
        n_tot = sel_n_h + sel_n_m
        cand = [w for w in cand if (sel_df_h[w] + sel_df_m[w]) / n_tot >= self.min_df]
        if self.selection in ("pos", "ratio_pos"):
            cand = [w for w in cand if "|" in w and w.rsplit("|", 1)[1] in self.pos_filter]
        if self.selection == "stable" and sel_groups is None:
            sel_groups = human_groups
        if self.selection in ("ratio", "excess", "ratio_pos", "stable"):
            ph = np.array([(sel_df_h[w] + self.smoothing) / (sel_n_h + 2 * self.smoothing) for w in cand])
            if self.model_weights:   # rank on the same weighted mixture that is estimated afterwards
                sel_models = selection_docs[1] if selection_docs is not None else llm_docs_by_model
                sel_pm = {m: (lambda c, n: np.array([(c[w] + self.smoothing) / (n + 2 * self.smoothing) for w in cand]))(doc_frequency(d), len(d))
                          for m, d in sel_models.items() if len(d)}
                pm = self._mix(sel_pm)
            else:
                pm = np.array([(sel_df_m[w] + self.smoothing) / (sel_n_m + 2 * self.smoothing) for w in cand])
            lr = np.log(pm / ph)
            score = np.abs(lr) if self.selection in ("ratio", "ratio_pos") else lr
            if self.selection == "stable":
                s_h_docs = selection_docs[0] if selection_docs is not None else human_docs
                grp = pd.Series(sel_groups)
                keep = grp.map(grp.value_counts()) >= self.min_group
                widx = {w: i for i, w in enumerate(cand)}
                lrg = []
                for g in grp[keep].unique():
                    docs_g = [s_h_docs[i] for i in np.flatnonzero((grp == g).values & keep.values)]
                    dfg = doc_frequency(docs_g)
                    phg = np.array([(dfg[w] + self.smoothing) / (len(docs_g) + 2 * self.smoothing) for w in cand])
                    lrg.append(np.log(pm / phg))
                if lrg:
                    L = np.vstack(lrg)
                    pos = np.quantile(L, self.stable_q, axis=0)          # LLM-overused in (almost) every group
                    neg = -np.quantile(L, 1 - self.stable_q, axis=0)     # LLM-underused in (almost) every group
                    score = np.maximum(np.maximum(pos, neg), 0.0)
            order = np.argsort(-score)[: self.top_k]
            cand = [cand[i] for i in order]
        vocab = sorted(cand)

        p_h = np.array([(df_h[w] + self.smoothing) / (n_h + 2 * self.smoothing) for w in vocab])
        per_model = {}
        for m in llm_docs_by_model:
            per_model[m] = np.array([(df_m_models[m][w] + self.smoothing) / (n_m_models[m] + 2 * self.smoothing)
                                     for w in vocab])
        p_m = self._mix(per_model)  # ensemble: weighted (model_weights) or equal-weight mixture of the generators
        ref = ReferenceDistribution(vocab, p_h, p_m, n_h, sum(n_m_models.values()), self.selection, per_model)

        if self.likelihood == "frequency":
            tc_h, N_h = token_counts(human_docs)
            ref.f_h = np.array([(tc_h[w] + self.smoothing) / (N_h + 2 * self.smoothing) for w in vocab])
            for m, docs in llm_docs_by_model.items():
                tc, N = token_counts(docs)
                ref.f_m[m] = np.array([(tc[w] + self.smoothing) / (N + 2 * self.smoothing) for w in vocab])
        elif self.likelihood == "length":
            ref.lam_h = calibrate_rates(p_h, np.array([max(len(d), 1) for d in human_docs]))
            for m, docs in llm_docs_by_model.items():
                ref.lam_m[m] = calibrate_rates(per_model[m], np.array([max(len(d), 1) for d in docs]))
        ref.len_h = np.array([max(len(d), 1) for d in human_docs])
        ref.len_m = {m: np.array([max(len(d), 1) for d in docs]) for m, docs in llm_docs_by_model.items()}
        self.ref = ref
        return self

    def adapt(self, human_docs, llm_docs_by_model: dict, kappa: float = 300.0, kappa_m: float | None = None):
        """Empirical-Bayes adaptation of a fitted (language-level) reference to a sub-corpus, e.g. one platform
        instance: the vocabulary is kept and the occurrence probabilities are re-estimated as
        (df + kappa * p_language) / (n + kappa), i.e. the language-level probabilities act as a prior worth
        kappa pseudo-documents, updated with the sub-corpus' own human documents (for P_H) and with the LLM
        documents seeded from it (for P_M, per generator; kappa_m defaults to kappa). kappa=0 gives a purely
        local reference, kappa=inf the language-level one. Returns a new estimator; the 'frequency'
        likelihood is not supported."""
        assert self.ref is not None, "call fit_reference first"
        if self.likelihood == "frequency":
            raise NotImplementedError("adapt() supports the occurrence and length likelihoods")
        import copy
        r = self.ref
        idx = self._index()
        km = kappa if kappa_m is None else kappa_m

        def counts(docs):
            c = np.zeros(len(r.vocab))
            for w, k in doc_frequency(docs).items():
                if w in idx:
                    c[idx[w]] = k
            return c

        human_docs = [list(d) for d in human_docs]
        n_h = len(human_docs)
        p_h = (counts(human_docs) + kappa * r.p_h) / (n_h + kappa) if np.isfinite(kappa) and (n_h + kappa) > 0 else r.p_h.copy()
        per_model, len_m, n_m = {}, {}, 0
        for m, p0 in r.per_model_p_m.items():
            docs = [list(d) for d in llm_docs_by_model.get(m, [])]
            n_m += len(docs)
            per_model[m] = (counts(docs) + km * p0) / (len(docs) + km) if np.isfinite(km) and (len(docs) + km) > 0 else p0.copy()
            len_m[m] = np.array([max(len(d), 1) for d in docs]) if len(docs) >= 50 else r.len_m.get(m)
        new = copy.copy(self)
        new.ref = ReferenceDistribution(list(r.vocab), p_h, self._mix(per_model), n_h, n_m, r.selection, per_model)
        new.ref.len_h = np.array([max(len(d), 1) for d in human_docs]) if n_h >= 50 else r.len_h
        new.ref.len_m = len_m
        if self.likelihood == "length":
            new.ref.lam_h = calibrate_rates(p_h, new.ref.len_h)
            for m in per_model:
                new.ref.lam_m[m] = calibrate_rates(per_model[m], len_m[m])
        return new

    # ------------------------------------------------------------------ estimation
    def _index(self):
        return {w: i for i, w in enumerate(self.ref.vocab)}

    def _rows(self, docs):
        """Per document: vocabulary indices of distinct tokens, of all tokens, and the content length."""
        idx = self._index()
        rows_set, rows_tok, lens = [], [], []
        for d in docs:
            d = list(d)
            rows_tok.append(np.fromiter((idx[w] for w in d if w in idx), dtype=np.int64))
            rows_set.append(np.unique(rows_tok[-1]))
            lens.append(max(len(d), 1))
        return rows_set, rows_tok, np.array(lens)

    def _p_components(self, model: str | None, L_bins: np.ndarray | None = None):
        r = self.ref
        if self.likelihood == "occurrence":
            return r.p_h, (r.per_model_p_m[model] if model else r.p_m)
        if self.likelihood == "frequency":
            f_m = r.f_m[model] if model else self._mix(r.f_m)
            return r.f_h, f_m
        # length: V x B matrices of occurrence probabilities at the bin lengths
        ph = 1 - np.exp(-np.outer(r.lam_h, L_bins))
        if model:
            pm = 1 - np.exp(-np.outer(r.lam_m[model], L_bins))
        else:
            pm = self._mix({m: 1 - np.exp(-np.outer(l, L_bins)) for m, l in r.lam_m.items()})
        return ph, pm

    def _nll(self, alpha: float, stats: dict, ph, pm) -> float:
        if self.likelihood == "occurrence":
            p = np.clip((1 - alpha) * ph + alpha * pm, EPS, 1 - EPS)
            n_w, n_docs = stats["n_w"], stats["n_docs"]
            return -float(np.sum(n_w * np.log(p) + (n_docs - n_w) * np.log(1 - p)))
        if self.likelihood == "frequency":
            p = np.clip((1 - alpha) * ph + alpha * pm, EPS, 1 - EPS)
            p_other = max(1 - p.sum(), EPS)
            return -float(np.sum(stats["c_w"] * np.log(p)) + stats["c_other"] * np.log(p_other))
        p = np.clip((1 - alpha) * ph + alpha * pm, EPS, 1 - EPS)          # V x B
        n_bw, N_b = stats["n_bw"], stats["N_b"]                            # V x B, B
        return -float(np.sum(n_bw * np.log(p) + (N_b[None, :] - n_bw) * np.log(1 - p)))

    def _stats(self, rows_set, rows_tok, lens, bins, sample=None):
        V = len(self.ref.vocab)
        idx = range(len(rows_set)) if sample is None else sample
        if self.likelihood == "occurrence":
            n_w = np.zeros(V)
            for i in idx:
                n_w[rows_set[i]] += 1
            return {"n_w": n_w, "n_docs": len(idx)}
        if self.likelihood == "frequency":
            c_w = np.zeros(V)
            tot = 0
            for i in idx:
                np.add.at(c_w, rows_tok[i], 1)
                tot += lens[i]
            return {"c_w": c_w, "c_other": max(tot - c_w.sum(), 0.0)}
        B = len(LENGTH_EDGES) - 1
        n_bw = np.zeros((V, B))
        N_b = np.zeros(B)
        for i in idx:
            n_bw[rows_set[i], bins[i]] += 1
            N_b[bins[i]] += 1
        return {"n_bw": n_bw, "N_b": N_b}

    def _fit(self, stats, ph, pm) -> float:
        res = minimize_scalar(self._nll, bounds=(0.0, 1.0), args=(stats, ph, pm), method="bounded",
                              options={"xatol": 1e-6})
        return float(res.x)

    def estimate_from_counts(self, n_w: np.ndarray, n_docs: int, p_m: np.ndarray | None = None) -> float:
        """Occurrence likelihood from aggregated indicator counts (kept for compatibility)."""
        p_m = self.ref.p_m if p_m is None else p_m
        return self._fit({"n_w": n_w, "n_docs": n_docs}, self.ref.p_h, p_m)

    def estimate(self, docs, n_boot: int = 0, ci: float = 0.95, model: str | None = None) -> dict:
        """Return alpha_hat and (optionally) a document-bootstrap percentile CI."""
        assert self.ref is not None, "call fit_reference first"
        rows_set, rows_tok, lens = self._rows(docs)
        n_docs = len(rows_set)
        bins = np.clip(np.searchsorted(LENGTH_EDGES, lens, side="right") - 1, 0, len(LENGTH_EDGES) - 2)
        L_bins = None
        if self.likelihood == "length":
            L_bins = np.array([lens[bins == b].mean() if np.any(bins == b) else (LENGTH_EDGES[b] + 1)
                               for b in range(len(LENGTH_EDGES) - 1)])
        ph, pm = self._p_components(model, L_bins)
        a = self._fit(self._stats(rows_set, rows_tok, lens, bins), ph, pm)
        out = {"alpha": a, "n_docs": n_docs, "vocab_size": len(self.ref.vocab), "likelihood": self.likelihood}
        if n_boot and n_docs > 1:
            boots = np.empty(n_boot)
            for b in range(n_boot):
                samp = self.rng.integers(0, n_docs, n_docs)
                boots[b] = self._fit(self._stats(rows_set, rows_tok, lens, bins, samp), ph, pm)
            lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
            out.update({"ci_lo": float(lo), "ci_hi": float(hi), "boot_sd": float(boots.std())})
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        import json
        r = self.ref
        json.dump({"vocab": r.vocab, "p_h": r.p_h.tolist(), "p_m": r.p_m.tolist(), "n_h": r.n_h, "n_m": r.n_m,
                   "selection": r.selection, "per_model": {m: v.tolist() for m, v in r.per_model_p_m.items()},
                   "f_h": None if r.f_h is None else r.f_h.tolist(), "f_m": {m: v.tolist() for m, v in r.f_m.items()},
                   "lam_h": None if r.lam_h is None else r.lam_h.tolist(),
                   "lam_m": {m: v.tolist() for m, v in r.lam_m.items()},
                   "params": {"selection": self.selection, "top_k": self.top_k, "min_df": self.min_df,
                              "smoothing": self.smoothing, "likelihood": self.likelihood,
                              "split_selection": self.split_selection}}, open(path, "w"))

    @classmethod
    def load(cls, path):
        import json
        d = json.load(open(path))
        est = cls(**d["params"])
        est.ref = ReferenceDistribution(d["vocab"], np.array(d["p_h"]), np.array(d["p_m"]), d["n_h"], d["n_m"],
                                        d["selection"], {m: np.array(v) for m, v in d["per_model"].items()},
                                        None if d.get("f_h") is None else np.array(d["f_h"]),
                                        {m: np.array(v) for m, v in d.get("f_m", {}).items()},
                                        None if d.get("lam_h") is None else np.array(d["lam_h"]),
                                        {m: np.array(v) for m, v in d.get("lam_m", {}).items()})
        return est


# ---------------------------------------------------------------------- baselines
def rogan_gladen(flag_rate: float, sens: float, spec: float, n: int | None = None,
                 n_sens: int | None = None, n_spec: int | None = None) -> dict:
    """Prevalence corrected for detector error: (p - (1-spec)) / (sens - (1-spec)),
    clipped to [0,1]. If counts are given, a delta-method variance is returned."""
    j = sens + spec - 1
    if j <= 0:
        return {"alpha": float("nan"), "se": float("nan")}
    a = (flag_rate - (1 - spec)) / j
    out = {"alpha": float(np.clip(a, 0, 1))}
    if n and n_sens and n_spec:
        var = (flag_rate * (1 - flag_rate) / n + a**2 * sens * (1 - sens) / n_sens
               + (1 - a) ** 2 * spec * (1 - spec) / n_spec) / j**2
        out["se"] = float(np.sqrt(var))
        out["ci_lo"], out["ci_hi"] = float(max(0, a - 1.96 * out["se"])), float(min(1, a + 1.96 * out["se"]))
    return out


def bayes_prevalence(k_flag: int, n: int, tp: int, n_pos: int, tn: int, n_neg: int,
                     grid: int = 401, draws: int = 4000, seed: int = 0) -> dict:
    """Posterior for prevalence pi with unknown sensitivity/specificity (Gelman & Carpenter 2020),
    computed by Monte Carlo over (sens, spec) ~ Beta posteriors from validation counts and a
    grid over pi with a uniform prior. Returns posterior mean and 95% credible interval."""
    rng = np.random.default_rng(seed)
    pis = np.linspace(0, 1, grid)
    sens = beta_dist(tp + 1, n_pos - tp + 1).rvs(draws, random_state=rng)
    spec = beta_dist(tn + 1, n_neg - tn + 1).rvs(draws, random_state=rng)
    # p(flag) = pi*sens + (1-pi)*(1-spec)
    p = pis[None, :] * sens[:, None] + (1 - pis[None, :]) * (1 - spec[:, None])
    p = np.clip(p, EPS, 1 - EPS)
    logp = k_flag * np.log(p) + (n - k_flag) * np.log(1 - p)
    logp -= logp.max()
    w = np.exp(logp)
    post = w.mean(axis=0)
    post /= post.sum()
    cdf = np.cumsum(post)
    mean = float(np.sum(pis * post))
    lo = float(pis[np.searchsorted(cdf, 0.025)])
    hi = float(pis[min(np.searchsorted(cdf, 0.975), grid - 1)])
    return {"alpha": mean, "ci_lo": lo, "ci_hi": hi}


def score_threshold_prevalence(scores_corpus: np.ndarray, scores_h: np.ndarray, scores_m: np.ndarray,
                               fpr_target: float = 0.05) -> dict:
    """Detector baseline: choose threshold at a target FPR on human reference scores, then
    compute raw flag rate, Rogan-Gladen correction and Bayesian estimate."""
    tau = float(np.quantile(scores_h, 1 - fpr_target))
    spec = float(np.mean(scores_h <= tau))
    sens = float(np.mean(scores_m > tau))
    flag = float(np.mean(scores_corpus > tau))
    rg = rogan_gladen(flag, sens, spec, len(scores_corpus), len(scores_m), len(scores_h))
    by = bayes_prevalence(int(np.sum(scores_corpus > tau)), len(scores_corpus),
                          int(np.sum(scores_m > tau)), len(scores_m), int(np.sum(scores_h <= tau)), len(scores_h))
    return {"tau": tau, "sens": sens, "spec": spec, "raw_flag_rate": flag,
            "rg_alpha": rg["alpha"], "rg_ci_lo": rg.get("ci_lo"), "rg_ci_hi": rg.get("ci_hi"),
            "bayes_alpha": by["alpha"], "bayes_ci_lo": by["ci_lo"], "bayes_ci_hi": by["ci_hi"]}


# ---------------------------------------------------------------------- toy test
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    V = 2000
    ph = rng.beta(0.5, 8, V)
    pm = ph * np.exp(rng.normal(0, 0.6, V))
    pm = np.clip(pm, 0, 0.95)

    def sample(p, n):
        return [set(np.where(rng.random(V) < p)[0].astype(str)) for _ in range(n)]

    H = sample(ph, 3000)
    M = {"m1": sample(pm, 1500), "m2": sample(np.clip(pm * np.exp(rng.normal(0, 0.2, V)), 0, 0.95), 1500)}
    for lik in LIKELIHOODS:
        est = DistributionalEstimator(selection="ratio", top_k=800, min_df=0.002, likelihood=lik).fit_reference(H, M)
        for a_true in [0.0, 0.05, 0.3]:
            n = 1000
            k = int(round(a_true * n))
            C = sample(ph, n - k) + sample(pm, k)
            r = est.estimate(C, n_boot=50)
            print(f"{lik:10s} true={a_true:.3f} est={r['alpha']:.3f} CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
    print(rogan_gladen(0.12, 0.8, 0.97, 1000, 500, 500))
    print(bayes_prevalence(120, 1000, 400, 500, 485, 500))
