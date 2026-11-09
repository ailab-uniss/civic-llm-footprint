"""Model-free lower bound on LLM-assisted writing by excess vocabulary (Kobak et al. 2024), adapted to civic corpora.

For each instance x language with enough documents per year, the document frequency of every content lemma is
tracked by calendar year. For a post-ChatGPT year y, the counterfactual frequency q_w(y) is the linear extrapolation
of the pre-ChatGPT years (at most the last `--trend-years` of them); the excess is d_w = p_w(y) - q_w(y) and the
ratio r_w = p_w / q_w. Excess words are selected on one half of the documents (fold A, by hash of the identifier)
and the bound is measured on the other half (fold B), as in the Korean adaptation of the method: the share of
fold-B documents of year y that contain at least one excess word, minus the counterfactual share of that indicator
extrapolated from the pre-ChatGPT years. This is a lower bound on the share of LLM-processed documents in that
year: documents whose LLM rewriting introduced none of the selected words are not counted.

A background-noise level is obtained by running exactly the same procedure with a pre-ChatGPT year treated as
"post" (its counterfactual extrapolated from the years before it): the bounds obtained for such years are topic
drift, not LLM use, and are reported next to the post-ChatGPT bounds.

Outputs: <out>/excess.csv (one row per instance, lang, year, kind=post|placebo) and <out>/excess_words.csv (the
excess words per instance-year with their frequencies, for inspection).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # allow `python src/civic/excess.py`

CHATGPT_YEAR = 2023
CONTENT_POS = {"ADJ", "ADV", "VERB", "NOUN", "PROPN"}


def fold_of(doc_id: str) -> int:
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 2


def content_set(tokens) -> frozenset:
    return frozenset(t for t in tokens if "|" in t and t.rsplit("|", 1)[1] in CONTENT_POS)


def doc_freq(sets: list[frozenset]) -> Counter:
    c = Counter()
    for s in sets:
        c.update(s)
    return c


MAX_HORIZON = 2   # years beyond the last trend year up to which the linear trend is extrapolated (then held constant)


def extrapolate(years: list[int], values: np.ndarray, target: int) -> np.ndarray:
    """Linear extrapolation (least squares over the given years) of each column to the target year, at most
    MAX_HORIZON years beyond the last trend year (as in Kobak et al., who extrapolate two years); with a single
    year, its value. Clipped to [0, 1]."""
    if len(years) == 1:
        return np.clip(values[0], 0, 1)
    x = np.array(years, dtype=float)
    xm = x.mean()
    slope = ((x - xm)[:, None] * (values - values.mean(axis=0))).sum(axis=0) / ((x - xm) ** 2).sum()
    t = min(target, max(years) + MAX_HORIZON)
    return np.clip(values.mean(axis=0) + slope * (t - xm), 0, 1)


def bound_for_year(by_year: dict[int, tuple[list[frozenset], list[frozenset]]], target: int, trend_years: int,
                   min_gap: float, min_ratio: float, z_min: float, n_boot: int, rng: np.random.Generator):
    """Excess-vocabulary bound for `target`, using the years < target as the trend. by_year[y] = (fold A, fold B)."""
    pre_years = sorted(y for y in by_year if y < min(target, CHATGPT_YEAR))[-trend_years:]   # trend: pre-ChatGPT years only
    if not pre_years:
        return None
    A_t, B_t = by_year[target]
    # --- selection on fold A
    vocab = Counter()
    for y in pre_years:
        vocab.update(doc_freq(by_year[y][0]))
    vocab.update(doc_freq(A_t))
    n_pre_A = sum(len(by_year[y][0]) for y in pre_years)
    words = [w for w, c in vocab.items() if c >= 5]
    idx = {w: i for i, w in enumerate(words)}
    P = np.zeros((len(pre_years), len(words)))
    for i, y in enumerate(pre_years):
        c = doc_freq(by_year[y][0])
        for w, k in c.items():
            if w in idx:
                P[i, idx[w]] = k / len(by_year[y][0])
    c = doc_freq(A_t)
    p_t = np.zeros(len(words))
    for w, k in c.items():
        if w in idx:
            p_t[idx[w]] = k / len(A_t)
    q_t = extrapolate(pre_years, P, target)
    floor = 0.5 / max(n_pre_A, 1)                       # pseudo-count so that unseen words get a finite ratio
    gap = p_t - q_t
    ratio = (p_t + floor) / (q_t + floor)
    # sampling error of the gap (binomial on the target year and on the pre years used for the trend)
    n_pre_mean = n_pre_A / len(pre_years)
    se = np.sqrt(p_t * (1 - p_t) / max(len(A_t), 1) + q_t * (1 - q_t) / max(n_pre_mean, 1) + 1e-12)
    # stability: a candidate must be a general word, present in every trend year at a comparable rate; words that
    # appear in bursts (the topic of one year's consultations) are excluded
    stable = (P.min(axis=0) >= 0.3 * P.mean(axis=0)) & (P.mean(axis=0) >= 0.002)
    cand = np.where((gap >= min_gap) & (ratio >= min_ratio) & (gap / se >= z_min) & stable)[0]
    cand = cand[np.argsort(-gap[cand])]
    # greedy choice, on fold A, of the subset of candidate words that maximises the bound (any valid subset gives a
    # lower bound; the choice on fold A keeps the measurement on fold B unbiased)
    def any_share(sets, wset):
        return np.array([1.0 if (s & wset) else 0.0 for s in sets])
    ind_A_pre = {y: np.zeros(len(by_year[y][0])) for y in pre_years}
    ind_A_t = np.zeros(len(A_t))
    best, excess_set = -1.0, set()
    for i in cand[:400]:
        w = words[i]
        trial_pre = {y: np.maximum(ind_A_pre[y], any_share(by_year[y][0], {w})) for y in pre_years}
        trial_t = np.maximum(ind_A_t, any_share(A_t, {w}))
        # conservative counterfactual for the selection step: the larger of the pre-year mean and the trend
        # extrapolation (the trend alone over-fits sampling noise when words are added one by one)
        means = np.array([[trial_pre[y].mean()] for y in pre_years])
        qa = max(float(extrapolate(pre_years, means, target)[0]), float(means.mean()))
        b = float(trial_t.mean()) - qa
        if b > best + 0.005:
            best, ind_A_pre, ind_A_t = b, trial_pre, trial_t
            excess_set.add(w)
    excess = sorted(excess_set)
    # --- measurement on fold B: share of documents containing at least one excess word
    def any_share(sets, wset=excess_set):   # noqa: F811
        return np.array([1.0 if (s & wset) else 0.0 for s in sets])
    ind_pre = {y: any_share(by_year[y][1]) for y in pre_years}
    ind_t = any_share(B_t)
    pre_vals = np.array([[v.mean()] for y, v in ind_pre.items()])
    q_any = float(extrapolate(pre_years, pre_vals, target)[0])
    p_any = float(ind_t.mean())
    bound = p_any - q_any
    boots = []
    for _ in range(n_boot):
        pv = np.array([[rng.choice(v, len(v)).mean()] for v in ind_pre.values()])
        qb = float(extrapolate(pre_years, pv, target)[0])
        pb = float(rng.choice(ind_t, len(ind_t)).mean())
        boots.append(pb - qb)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    top = sorted(((w, p_t[idx[w]], q_t[idx[w]]) for w in excess), key=lambda t: -(t[1] - t[2]))[:60]
    return dict(year=target, trend_years=",".join(map(str, pre_years)), n_docs=len(A_t) + len(B_t), n_fold_b=len(B_t),
                n_excess_words=len(excess), p_any=p_any, q_any=q_any, bound=bound, ci_lo=lo, ci_hi=hi), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-docs-year", type=int, default=300, help="documents a year needs (both folds) to be used")
    ap.add_argument("--min-pre-years", type=int, default=2)
    ap.add_argument("--trend-years", type=int, default=3, help="pre years used for the linear counterfactual")
    ap.add_argument("--min-gap", type=float, default=0.01, help="excess word: document-frequency gap p - q >= this")
    ap.add_argument("--min-ratio", type=float, default=1.5, help="excess word: ratio p / q >= this")
    ap.add_argument("--z-min", type=float, default=3.0, help="excess word: gap at least this many standard errors")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos", nargs="+", default=["ADJ", "ADV", "VERB"],
                    help="POS tags of the candidate lemmas; nouns and proper nouns carry the topic of the year's "
                         "consultations and are excluded by default (civic corpora change topic every year)")
    a = ap.parse_args()
    CONTENT_POS.clear()
    CONTENT_POS.update(a.pos)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    from civic.analysis import split_author_type   # HYS citizens / organisations split
    frames = []
    for p in a.processed:
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(p).schema_arrow.names
        d = pd.read_parquet(p, columns=[c for c in ["instance", "type", "id", "lang", "created_at", "is_dup_exact", "tokens", "user_type"] if c in cols])
        d = d[~d.is_dup_exact]
        d["doc_id"] = d.instance + ":" + d.type + ":" + d.id.astype(str) + ":" + d.lang
        d["instance"] = split_author_type(d.instance, d["user_type"] if "user_type" in d.columns else None)
        d["year"] = d.created_at.dt.year
        ok = d.groupby(["instance", "lang", "year"]).size()
        ok = ok[ok >= a.min_docs_year].reset_index()[["instance", "lang", "year"]]
        d = d.merge(ok, on=["instance", "lang", "year"])
        frames.append(d[["doc_id", "instance", "lang", "year", "tokens"]])
    D = pd.concat(frames, ignore_index=True)
    D["fold"] = D.doc_id.map(fold_of)
    rows, word_rows = [], []
    for (inst, lang), g in D.groupby(["instance", "lang"]):
        years = sorted(g.year.unique())
        pre = [y for y in years if y < CHATGPT_YEAR]
        post = [y for y in years if y >= CHATGPT_YEAR]
        if len(pre) < a.min_pre_years or not post:
            continue
        by_year = {}
        for y, gy in g.groupby("year"):
            sets = [content_set(t) for t in gy.tokens]
            f = gy.fold.values
            by_year[y] = ([s for s, k in zip(sets, f) if k == 0], [s for s, k in zip(sets, f) if k == 1])
        print(f"[{inst} {lang}] years {years}", flush=True)
        for y in post:
            r = bound_for_year(by_year, y, a.trend_years, a.min_gap, a.min_ratio, a.z_min, a.n_boot, rng)
            if r:
                rows.append(dict(instance=inst, lang=lang, kind="post", **r[0]))
                word_rows += [dict(instance=inst, lang=lang, year=y, word=w, p=p, q=q) for w, p, q in r[1]]
        # background noise: each pre year with at least min_pre_years earlier years, treated as "post"
        for y in pre:
            if len([z for z in pre if z < y]) >= a.min_pre_years:
                r = bound_for_year({z: v for z, v in by_year.items() if z <= y}, y, a.trend_years, a.min_gap, a.min_ratio, a.z_min, a.n_boot, rng)
                if r:
                    rows.append(dict(instance=inst, lang=lang, kind="placebo", **r[0]))   # not "null": pandas reads it as NaN
        sub = pd.DataFrame([r for r in rows if r["instance"] == inst and r["lang"] == lang])
        print(sub[["kind", "year", "n_docs", "n_excess_words", "p_any", "q_any", "bound", "ci_lo", "ci_hi"]].round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(out / "excess.csv", index=False)
    pd.DataFrame(word_rows).to_csv(out / "excess_words.csv", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
