"""Cross-country empirical study.

Steps
  1. For each language, fit the language-level reference: P_H on pre-ChatGPT human texts sub-sampled so
     that the platform composition matches that of the generation seeds (otherwise topic words of the
     seed platforms enter the vocabulary), P_M on the multi-generator LLM texts.
  2. Adapt the reference to every platform instance (and, for EU Have Your Say, to the author type:
     citizens vs organisations) by empirical-Bayes shrinkage: the language-level probabilities act as a
     prior worth kappa pseudo-documents, updated with the instance's own pre-ChatGPT texts (P_H) and with
     the LLM texts seeded from the instance (P_M). Instances with fewer than --min-ref pre-ChatGPT texts
     keep the language-level reference ("pooled").
  3. Null calibration per instance: the adaptation is re-fitted on one random half of the pre-ChatGPT texts
     and alpha is estimated on the other half, overall, by year and by type (empirical false-positive
     floor and topic-drift range), with the pooled reference for comparison.
  4. Estimate alpha with bootstrap CI for every cell: instance x language x type x period, instance x
     language x half-year, instance x language x year (aggregated to language x year by document weight).
  5. Interrupted time series per instance, multilevel model across instances, placebo break (Dec 2020),
     homogenisation metrics, single-generator references, pooled vs adapted comparison.

Outputs: results/alpha_cells_*.csv, its.csv, mixed_model.txt, placebo.txt, homogenisation.csv,
         null_calibration.csv, alpha_single_generator.csv, alpha_pooled_vs_adapted.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator import DistributionalEstimator  # noqa: E402

CHATGPT = pd.Timestamp("2022-11-30", tz="UTC")
HYS_CITIZEN = {"EU_CITIZEN", "NON_EU_CITIZEN"}


def to_sets(tokens_series):
    return [list(t) if t is not None else [] for t in tokens_series]


def half_year(ts: pd.Series) -> pd.Series:
    return ts.dt.year.astype(str) + "H" + ((ts.dt.month > 6).astype(int) + 1).astype(str)


def fold_of(doc_ids: pd.Series) -> pd.Series:
    return doc_ids.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 2)


def split_author_type(instance: pd.Series, user_type: pd.Series | None) -> pd.Series:
    """EU Have Your Say feedback is analysed separately for citizens and organisations (companies,
    business associations, NGOs, public authorities, academia, trade unions): the register of
    organisational submissions differs from citizen writing and would otherwise be read as LLM share."""
    if user_type is None:
        return instance
    ut = user_type.fillna("").astype(str)
    out = instance.copy()
    hys = instance == "eu_hys"
    out[hys & ut.isin(HYS_CITIZEN)] = "eu_hys_citizens"
    out[hys & ~ut.isin(HYS_CITIZEN) & (ut != "")] = "eu_hys_organisations"
    out[hys & (ut == "")] = "eu_hys_unknown"
    return out


def matched_reference(h0: pd.DataFrame, seeds: pd.Series, n_max: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Sub-sample human reference documents per instance in proportion to the LLM seeds' instance
    composition (instances holding 80% of the seed mass are never short of documents)."""
    rng = np.random.default_rng(seed)
    avail = h0.groupby("instance").size()
    share = (seeds / seeds.sum()).reindex(avail.index).fillna(0)
    ratios = (avail / share.replace(0, np.nan)).dropna().sort_values()
    if ratios.empty:
        return h0
    cum = share.reindex(ratios.index).cumsum()
    n_tot = int(min(n_max, ratios[cum >= 0.2].iloc[0] if (cum >= 0.2).any() else ratios.iloc[-1]))
    n_tot = max(n_tot, 2000)
    parts = []
    for inst, a in avail.items():
        k = min(a, int(round(n_tot * share.get(inst, 0))))
        if k > 0:
            parts.append(h0[h0.instance == inst].sample(k, random_state=int(rng.integers(1e9))))
    return pd.concat(parts) if parts else h0


def llm_by_model(g: pd.DataFrame, min_docs: int = 0) -> dict:
    d = {m: to_sets(x.tokens) for m, x in g.groupby("model")}
    return {m: x for m, x in d.items() if len(x) >= min_docs}


class References:
    """Language-level references and their per-instance adaptations."""

    def __init__(self, H: pd.DataFrame, G: pd.DataFrame, S: pd.DataFrame, a):
        self.a = a
        self.base: dict[str, DistributionalEstimator] = {}
        self.inst: dict[tuple[str, str], DistributionalEstimator] = {}
        self.pooled: set[tuple[str, str]] = set()
        pre = H[H.period == "pre"]
        # generator weights in P_M (market shares); generators without a weight are left out of the reference
        self.weights = getattr(a, "weights", None) or None
        if self.weights:
            G = G[G.model.isin(self.weights)]
        for lang in sorted(set(pre.lang) & set(G.lang)):
            h = pre[pre.lang == lang]
            by_model = llm_by_model(G[G.lang == lang], 200)
            if len(h) < 300 or not by_model:
                continue
            hm = matched_reference(h, S[S.lang == lang].groupby("instance").size())
            est = DistributionalEstimator(selection=a.selection, top_k=a.top_k, min_df=a.min_df, likelihood=a.likelihood,
                                          model_weights=self.weights)
            est.fit_reference(to_sets(hm.tokens), by_model)
            self.base[lang] = est
            print(f"[{lang}] language reference: {len(hm)} human docs (of {len(h)}), "
                  f"{sum(len(v) for v in by_model.values())} LLM docs, vocab {len(est.ref.vocab)}", flush=True)
            for inst, hi in h.groupby("instance"):
                gi = G[(G.lang == lang) & (G.inst == inst) & G["mode"].isin(["rewrite", "generate"])]
                if len(hi) >= a.min_ref:
                    self.inst[(inst, lang)] = est.adapt(to_sets(hi.tokens), llm_by_model(gi), kappa=a.kappa)
                else:
                    self.inst[(inst, lang)] = est
                    self.pooled.add((inst, lang))

    def get(self, inst: str, lang: str):
        return self.inst.get((inst, lang), self.base.get(lang))


def estimate_cells(df: pd.DataFrame, refs: References, min_docs: int, n_boot: int, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols):
        key = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        est = refs.get(key.get("instance"), key.get("lang"))
        if est is None or len(g) < min_docs:
            continue
        r = est.estimate(to_sets(g.tokens), n_boot=n_boot)
        rows.append({**key, "n_docs": len(g), "alpha": r["alpha"], "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi"),
                     "pooled_ref": (key.get("instance"), key.get("lang")) in refs.pooled,
                     "mean_words": g.n_words.mean(), "median_words": g.n_words.median(),
                     "n_authors": g.author_hash[g.author_hash != ""].nunique() if "author_hash" in g else np.nan,
                     "campaign_share": g.is_campaign.mean() if "is_campaign" in g else np.nan})
    return pd.DataFrame(rows)


def its_per_instance(cells: pd.DataFrame) -> pd.DataFrame:
    """Segmented regression of alpha on time with a level shift and slope change at the break."""
    import statsmodels.api as sm
    out = []
    for (inst, lang), g in cells.groupby(["instance", "lang"]):
        g = g.dropna(subset=["alpha"]).sort_values("t")
        if g.t.nunique() < 4 or (g.post == 1).sum() < 1 or (g.post == 0).sum() < 2:
            continue
        X = pd.DataFrame({"const": 1.0, "t": g.t, "post": g.post, "t_post": g.t * g.post})
        w = g.n_docs.values
        m = sm.WLS(g.alpha.values, X, weights=w).fit()
        out.append(dict(instance=inst, lang=lang, n_cells=len(g), level_shift=m.params["post"],
                        level_p=m.pvalues["post"], slope_change=m.params["t_post"], slope_p=m.pvalues["t_post"],
                        pre_mean=g[g.post == 0].alpha.mean(), post_mean=g[g.post == 1].alpha.mean()))
    return pd.DataFrame(out)


def mixed_model(cells: pd.DataFrame, formula: str = "alpha ~ post + C(lang) + C(type)") -> str:
    import statsmodels.formula.api as smf
    d = cells.dropna(subset=["alpha"]).copy()
    d = d[d.n_docs >= 1]
    try:
        m = smf.mixedlm(formula, d, groups=d["instance"]).fit(reml=True)
        return m.summary().as_text()
    except Exception as e:  # noqa: BLE001
        return f"MixedLM failed: {e}"


def homogenisation(df: pd.DataFrame, emb: pd.DataFrame | None, group_cols: list[str], min_docs: int = 50,
                   max_pairs: int = 20000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    E = None
    if emb is not None:
        E = emb.set_index("doc_id")["emb"]
    for key, g in df.groupby(group_cols):
        if len(g) < min_docs:
            continue
        key = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        # distinct-2 over lemma tokens
        bigrams, total = set(), 0
        for toks in g.tokens:
            t = [x.split("|")[0] for x in (toks if toks is not None else [])]
            total += max(len(t) - 1, 0)
            bigrams.update(zip(t[:-1], t[1:]))
        d2 = len(bigrams) / total if total else np.nan
        cos = np.nan
        if E is not None:
            ids = [i for i in g.doc_id if i in E.index]
            if len(ids) >= min_docs:
                M = np.stack(E.loc[ids].values)
                n = len(ids)
                npairs = min(max_pairs, n * (n - 1) // 2)
                a = rng.integers(0, n, npairs)
                b = rng.integers(0, n, npairs)
                ok = a != b
                cos = float(np.mean(np.sum(M[a[ok]] * M[b[ok]], axis=1)))
        rows.append({**key, "n_docs": len(g), "distinct2": d2, "mean_cos": cos, "mean_words": g.n_words.mean(),
                     "sd_words": g.n_words.std()})
    return pd.DataFrame(rows)


def null_calibration(H: pd.DataFrame, G: pd.DataFrame, S: pd.DataFrame, a) -> pd.DataFrame:
    """Held-out pre-ChatGPT slices: language reference and adaptations re-fitted on fold 0, alpha on fold 1."""
    pre = H[H.period == "pre"].copy()
    pre["fold"] = fold_of(pre.doc_id)
    pre["year"] = pre.created_at.dt.year
    rows = []
    weights = getattr(a, "weights", None) or None
    if weights:   # same generator set and weights as the study references
        G = G[G.model.isin(weights)]
    for lang in sorted(set(pre.lang) & set(G.lang)):
        h = pre[pre.lang == lang]
        by_model = llm_by_model(G[G.lang == lang], 200)
        if (h.fold == 0).sum() < 300 or not by_model:
            continue
        hm = matched_reference(h[h.fold == 0], S[S.lang == lang].groupby("instance").size())
        base = DistributionalEstimator(selection=a.selection, top_k=a.top_k, min_df=a.min_df, likelihood=a.likelihood,
                                       model_weights=weights)
        base.fit_reference(to_sets(hm.tokens), by_model)
        test_all = h[h.fold == 1]
        if len(test_all) >= a.min_docs:
            r = base.estimate(to_sets(test_all.tokens), n_boot=a.n_boot)
            rows.append(dict(lang=lang, instance="ALL", slice="fold1", reference="pooled", n_docs=len(test_all),
                             alpha=r["alpha"], ci_lo=r.get("ci_lo"), ci_hi=r.get("ci_hi")))
        for inst, hi in h.groupby("instance"):
            fit, test = hi[hi.fold == 0], hi[hi.fold == 1]
            if len(test) < a.min_docs:
                continue
            gi = G[(G.lang == lang) & (G.inst == inst) & G["mode"].isin(["rewrite", "generate"])]
            ests = {"pooled": base}
            if len(fit) >= a.min_ref // 2:
                ests["adapted"] = base.adapt(to_sets(fit.tokens), llm_by_model(gi), kappa=a.kappa)
            for ref, e in ests.items():
                r = e.estimate(to_sets(test.tokens), n_boot=a.n_boot)
                rows.append(dict(lang=lang, instance=inst, slice="fold1", reference=ref, n_docs=len(test),
                                 alpha=r["alpha"], ci_lo=r.get("ci_lo"), ci_hi=r.get("ci_hi")))
                for y, gy in test.groupby("year"):
                    if len(gy) >= a.min_docs:
                        rows.append(dict(lang=lang, instance=inst, slice=f"year={y}", reference=ref, n_docs=len(gy),
                                         alpha=e.estimate(to_sets(gy.tokens), n_boot=0)["alpha"]))
                for ty, gt in test.groupby("type"):
                    if len(gt) >= a.min_docs:
                        rows.append(dict(lang=lang, instance=inst, slice=f"type={ty}", reference=ref, n_docs=len(gt),
                                         alpha=e.estimate(to_sets(gt.tokens), n_boot=0)["alpha"]))
                late = test[test.created_at >= pd.Timestamp("2021-01-01", tz="UTC")]
                if len(late) >= a.min_docs:
                    rows.append(dict(lang=lang, instance=inst, slice="2021-22", reference=ref, n_docs=len(late),
                                     alpha=e.estimate(to_sets(late.tokens), n_boot=0)["alpha"]))
        print(f"[{lang}] null calibration done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--sample", default="data/generated/human_sample.parquet", help="generation seeds (instance composition)")
    ap.add_argument("--campaigns", default="")
    ap.add_argument("--emb", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", default="ratio")
    ap.add_argument("--likelihood", default="occurrence", help="occurrence | length")
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--min-df", type=float, default=0.003)
    ap.add_argument("--min-docs", type=int, default=150)
    ap.add_argument("--min-ref", type=int, default=300, help="pre-ChatGPT texts an instance needs for its own adapted reference")
    ap.add_argument("--kappa", type=float, default=1000.0, help="prior weight (pseudo-documents) of the language reference")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--weights", default="", help="YAML {generator: weight}: market-share-weighted P_M restricted to these generators "
                                                 "(default: equal-weight mixture of all generators)")
    ap.add_argument("--exclude-models", nargs="*", default=[], help="generators left out of the reference (e.g. legacy models kept for the benchmark only)")
    a = ap.parse_args()
    if a.weights:
        import yaml
        a.weights = {str(k): float(v) for k, v in yaml.safe_load(open(a.weights, encoding="utf-8"))["weights"].items()}
        print("reference generators and weights:", a.weights, flush=True)
    else:
        a.weights = None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H = pd.concat([pd.read_parquet(p) for p in a.processed], ignore_index=True)
    if "doc_id" not in H.columns:
        H["doc_id"] = H["instance"] + ":" + H["type"] + ":" + H["id"].astype(str) + ":" + H["lang"]
    H = H[~H.is_dup_exact].drop_duplicates("doc_id").copy()
    H["instance_raw"] = H.instance
    H["instance"] = split_author_type(H.instance, H["user_type"] if "user_type" in H.columns else None)
    if a.campaigns:
        C = pd.read_parquet(a.campaigns)[["doc_id", "is_campaign", "cluster_id"]].drop_duplicates("doc_id")
        H = H.merge(C, on="doc_id", how="left")
        H["is_campaign"] = H["is_campaign"].fillna(False)
    G = pd.read_parquet(a.gen)
    if a.exclude_models:
        G = G[~G.model.isin(a.exclude_models)]
    seed_inst = H.set_index("doc_id")["instance"]
    G["inst"] = G.doc_id.map(seed_inst).fillna(G.doc_id.str.split(":").str[0])  # seed's instance (HYS split by author type)
    S = pd.read_parquet(a.sample, columns=["doc_id", "lang", "instance"])
    S["instance"] = S.doc_id.map(seed_inst).fillna(S.instance)

    refs = References(H, G, S, a)
    print("languages with references:", list(refs.base), "| pooled instance-language pairs:", len(refs.pooled), flush=True)
    D = H.copy()
    D["half"] = half_year(D.created_at)
    D["year"] = D.created_at.dt.year
    D["post"] = (D.created_at >= CHATGPT).astype(int)
    D["t"] = (D.created_at - CHATGPT).dt.days / 365.25

    cells_half = estimate_cells(D, refs, a.min_docs, a.n_boot, ["instance", "lang", "half"])
    cells_half["t"] = cells_half["half"].map(lambda h: int(h[:4]) + (0.25 if h.endswith("1") else 0.75)) - 2022.91
    cells_half["post"] = (cells_half.t > 0).astype(int)
    cells_half.to_csv(out / "alpha_cells_half.csv", index=False)

    cells_period = estimate_cells(D, refs, a.min_docs, a.n_boot, ["instance", "lang", "type", "period"])
    cells_period.to_csv(out / "alpha_cells_period.csv", index=False)

    cells_iy = estimate_cells(D, refs, a.min_docs, a.n_boot, ["instance", "lang", "year"])
    cells_iy.to_csv(out / "alpha_cells_instance_year.csv", index=False)
    w = cells_iy.assign(wa=cells_iy.alpha * cells_iy.n_docs, wlo=cells_iy.ci_lo * cells_iy.n_docs, whi=cells_iy.ci_hi * cells_iy.n_docs)
    ly = w.groupby(["lang", "year"]).agg(n_docs=("n_docs", "sum"), wa=("wa", "sum"), wlo=("wlo", "sum"), whi=("whi", "sum"),
                                         mean_words=("mean_words", "mean"), n_instances=("instance", "nunique")).reset_index()
    ly["alpha"], ly["ci_lo"], ly["ci_hi"] = ly.wa / ly.n_docs, ly.wlo / ly.n_docs, ly.whi / ly.n_docs
    ly.drop(columns=["wa", "wlo", "whi"]).to_csv(out / "alpha_lang_year.csv", index=False)

    if "is_campaign" in D:
        cells_camp = estimate_cells(D[D.post == 1], refs, a.min_docs, a.n_boot, ["instance", "lang", "is_campaign"])
        cells_camp.to_csv(out / "alpha_campaign.csv", index=False)

    its_per_instance(cells_half).to_csv(out / "its.csv", index=False)
    with open(out / "mixed_model.txt", "w") as f:
        f.write(mixed_model(cells_period.assign(post=(cells_period.period == "post").astype(int))))
        f.write("\n\n=== half-year cells ===\n")
        f.write(mixed_model(cells_half, "alpha ~ post + t + C(lang)"))
    pl = cells_half[cells_half.post == 0].copy()   # placebo: pretend break at 2020-12 using only pre-ChatGPT cells
    pl["post"] = (pl.t > -2.0).astype(int)
    with open(out / "placebo.txt", "w") as f:
        f.write(mixed_model(pl, "alpha ~ post + t + C(lang)"))
    emb = pd.read_parquet(a.emb) if a.emb else None
    homogenisation(D, emb, ["instance", "lang", "period"]).to_csv(out / "homogenisation.csv", index=False)

    null_calibration(H, G, S, a).to_csv(out / "null_calibration.csv", index=False)

    # robustness: single-generator references (adapted) on pre/post cells
    rows = []
    for (inst, lang), g in D.groupby(["instance", "lang"]):
        e = refs.get(inst, lang)
        if e is None:
            continue
        for m in sorted(e.ref.per_model_p_m):
            for period, gp in g.groupby("period"):
                if len(gp) >= a.min_docs:
                    rows.append(dict(model=m, lang=lang, instance=inst, period=period, n_docs=len(gp),
                                     alpha=e.estimate(to_sets(gp.tokens), n_boot=0, model=m)["alpha"]))
    pd.DataFrame(rows).to_csv(out / "alpha_single_generator.csv", index=False)

    # robustness: pooled (language-level) vs adapted reference on the same cells
    rows = []
    for (inst, lang), g in D.groupby(["instance", "lang"]):
        if (inst, lang) in refs.pooled or lang not in refs.base:
            continue
        for period, gp in g.groupby("period"):
            if len(gp) >= a.min_docs:
                sets = to_sets(gp.tokens)
                rows.append(dict(lang=lang, instance=inst, period=period, n_docs=len(gp),
                                 alpha_pooled=refs.base[lang].estimate(sets, n_boot=0)["alpha"],
                                 alpha_adapted=refs.get(inst, lang).estimate(sets, n_boot=0)["alpha"]))
    pd.DataFrame(rows).to_csv(out / "alpha_pooled_vs_adapted.csv", index=False)
    print(cells_period.to_string())


if __name__ == "__main__":
    main()
