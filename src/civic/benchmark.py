"""Validation benchmark: synthetic mixtures with known alpha, per language, leave-one-model-out.

Inputs
  --human   processed parquet(s) of pre-ChatGPT human texts with 'tokens' (list of 'lemma|POS'), 'lang', 'doc_id'
  --gen     processed parquet of generated texts with 'tokens', 'lang', 'model', 'mode', 'doc_id'
  --scores  optional parquet(s) with per-document detector scores (doc_id, score_<det>) for human and generated docs

Protocol (per language)
  1. Split human docs 50/50 into REF (reference distribution) and TEST (mixture pool), by doc_id hash.
     Generated docs inherit the split of their source doc (rewrites of REF docs go to REF, etc.),
     so no test text (or its LLM rewrite) is ever used to build the reference distribution.
  2. Leave-one-model-out: for each held-out model m*, P_M is built from the other models;
     test mixtures use LLM texts from m* only.
  3. For alpha in ALPHAS and R repetitions: draw n docs = (1-alpha) human TEST + alpha LLM(m*) docs,
     estimate alpha_hat with CI; record error.
  Scenarios: mode subsets (rewrite/generate/translate/all), short texts, vocabulary selection strategies.

Outputs: long-format parquet of results + summary CSV (MAE, bias, coverage, RMSE per language x scenario).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator import DistributionalEstimator, score_threshold_prevalence  # noqa: E402

ALPHAS = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50]


def split_key(doc_id: str) -> int:
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 2


def to_sets(tokens_series) -> list[list[str]]:
    """Token lists per document (duplicates kept: the length and frequency likelihoods need them)."""
    return [list(t) if t is not None else [] for t in tokens_series]


HELDOUT_ONLY: set[str] = set()   # generators used only as held-out test generators (set from --heldout-only)


def run_language(lang: str, H: pd.DataFrame, G: pd.DataFrame, scores: pd.DataFrame | None, n_mix: int, reps: int,
                 n_boot: int, rng: np.random.Generator, selections: list[str], modes_scenarios: dict,
                 short_max: int = 80, likelihoods: tuple[str, ...] = ("occurrence",), fixed_ref: bool = False) -> list[dict]:
    """fixed_ref: build the reference from every mode whatever the scenario (the scenario then selects the test texts only)."""
    res = []
    H = H.copy()
    H["split"] = H.doc_id.map(split_key)
    G = G.copy()
    # generated texts inherit the split of their target-language seed (first part of the composite id)
    seed_id = G["src_doc_id"] if "src_doc_id" in G.columns else G.doc_id.str.split("|").str[0]
    G["split"] = seed_id.map(split_key)
    models = sorted(G.model.unique())
    H_ref, H_test = H[H.split == 0], H[H.split == 1]
    if len(H_ref) < 200 or len(H_test) < 200:
        print(f"[{lang}] too few human docs ({len(H)}); skipping")
        return res
    h_ref_sets = to_sets(H_ref.tokens)
    h_test_sets = to_sets(H_test.tokens)
    configs = [(sel, "occurrence") for sel in selections] + [(selections[0], l) for l in likelihoods if l != "occurrence"]
    for sel, lik in configs:
        for scen, modes in modes_scenarios.items():
            Gs = G[G["mode"].isin(modes)]
            Gref = G if fixed_ref else Gs
            for m_star in models:
                # HELDOUT_ONLY generators (e.g. legacy commercial models) are only ever held out, never part of P_M
                ref_models = [m for m in models if m != m_star and m not in HELDOUT_ONLY]
                if not ref_models:
                    ref_models = [m_star]
                llm_ref = {m: to_sets(Gref[(Gref.model == m) & (Gref.split == 0)].tokens) for m in ref_models}
                llm_ref = {m: d for m, d in llm_ref.items() if len(d) >= 100}
                if not llm_ref:
                    continue
                est = DistributionalEstimator(selection=sel, top_k=1000, min_df=0.003, likelihood=lik)
                est.fit_reference(h_ref_sets, llm_ref)
                pool_llm = Gs[(Gs.model == m_star) & (Gs.split == 1)]
                if scen == "short":
                    pool_llm = pool_llm[pool_llm.n_words <= short_max]
                    pool_h = H_test[H_test.n_words <= short_max]
                    h_pool_sets = to_sets(pool_h.tokens)
                else:
                    pool_h = H_test
                    h_pool_sets = h_test_sets
                llm_pool_sets = to_sets(pool_llm.tokens)
                if len(llm_pool_sets) < 50 or len(h_pool_sets) < 100:
                    continue
                for alpha in ALPHAS:
                    for r in range(reps):
                        n = min(n_mix, len(h_pool_sets), int(len(llm_pool_sets) / max(alpha, 0.5)))
                        k = int(round(alpha * n))
                        hi = rng.choice(len(h_pool_sets), n - k, replace=False)
                        li = rng.choice(len(llm_pool_sets), k, replace=False) if k else []
                        docs = [h_pool_sets[i] for i in hi] + [llm_pool_sets[i] for i in li]
                        out = est.estimate(docs, n_boot=n_boot)
                        row = dict(lang=lang, selection=sel, scenario=scen, heldout_model=m_star, alpha=alpha, rep=r,
                                   n=n, alpha_hat=out["alpha"], ci_lo=out.get("ci_lo"), ci_hi=out.get("ci_hi"),
                                   vocab=out["vocab_size"], method="distributional", likelihood=lik)
                        res.append(row)
                        # detector baselines on the same mixture
                        if scores is not None and sel == selections[0] and lik == "occurrence":
                            h_ids = pool_h.doc_id.values[hi]
                            l_ids = pool_llm.doc_id.values[li] if k else np.array([])
                            for det in [c for c in scores.columns if c.startswith("score_")]:
                                s = scores[det]
                                sc = s.reindex(np.concatenate([h_ids, l_ids])).dropna().values
                                s_h = s.reindex(H_ref.doc_id).dropna().values
                                s_m = s.reindex(Gs[(Gs.split == 0) & (Gs.model.isin(ref_models))].doc_id).dropna().values
                                if len(sc) < 50 or len(s_h) < 100 or len(s_m) < 100:
                                    continue
                                b = score_threshold_prevalence(sc, s_h, s_m, fpr_target=0.05)
                                for meth, key in [("raw", "raw_flag_rate"), ("rogan_gladen", "rg_alpha"),
                                                  ("bayes", "bayes_alpha")]:
                                    res.append(dict(lang=lang, selection=sel, scenario=scen, heldout_model=m_star,
                                                    alpha=alpha, rep=r, n=n, alpha_hat=b[key],
                                                    ci_lo=b.get(f"{key.split('_')[0]}_ci_lo"),
                                                    ci_hi=b.get(f"{key.split('_')[0]}_ci_hi"), vocab=None,
                                                    method=f"{det[6:]}+{meth}", likelihood=""))
                print(f"[{lang}] sel={sel} lik={lik} scen={scen} heldout={m_star}: done", flush=True)
    return res


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["err"] = df.alpha_hat - df.alpha
    df["cover"] = (df.ci_lo <= df.alpha) & (df.alpha <= df.ci_hi)
    g = df.groupby(["lang", "method", "likelihood", "selection", "scenario"])
    s = g.agg(mae=("err", lambda e: e.abs().mean()), bias=("err", "mean"), rmse=("err", lambda e: np.sqrt((e**2).mean())),
              coverage=("cover", "mean"), n=("err", "size")).reset_index()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--scores", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--n-mix", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--selections", nargs="+", default=["ratio", "pos", "all"])
    ap.add_argument("--likelihoods", nargs="+", default=["occurrence", "length", "frequency"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1, help="languages processed in parallel")
    ap.add_argument("--heldout-only", nargs="*", default=[], help="generators never used in the reference (held out only)")
    ap.add_argument("--scenarios", nargs="*", default=[], help="replace the default scenarios: name=mode1+mode2 ...")
    ap.add_argument("--fixed-ref", action="store_true", help="reference from all modes in every scenario (scenario = test texts only)")
    a = ap.parse_args()
    HELDOUT_ONLY.update(a.heldout_only)
    rng = np.random.default_rng(a.seed)
    H = pd.concat([pd.read_parquet(p) for p in a.human], ignore_index=True)
    H = H[(H.period == "pre") & (~H.is_dup_exact)]
    if "doc_id" not in H.columns:
        H["doc_id"] = H["instance"] + ":" + H["type"] + ":" + H["id"].astype(str) + ":" + H["lang"]
    G = pd.read_parquet(a.gen)
    scores = None
    if a.scores:  # one parquet per detector (doc_id, score_<det>): align on doc_id
        scores = pd.read_parquet(a.scores[0]).drop_duplicates("doc_id").set_index("doc_id")
        for p in a.scores[1:]:
            scores = scores.join(pd.read_parquet(p).drop_duplicates("doc_id").set_index("doc_id"), how="outer")
    modes_scenarios = {"all": ["rewrite", "generate", "translate"], "rewrite": ["rewrite"], "generate": ["generate"],
                       "translate": ["translate"], "short": ["rewrite", "generate", "translate"]}
    if a.scenarios:
        modes_scenarios = {sc.split("=")[0]: sc.split("=")[1].split("+") for sc in a.scenarios}
    langs = a.langs or sorted(set(H.lang) & set(G.lang))
    results = []
    if a.jobs > 1:  # one process per language, each with its own seeded generator
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(lang, H[H.lang == lang], G[G.lang == lang], scores, a.n_mix, a.reps, a.n_boot,
                 np.random.default_rng([a.seed, i]), a.selections, modes_scenarios, 80, tuple(a.likelihoods), a.fixed_ref)
                for i, lang in enumerate(langs)]
        with ProcessPoolExecutor(max_workers=min(a.jobs, len(langs))) as ex:
            futs = [ex.submit(run_language, *j) for j in jobs]
            for f in futs:
                results += f.result()
                pd.DataFrame(results).to_parquet(a.out, index=False)
    else:
        for lang in langs:
            results += run_language(lang, H[H.lang == lang], G[G.lang == lang], scores, a.n_mix, a.reps, a.n_boot, rng,
                                    a.selections, modes_scenarios, likelihoods=tuple(a.likelihoods), fixed_ref=a.fixed_ref)
            pd.DataFrame(results).to_parquet(a.out, index=False)
    df = pd.DataFrame(results)
    df.to_parquet(a.out, index=False)
    s = summarise(df)
    s.to_csv(Path(a.out).with_suffix(".summary.csv"), index=False)
    print(s[s.method == "distributional"].to_string())


if __name__ == "__main__":
    main()
