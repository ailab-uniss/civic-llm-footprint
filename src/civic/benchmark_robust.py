"""Robustness benchmarks requested in review: the study configuration and realistic shifts.

Experiments (per language, held-out generator m* in turn; selection=ratio, likelihood=length as in the study):
  weighted    the study reference (commercial generators weighted by market share, configs/reference_weights.yaml),
              hash split as in benchmark.py, n = 1000; m* removed from the reference (weights renormalised)
  smalln      the same with mixtures of n = 150 and n = 300 documents
  instance    composition shift: language reference on fold 0 of all instances, adapted (Eq. adapt) on fold 0 of one
              instance; mixtures of that instance's fold-1 human texts and m* texts derived from fold-1 seeds
              (of the same instance when at least 60 exist, of the language otherwise); also without adaptation
  period      temporal shift: reference and adaptation fitted on texts of 2020 and earlier; mixtures of the
              instance's 2021-2022 (pre-ChatGPT) human texts and m* texts derived from 2021-2022 seeds
  modeout     intervention mode not in the reference: P_M from the other two modes, test texts from the third
              (equal-weight reference of the other generators, as in the main benchmark)
  variantout  prompt variant not in the reference: P_M from the other prompt variants, test texts from the third
For instance/period the generators of the study reference are also tested while kept in the reference ("in-reference").

  python src/civic/benchmark_robust.py --processed data/processed/*.parquet --gen data/generated/generated_processed.parquet \
      --weights configs/reference_weights.yaml --out results/robust/robust.parquet --jobs 9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator import DistributionalEstimator  # noqa: E402
from analysis import fold_of, matched_reference, split_author_type, to_sets, llm_by_model  # noqa: E402

ALPHAS = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50]
ALPHAS_SHIFT = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
SEL, LIK, TOPK, MINDF, KAPPA = "ratio", "length", 1000, 0.003, 1000.0


CFG = {"selection": SEL, "stable_q": 0.1}


def est_new(weights):
    return DistributionalEstimator(selection=CFG["selection"], top_k=TOPK, min_df=MINDF, likelihood=LIK, model_weights=weights,
                                   stable_q=CFG["stable_q"])


def groups(df):
    return (df.instance.astype(str) + "|" + df.created_at.dt.year.astype(str)).tolist()


def mixtures(est, h_sets, l_sets, alphas, reps, n_max, n_boot, rng, base: dict, out: list):
    if len(l_sets) < 30 or len(h_sets) < 100:
        return
    for alpha in alphas:
        for r in range(reps):
            n = min(n_max, len(h_sets), int(len(l_sets) / max(alpha, 0.5)))
            k = int(round(alpha * n))
            hi = rng.choice(len(h_sets), n - k, replace=False)
            li = rng.choice(len(l_sets), k, replace=False) if k else []
            docs = [h_sets[i] for i in hi] + [l_sets[i] for i in li]
            o = est.estimate(docs, n_boot=n_boot)
            out.append(dict(base, alpha=alpha, rep=r, n=n, alpha_hat=o["alpha"], ci_lo=o.get("ci_lo"), ci_hi=o.get("ci_hi")))


def run_language(lang, H, G, S, weights, exps, reps, n_boot, seed):
    rng = np.random.default_rng(seed)
    res = []
    H = H.copy(); G = G.copy()
    H["fold"] = fold_of(H.doc_id); H["year"] = H.created_at.dt.year
    G["fold"] = fold_of(G.doc_id)
    seed_year = H.set_index("doc_id")["year"]
    G["year"] = G.doc_id.map(seed_year)
    models = sorted(G.model.unique())
    wmodels = [m for m in models if m in weights]
    pre = H[H.period == "pre"]

    def log(msg):
        print(f"[{lang}] {msg}", flush=True)

    # ---- weighted study reference, hash split (as benchmark.py), n = 1000 and small n
    if {"weighted", "smalln"} & set(exps):
        h_ref, h_test = pre[pre.fold == 0], pre[pre.fold == 1]
        hr, ht = to_sets(h_ref.tokens), to_sets(h_test.tokens)
        for m in models:
            w = {k: v for k, v in weights.items() if k != m}
            by = llm_by_model(G[(G.fold == 0) & G.model.isin(w)], 100)
            if not by:
                continue
            est = est_new(w); est.fit_reference(hr, by, human_groups=groups(h_ref))
            lt = to_sets(G[(G.fold == 1) & (G.model == m)].tokens)
            if "weighted" in exps:
                mixtures(est, ht, lt, ALPHAS, reps, 1000, n_boot, rng, dict(exp="weighted", lang=lang, heldout_model=m, setting="n1000"), res)
            if "smalln" in exps:
                for nm in (150, 300):
                    mixtures(est, ht, lt, ALPHAS, reps, nm, n_boot, rng, dict(exp="smalln", lang=lang, heldout_model=m, setting=f"n{nm}"), res)
            log(f"weighted/smalln heldout={m}")

    # ---- composition shift (instance) and temporal shift (period), with the study pipeline (matched + adapted)
    for exp in [e for e in ("instance", "period") if e in exps]:
        if exp == "instance":
            fit_h, test_h = pre[pre.fold == 0], pre[pre.fold == 1]
            fit_g, test_g = G[G.fold == 0], G[G.fold == 1]
        else:
            fit_h, test_h = pre[pre.year <= 2020], pre[pre.year.isin([2021, 2022])]
            fit_g, test_g = G[G.year <= 2020], G[G.year.isin([2021, 2022])]
        insts = [i for i in test_h.instance.unique()
                 if (fit_h.instance == i).sum() >= 300 and (test_h.instance == i).sum() >= 150]
        if not insts or len(fit_h) < 300:
            continue
        hm = matched_reference(fit_h, S.groupby("instance").size())
        hm_sets = to_sets(hm.tokens)
        cases = [(m, False) for m in models] + [(m, True) for m in wmodels]   # held out / kept in the reference
        for m, inref in cases:
            w = dict(weights) if inref else {k: v for k, v in weights.items() if k != m}
            by = llm_by_model(fit_g[fit_g.model.isin(w)], 100)
            if not by:
                continue
            base = est_new(w); base.fit_reference(hm_sets, by, human_groups=groups(hm))
            for inst in insts:
                gi = fit_g[(fit_g.inst == inst) & fit_g.model.isin(w) & fit_g["mode"].isin(["rewrite", "generate"])]
                adapted = base.adapt(to_sets(fit_h[fit_h.instance == inst].tokens), llm_by_model(gi), kappa=KAPPA)
                pool = test_g[(test_g.model == m) & (test_g.inst == inst)]
                src = "instance"
                if len(pool) < 60:
                    pool, src = test_g[test_g.model == m], "language"
                ht = to_sets(test_h[test_h.instance == inst].tokens)
                lt = to_sets(pool.tokens)
                for refname, e in (("adapted", adapted), ("unadapted", base)):
                    mixtures(e, ht, lt, ALPHAS_SHIFT, reps, 500, n_boot, rng,
                             dict(exp=exp, lang=lang, heldout_model=m, setting=f"{refname}|{'in' if inref else 'out'}",
                                  instance=inst, llm_source=src), res)
            log(f"{exp} {m} inref={inref}")

    # ---- mode and prompt variant not represented in the reference (main-benchmark reference: other generators, equal weights)
    if {"modeout", "variantout"} & set(exps):
        h_ref, h_test = pre[pre.fold == 0], pre[pre.fold == 1]
        hr, ht = to_sets(h_ref.tokens), to_sets(h_test.tokens)
        open_models = [m for m in models if m != "gpt-4o-mini"]
        for exp, col in (("modeout", "mode"), ("variantout", "variant")):
            if exp not in exps:
                continue
            for v in sorted(G[col].dropna().unique()):
                for m in open_models:
                    ref_g = G[(G.fold == 0) & (G[col] != v) & G.model.isin([x for x in open_models if x != m])]
                    by = llm_by_model(ref_g, 100)
                    if not by:
                        continue
                    est = est_new(None); est.fit_reference(hr, by, human_groups=groups(h_ref))
                    lt = to_sets(G[(G.fold == 1) & (G.model == m) & (G[col] == v)].tokens)
                    mixtures(est, ht, lt, ALPHAS, max(1, reps // 2), 1000, 0, rng,
                             dict(exp=exp, lang=lang, heldout_model=m, setting=f"{col}={v}"), res)
                log(f"{exp} {col}={v}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--sample", default="data/generated/human_sample.parquet")
    ap.add_argument("--weights", default="configs/reference_weights.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exps", nargs="+", default=["weighted", "smalln", "instance", "period", "modeout", "variantout"])
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--selection", default=SEL)
    ap.add_argument("--stable-q", type=float, default=0.1)
    a = ap.parse_args()
    CFG.update(selection=a.selection, stable_q=a.stable_q)
    weights = {str(k): float(v) for k, v in yaml.safe_load(open(a.weights, encoding="utf-8"))["weights"].items()}
    H = pd.concat([pd.read_parquet(p, columns=None) for p in a.processed], ignore_index=True)
    if "doc_id" not in H.columns:
        H["doc_id"] = H["instance"] + ":" + H["type"] + ":" + H["id"].astype(str) + ":" + H["lang"]
    H = H[~H.is_dup_exact].drop_duplicates("doc_id").copy()
    H["instance"] = split_author_type(H.instance, H["user_type"] if "user_type" in H.columns else None)
    G = pd.read_parquet(a.gen)
    seed_inst = H.set_index("doc_id")["instance"]
    G["inst"] = G.doc_id.map(seed_inst).fillna(G.doc_id.str.split(":").str[0])
    S = pd.read_parquet(a.sample, columns=["doc_id", "lang", "instance"])
    S["instance"] = S.doc_id.map(seed_inst).fillna(S.instance)
    langs = a.langs or sorted(set(H[H.period == "pre"].lang) & set(G.lang))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    results = []
    jobs = [(l, H[H.lang == l], G[G.lang == l], S[S.lang == l], weights, a.exps, a.reps, a.n_boot, [7, i]) for i, l in enumerate(langs)]
    with ProcessPoolExecutor(max_workers=min(a.jobs, len(langs))) as ex:
        futs = {ex.submit(run_language, *j): j[0] for j in jobs}
        for f in futs:
            results += f.result()
            pd.DataFrame(results).to_parquet(a.out, index=False)
    df = pd.DataFrame(results)
    df.to_parquet(a.out, index=False)
    df["err"] = df.alpha_hat - df.alpha
    df["cover"] = (df.ci_lo <= df.alpha) & (df.alpha <= df.ci_hi)
    s = df.groupby(["exp", "setting", "lang"]).agg(mae=("err", lambda e: e.abs().mean()), bias=("err", "mean"),
                                                   coverage=("cover", "mean"), n=("err", "size")).reset_index()
    s.to_csv(Path(a.out).with_suffix(".summary.csv"), index=False)
    print(df.groupby(["exp", "setting"]).agg(mae=("err", lambda e: 100 * e.abs().mean()), bias=("err", lambda e: 100 * e.mean()),
                                             cover=("cover", "mean"), n=("err", "size")).round(2).to_string())


if __name__ == "__main__":
    main()
