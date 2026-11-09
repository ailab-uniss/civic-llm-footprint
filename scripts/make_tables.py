"""Generate LaTeX tables in paper/tables from configs and results.  python scripts/make_tables.py"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

R, T = Path("results"), Path("paper/tables")
T.mkdir(parents=True, exist_ok=True)
LANG = {"ca": "Catalan", "es": "Spanish", "fr": "French", "it": "Italian", "fi": "Finnish", "en": "English", "pt": "Portuguese", "sv": "Swedish", "de": "German"}
METH = {"distributional": "Distributional (ours)", "pangram+raw": "Open Pangram, raw rate", "pangram+rogan_gladen": "Open Pangram + RG",
        "pangram+bayes": "Open Pangram + Bayes", "binoculars+raw": "Binoculars, raw rate", "binoculars+rogan_gladen": "Binoculars + RG",
        "binoculars+bayes": "Binoculars + Bayes", "fastdetect+raw": "Fast-DetectGPT, raw rate", "fastdetect+rogan_gladen": "Fast-DetectGPT + RG",
        "fastdetect+bayes": "Fast-DetectGPT + Bayes"}


def w(name, lines):
    (T / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


# ---- models table
cfg = yaml.safe_load(open("configs/models.yaml", encoding="utf-8"))["generators"]
used = set()
if (R / "generation_stats.csv").exists():
    used = set(pd.read_csv(R / "generation_stats.csv").model.unique())
L = [r"\begin{table}[t]", r"\caption{Generators. Open models and current commercial assistants build the reference distributions and are held out in turn in the benchmark; the legacy generator (GPT-4o mini, July 2024) is only ever held out. Weights: bf16, native MXFP4, or 4-bit NF4 (bitsandbytes); commercial models are accessed through their APIs.}",
     r"\label{tab:models}", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lllll@{}}", r"\toprule",
     r"Generator & Developer & Parameters & Weights & Role \\", r"\midrule"]
ROLE = {"ref": "ref., open", "extra": "ref., study", "legacy": "held out"}
SHORT = {"NF4 (bitsandbytes)": "NF4 4-bit", "MXFP4 (native)": "MXFP4", "n/a": "API", "21B (3.6B active)": "21B (3.6B act.)",
         "OpenAI (API)": "OpenAI", "Anthropic (API)": "Anthropic", "Google (API)": "Google", "OpenAI (API, July 2024)": "OpenAI (Jul 2024)"}
for tag, m in cfg.items():
    if used and tag not in used:
        continue
    cells = [tag, str(m['family']), str(m['params']), str(m['quant']), ROLE.get(m.get('role'), m.get('role'))]
    L.append(" & ".join(SHORT.get(c, c) for c in cells) + " \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_models.tex", L)

# ---- benchmark table: MAE (pp) per language x method, scenario all, best selection for ours
if (R / "benchmark.summary.csv").exists():
    b = pd.read_csv(R / "benchmark.summary.csv")
    if "likelihood" not in b.columns:
        b["likelihood"] = "occurrence"
    b["likelihood"] = b.likelihood.fillna("")
    d = b[(b.scenario == "all")]
    ours = d[d.method == "distributional"]
    # configuration (vocabulary selection x likelihood) with the lowest MAE averaged over languages and scenarios
    allsc = b[b.method == "distributional"]
    best_sel, best_lik = allsc.groupby(["selection", "likelihood"]).mae.mean().idxmin()
    Path("results/best_config.txt").write_text(f"{best_sel} {best_lik}\n")
    print("best configuration:", best_sel, best_lik)
    rows = []
    for meth in ["distributional", "pangram+raw", "pangram+rogan_gladen", "pangram+bayes", "binoculars+rogan_gladen", "fastdetect+rogan_gladen"]:
        dd = d[(d.method == meth)]
        if meth == "distributional":
            dd = dd[(dd.selection == best_sel) & (dd.likelihood == best_lik)]
        if dd.empty:
            continue
        rows.append((meth, dd.set_index("lang").mae * 100, dd.set_index("lang").coverage * 100))
    langs = sorted(ours.lang.unique())
    L = [r"\begin{table*}[t]", r"\caption{Benchmark results: mean absolute error of the estimated LLM share (percentage points) over $\alpha\in\{0,\dots,0.5\}$, ten repetitions and all held-out generators, per language and method (all intervention modes). The last column is the empirical coverage of the 95\% intervals for our estimator.}",
         r"\label{tab:benchmark}", r"\centering\footnotesize", r"\begin{tabular}{l" + "r" * len(langs) + "r}", r"\toprule",
         "Method & " + " & ".join(LANG.get(l, l) for l in langs) + r" & Mean \\", r"\midrule"]
    for meth, mae, cov in rows:
        vals = [f"{mae.get(l, float('nan')):.1f}" for l in langs]
        L.append(f"{METH.get(meth, meth)} & " + " & ".join(vals) + f" & {mae.reindex(langs).mean():.1f} \\\\")
    cov = rows[0][2]
    L.append(r"\midrule")
    L.append("CI coverage (ours), \\% & " + " & ".join(f"{cov.get(l, float('nan')):.0f}" for l in langs) + f" & {cov.reindex(langs).mean():.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    w("tab_benchmark.tex", L)

    # ---- scenario table: MAE (pp) per scenario x configuration of our estimator, averaged over languages
    SEL = {"ratio": r"\textsc{ratio}", "all": r"\textsc{all}", "pos": r"\textsc{pos}", "excess": r"\textsc{excess}"}
    LIK = {"occurrence": "occurrence", "length": "length-calibrated", "frequency": "token frequency"}
    SCEN = [("all", "All"), ("rewrite", "Rewrite"), ("generate", "Generate"), ("translate", "Translate"), ("short", r"Short")]
    t = allsc.groupby(["likelihood", "selection", "scenario"]).mae.mean().unstack("scenario") * 100
    cov = allsc.groupby(["likelihood", "selection", "scenario"]).coverage.mean().unstack("scenario") * 100
    order = [(l, s) for l in ["occurrence", "length", "frequency"] for s in ["ratio", "all", "pos", "excess"] if (l, s) in t.index]
    L = [r"\begin{table}[t]", r"\caption{Mean absolute error (percentage points) of the distributional estimator by likelihood, vocabulary selection and benchmark scenario (intervention modes; ``short'': texts of at most 80 words), averaged over languages and held-out generators; in parentheses, the empirical coverage (\%) of the 95\% bootstrap intervals.}",
         r"\label{tab:bench_scenarios}", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}ll" + "r" * len(SCEN) + "@{}}", r"\toprule",
         "Likelihood & Vocab. & " + " & ".join(n for _, n in SCEN) + r" \\", r"\midrule"]
    for lik, sel in order:
        vals = [f"{t.loc[(lik, sel)].get(s, float('nan')):.1f} ({cov.loc[(lik, sel)].get(s, float('nan')):.0f})" for s, _ in SCEN]
        L.append(f"{LIK.get(lik, lik)} & {SEL.get(sel, sel)} & " + " & ".join(vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_bench_scenarios.tex", L)

# ---- mixed model table (parse statsmodels summary saved as text)
if (R / "mixed_model.txt").exists():
    txt = (R / "mixed_model.txt").read_text()
    sec = txt.split("=== half-year cells ===")[0]
    lines = [l for l in sec.splitlines() if l.strip()]
    rows = []
    start = False
    for l in lines:
        if l.startswith("Intercept") or start:
            start = True
            parts = l.split()
            if len(parts) >= 6 and parts[0] not in ("Group", "Var"):
                try:
                    rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[4])))
                except ValueError:
                    pass
    L = [r"\begin{table}[t]", r"\caption{Linear mixed model of the estimated LLM share per instance $\times$ language $\times$ type $\times$ period cell, with instance random intercepts.}",
         r"\label{tab:mixed}", r"\centering\footnotesize", r"\begin{tabular}{lrrr}", r"\toprule", r"Term & Coef. & SE & $p$ \\", r"\midrule"]
    for t, c, s, p in rows:
        t = t.replace("C(lang)[T.", "lang=").replace("C(type)[T.", "type=").replace("]", "").replace("_", r"\_")
        L.append(f"{t} & {100 * c:.2f} & {100 * s:.2f} & {p:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_mixed.tex", L)

# ---- audit table: mean metrics by scenario (rows) x metric, averaged over models and languages
if (R / "audit_metrics.parquet").exists():
    a = pd.read_parquet(R / "audit_metrics.parquet")
    cols = [c for c in ["l1_any_new", "l6_any_foreign", "l2_ents_kept", "l2_nums_kept", "l4_len_ratio", "l5_claim_coverage", "l5_output_grounded"] if c in a.columns]
    names = {"l1_any_new": "L1 new ref.\\ (\\%)", "l6_any_foreign": "L6 foreign (\\%)", "l2_ents_kept": "L2 ent.\\ kept (\\%)",
             "l2_nums_kept": "L2 num.\\ kept (\\%)", "l4_len_ratio": "L4 len.\\ ratio", "l5_claim_coverage": "L5 coverage (\\%)",
             "l5_output_grounded": "L5 grounded (\\%)"}
    g = a.groupby("mode")[cols].mean()
    if (R / "audit_homog.csv").exists():   # L3, set-level metric on the paired design
        hh = pd.read_csv(R / "audit_homog.csv").groupby("mode")[["dcos_trunc", "d2_orig", "d2_out_trunc"]].mean()
        g["l3_dcos"] = hh.dcos_trunc; g["l3_d2"] = hh.d2_out_trunc / hh.d2_orig
        cols = cols + ["l3_dcos", "l3_d2"]
        names.update({"l3_dcos": "L3 $\\Delta$cos", "l3_d2": "L3 dist-2 ratio"})
    L = [r"\begin{table*}[t]", r"\caption{Audit metrics by scenario, averaged over generators and languages: share of outputs with a new legal reference (L1) or a foreign one (L6), named entities and numbers of the input kept (L2), output/input length ratio (L4), share of input claims covered by the output and of output claims grounded in the input (L5), and homogenisation (L3): mean pairwise cosine similarity among the outputs minus that among their own originals, and distinct-2 ratio of the outputs over that of the originals, both with the outputs truncated to the length of their original. Per-generator values in Fig.~\ref{fig:audit}.}",
         r"\label{tab:audit}", r"\centering\scriptsize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", r"\toprule",
         "Scenario & " + " & ".join(names[c] for c in cols) + r" \\", r"\midrule"]
    for mode, r in g.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if c == "l3_dcos":
                vals.append(f"{v:+.3f}".replace("-", "$-$"))
            else:
                vals.append(f"{v:.2f}" if c in ("l4_len_ratio", "l3_d2") else f"{100 * v:.0f}")
        MODE = {"s1_improve": "S1 improve", "s2_notes": "S2 notes", "s3_persuasive": "S3 persuasive"}
        L.append(f"{MODE.get(mode, mode.replace('_', ' '))} & " + " & ".join(vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    w("tab_audit.tex", L)

# ---- null calibration table: held-out pre-ChatGPT half per instance, adapted vs language-level reference
if (R / "null_calibration.csv").exists():
    n = pd.read_csv(R / "null_calibration.csv")
    if "reference" in n.columns and "slice" in n.columns:
        f = n[(n.slice == "fold1") & (n.instance != "ALL")]
        ad = f[f.reference == "adapted"].set_index(["instance", "lang"])
        po = f[f.reference == "pooled"].set_index(["instance", "lang"])
        yr = n[(n.reference == "adapted") & n.slice.str.startswith("year=")].groupby(["instance", "lang"]).alpha.agg(["min", "max"])
        rows = ad.join(po[["alpha"]].rename(columns={"alpha": "alpha_pooled"}), how="left").join(yr, how="left")
        rows = rows.sort_values("n_docs", ascending=False)
        L = [r"\begin{table}[t]", r"\caption{Null calibration: estimated LLM share (\%) on a held-out half of the pre-ChatGPT texts of each instance, with the instance-adapted reference (Eq.~\eqref{eq:adapt}) and with the unadapted language-level reference, and the range over single pre-ChatGPT years with the adapted reference (years with at least \minDocs{} texts). Values are false-positive floors and topic-drift ranges, not LLM use.}",
             r"\label{tab:null}", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Instance & Lang. & $n$ & Adapted & Unadapted & Years \\", r"\midrule"]
        INAME = {k: v["name"] for k, v in yaml.safe_load(open("configs/instances.yaml", encoding="utf-8"))["instances"].items()}
        INAME.update({"eu_hys_citizens": "HYS citizens", "eu_hys_organisations": "HYS organisations", "eu_hys_unknown": "HYS unknown",
                      "regulations_gov_epa": "Regulations.gov EPA", "regulations_gov_dot": "Regulations.gov DOT",
                      "regulations_gov_usda": "Regulations.gov USDA", "regulations_gov_doi": "Regulations.gov DOI"})
        for (inst, lang), r in rows.iterrows():
            name = INAME.get(inst, inst).replace("_", r"\_").replace("&", r"\&")
            yrs = f"{100 * r['min']:.0f}--{100 * r['max']:.0f}" if pd.notna(r.get("max")) else "--"
            L.append(f"{name} & {lang} & {int(r.n_docs):,} & {100 * r.alpha:.1f} & {100 * r.alpha_pooled:.1f} & {yrs} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        w("tab_null.tex", L)

# ---- excess-vocabulary lower bounds: per instance-language, post years (bound, CI) next to the max pre-year noise bound
if (R / "excess" / "excess.csv").exists():
    e = pd.read_csv(R / "excess" / "excess.csv")
    yrs = sorted(e[e.kind == "post"].year.unique())
    rows = []
    for (inst, lang), g in e.groupby(["instance", "lang"]):
        nul = g[g.kind == "placebo"]
        if nul.empty:
            continue
        r = {"instance": inst, "lang": lang, "noise": nul.bound.max(), "n_null": len(nul)}
        for y in yrs:
            x = g[(g.kind == "post") & (g.year == y)]
            r[y] = (x.bound.iloc[0], x.ci_lo.iloc[0], x.ci_hi.iloc[0], x.n_docs.iloc[0]) if len(x) else None
        rows.append(r)
    rows.sort(key=lambda r: r["noise"])
    L = [r"\begin{table*}[t]", r"\caption{Excess-vocabulary indicator per year (\%, 95\% document-bootstrap interval in brackets), by instance and language. Under the stable-vocabulary counterfactual it is a lower-bound proxy for the share of LLM-processed documents. ``Noise'' is the largest value obtained when a pre-ChatGPT year is treated as post-ChatGPT (number of placebo years in parentheses); values below it are not distinguishable from topic drift. Years with fewer than 300 documents are omitted.}",
         r"\label{tab:excess}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llr" + "r" * len(yrs) + "@{}}", r"\toprule",
         "Instance & Lang. & Noise & " + " & ".join(str(y) for y in yrs) + r" \\", r"\midrule"]
    for r in rows:
        cells = []
        for y in yrs:
            v = r[y]
            cells.append("--" if v is None else f"{100 * v[0]:.0f} [{100 * v[1]:.0f}, {100 * v[2]:.0f}]")
        L.append(f"{r['instance'].replace('_', chr(92) + '_')} & {r['lang']} & {100 * r['noise']:.0f} ({r['n_null']}) & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    w("tab_excess.tex", L)

# ---- temporal-transfer test on MULTITuDE: 2023 generators held out against a reference of today's models (news domain)
if (R / "multitude" / "benchmark_all.parquet").exists():
    mb = pd.read_parquet(R / "multitude" / "benchmark_all.parquet")
    mb = mb[(mb.method == "distributional") & (mb.selection == "ratio") & (mb.likelihood == "length") & (mb.scenario == "generate")].copy()
    mb["err"] = (mb.alpha_hat - mb.alpha) * 100
    hi = mb[mb.alpha >= 0.05]
    mae, bias = hi.groupby("heldout_model").err.apply(lambda s: s.abs().mean()), hi.groupby("heldout_model").err.mean()
    half = mb[mb.alpha == 0.5].groupby("heldout_model").err.mean()
    neg = hi.groupby(["heldout_model", "lang"]).err.mean().groupby(level=0).apply(lambda s: int((s < -1).sum()))
    GROUPS = [("OpenAI models of 2023--2024", [("mt1_gpt-4", "GPT-4 (2023)"), ("mt1_gpt-3.5-turbo", "GPT-3.5-turbo (2023)"),
                                                ("mt1_text-davinci-003", "text-davinci-003 (2023)"), ("mt_gpt-3.5-turbo-0125", "GPT-3.5-turbo (Jan 2024)"),
                                                ("gpt-4o-mini", "GPT-4o mini (Jul 2024)")]),
              ("Open models of 2023--2024", [("mt1_llama-65b", "Llama 65B"), ("mt1_alpaca-lora-30b", "Alpaca-LoRA 30B"), ("mt1_vicuna-13b", "Vicuna 13B"),
                                              ("mt1_opt-66b", "OPT 66B"), ("mt_Llama-2-70b-chat-hf", "Llama~2 70B chat"), ("mt_Mistral-7B-Instruct-v0.2", "Mistral 7B v0.2"),
                                              ("mt_aya-101", "Aya 101"), ("mt_v5-Eagle-7B-HF", "Eagle 7B")]),
              ("Contemporary generators (leave-one-out)", [("gpt-5.4-mini", "GPT-5.4 mini"), ("claude-haiku-5-5", "Claude Haiku 5.5"), ("gpt-oss-20b", "gpt-oss-20b"),
                                                            ("qwen2.5-7b", "Qwen~2.5 7B"), ("mistral-small-24b", "Mistral Small~3 24B"), ("llama-3.1-8b", "Llama~3.1 8B"),
                                                            ("eurollm-9b", "EuroLLM 9B"), ("salamandra-7b", "Salamandra 7B")])]
    L = [r"\begin{table}[t]", r"\caption{Temporal transfer on MULTITuDE news texts (five languages): error of the estimator when the held-out generator is a model of 2023--2024 and the reference is built from today's generators (generation mode). MAE and mean bias over $\alpha\geq 0.05$, bias at $\alpha=0.5$ (pp), and number of the five languages with a mean bias below $-1$~pp.}",
         r"\label{tab:multitude}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Held-out generator & MAE & Bias & Bias at 0.5 & Under \\", r"\midrule"]
    for title, rows in GROUPS:
        L.append(r"\multicolumn{5}{@{}l}{\emph{" + title + r"}} \\")
        for tag, name in rows:
            if tag in mae.index:
                L.append(f"{name} & {mae[tag]:.1f} & {bias[tag]:+.1f} & {half[tag]:+.1f} & {neg[tag]}/5 \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_multitude.tex", L)


# ---- generator switch at constant share: spurious shift between two held-out generators estimated with the same reference
if (R / "multitude" / "benchmark_all.parquet").exists():
    mb = pd.read_parquet(R / "multitude" / "benchmark_all.parquet")
    mb = mb[(mb.method == "distributional") & (mb.selection == "ratio") & (mb.likelihood == "length") & (mb.scenario == "generate")]
    m = mb.groupby(["heldout_model", "alpha", "lang"]).alpha_hat.mean().unstack("lang") * 100
    PAIRS = [("mt1_gpt-3.5-turbo", "mt1_gpt-4", "GPT-3.5 '23 $\\to$ GPT-4 '23"),
             ("mt1_gpt-3.5-turbo", "mt_gpt-3.5-turbo-0125", "GPT-3.5 '23 $\\to$ GPT-3.5 '24"),
             ("mt1_gpt-3.5-turbo", "gpt-4o-mini", "GPT-3.5 '23 $\\to$ GPT-4o mini '24"),
             ("gpt-4o-mini", "gpt-5.4-mini", "GPT-4o mini '24 $\\to$ GPT-5.4 mini '26"),
             ("mt1_gpt-3.5-turbo", "gpt-5.4-mini", "GPT-3.5 '23 $\\to$ GPT-5.4 mini '26")]
    AL = [0.05, 0.10, 0.20]
    L = [r"\begin{table}[t]", r"\caption{Generator switch at constant share (MULTITuDE news texts, five languages, same reference and human pool): apparent change of the estimate, in pp, when the LLM component of a mixture is produced by the second generator instead of the first, for true shares of 5, 10 and 20\%. Mean over languages; the last column gives the range over languages at a true share of 10\%. The last row is the change between two independent resamplings of the same generator.}",
         r"\label{tab:switch}", r"\centering\scriptsize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         "Switch & " + " & ".join(f"$\\alpha={a:.2f}$" for a in AL) + r" & Range at 0.10 \\", r"\midrule"]
    for ga, gb, name in PAIRS:
        if (ga, AL[0]) in m.index and (gb, AL[0]) in m.index:
            cells = []
            for a in AL:
                d = m.loc[(gb, a)] - m.loc[(ga, a)]
                cells.append(f"{d.mean():+.1f}")
            d10 = m.loc[(gb, 0.10)] - m.loc[(ga, 0.10)]
            cells.append(f"{d10.min():+.1f} to {d10.max():+.1f}")
            L.append(f"{name} & " + " & ".join(cells) + r" \\")
    # resampling-only control: same generator, reps 0-4 vs 5-9
    g0 = mb[mb.heldout_model == "mt1_gpt-3.5-turbo"]
    cells = []
    for a in AL:
        x = g0[g0.alpha == a]
        h1 = x[x.rep < 5].groupby("lang").alpha_hat.mean() * 100; h2 = x[x.rep >= 5].groupby("lang").alpha_hat.mean() * 100
        d = h2 - h1
        cells.append(f"{d.mean():+.1f}")
        if a == 0.10:
            r10 = f"{d.min():+.1f} to {d.max():+.1f}"
    cells.append(r10)
    L.append(r"\midrule"); L.append("No switch (resampling only) & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_switch.tex", L)


# ---- identifiability: prevalence x editing intensity, fixed reference (results/intensity)
if (R / "intensity" / "benchmark.parquet").exists():
    ib = pd.read_parquet(R / "intensity" / "benchmark.parquet")
    IL = ["en", "de", "fr", "it", "fi"]   # languages whose rewrites could be tiered by lexical overlap with the seed
    ib = ib[(ib.method == "distributional") & ib.lang.isin(IL) & (ib.alpha > 0)].copy()
    ib["ratio"] = ib.alpha_hat / ib.alpha
    jt = pd.read_csv(R / "intensity" / "jaccard_tiers.csv")
    jt = jt[jt.lang.isin(IL)].groupby("mode")["50%"].agg(["min", "max"])
    TIER = [("light", "Rewrite, light edit", "rewrite_light"), ("mid", "Rewrite, medium edit", "rewrite_mid"), ("heavy", "Rewrite, heavy edit", "rewrite_heavy"),
            ("translate", "Translation", None), ("generate", "Generation from title", None)]
    AL = [0.1, 0.2, 0.3]
    L = [r"\begin{table}[t]", r"\caption{Identifiability of $\alpha$: recovered fraction $\hat\alpha/\alpha$ when the LLM component of the mixture is produced with a given intervention intensity and the reference is fixed (all modes of the other generators). Rewrites are split into terciles of lexical overlap with the original (Jaccard of content lemmas; range of the per-language medians). Mean over " + str(len(IL)) + r" languages and held-out generators.}",
         r"\label{tab:intensity}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         "Test component & Overlap & " + " & ".join(f"$\\alpha={a:.1f}$" for a in AL) + r" \\", r"\midrule"]
    for sc, name, mode in TIER:
        g = ib[ib.scenario == sc]
        if g.empty:
            continue
        ov = f"{jt.loc[mode, 'min']:.2f}--{jt.loc[mode, 'max']:.2f}" if mode in jt.index else "--"
        L.append(f"{name} & {ov} & " + " & ".join(f"{g[g.alpha == a].ratio.mean():.2f}" for a in AL) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_intensity.tex", L)
