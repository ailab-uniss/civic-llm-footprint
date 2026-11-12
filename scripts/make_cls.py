"""Numbers and tables of the classifier estimator (score quantification), from the results directory.

Reads the benchmark of the score quantifiers (results/supervised), the external tests (results/multitude, results/raid_lb.csv,
results/intensity) and the study cells (results/analysis_cls3), restricts the study to the instance-language pairs that
meet the inclusion criteria (results/processed_stats.csv), attaches to every cell the benchmark-calibrated error band
(results/supervised/calibration_coefs.csv) and writes

  paper/numbers_cls.tex, paper/numbers_cls.json      LaTeX macros (prefix \\cls)
  paper/tables/tab_cls_*.tex                         tables
  results/study_cls/*.csv                            study cells with bands, models, summaries (released with the data)

  python scripts/make_cls.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml

R, O, T = Path("results"), Path("paper"), Path("paper/tables")
A, S, OUT = R / os.environ.get("CLS_DIR", "analysis_cls5"), R / "supervised", R / "study_cls"
AE = A.with_name(A.name + "_equal")
METH = os.environ.get("CLS_METH", "hdyv")    # estimator of the study in the benchmark files: HDy with the vendor-weighted reference
MI = f"{METH}_inst"
OUT.mkdir(parents=True, exist_ok=True)
CFG = yaml.safe_load(open("configs/instances.yaml", encoding="utf-8"))["instances"]
NAME = {k: v["name"] for k, v in CFG.items()}
NAME.update({"eu_hys_citizens": "HYS citizens", "eu_hys_organisations": "HYS organisations", "eu_hys_unknown": "HYS unknown",
             "regulations_gov_epa": "Regulations.gov EPA", "regulations_gov_dot": "Regulations.gov DOT",
             "regulations_gov_usda": "Regulations.gov USDA", "regulations_gov_doi": "Regulations.gov DOI"})
GEN = {"gpt-5.4-mini": "GPT-5.4 mini", "gemini-3.6-flash": "Gemini 3.6 Flash", "claude-haiku-5-5": "Claude Haiku 5.5",
       "gpt-4o": "GPT-4o (Aug 2024)", "gpt-4o-mini": "GPT-4o mini (Jul 2024)", "gpt-oss-20b": "gpt-oss-20b",
       "gemma-2-9b-it": "Gemma~2 9B", "mistral-small-24b": "Mistral Small~3 24B", "qwen2.5-7b": "Qwen~2.5 7B",
       "eurollm-9b": "EuroLLM 9B", "llama-3.1-8b": "Llama~3.1 8B", "salamandra-7b": "Salamandra 7B"}
OUTSIDE = "salamandra-7b"
nums: dict = {}


def tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&")


def w(name, lines):
    (T / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


def p1(v):   # share -> percent with one decimal
    return f"{100 * v:.1f}"


def p0(v):
    return f"{100 * v:.0f}"


def sgn(v, d=1):
    t = f"{v:+.{d}f}"
    return t[1:] if float(t) == 0 else t


def fint(n):
    return f"{int(n):,}".replace(",", "{,}")


# ------------------------------------------------------------------ error band of the forward-validation benchmark
COEF = pd.read_csv(S / "calibration_coefs.csv").set_index("calibration")


def band(a, pooled):
    """Interval of an estimate a (array) from the benchmark error quantiles (scripts/calibrate_scores.py)."""
    a = np.asarray(a, float)
    pooled = np.broadcast_to(np.asarray(pooled, bool), a.shape)
    lo, hi = np.empty_like(a), np.empty_like(a)
    for flag, key in ((False, "own"), (True, "pooled")):
        c, m = COEF.loc[key], pooled == flag
        lo[m] = a[m] - np.maximum(c.hi0 + c.hi1 * a[m], 0)
        hi[m] = a[m] - np.minimum(c.lo0 + c.lo1 * a[m], 0)
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


for key, tag in (("own", "Own"), ("pooled", "Pooled")):
    for a, t in ((0.0, "Zero"), (0.05, "Five"), (0.10, "Ten"), (0.20, "Twenty"), (0.40, "Forty")):
        lo, hi = band([a], key == "pooled")
        nums[f"clsBand{tag}{t}Lo"], nums[f"clsBand{tag}{t}Hi"] = p1(lo[0]), p1(hi[0])

# ------------------------------------------------------------------ benchmark of the score quantifiers
Q = pd.read_parquet(S / "quant_aug.parquet")
Q["ae"], Q["b"] = (Q.alpha_hat - Q.alpha).abs() * 100, (Q.alpha_hat - Q.alpha) * 100
QG = Q[Q.heldout_model.str.startswith("group:")]
Q = Q[~Q.heldout_model.str.startswith("group:")]
M = Q[Q.method == MI].copy()
M["own"] = M.own.astype(bool)
for exp, t in (("instance", "Inst"), ("period", "Fwd")):
    g = M[M.exp == exp]
    nums[f"clsMae{t}"] = f"{g.ae.mean():.1f}"
    nums[f"clsMae{t}Own"] = f"{g[g.own].ae.mean():.1f}"
    nums[f"clsMae{t}Pooled"] = f"{g[~g.own].ae.mean():.1f}"
    nums[f"clsFloor{t}"] = f"{100 * g[g.alpha == 0].alpha_hat.mean():.1f}"
    nums[f"clsFloor{t}Own"] = f"{100 * g[(g.alpha == 0) & g.own].alpha_hat.mean():.1f}"
    nums[f"clsFloor{t}Pooled"] = f"{100 * g[(g.alpha == 0) & ~g.own].alpha_hat.mean():.1f}"
    nums[f"clsFloor{t}Max"] = f"{100 * g[(g.alpha == 0) & g.own].groupby(['lang', 'instance']).alpha_hat.mean().max():.1f}"
    pm = g[g.heldout_model != OUTSIDE].groupby("heldout_model").ae.mean()
    nums[f"clsMae{t}GenMin"], nums[f"clsMae{t}GenMax"] = f"{pm.min():.1f}", f"{pm.max():.1f}"
    nums[f"clsMae{t}Salamandra"] = f"{g[g.heldout_model == OUTSIDE].ae.mean():.1f}"
    nums[f"clsMae{t}OpenAI"] = f"{QG[(QG.method == MI) & (QG.exp == exp)].ae.mean():.1f}"
    hi = g[g.alpha >= 0.05]
    nums[f"clsBias{t}"] = sgn(hi.b.mean())
    nums[f"clsAtHalf{t}"] = sgn(g[g.alpha == 0.5].b.mean())
    b = g[g.ci_lo.notna()]
    nums[f"clsCovBoot{t}"] = p0(((b.ci_lo <= b.alpha + 1e-9) & (b.alpha <= b.ci_hi + 1e-9)).mean())
    nums[f"clsCovRegion{t}"] = p0(((b.lb_lo <= b.alpha + 1e-9) & (b.alpha <= b.gmax_hi + 1e-9)).mean())
    nums[f"clsLbValid{t}"] = p1((b.lb_lo <= b.alpha + 1e-9).mean())
    # the bound presumes the human score distribution of the instance: validity by kind of calibration
    bo, bp = b[b.own], b[~b.own]
    nums[f"clsLbValid{t}Own"], nums[f"clsLbValid{t}Pooled"] = p1((bo.lb_lo <= bo.alpha + 1e-9).mean()), p1((bp.lb_lo <= bp.alpha + 1e-9).mean())
    vi = (bo.lb_lo > bo.alpha + 1e-9).groupby([bo.lang, bo.instance]).mean()
    nums[f"clsLbViolPairs{t}"] = int((vi > 0.10).sum())
    nums[f"clsLbViolPairMax{t}"] = p0(vi.max())
    ex = (bo.lb_lo - bo.alpha)[bo.lb_lo > bo.alpha + 1e-9] * 100
    nums[f"clsLbViolExcess{t}"] = f"{ex.median():.1f}" if len(ex) else "0.0"
    nums[f"clsLbAbove{t}"] = p1((bo.lb_lo > bo.alpha_hat + 1e-9).mean())
    # adjusted count with the same classifier and calibration, by kind of calibration and at low shares
    ac = Q[(Q.method == "acc_sp_inst") & (Q.exp == exp)]
    ownpair = g.groupby(["lang", "instance"]).own.first()
    aown = pd.Series(ownpair.reindex(pd.MultiIndex.from_frame(ac[["lang", "instance"]])).values, index=ac.index).fillna(False).astype(bool)
    nums[f"clsAccMae{t}"], nums[f"clsAccMae{t}Own"], nums[f"clsAccMae{t}Pooled"] = (f"{ac.ae.mean():.1f}", f"{ac[aown].ae.mean():.1f}",
                                                                                 f"{ac[~aown].ae.mean():.1f}")
    nums[f"clsAccFloor{t}"] = f"{100 * ac[ac.alpha == 0].alpha_hat.mean():.1f}"
    nums[f"clsAccMaeLow{t}"], nums[f"clsMaeLow{t}"] = f"{ac[ac.alpha <= 0.05].ae.mean():.1f}", f"{g[g.alpha <= 0.05].ae.mean():.1f}"
    nums[f"clsAccMaeHigh{t}"], nums[f"clsMaeHigh{t}"] = f"{ac[ac.alpha >= 0.2].ae.mean():.1f}", f"{g[g.alpha >= 0.2].ae.mean():.1f}"
    lbm = Q[(Q.method == "lb_inst") & (Q.exp == exp) & (Q.alpha > 0)]
    nums[f"clsLbRec{t}"] = f"{(lbm.alpha_hat / lbm.alpha).mean():.2f}"
    bb = Q[(Q.method == "bbe_inst") & (Q.exp == exp)]
    nums[f"clsBbeOver{t}"] = p0((bb.alpha_hat > bb.alpha + 0.01).mean())
ccf = Q[(Q.method == "cc_inst") & (Q.exp == "period") & (Q.alpha == 0)]
nums["clsCcFloorFwd"] = f"{100 * ccf.alpha_hat.mean():.0f}"
fq = {m: Q[(Q.method == f"{m}_inst") & (Q.exp == "period")].ae.mean() for m in ("acc_sp", "ms", "em", METH)}
nums["clsQuantMinFwd"], nums["clsQuantMaxFwd"] = f"{min(fq.values()):.1f}", f"{max(fq.values()):.1f}"
nums["clsHdyEqDiffFwd"] = f"{abs(fq[METH] - Q[(Q.method == 'hdyx_inst') & (Q.exp == 'period')].ae.mean()):.1f}"
sal = M[(M.heldout_model == OUTSIDE) & (M.alpha > 0)]
nums["clsSalamandraRec"] = f"{(sal.alpha_hat / sal.alpha).mean():.2f}"
salf = sal[sal.exp == "period"]
nums["clsSalamandraRecFwd"] = f"{salf.alpha_hat.sum() / salf.alpha.sum():.2f}"
nums["clsNInstBench"] = int(M[M.exp == "instance"].groupby(["lang", "instance"]).ngroups)
nums["clsNInstBenchFwd"] = int(M[M.exp == "period"].groupby(["lang", "instance"]).ngroups)
nums["clsNInstBenchOwn"] = int(M[(M.exp == "period") & M.own].groupby(["lang", "instance"]).ngroups)
nums["clsNMixtures"] = fint(len(M))
nums["clsNLangsBench"] = int(M.lang.nunique())
fl = M[M.exp == "period"].groupby("lang").ae.mean()
flo = fl.drop("en")
nums["clsMaeFwdLangMin"], nums["clsMaeFwdLangMax"], nums["clsMaeFwdEn"] = f"{flo.min():.1f}", f"{flo.max():.1f}", f"{fl['en']:.1f}"
nums["clsMaeFwdEnOwn"] = f"{M[(M.exp == 'period') & (M.lang == 'en') & M.own].ae.mean():.1f}"
nums["clsFloorFwdEn"] = f"{100 * M[(M.exp == 'period') & (M.lang == 'en') & (M.alpha == 0)].alpha_hat.mean():.1f}"
wi = M.groupby(["exp", "lang", "instance"]).ae.mean().sort_values(ascending=False)
nums["clsWorstInst"], nums["clsWorstInstMae"] = tex(NAME.get(wi.index[0][2], wi.index[0][2])), f"{wi.iloc[0]:.1f}"

CC = pd.read_csv(S / "calibration_coverage.csv")
for cal, tc in (("own", "Own"), ("pooled", "Pooled")):
    for exp, t in (("instance", "Inst"), ("period", "Fwd")):
        g = CC[(CC.calibration == cal) & (CC.exp == exp)]
        nums[f"clsCovBand{tc}{t}"] = p0(g[g.heldout_model != OUTSIDE].coverage_cal.mean())
        nums[f"clsCovBand{tc}{t}Min"] = p0(g[g.heldout_model != OUTSIDE].coverage_cal.min())
        nums[f"clsCovBand{tc}{t}Salamandra"] = p0(g[g.heldout_model == OUTSIDE].coverage_cal.mean())
        nums[f"clsBandWidth{tc}{t}"] = f"{100 * g[g.heldout_model != OUTSIDE].width_cal.mean():.1f}"

# lexical estimator and Binoculars under the same protocols
RB = pd.read_parquet(R / "robust" / "robust.parquet")
RB["ae"] = (RB.alpha_hat - RB.alpha).abs() * 100
lexi = RB[(RB.exp == "instance") & (RB.setting == "adapted|out")]
lexf = RB[(RB.exp == "period") & (RB.setting == "adapted|out")]
nums["lexMaeInst"], nums["lexMaeFwd"] = f"{lexi.ae.mean():.1f}", f"{lexf.ae.mean():.1f}"
nums["lexFloorInst"], nums["lexFloorFwd"] = f"{100 * lexi[lexi.alpha == 0].alpha_hat.mean():.1f}", f"{100 * lexf[lexf.alpha == 0].alpha_hat.mean():.1f}"
BQ = pd.read_parquet(S / "quant_binoculars.parquet")
BQ = BQ[~BQ.heldout_model.str.startswith("group:")]
BQ["ae"] = (BQ.alpha_hat - BQ.alpha).abs() * 100
old = json.load(open(O / "numbers.json"))


def cell(df, meth, exp):
    g = df[(df.method == meth) & (df.exp == exp)]
    return (f"{g.ae.mean():.1f}", f"{100 * g[g.alpha == 0].alpha_hat.mean():.1f}") if len(g) else ("--", "--")


KEYS = {"hdyx": "HdyEq", "hdyw": "HdyCur", "acc_sp": "Acc", "ms": "Ms", "em": "Em", "cc": "Cc", METH: "Hdy"}
ROWS = [("Classifier + HDy, instance calibration (ours)", METH), ("\\quad equal weights, all generators", "hdyx"),
        ("\\quad vendor shares, current models only", "hdyw"),
        ("Classifier + adjusted count at 2\\% FPR", "acc_sp"), ("Classifier + median sweep", "ms"), ("Classifier + EM prior shift", "em"),
        ("Classifier, classify and count", "cc")]
L = [r"\begin{table}[t]",
     r"\caption{Benchmark with the generator held out: mean absolute error of the estimated share (pp) over $\alpha\in\{0, 0.05, 0.1, 0.2, 0.3, 0.5\}$ and estimate at $\alpha=0$ (floor, \%), under the three protocols. Language: calibration and test at the language level; Instance: mixtures of the texts of one instance, calibration on a disjoint part of its pre-ChatGPT texts; Forward: calibration on the texts of 2020 and earlier, mixtures of the texts of 2021--2022. The lexical estimator and the detectors with error correction were evaluated in the language protocol on a finer grid of shares.}",
     r"\label{tab:cls_quant}", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{MAE} & \multicolumn{2}{c}{Floor} \\", r"Method & Lang. & Inst. & Forw. & Inst. & Forw. \\", r"\midrule"]
for label, m in ROWS:
    if m == "hdyw" and METH == "hdyw" and "current models" in label:
        continue
    a, b, c = cell(Q, f"{m}_lang", "main"), cell(Q, f"{m}_inst", "instance"), cell(Q, f"{m}_inst", "period")
    L.append(f"{label} & {a[0]} & {b[0]} & {c[0]} & {b[1]} & {c[1]} \\\\")
    nums["clsMae" + KEYS[m] + "Lang"] = a[0]
    for t, v in (("Inst", b), ("Fwd", c)):
        nums["clsMae" + KEYS[m] + t] = v[0]
    if m == METH:   # the same estimator, separately for the pairs calibrated on their own texts and on the language pool
        L.append(f"\\quad pairs calibrated on their own texts & -- & {nums['clsMaeInstOwn']} & {nums['clsMaeFwdOwn']} & {nums['clsFloorInstOwn']} & {nums['clsFloorFwdOwn']} \\\\")
        L.append(f"\\quad pairs calibrated on the language pool & -- & {nums['clsMaeInstPooled']} & {nums['clsMaeFwdPooled']} & {nums['clsFloorInstPooled']} & {nums['clsFloorFwdPooled']} \\\\")
a, b, c = cell(Q, f"{METH}_lang", "main"), cell(Q, f"{METH}_lang", "instance"), cell(Q, f"{METH}_lang", "period")
L.append(f"Classifier + HDy, language calibration & {a[0]} & {b[0]} & {c[0]} & {b[1]} & {c[1]} \\\\")
nums["clsMaeHdyLangCalInst"], nums["clsMaeHdyLangCalFwd"] = b[0], c[0]
L.append(r"\midrule")
L.append(f"Lexical distributional estimator & {old['benchMaeMean']} & {nums['lexMaeInst']} & {nums['lexMaeFwd']} & {nums['lexFloorInst']} & {nums['lexFloorFwd']} \\\\")
BM = METH if (BQ.method == MI).any() else "hdyw"
a, b, c = cell(BQ, f"{BM}_lang", "main"), cell(BQ, f"{BM}_inst", "instance"), cell(BQ, f"{BM}_inst", "period")
L.append(f"Binoculars score + HDy, instance calibration & {a[0]} & {b[0]} & {c[0]} & {b[1]} & {c[1]} \\\\")
nums["binoMaeLang"], nums["binoMaeInst"], nums["binoMaeFwd"], nums["binoFloorInst"], nums["binoFloorFwd"] = a[0], b[0], c[0], b[1], c[1]
for lab, k in (("Open Pangram + Rogan--Gladen", "benchMaePangramRG"), ("Binoculars + Rogan--Gladen", "benchMaeBinocularsRG"),
               ("Fast-DetectGPT + Rogan--Gladen", "benchMaeFastdetectRG")):
    L.append(f"{lab} & {old[k]} & -- & -- & -- & -- \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_cls_quant.tex", L)

# per held-out generator, held-out vendor, held-out mode and prompt variant
QH = pd.read_parquet(S / "quant_hold.parquet")
QH["ae"], QH["b"] = (QH.alpha_hat - QH.alpha).abs() * 100, (QH.alpha_hat - QH.alpha) * 100


def cov_final(g):
    lo, hi = band(g.alpha_hat.values, ~g.own.astype(bool).values)
    if "ci_lo" in g:  # union with the bootstrap interval, as in the study
        lo, hi = np.fmin(lo, g.ci_lo.values), np.fmax(hi, g.ci_hi.values)
    return ((lo <= g.alpha + 1e-9) & (g.alpha <= hi + 1e-9)).mean()


# errors by language (forward protocol unless stated)
LANGN = {"ca": "Catalan", "de": "German", "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French", "it": "Italian"}
L = [r"\begin{table}[t]",
     r"\caption{Error of the estimator by language (generator held out): number of instance--language pairs in the forward protocol and of those calibrated on their own texts, mean absolute error (pp) in the instance and forward protocols, estimate at $\alpha=0$ in the forward protocol (\%), coverage of the interval (\%) and share of mixtures in which the lower bound does not exceed the true share (\%, pairs with own calibration). Swedish and Portuguese have too few pre-ChatGPT texts for the benchmark.}",
     r"\label{tab:cls_langs}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{Pairs} & \multicolumn{2}{c}{MAE} & Floor & Cover. & Bound \\", r"Language & all & own & Inst. & Forw. & Forw. & Forw. & Forw. \\", r"\midrule"]
for lg, name in LANGN.items():
    gi, gf = M[(M.exp == "instance") & (M.lang == lg)], M[(M.exp == "period") & (M.lang == lg)]
    if not len(gf):
        continue
    gb = gf[gf.ci_lo.notna()]
    L.append(f"{name} & {gf.instance.nunique()} & {gf[gf.own].instance.nunique()} & {gi.ae.mean():.1f} & {gf.ae.mean():.1f} & "
             f"{100 * gf[gf.alpha == 0].alpha_hat.mean():.1f} & {p0(cov_final(gb))} & {p0((gb[gb.own].lb_lo <= gb[gb.own].alpha + 1e-9).mean()) if gb.own.any() else '--'} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_cls_langs.tex", L)
gb = M[(M.exp == "period") & M.ci_lo.notna()]
nums["clsCovUnionFwd"] = p0(cov_final(gb))
nums["clsCovUnionFwdOwn"] = p0(cov_final(gb[gb.own]))
nums["clsCovUnionFwdNoSal"] = p0(cov_final(gb[gb.heldout_model != OUTSIDE]))


def lbv(g):
    g = g[g.lb_lo.notna() & g.own.astype(bool)]
    return (g.lb_lo <= g.alpha + 1e-9).mean()


L = [r"\begin{table}[t]",
     r"\caption{Error of the estimator by held-out component (instance calibration): mean absolute error (pp) in the instance and forward protocols, mean bias in the forward protocol for $\alpha\geq 0.05$ (pp), coverage of the calibrated interval (\%, band fitted without the held-out generator; instances calibrated on their own texts) and share of mixtures in which the generator-free lower bound does not exceed the true share (\%, same instances). The last rows hold out all the models of one vendor and, marked with an asterisk, one intervention mode or one prompt variant for every generator; the values with an asterisk refer to the instance protocol.}",
     r"\label{tab:cls_generators}", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{MAE} & Bias & Cover. & Bound \\", r"Held out & Inst. & Forw. & Forw. & Forw. & Forw. \\", r"\midrule"]
for m, lab in GEN.items():
    gi, gf = M[(M.exp == "instance") & (M.heldout_model == m)], M[(M.exp == "period") & (M.heldout_model == m)]
    cv = CC[(CC.calibration == "own") & (CC.exp == "period") & (CC.heldout_model == m)].coverage_cal.mean()
    L.append(f"{lab} & {gi.ae.mean():.1f} & {gf.ae.mean():.1f} & ${sgn(gf[gf.alpha >= 0.05].b.mean())}$ & {p0(cv)} & {p0(lbv(gf))} \\\\")
L.append(r"\midrule")
g = QG[QG.method == MI]
gi, gf = g[g.exp == "instance"], g[g.exp == "period"]
L.append(f"All OpenAI models & {gi.ae.mean():.1f} & {gf.ae.mean():.1f} & ${sgn(gf[gf.alpha >= 0.05].b.mean())}$ & {p0(cov_final(gf[gf.own.astype(bool)]))} & {p0(lbv(gf))} \\\\")
nums["clsCovBandOpenAI"] = p0(cov_final(gf[gf.own.astype(bool)]))
HH = QH[(QH.method == MI) & (QH.exp == "instance")]
for key, lab, t in (("mode:rewrite", "Mode: rewrite", "Rewrite"), ("mode:generate", "Mode: generate", "Generate"),
                    ("mode:translate", "Mode: translate", "Translate"), ("variant:0", "Prompt variant 1", "VarA"),
                    ("variant:1", "Prompt variant 2", "VarB"), ("variant:2", "Prompt variant 3", "VarC")):
    g = HH[HH.heldout_model == key]
    L.append(f"{lab} & {g.ae.mean():.1f} & -- & ${sgn(g[g.alpha >= 0.05].b.mean())}{{}}^\\ast$ & {p0(cov_final(g[g.own.astype(bool)]))}$^\\ast$ & {p0(lbv(g))}$^\\ast$ \\\\")
    nums[f"clsHold{t}Mae"] = f"{g.ae.mean():.1f}"
    nums[f"clsHold{t}Rec"] = f"{(g[g.alpha > 0].alpha_hat / g[g.alpha > 0].alpha).mean():.2f}"
    nums[f"clsHold{t}Cov"] = p0(cov_final(g[g.own.astype(bool)]))
    nums[f"clsHold{t}Lb"] = p0(lbv(g))
L += [r"\bottomrule", r"\multicolumn{6}{@{}l}{$^\ast$ instance protocol.}", r"\end{tabular}", r"\end{table}"]
w("tab_cls_generators.tex", L)

# ------------------------------------------------------------------ external tests: MULTITuDE, RAID, intensity
MT = pd.read_parquet(R / "multitude" / "cls_benchmark_aug.parquet")
MT = MT[MT.reference == "weighted"].copy()
MT["ae"] = (MT.alpha_hat - MT.alpha).abs() * 100
pos = MT[MT.alpha > 0].assign(rec=lambda x: x.alpha_hat / x.alpha, rec_lb=lambda x: x.lb / x.alpha)
mt = pd.DataFrame({"mae": MT.groupby("heldout_model").ae.mean(), "rec": pos.groupby("heldout_model").rec.mean(),
                   "rec_lb": pos.groupby("heldout_model").rec_lb.mean()})
nums["clsMtFloor"] = f"{100 * MT[MT.alpha == 0].alpha_hat.mean():.1f}"
GROUPS = [("Assistants of 2023 (MULTITuDE v1), absent from training",
           [("mt1_gpt-4", "GPT-4", "MtGptFour"), ("mt1_gpt-3.5-turbo", "GPT-3.5-turbo", "MtGptThree"), ("mt1_text-davinci-003", "text-davinci-003", "MtDavinci")]),
          ("Open and completion models of 2023, absent from training",
           [("mt1_vicuna-13b", "Vicuna 13B", "MtVicuna"), ("mt1_alpaca-lora-30b", "Alpaca-LoRA 30B", "MtAlpaca"), ("mt1_llama-65b", "Llama 65B (completion)", "MtLlama"),
            ("mt1_opt-66b", "OPT 66B (completion)", "MtOpt")]),
          ("Generators of the study, held out",
           [("gpt-5.4-mini", "GPT-5.4 mini", "MtGptFive"), ("gemini-3.6-flash", "Gemini 3.6 Flash", "MtGemini"), ("claude-haiku-5-5", "Claude Haiku 5.5", "MtClaude"),
            ("gpt-4o", "GPT-4o", "MtFourO"), ("gpt-4o-mini", "GPT-4o mini", "MtFourOMini"), ("gpt-oss-20b", "gpt-oss-20b", "MtGptOss"),
            ("mistral-small-24b", "Mistral Small 3 24B", "MtMistral"), ("qwen2.5-7b", "Qwen 2.5 7B", "MtQwen"), ("eurollm-9b", "EuroLLM 9B", "MtEuro"),
            ("llama-3.1-8b", "Llama 3.1 8B", "MtLlamaThree"), ("salamandra-7b", "Salamandra 7B", "MtSalamandra")])]
L = [r"\begin{table}[t]",
     r"\caption{External tests on texts of the assistants of 2023. Top: MULTITuDE news in five languages (Catalan, German, English, Spanish, Portuguese), mixtures with human news; MAE (pp) over $\alpha\in\{0,\dots,0.5\}$, recovered fraction $\hat\alpha/\alpha$ and recovered fraction of the generator-free lower bound. Bottom: RAID, English Reddit posts and reviews, a domain and models absent from the training data; recovered fraction of the lower bound, by decoding.}",
     r"\label{tab:cls_external}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
     r"Held-out generator & MAE & $\hat\alpha/\alpha$ & Bound$/\alpha$ \\", r"\midrule"]
for title, rows in GROUPS:
    L.append(r"\multicolumn{4}{@{}l}{\emph{" + title + r"}} \\")
    for tag, lab, key in rows:
        if tag in mt.index:
            r = mt.loc[tag]
            L.append(f"{lab} & {r.mae:.1f} & {r.rec:.2f} & {r.rec_lb:.2f} \\\\")
            nums[f"cls{key}Mae"], nums[f"cls{key}Rec"], nums[f"cls{key}Lb"] = f"{r.mae:.1f}", f"{r.rec:.2f}", f"{r.rec_lb:.2f}"
cur = mt.loc[[t for t, _, _ in GROUPS[2][1] if t in mt.index and t != "salamandra-7b"]]
nums["clsMtCurMin"], nums["clsMtCurMax"] = f"{cur.mae.min():.1f}", f"{cur.mae.max():.1f}"
RD = pd.read_csv(R / "raid_lb.csv")
rp = RD[RD.alpha > 0].assign(rec=lambda x: x.lb / x.alpha, over=lambda x: x.lb > x.alpha + 0.01)
rr = rp.groupby(["model", "decoding"]).agg(rec=("rec", "mean"), over=("over", "mean"))
L.append(r"\midrule")
L.append(r"\multicolumn{4}{@{}l}{\emph{RAID (English, 2023): lower bound only}} \\")
for m, lab, key in (("chatgpt", "ChatGPT (GPT-3.5-turbo)", "RaidChatgpt"), ("gpt4", "GPT-4", "RaidGptFour"), ("gpt3", "GPT-3 (text-davinci-002)", "RaidGptThree")):
    a, b = rr.loc[(m, "greedy")].rec, rr.loc[(m, "sampling")].rec
    L.append(f"{lab}, greedy / sampling & -- & -- & {a:.2f} / {b:.2f} \\\\")
    nums[f"cls{key}Greedy"], nums[f"cls{key}Sampling"] = f"{a:.2f}", f"{b:.2f}"
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_cls_external.tex", L)
nums["clsRaidOver"] = p0(rp.over.mean())
nums["clsRaidOverMax"] = p1((RD[RD.alpha > 0].lb - RD[RD.alpha > 0].alpha).max())
nums["clsRaidFloor"] = p1(RD[RD.alpha == 0].lb.mean())
nums["clsRaidHumanAbove"] = p1(RD.human_above.mean())

IT = pd.read_csv(R / "intensity" / "cls_intensity.csv")
IT["rec"], IT["rec_lb"] = IT.alpha_hat / IT.alpha, IT.lb / IT.alpha
jt = pd.read_csv(R / "intensity" / "jaccard_tiers.csv")
jt = jt[jt.lang.isin(["en", "de", "fr", "it", "fi"])].groupby("mode")["50%"].agg(["min", "max"])
L = [r"\begin{table}[t]",
     r"\caption{Contribution of a text to the estimate by editing intensity: recovered fraction $\hat\alpha/\alpha$ of the estimate and of the generator-free lower bound when the generated component of a mixture has a given editing intensity and the calibration is fixed (rewrites and generations of the reference generators). Rewrites are split into terciles of lexical overlap with the original (Jaccard of content lemmas; range of the per-language medians). Mean over five languages and $\alpha\in\{0.1, 0.2, 0.3\}$.}",
     r"\label{tab:cls_intensity}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
     r"Generated component & Overlap & $\hat\alpha/\alpha$ & Bound$/\alpha$ \\", r"\midrule"]
for tier, lab, key in (("rewrite_light", "Rewrite, light edit", "Light"), ("rewrite_mid", "Rewrite, medium edit", "Mid"), ("rewrite_heavy", "Rewrite, heavy edit", "Heavy"),
                       ("translate", "Translation", "Translate"), ("generate", "Generation from title", "Generate")):
    g = IT[IT.tier == tier]
    ov = f"{jt.loc[tier, 'min']:.2f}--{jt.loc[tier, 'max']:.2f}" if tier in jt.index else "--"
    L.append(f"{lab} & {ov} & {g.rec.mean():.2f} & {g.rec_lb.mean():.2f} \\\\")
    nums[f"clsInt{key}"], nums[f"clsInt{key}Lb"] = f"{g.rec.mean():.2f}", f"{g.rec_lb.mean():.2f}"
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_cls_intensity.tex", L)
nums["clsIntLightAtThirty"] = f"{100 * IT[(IT.tier == 'rewrite_light') & (IT.alpha == 0.3)].alpha_hat.mean():.0f}"

# ------------------------------------------------------------------ study cells (analysis_cls3), study set, bands
ST = pd.read_csv(R / "processed_stats.csv")
INC = set(zip(ST.instance, ST.lang))


def base(i):
    return "eu_hys" if i.startswith("eu_hys") else "regulations_gov" if i.startswith("regulations_gov") else i


def study(df):
    k = [(base(i), l) in INC for i, l in zip(df.instance, df.lang)]
    return df[k].copy()


def with_band(df):
    df = df.copy()
    df["band_lo"], df["band_hi"] = band(df.alpha.values, df.pooled_ref.values)
    # the band depends on the estimate only; its union with the bootstrap interval adds the sampling error of small slices
    if {"ci_lo", "ci_hi"} <= set(df.columns):
        df["band_only_lo"], df["band_only_hi"] = df.band_lo, df.band_hi
        df["band_lo"] = np.fmin(df.band_lo, df.ci_lo)
        df["band_hi"] = np.fmax(df.band_hi, df.ci_hi)
    # the lower bound needs the human scores of the instance itself: with the calibration on the language pool it was not
    # valid in the benchmark, so it is not reported for those cells and counts as zero in the aggregates
    for c in ("lb", "lb_lo"):
        if c in df:
            df.loc[df.pooled_ref.astype(bool), c] = 0.0
    return df


def source(i):
    return "hys" if i.startswith("eu_hys") else "regs" if i.startswith("regulations_gov") else "madrid" if i == "decide_madrid" else "decidim"


cells = {k: with_band(study(pd.read_csv(A / f"alpha_cells_{k}.csv"))) for k in
         ("period", "instance_year", "half", "period_notemplate", "instance_year_notemplate", "half_notemplate", "consultation")}
for k, d in cells.items():
    d.to_csv(OUT / f"cells_{k}.csv", index=False)
cp, cy, ch = cells["period"], cells["instance_year"], cells["half"]
cpn, cyn = cells["period_notemplate"], cells["instance_year_notemplate"]
pairs = cp.groupby(["instance", "lang"]).ngroups
nums["clsStudyPairs"] = int(pairs)
nums["clsStudyPooledPairs"] = int(cp[cp.pooled_ref].groupby(["instance", "lang"]).ngroups)
nums["clsStudyTexts"] = fint(cp.n_docs.sum())
nums["clsNLangsStudy"] = int(cp.lang.nunique())


def wmean(d, col="alpha"):
    return float((d[col] * d.n_docs).sum() / d.n_docs.sum()) if len(d) else np.nan


for per, t in (("pre", "Pre"), ("post", "Post")):
    d, dn = cp[cp.period == per], cpn[cpn.period == per]
    nums[f"clsAlpha{t}All"] = p1(wmean(d))
    nums[f"clsLb{t}All"] = p1(wmean(d, "lb_lo"))
    nums[f"clsAlpha{t}AllNoTpl"] = p1(wmean(dn))
    for src, ts in (("hys", "Hys"), ("regs", "Regs"), ("decidim", "Decidim"), ("madrid", "Madrid")):
        ds, dns = d[d.instance.map(source) == src], dn[dn.instance.map(source) == src]
        nums[f"clsAlpha{t}{ts}"] = p1(wmean(ds))
        nums[f"clsLb{t}{ts}"] = p1(wmean(ds, "lb_lo"))
        nums[f"clsAlpha{t}{ts}NoTpl"] = p1(wmean(dns))
post = cp[cp.period == "post"]
nums["clsPostShareHys"] = p0(post[post.instance.map(source) == "hys"].n_docs.sum() / post.n_docs.sum())
nums["clsTplSharePost"] = p1(wmean(post, "template_share"))
nums["clsTplSharePre"] = p1(wmean(cp[cp.period == "pre"], "template_share"))
YEAR = {2023: "TwentyThree", 2024: "TwentyFour", 2025: "TwentyFive", 2026: "TwentySix"}
# pooled yearly figures over the study set, with and without templated letters
for y, t in YEAR.items():
    d, dn = cy[cy.year == y], cyn[cyn.year == y]
    nums[f"clsYear{t}"], nums[f"clsYear{t}NoTpl"], nums[f"clsYear{t}Lb"] = p1(wmean(d)), p1(wmean(dn)), p1(wmean(d, "lb_lo"))
    dd = d[d.instance.map(source).isin(["decidim", "madrid"])]
    nums[f"clsYear{t}Local"] = p1(wmean(dd))
    nums[f"clsYear{t}LocalLb"] = p1(wmean(dd, "lb_lo"))
pre_y = cy[cy.year <= 2022]
nums["clsPreYearMax"] = p1(pre_y.alpha.max())
nums["clsPreYearCells"] = int(len(pre_y))
nums["clsPreYearAboveOne"] = int((pre_y.alpha > 0.01).sum())


def key_cell(inst, lang, year, d=None):
    d = cy if d is None else d
    r = d[(d.instance == inst) & (d.lang == lang) & (d.year == year)]
    return r.iloc[0] if len(r) else None


KEY = {"Madrid": ("decide_madrid", "es"), "Bcn": ("barcelona", "ca"), "Gencat": ("participa_gencat", "ca"), "Ccomptes": ("participationcitoyenne_ccomptes", "fr"),
       "HysEn": ("eu_hys_citizens", "en"), "HysFr": ("eu_hys_citizens", "fr"), "HysDe": ("eu_hys_citizens", "de"), "HysEs": ("eu_hys_citizens", "es"),
       "HysIt": ("eu_hys_citizens", "it"), "HysSv": ("eu_hys_citizens", "sv"), "HysPt": ("eu_hys_citizens", "pt"),
       "OrgEn": ("eu_hys_organisations", "en"), "OrgDe": ("eu_hys_organisations", "de"), "OrgFr": ("eu_hys_organisations", "fr"),
       "Epa": ("regulations_gov_epa", "en"), "Dot": ("regulations_gov_dot", "en"), "Meta": ("meta_decidim", "en"), "Turku": ("asukasbudjetti_turku", "fi")}
for tag, (inst, lang) in KEY.items():
    for y, t in YEAR.items():
        r, rn = key_cell(inst, lang, y), key_cell(inst, lang, y, cyn)
        if r is not None:
            nums[f"cls{tag}{t}"], nums[f"cls{tag}{t}Lo"], nums[f"cls{tag}{t}Hi"] = p1(r.alpha), p1(r.band_lo), p1(r.band_hi)
            nums[f"cls{tag}{t}Lb"] = p1(r.lb_lo)
            if "alpha_gmin" in r.index:
                nums[f"cls{tag}{t}Gmin"], nums[f"cls{tag}{t}Gmax"] = p1(r.alpha_gmin), p1(r.alpha_gmax)
        if rn is not None:
            nums[f"cls{tag}{t}NoTpl"], nums[f"cls{tag}{t}NoTplLb"] = p1(rn.alpha), p1(rn.lb_lo)

# pairs whose estimate on pre-ChatGPT texts in the forward benchmark exceeds 5%: reported, marked and not interpreted
_fb = M[(M.exp == "period") & (M.alpha == 0)].groupby(["instance", "lang"]).alpha_hat.mean()
BADPAIRS = set(_fb[_fb > 0.05].index) & set(zip(cp.instance, cp.lang))
nums["clsBadPairs"] = len(BADPAIRS)
for (_i, _l) in BADPAIRS:
    nums["clsBadPairFloor"] = p1(_fb[(_i, _l)])
    nums["clsBadPairMae"] = f"{M[(M.exp == 'period') & (M.instance == _i) & (M.lang == _l)].ae.mean():.1f}"
    nums["clsBadPairPostN"] = fint(cp[(cp.instance == _i) & (cp.lang == _l) & (cp.period == "post")].n_docs.sum())
print("pairs with a forward benchmark floor above 5%:", BADPAIRS)
# table: yearly estimates with bands
piv = cy[cy.year >= 2023]
keep = piv.groupby(["instance", "lang"]).filter(lambda g: len(g) >= 2)
order = keep.groupby(["instance", "lang"]).n_docs.sum().sort_values(ascending=False).index
ORDER_SRC = {"madrid": 0, "decidim": 1, "hys": 2, "regs": 3}
order = sorted(order, key=lambda k: (ORDER_SRC[source(k[0])], k[0].replace("eu_hys_unknown", "eu_hys_zz"), -keep[(keep.instance == k[0]) & (keep.lang == k[1])].n_docs.sum()))
N = pd.read_csv(A / "null_calibration.csv")
N = study(N)
fwd = N[N.slice == "2021-22"].set_index(["instance", "lang"]).alpha
yrmax = N[N.slice.str.startswith("year=")].groupby(["instance", "lang"]).alpha.max()
L = [r"\begin{table*}[t]",
     r"\caption{Estimated share of LLM-written or LLM-rewritten texts per year (\%), with the benchmark-calibrated interval (nominal 95\%) in brackets, for the instance--language pairs with at least two post-ChatGPT years of at least \minDocs{} texts. Floor: largest estimate on a single pre-ChatGPT year of the pair, calibrated without that year. $^\dagger$: pair calibrated on the language pool (fewer than \minRef{} pre-ChatGPT texts), with the wider interval of that case. $^\ddagger$: pair whose estimate on pre-ChatGPT texts exceeds 5\% in the forward benchmark; its estimates are not interpreted. Exact duplicates are counted once; templated letters are included (Table~\ref{tab:cls_template}). 2026 covers January to November.}",
     r"\label{tab:cls_years}", r"\centering\scriptsize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}llrllll@{}}", r"\toprule",
     r"Instance & Lang. & Floor & 2023 & 2024 & 2025 & 2026 \\", r"\midrule"]
last = None
for inst, lang in order:
    if last is not None and source(inst) != last:
        L.append(r"\addlinespace")
    last = source(inst)
    g = keep[(keep.instance == inst) & (keep.lang == lang)].set_index("year")
    vals = [f"{p1(g.loc[y].alpha)} [{p1(g.loc[y].band_lo)}, {p1(g.loc[y].band_hi)}]" if y in g.index else "--" for y in YEAR]
    fl = p1(yrmax[(inst, lang)]) if (inst, lang) in yrmax.index else "--"
    L.append(f"{tex(NAME.get(inst, inst))}{'$^' + chr(92) + 'dagger$' if g.pooled_ref.any() else ''}{'$^' + chr(92) + 'ddagger$' if (inst, lang) in BADPAIRS else ''} & {lang} & {fl} & " + " & ".join(vals) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
w("tab_cls_years.tex", L)
nums["clsYearsPairs"] = len(order)
# how many pairs rise from their first to their last post-ChatGPT year, and for how many the last interval excludes zero
first_last = keep.sort_values("year").groupby(["instance", "lang"]).agg(first=("alpha", "first"), last=("alpha", "last"), lo=("band_lo", "last"), lb=("lb_lo", "last"))
nums["clsYearsUp"] = int((first_last["last"] > first_last["first"]).sum())
nums["clsYearsLastAboveZero"] = int((first_last.lo > 0).sum())
nums["clsYearsLastLbAboveZero"] = int((first_last.lb > 0).sum())
last_cells = cy[cy.year >= 2025]
nums["clsLateCells"] = int(len(last_cells))
nums["clsLateMedian"] = p1(last_cells.alpha.median())
nums["clsLateBelowHalf"] = int((last_cells.alpha < 0.5).sum())
nums["clsLateLbAbove"] = int((last_cells.lb_lo > last_cells.alpha + 1e-9).sum())
nums["clsYearCells"] = int((cy.year >= 2023).sum())
nums["clsBootWidened"] = int((((cy.band_lo < cy.band_only_lo - 0.005) | (cy.band_hi > cy.band_only_hi + 0.005)) & (cy.year >= 2023)).sum())
nums["clsLateLbAboveMax"] = f"{100 * max((last_cells.lb_lo - last_cells.alpha).max(), 0):.1f}"
FRM = ["jeparticipe_toulouse", "oye_participer_lyon", "ecrivons_angers", "decidonsensemble_tours", "participation_bordeaux_metropole",
       "participez_nouvelle_aquitaine", "participez_nanterre"]
frm = cy[cy.instance.isin(FRM) & (cy.lang == "fr") & (cy.year >= 2023)]
nums["clsFrMuniMax"] = f"{np.ceil(100 * frm.alpha.max()):.0f}"

# null calibration table
half = N[N.slice == "fold1"].set_index(["instance", "lang"])
loyo = N[N.slice.str.startswith("year=")]
for sl, t in (("fold1", "Half"), ("2021-22", "Fwd")):
    g = N[N.slice == sl]
    nums[f"clsNull{t}N"], nums[f"clsNull{t}Mean"], nums[f"clsNull{t}Max"] = int(len(g)), p1(g.alpha.mean()), p1(g.alpha.max())
nums["clsNullYearN"], nums["clsNullYearMean"], nums["clsNullYearMax"] = int(len(loyo)), p1(loyo.alpha.mean()), p1(loyo.alpha.max())
nums["clsNullYearAboveOne"] = int((loyo.alpha > 0.01).sum())
nums["clsNullLbMax"] = p1(N.lb_lo.max())
nums["clsNullMadridMax"] = p1(loyo[loyo.instance == "decide_madrid"].alpha.max())
rows = half.join(fwd.rename("fwd"), how="left").join(yrmax.rename("ymax"), how="left").sort_values("n_docs", ascending=False)
L = [r"\begin{table}[t]",
     r"\caption{False-positive floors on pre-ChatGPT texts: estimated share (\%) on a held-out half of the pre-ChatGPT texts of each pair (calibration on the other half), on the texts of 2021--2022 (calibration on 2020 and earlier) and largest value over single pre-ChatGPT years (calibration without that year; years with at least \minDocs{} texts). $n$: held-out texts of the first test; --: not enough texts. HYS: EU Have Your Say portal, by respondent type (unknown: type not recorded).}",
     r"\label{tab:cls_null}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
     r"Instance & Lang. & $n$ & Half & 2021--22 & Year max \\", r"\midrule"]
for (inst, lang), r in rows.iterrows():
    f = lambda v: p1(v) if pd.notna(v) else "--"
    L.append(f"{tex(NAME.get(inst, inst))} & {lang} & {int(r.n_docs):,} & {p1(r.alpha)} & {f(r.fwd)} & {f(r.ymax)} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_cls_null.tex", L)

# templated letters
TP = study(pd.read_csv(A / "alpha_template.csv"))
tp = TP.pivot_table(index=["instance", "lang"], columns="is_template", values=["alpha", "n_docs"])
allp = cp[cp.period == "post"].groupby(["instance", "lang"]).apply(lambda d: pd.Series({"n": d.n_docs.sum(), "alpha": wmean(d), "tpl": wmean(d, "template_share")}))
tpl = allp.join(tp["alpha"].rename(columns={False: "a_out", True: "a_in"}), how="left")
tn = tp["n_docs"].rename(columns={False: "n_out", True: "n_in"}).fillna(0)
tpl = tpl.join(tn, how="left")
tpl["n"] = tpl.n_out.fillna(0) + tpl.n_in.fillna(0)
tpl["tpl"] = tpl.n_in / tpl.n
tpl.loc[tpl.n_out < 150, "a_out"] = np.nan
tpl.loc[tpl.n_in < 150, "a_in"] = np.nan
tpl = tpl[tpl.tpl >= 0.05].sort_values("n", ascending=False)
L = [r"\begin{table}[t]",
     r"\caption{Templated letters after ChatGPT: share of texts assembled from sentences shared by more than five texts of the instance (\%), and estimated share of LLM-written texts (\%) over all texts, without the templated letters and among them, for the pairs in which they are at least 5\% of the texts. --: fewer than \minDocs{} templated texts.}",
     r"\label{tab:cls_template}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
     r"Instance & Lang. & Texts & Templ. & All & Without & Among \\", r"\midrule"]
for (inst, lang), r in tpl.iterrows():
    f = lambda v: p1(v) if pd.notna(v) else "--"
    L.append(f"{tex(NAME.get(inst, inst))} & {lang} & {int(r.n):,} & {p1(r.tpl)} & {p1(r.alpha)} & {f(r.a_out)} & {f(r.a_in)} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_cls_template.tex", L)
nums["clsTplPairs"] = int(len(tpl))
hy = TP[TP.instance == "eu_hys_citizens"]
nums["clsTplHysIn"], nums["clsTplHysOut"] = p0(wmean(hy[hy.is_template])), p0(wmean(hy[~hy.is_template]))
ot = TP[~TP.instance.str.startswith("eu_hys") & TP.is_template]
nums["clsTplOtherIn"] = p1(wmean(ot))
nums["clsTplHysCamp"] = p0(wmean(hy[hy.is_template], "campaign_share"))  # templated letters also in a near-duplicate cluster
TPL = pd.read_parquet(R / "templates.parquet")
nums["clsTplTexts"], nums["clsTplAllTexts"] = fint(TPL.is_template.sum()), fint(len(TPL))

# the two Have Your Say consultations of autumn 2025
CO = cells["consultation"]
for pid, tag in (("20315", "Tobacco"), ("20100", "Simplif")):
    g = CO[(CO.consultation.astype(str).str.replace(".0", "", regex=False) == pid) & (CO.instance == "eu_hys_citizens")]
    nums[f"cls{tag}N"] = fint(g.n_docs.sum())
    nums[f"cls{tag}Alpha"], nums[f"cls{tag}Tpl"] = p0(wmean(g)), p0(wmean(g, "template_share"))
    for lang in ("fr", "sv", "de", "en", "es", "it"):
        r = g[g.lang == lang]
        if len(r):
            r = r.iloc[0]
            L2 = lang.capitalize()
            nums[f"cls{tag}{L2}N"], nums[f"cls{tag}{L2}Alpha"], nums[f"cls{tag}{L2}Lb"] = fint(r.n_docs), p0(r.alpha), p0(r.lb_lo)
            nums[f"cls{tag}{L2}Tpl"] = p0(r.template_share)
            nums[f"cls{tag}{L2}Lo"], nums[f"cls{tag}{L2}Hi"] = p0(r.band_lo), p0(r.band_hi)

if (A / "alpha_acc_instance_year.csv").exists():
    AC = pd.read_csv(A / "alpha_acc_instance_year.csv").merge(cy[["instance", "lang", "year", "alpha"]], on=["instance", "lang", "year"])
    AC = AC[AC.year >= 2023]
    nums["clsAccCells"], nums["clsAccCorr"] = int(len(AC)), f"{np.corrcoef(AC.alpha, AC.alpha_acc)[0, 1]:.2f}"
    nums["clsAccMeanDiff"] = sgn(100 * (AC.alpha_acc - AC.alpha).mean())
    nums["clsAccMaxDiff"] = f"{100 * (AC.alpha_acc - AC.alpha).abs().max():.1f}"
    for tag, (inst, lang) in (("Madrid", ("decide_madrid", "es")), ("Bcn", ("barcelona", "ca")), ("Gencat", ("participa_gencat", "ca"))):
        r = AC[(AC.instance == inst) & (AC.lang == lang) & (AC.year == 2026)]
        if len(r):
            nums[f"clsAcc{tag}TwentySix"] = p1(r.alpha_acc.iloc[0])
            nums[f"cls{tag}TwentySixN"] = fint(r.n_docs.iloc[0])
gp = CO[(CO.instance == "participa_gencat") & (CO.lang == "ca")].copy()
dc = [c for c in gp.columns if "created" in c or "date" in c]
if dc and len(gp):
    gp = gp[pd.to_datetime(gp[dc[0]], errors="coerce").dt.year == 2026].sort_values("alpha", ascending=False)
    if len(gp):
        nums["clsGencatTopProcN"], nums["clsGencatTopProcAlpha"] = fint(gp.n_docs.iloc[0]), p0(gp.alpha.iloc[0])

# consultations: yearly medians across processes
YC = study(pd.read_csv(A / "alpha_cells_year_consult.csv"))
for tag, (inst, lang) in KEY.items():
    g = YC[(YC.instance == inst) & (YC.lang == lang)].set_index("year")
    for y, t in {2022: "TwentyTwo", **YEAR}.items():
        if y in g.index and g.loc[y].n_consultations >= 3:
            nums[f"clsMed{tag}{t}"] = p1(g.loc[y].alpha)
nums["clsConsultCells"] = int(len(CO))
nums["clsConsultPost"] = int((CO.post == 1).sum())
cpost = CO[CO.post == 1]
nums["clsConsultPostAboveTen"] = p0((cpost.alpha > 0.10).mean())
nums["clsConsultPostAboveHalf"] = int((cpost.alpha > 0.5).sum())
nums["clsConsultPreMax"] = p1(CO[CO.post == 0].alpha.max())
nums["clsConsultPreAboveFive"] = int((CO[CO.post == 0].alpha > 0.05).sum())
nums["clsConsultPre"] = int((CO.post == 0).sum())

# citizens and organisations (Have Your Say), with and without templated letters
co = []
for lang in sorted(set(post.lang)):
    row = {"lang": lang}
    for who in ("citizens", "organisations"):
        for d, t in ((post, ""), (cpn[cpn.period == "post"], "_nt")):
            g = d[(d.instance == f"eu_hys_{who}") & (d.lang == lang)]
            row[who + t] = wmean(g) if len(g) else np.nan
    co.append(row)
co = pd.DataFrame(co).dropna(subset=["citizens", "organisations"])
co.to_csv(OUT / "citizens_organisations.csv", index=False)
nums["clsOrgLangs"] = int(len(co))
nums["clsOrgLower"] = int((co.organisations < co.citizens).sum())
nums["clsOrgLowerNoTpl"] = int((co.organisations_nt < co.citizens_nt).sum())
for who, t in (("citizens", "Cit"), ("organisations", "Org")):
    for d, s in ((post, ""), (cpn[cpn.period == "post"], "NoTpl")):
        nums[f"clsHys{t}Post{s}"] = p1(wmean(d[d.instance == f"eu_hys_{who}"]))
ty = cp[(cp.period == "post") & ~cp.instance.str.startswith(("eu_hys", "regulations"))]
tyv = ty.pivot_table(index=["instance", "lang"], columns="type", values="alpha")
if {"proposal", "comment"} <= set(tyv.columns):
    both = tyv.dropna(subset=["proposal", "comment"])
    nums["clsTypePairs"] = int(len(both))
    nums["clsTypePropHigher"] = int((both.proposal > both.comment + 0.02).sum())
    nums["clsTypeCommHigher"] = int((both.comment > both.proposal + 0.02).sum())

# interrupted time series, multilevel models, placebo (study set)
its, hac = [], []
for (inst, lang), g in ch.groupby(["instance", "lang"]):
    g = g.dropna(subset=["alpha"]).sort_values("t")
    if g.t.nunique() < 8 or (g.post == 1).sum() < 3 or (g.post == 0).sum() < 3:
        continue
    X = pd.DataFrame({"const": 1.0, "t": g.t, "post": g.post, "t_post": g.t * g.post})
    m = sm.WLS(g.alpha.values, X, weights=g.n_docs.values).fit()
    row = dict(instance=inst, lang=lang, n_cells=len(g), level_shift=m.params["post"], level_p=m.pvalues["post"], slope_change=m.params["t_post"],
               slope_p=m.pvalues["t_post"], pre_mean=g[g.post == 0].alpha.mean(), post_mean=g[g.post == 1].alpha.mean())
    if (g.post == 1).sum() >= 3 and (g.post == 0).sum() >= 3:
        h = sm.OLS(g.alpha.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        row.update(hac_level=h.params["post"], hac_level_p=h.pvalues["post"], hac_slope=h.params["t_post"], hac_slope_p=h.pvalues["t_post"])
    its.append(row)
its = pd.DataFrame(its)
its.to_csv(OUT / "its.csv", index=False)
nums["clsItsPairs"] = int(len(its))
nums["clsItsPostAbove"] = int((its.post_mean > its.pre_mean).sum())
nums["clsItsSlopeSig"] = int((its.slope_p < 0.05).sum())
nums["clsItsLevelSig"] = int((its.level_p < 0.05).sum())
hh = its.dropna(subset=["hac_level_p"])
nums["clsItsHacPairs"] = int(len(hh))
nums["clsItsHacSlopeSig"] = int(((hh.hac_slope_p < 0.05) & (hh.hac_slope > 0)).sum())
nums["clsItsHacLevelSig"] = int((hh.hac_level_p < 0.05).sum())
nums["clsItsHacLevelPos"] = int(((hh.hac_level_p < 0.05) & (hh.hac_level > 0)).sum())


def mixed(d, formula):
    d = d.dropna(subset=["alpha"])
    return smf.mixedlm(formula, d, groups=d["instance"]).fit(reml=True)


def coef(m, k, tag, scale=100):
    ci = m.conf_int().loc[k]
    nums[f"cls{tag}"], nums[f"cls{tag}Lo"], nums[f"cls{tag}Hi"] = [f"{scale * v:.1f}".replace("-0.0", "0.0") for v in (m.params[k], ci[0], ci[1])]
    nums[f"cls{tag}SE"] = f"{scale * m.bse[k]:.1f}"
    nums[f"cls{tag}P"] = "<0.001" if m.pvalues[k] < 0.001 else f"={m.pvalues[k]:.3f}"


txt = []
for label, d, tag in (("all texts", cp, ""), ("without templated letters", cpn, "NoTpl")):
    m = mixed(d.assign(post=(d.period == "post").astype(int)), "alpha ~ post + C(lang) + C(type)")
    coef(m, "post", f"MixedPost{tag}")
    nums[f"clsMixedCells{tag}"] = int(m.nobs)
    txt.append(f"=== period cells, {label} ===\n{m.summary().as_text()}")
for label, d, tag in (("all texts", ch, ""), ("without templated letters", cells["half_notemplate"], "NoTpl")):
    d = d.copy()
    if "t" not in d:
        d["t"] = d["half"].map(lambda h: int(h[:4]) + (0.25 if h.endswith("1") else 0.75)) - 2022.91
        d["post"] = (d.t > 0).astype(int)
    d["tp"] = d.t * d.post
    m = mixed(d, "alpha ~ post + tp + C(lang)")
    coef(m, "post", f"RampLevel{tag}"); coef(m, "tp", f"RampSlope{tag}")
    if not tag:
        # convex alternative, and the same ramp with cells weighted by their precision (errors clustered by instance)
        d["tpq"] = d.tp ** 2
        mq = mixed(d, "alpha ~ post + tp + tpq + C(lang)")
        coef(mq, "tpq", "RampQuad")
        dw = d.dropna(subset=["alpha", "ci_lo", "ci_hi"]).copy()
        dw["se"] = ((dw.ci_hi - dw.ci_lo) / 3.92).clip(lower=0.005)
        o = smf.ols("alpha ~ post + tp + C(lang)", dw).fit()
        tau2 = max(0.0, float(np.var(o.resid)) - float((dw.se ** 2).mean()))
        mw = smf.wls("alpha ~ post + tp + C(lang)", dw, weights=1.0 / (dw.se ** 2 + tau2)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(dw.instance)[0]})
        coef(mw, "tp", "RampSlopeW"); coef(mw, "post", "RampLevelW")
        txt.append(f"=== half-year cells, ramp with quadratic term ===\n{mq.summary().as_text()}")
        txt.append(f"=== half-year cells, ramp, precision-weighted, clustered by instance ===\n{mw.summary().as_text()}")
    txt.append(f"=== half-year cells, ramp, {label} ===\n{m.summary().as_text()}")
pl = ch[ch.post == 0].copy()
pl["post"] = (pl.t > -2.0).astype(int)
pl["tp"] = (pl.t + 2.0) * pl.post
m = mixed(pl, "alpha ~ post + tp + C(lang)")
coef(m, "post", "Placebo")
txt.append(f"=== placebo (break in December 2020, pre-ChatGPT cells) ===\n{m.summary().as_text()}")
(OUT / "models.txt").write_text("\n\n".join(txt), encoding="utf-8")

# sensitivity: equal weights over all generators, single generators, language calibration, lexical estimator
ce = study(pd.read_csv(AE / "alpha_cells_period.csv"))
mm = cp.merge(ce, on=["instance", "lang", "type", "period"], suffixes=("", "_eq"))
nums["clsEqCorr"] = f"{np.corrcoef(mm.alpha, mm.alpha_eq)[0, 1]:.3f}"
nums["clsEqPost"] = p1(wmean(ce[ce.period == "post"]))
nums["clsEqPre"] = p1(wmean(ce[ce.period == "pre"]))
nums["clsEqMaxDiff"] = p1((mm.alpha - mm.alpha_eq).abs().max())
sg = study(pd.read_csv(A / "alpha_single_generator.csv"))
sgp = sg[sg.period == "post"].groupby("model").apply(wmean)
nums["clsSgMin"], nums["clsSgMax"] = p1(sgp.min()), p1(sgp.max())
nums["clsSgMinModel"], nums["clsSgMaxModel"] = GEN.get(sgp.idxmin(), sgp.idxmin()), GEN.get(sgp.idxmax(), sgp.idxmax())
nums["clsSgModels"] = int(len(sgp))
nums["clsRegionPostLo"], nums["clsRegionPostHi"] = p1(wmean(post, "alpha_gmin")), p1(wmean(post, "alpha_gmax"))
pa = study(pd.read_csv(A / "alpha_pooled_vs_adapted.csv"))
pap = pa[pa.period == "post"]
nums["clsLangCalPost"] = p1(wmean(pap, "alpha_pooled"))
nums["clsLangCalCorr"] = f"{np.corrcoef(pap.alpha_pooled, pap.alpha_adapted)[0, 1]:.2f}"
nums["clsLangCalPre"] = p1(wmean(pa[pa.period == "pre"], "alpha_pooled"))
lx = study(pd.read_csv(R / "alpha_cells_instance_year.csv"))
lm = cy.merge(lx[["instance", "lang", "year", "alpha"]], on=["instance", "lang", "year"], suffixes=("", "_lex"))
lpost = lm[lm.year >= 2023]
nums["clsLexCorrPost"] = f"{np.corrcoef(lpost.alpha, lpost.alpha_lex)[0, 1]:.2f}"
nums["clsLexSpearmanPost"] = f"{lpost[['alpha', 'alpha_lex']].corr(method='spearman').iloc[0, 1]:.2f}"
nums["clsLexCells"] = int(len(lpost))
lpre = lm[lm.year <= 2022]
nums["clsLexPreMax"], nums["clsLexPreAboveFive"], nums["clsLexPreCells"] = p0(lpre.alpha_lex.max()), int((lpre.alpha_lex > 0.05).sum()), int(len(lpre))
nums["clsPreAboveFive"] = int((lpre.alpha > 0.05).sum())
nums["clsLexHigher"] = int((lpost.alpha_lex > lpost.alpha + 0.10).sum())
nums["clsLexLower"] = int((lpost.alpha_lex < lpost.alpha - 0.10).sum())
for tag, (inst, lang) in KEY.items():
    for y, t in YEAR.items():
        r = lm[(lm.instance == inst) & (lm.lang == lang) & (lm.year == y)]
        if len(r):
            nums[f"lex{tag}{t}"] = p1(r.alpha_lex.iloc[0])
lm.to_csv(OUT / "classifier_vs_lexical.csv", index=False)
EX = pd.read_csv(R / "excess" / "excess.csv")

# campaigns (near-duplicates)
ca = study(pd.read_csv(A / "alpha_campaign.csv"))
cpv = ca.pivot_table(index=["instance", "lang"], columns="is_campaign", values=["alpha", "n_docs"])
both = cpv.dropna()
if len(both):
    d = both[("alpha", True)] - both[("alpha", False)]
    nums["clsCampPairs"] = int(len(both))
    nums["clsCampHigher"], nums["clsCampLower"] = int((d > 0.02).sum()), int((d < -0.02).sum())
    nums["clsCampIn"] = p1((both[("alpha", True)] * both[("n_docs", True)]).sum() / both[("n_docs", True)].sum())
    nums["clsCampOut"] = p1((both[("alpha", False)] * both[("n_docs", False)]).sum() / both[("n_docs", False)].sum())
    nums["clsCampDiffMin"], nums["clsCampDiffMax"] = f"{100 * d.min():.0f}", f"{100 * d.max():.0f}"
nums["clsCampShare"] = p1(ca[ca.is_campaign].n_docs.sum() / ca.n_docs.sum())

# homogenisation against the change of the estimate
hm = study(pd.read_csv(A / "homogenisation.csv"))
hv = hm.pivot_table(index=["instance", "lang"], columns="period", values=["distinct2", "mean_cos", "mean_words"]).dropna()
al = cp.groupby(["instance", "lang", "period"]).apply(wmean).unstack("period")
da = (al["post"] - al["pre"]).reindex(hv.index)
hv, da = hv[da.notna().values], da[da.notna()]
nums["clsHomPairs"] = int(len(hv))
for col, t in (("mean_words", "Words"), ("distinct2", "Distinct"), ("mean_cos", "Cos")):
    dd = hv[(col, "post")] - hv[(col, "pre")]
    nums[f"clsHomCorr{t}"] = sgn(np.corrcoef(dd, da)[0, 1], 2)
    nums[f"clsHom{t}Pre"], nums[f"clsHom{t}Post"] = (f"{hv[(col, 'pre')].mean():.0f}", f"{hv[(col, 'post')].mean():.0f}") if col == "mean_words" else \
        (f"{hv[(col, 'pre')].mean():.3f}", f"{hv[(col, 'post')].mean():.3f}")
nums["clsHomWordsUp"] = int((hv[("mean_words", "post")] > hv[("mean_words", "pre")]).sum())

# off-language texts (not in the study set)
allp_ = pd.read_csv(A / "alpha_cells_period.csv")
for inst, t in (("participa_gencat", "Gencat"), ("participez_nanterre", "Nanterre"), ("ecrivons_angers", "Angers")):
    g = allp_[(allp_.instance == inst) & (allp_.lang == "en") & (allp_.period == "post")]
    if len(g):
        nums[f"clsSpam{t}N"], nums[f"clsSpam{t}Alpha"], nums[f"clsSpam{t}Lb"] = int(g.n_docs.sum()), p0(wmean(g)), p0(wmean(g, "lb_lo"))

# countries (Have Your Say, citizens)
if (A / "alpha_country.csv").exists():
    C = pd.read_csv(A / "alpha_country.csv")
    C.to_csv(OUT / "country.csv", index=False)
    en = C[(C.lang == "en") & (C.years == "2025-26")].set_index("country")
    native = [c for c in ("IRL", "GBR", "USA") if c in en.index]
    other = en.drop(index=native)
    nums["clsCountryEnNative"] = p1(wmean(en.loc[native]))
    nums["clsCountryEnOther"] = p1(wmean(other))
    nums["clsCountryEnOtherMin"], nums["clsCountryEnOtherMax"] = p1(other.alpha.min()), p1(other.alpha.max())
    nums["clsCountryEnNativeMax"] = p1(en.loc[native].alpha.max())
    nums["clsCountryEnOtherN"] = int(len(other))
    nums["clsCountryEnOtherAbove"] = int((other.alpha > en.loc[native].alpha.max()).sum())
    pre = C[(C.lang == "en") & (C.years == "pre")]
    if len(pre):
        nums["clsCountryEnPreMax"], nums["clsCountryEnPreN"] = p1(pre.alpha.max()), int(len(pre))
    prea = C[C.years == "pre"]
    if len(prea):
        nums["clsCountryPreMax"], nums["clsCountryPreN"] = p1(prea.alpha.max()), int(len(prea))

# post-hoc description of the texts above the human threshold
if (A / "high_lemmas.csv").exists():
    HL = pd.read_csv(A / "high_lemmas.csv")
    SEL = [("decide_madrid", "es"), ("participa_gencat", "ca"), ("eu_hys_organisations", "en")]
    L = [r"\begin{table}[t]",
         r"\caption{Post-hoc description of the texts scored above the human threshold (the 98th percentile of the pre-ChatGPT scores of the instance) after ChatGPT, templated letters excluded: content lemmas with the highest ratio between their document frequency among these texts and among the pre-ChatGPT texts of the same instance (lemmas present in at least 5\% of the texts above the threshold; document frequencies in \% in parentheses, above threshold / pre-ChatGPT). Proper nouns and topic words of single consultations are omitted.}",
         r"\label{tab:cls_lemmas}", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.26\columnwidth}p{0.68\columnwidth}@{}}", r"\toprule",
         r"Instance & Lemmas \\", r"\midrule"]
    for inst, lang in SEL:
        g = HL[(HL.instance == inst) & (HL.lang == lang) & ~HL.lemma.str.endswith("|PROPN")].sort_values("rk").head(12)
        ws = ", ".join(f"\\emph{{{tex(x.lemma.split('|')[0])}}} ({100 * x.df_high:.0f}/{100 * x.df_pre:.0f})" for x in g.itertuples())
        L.append(f"{tex(NAME.get(inst, inst))} ({lang}), $n={int(g.n_high.iloc[0]):,}$ & {ws} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("tab_cls_lemmas.tex", L)

# generators table
MC = yaml.safe_load(open("configs/models.yaml", encoding="utf-8"))["generators"]
WV = yaml.safe_load(open("configs/reference_weights_vendor.yaml", encoding="utf-8"))["weights"]
SC = pd.read_parquet(S / "study_scores_aug.parquet", columns=["set", "model", "doc_id", "mode"])
nums["clsNSeeds"] = fint(SC[(SC.set == "civic_g") & (SC["mode"] != "translate")].doc_id.nunique())
ng = SC[SC.set == "civic_g"].groupby("model").size()
nums["clsNModels"], nums["clsNGenerated"] = int(len(ng)), fint(ng.sum())
SHORT = {"NF4 (bitsandbytes)": "NF4 4-bit", "MXFP4 (native)": "MXFP4", "n/a": "API", "21B (3.6B active)": "21B (3.6B act.)",
         "OpenAI (API)": "OpenAI", "Anthropic (API)": "Anthropic", "Google (API)": "Google", "OpenAI (API, July 2024)": "OpenAI (Jul 2024)",
         "OpenAI (API, August 2024)": "OpenAI (Aug 2024)"}
L = [r"\begin{table}[t]",
     r"\caption{Generators. All of them provide training texts for the classifier and are held out in turn in the benchmark. Weight: weight in the reference distribution $F_M$ of the study (vendor share of chatbot traffic, the share of ChatGPT divided among the three OpenAI models); the open models have zero weight. Texts: generated texts retained. Weights of the open models: bf16, native MXFP4, or 4-bit NF4 (bitsandbytes).}",
     r"\label{tab:models}", r"\centering\scriptsize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}llllrr@{}}", r"\toprule",
     r"Generator & Developer & Parameters & Format & Texts & Weight \\", r"\midrule"]
for tag, m in MC.items():
    if tag not in ng.index:
        continue
    cells_ = [tag, str(m["family"]), str(m["params"]), str(m["quant"])]
    wt = f"{100 * WV[tag]:.1f}\\%" if tag in WV else "--"
    L.append(" & ".join(SHORT.get(c, c) for c in cells_) + f" & {int(ng[tag]):,} & {wt} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("tab_models.tex", L)
nums["clsWOpenAIEach"] = f"{100 * WV['gpt-4o']:.1f}"

# ------------------------------------------------------------------ additional checks (scripts/review_*.py, results/review)
RV = R / "review"
if RV.exists():
    d = pd.read_csv(RV / "review_1_drift_gap_summary.csv")
    nums["clsDriftCells"], nums["clsDriftPairs"] = int(d.n_cells.sum()), int(pd.read_csv(RV / "review_1_drift_gap.csv").groupby(["instance", "lang"]).ngroups)
    nums["clsDriftMax"] = p1(d.alpha_max.max())
    long = d[d.gap.astype(str).str[0].astype(int) >= 3]
    nums["clsDriftLongCells"], nums["clsDriftLongMean"], nums["clsDriftLongMax"] = int(long.n_cells.sum()), p1((long.alpha_mean * long.n_cells).sum() / long.n_cells.sum()), p1(long.alpha_max.max())
    nums["clsDriftLbMax"] = p1(d.lb_lo_max.max())
    d = pd.read_csv(RV / "review_2_process_oos_summary.csv").set_index("subset").loc["all"]
    nums["clsProcOosN"], nums["clsProcOosMean"], nums["clsProcOosMax"], nums["clsProcOosAboveFive"] = int(d.n), p1(d.mean_out), p1(d.max_out), int(d.n_above_5)
    d = pd.read_csv(RV / "review_3_country_within.csv").set_index("estimate")
    a, b = d.loc["(a) NATIVE, publications with >= 5 NATIVE texts"], d.loc["(a) OTHER, publications with >= 5 NATIVE texts"]
    nums["clsCountryWithinPubs"], nums["clsCountryWithinNative"], nums["clsCountryWithinOther"] = int(a.n_publications), p1(a.alpha), p1(b.alpha)
    nums["clsCountryWithinOtherMatched"] = p1(d[d.index.str.startswith("(b) OTHER re-sampled")].alpha.iloc[0])
    d = pd.read_csv(RV / "review_4_hys_groups.csv")
    d = d[[len(l) == 2 and (i, l) in set(zip(cp.instance, cp.lang)) for i, l in zip(d.instance, d.lang)]]   # pairs of the study only
    tot = d.groupby(["variant", "instance"]).apply(wmean)
    for v, t in (("without publications 20100 and 20315", "NoCamp"), ("without publications 20100 and 20315 and without templated letters", "NoCampNoTpl")):
        nums[f"clsHysCit{t}"], nums[f"clsHysOrg{t}"] = p1(tot[(v, "eu_hys_citizens")]), p1(tot[(v, "eu_hys_organisations")])
        pl = d[(d.variant == v) & (d.lang.str.len() == 2)].pivot_table(index="lang", columns="instance", values="alpha").dropna()
        nums[f"clsOrgHigher{t}"], nums[f"clsOrgLangs{t}"] = int((pl.eu_hys_organisations > pl.eu_hys_citizens + 0.02).sum()), int(len(pl))
        nums[f"clsCitHigher{t}"] = int((pl.eu_hys_citizens > pl.eu_hys_organisations + 0.02).sum())
    d = pd.read_csv(RV / "review_5a_auroc.csv")
    al = d[(d.generators == "all generators") & (d["mode"] == "all modes")].set_index("lang")
    nums["clsAuc"], nums["clsAucMin"] = f"{al.loc['ALL'].auroc:.3f}", f"{al.drop('ALL').auroc.min():.3f}"
    rw = d[(d.generators == "all generators") & (d["mode"] == "rewrite")].set_index("lang")
    nums["clsAucRewrite"], nums["clsAucRewriteMin"] = f"{rw.loc['ALL'].auroc:.3f}", f"{rw.drop('ALL').auroc.min():.3f}"
    nums["clsHumanPos"] = p0(al.loc["ALL"].share_human_pre_score_gt0)
    nums["clsHumanPosMin"], nums["clsHumanPosMax"] = p0(al.drop("ALL").share_human_pre_score_gt0.min()), p0(al.drop("ALL").share_human_pre_score_gt0.max())
    d = pd.read_csv(RV / "review_5b_lengths.csv")
    d = d[d.scope.str.startswith("all")].set_index("group")
    nums["clsLongAll"], nums["clsLongHys"], nums["clsLongDecidim"] = p0(d.loc["ALL"].share_gt190_words), p0(d.loc["eu_hys"].share_gt190_words), p0(d.loc["Decidim instances"].share_gt190_words)
    nums["clsShortDecidim"], nums["clsShortMadrid"], nums["clsShortAll"] = p0(d.loc["Decidim instances"].share_lt500_chars), p0(d.loc["decide_madrid"].share_lt500_chars), p0(d.loc["ALL"].share_lt500_chars)
    d = pd.read_csv(RV / "review_5c_minwords.csv")
    d = d[d.file.str.startswith("ALL files")].iloc[0]
    nums["clsMinWordsRemoved"] = p0(d.n_lt30_words_after_cleaning / d.n_after_language_allowlist)
    d = pd.read_csv(RV / "review_6_fallback_summary.csv").set_index("generator")
    nums["clsOwnRefCurrent"] = p0(d.loc[["gpt-5.4-mini", "gemini-3.6-flash", "claude-haiku-5-5"]].share_post_texts_own.min())
    nums["clsOwnRefFourO"] = p0(d.loc[["gpt-4o", "gpt-4o-mini"]].share_post_texts_own.max())
N7 = R / "review" / "review_7_natural_summary.csv"
if N7.exists():
    d = pd.read_csv(N7)

    def row(table, subset, group="all"):
        return d[(d.table == table) & (d.subset == subset) & (d.group == group)].iloc[0]

    pr = d[(d.table == "precision")].set_index("group")
    nums["clsResPostMatches"], nums["clsResPostA"] = int(pr.loc["TIER dialogue+format (texts)"].n), int(pr.loc["TIER dialogue+format (texts)"].k)
    nums["clsResDialogueA"] = int(pr.loc["TIER dialogue (texts)"].k)
    nums["clsResFormatA"] = nums["clsResPostA"] - nums["clsResDialogueA"]
    nums["clsResDisclosed"] = int(pr.loc["TIER disclosed (texts)"].k)
    fam = d[d.table == "specificity_family"]
    nums["clsResPreMatches"] = int(d[d.table == "final_pre_matches"].n.sum())
    nums["clsResPreTexts"], nums["clsResPostTexts"] = fint(fam.n_pre.max()), fint(fam.n_post.max())
    al = row("final_above", "dialogue+format+disclosed, label A")
    nums["clsResAllN"], nums["clsResAllAbove"], nums["clsResAllShare"] = int(al.n), int(al.k), p1(al.share)
    nums["clsResAllLo"], nums["clsResAllHi"] = p1(al.lo), p1(al.hi)
    nums["clsResBaseline"] = p1(row("final_baseline_post_same_pairs", "dialogue+format+disclosed, label A").share)
    dl = row("final_above", "dialogue, label A")
    nums["clsResDialN"], nums["clsResDialAbove"], nums["clsResDialShare"] = int(dl.n), int(dl.k), p1(dl.share)
    nums["clsResDialLo"], nums["clsResDialHi"] = p1(dl.lo), p1(dl.hi)
    no = row("final_above", "dialogue+format, labels B and C (not assistant residue)")
    nums["clsResNotN"], nums["clsResNotShare"] = int(no.n), p1(no.share)
NP = R / "natural_positives.csv"
if NP.exists():
    d = pd.read_csv(NP)
    po, pr = d[d.period == "post"], d[d.period == "pre"]
    nums["clsNatPostN"], nums["clsNatPostAbove"], nums["clsNatPreN"], nums["clsNatPreAbove"] = int(len(po)), int(po.above.sum()), int(len(pr)), int(pr.above.sum())

# ---------------------------------------------------------------- additions after the review of the post-fill version
# English forward error by pair: the mean is driven by pairs calibrated on the language pool
ep = M[(M.exp == "period") & (M.lang == "en")].groupby("instance").ae.mean().sort_values()
eo, er = ep[ep > 10], ep[ep <= 10]
nums["clsMaeFwdEnOutN"], nums["clsMaeFwdEnRestN"] = int(len(eo)), int(len(er))
nums["clsMaeFwdEnOutMin"], nums["clsMaeFwdEnOutMax"] = f"{eo.min():.1f}", f"{eo.max():.1f}"
nums["clsMaeFwdEnRestMin"], nums["clsMaeFwdEnRestMax"] = f"{er.min():.1f}", f"{er.max():.1f}"
# forward benchmark by pair (Salamandra excluded): coverage of the interval used in the study, floor, error
gbp = M[(M.exp == "period") & M.ci_lo.notna() & (M.heldout_model != OUTSIDE)]
pc = pd.DataFrame([dict(instance=i, lang=l, own=bool(g.own.iloc[0]), in_study=(base(i), l) in INC, coverage=cov_final(g), coverage_low=cov_final(g[g.alpha <= 0.05]),
                        floor=g[g.alpha == 0].alpha_hat.mean(), mae=g.ae.mean()) for (i, l), g in gbp.groupby(["instance", "lang"])])
pc.to_csv(OUT / "benchmark_pairs.csv", index=False)
print(pc.sort_values("coverage").head(12).round(3).to_string())
for own, t in ((True, "Own"), (False, "Pooled")):
    g = pc[pc.own == own]
    nums[f"clsPairCov{t}N"], nums[f"clsPairCov{t}Min"], nums[f"clsPairCov{t}Median"] = int(len(g)), p0(g.coverage.min()), p0(g.coverage.median())
    nums[f"clsPairCov{t}Below"] = int((g.coverage < 0.8).sum())
    nums[f"clsPairCov{t}BelowStudy"] = int(((g.coverage < 0.8) & g.in_study).sum())
bad = pc[(pc.coverage < 0.8) & pc.in_study]
print("study pairs with forward coverage below 80%:", bad[["instance", "lang", "own", "coverage", "floor", "mae"]].round(3).to_dict("records"))
nums["clsNInstBenchStudy"] = int(pc.in_study.sum())
nums["clsNInstBenchOutside"] = int((~pc.in_study).sum())
# signs of the significant terms of the interrupted time series
nums["clsItsSlopeSigPos"] = int(((its.slope_p < 0.05) & (its.slope_change > 0)).sum())
nums["clsItsLevelSigNeg"] = int(((its.level_p < 0.05) & (its.level_shift < 0)).sum())
nums["clsItsMinCells"] = int(its.n_cells.min())
# text length weighted by texts
hw = hm[hm.set_index(["instance", "lang"]).index.isin(hv.index)]
for per, t in (("pre", "Pre"), ("post", "Post")):
    g = hw[hw.period == per]
    nums[f"clsHomWordsW{t}"] = f"{(g.mean_words * g.n_docs).sum() / g.n_docs.sum():.0f}"
# post-ChatGPT yearly cells in which the lower limit of the bound exceeds the estimate
la = cy[(cy.year >= 2023) & (cy.lb_lo > cy.alpha + 1e-9)]
nums["clsLbAboveCells"], nums["clsLbAboveCellsMax"] = int(len(la)), f"{100 * (la.lb_lo - la.alpha).max():.1f}" if len(la) else "0.0"
print("cells with lb_lo > alpha:", la[["instance", "lang", "year", "n_docs", "alpha", "lb_lo"]].round(3).to_dict("records"))
# classifier against lexical estimator by kind of source
lpost = lpost.assign(src=lpost.instance.map(source))
portal = lpost.src.isin(["hys", "regs"])
hi_, lo_ = lpost.alpha_lex > lpost.alpha + 0.10, lpost.alpha_lex < lpost.alpha - 0.10
nums["clsLexHigherPortal"], nums["clsLexLowerPortal"], nums["clsLexCellsPortal"] = int((hi_ & portal).sum()), int((lo_ & portal).sum()), int(portal.sum())
nums["clsLexHigherOther"], nums["clsLexLowerOther"], nums["clsLexCellsOther"] = int((hi_ & ~portal).sum()), int((lo_ & ~portal).sum()), int((~portal).sum())
print("lexical lower by > 10 pp:", lpost[lo_][["instance", "lang", "year", "alpha", "alpha_lex"]].round(3).to_dict("records"))
print("lexical higher, not portal:", lpost[hi_ & ~portal][["instance", "lang", "year", "alpha", "alpha_lex"]].round(3).to_dict("records"))
for tag_, inst_ in (("Gencat", "participa_gencat"), ("Bcn", "barcelona")):
    r = lm[(lm.instance == inst_) & (lm.lang == "ca") & (lm.year == 2026)]
    if len(r):
        nums[f"clsLex{tag_}TwentySix"] = p1(r.alpha_lex.iloc[0])
# largest post-ChatGPT yearly estimate of U.S. rulemaking; pre-ChatGPT yearly cells by kind of calibration
rg = cy[cy.instance.str.startswith("regulations_gov") & (cy.year >= 2023)]
nums["clsRegsYearMax"] = p1(rg.alpha.max())
pre_own, pre_pool = cy[(cy.year <= 2022) & ~cy.pooled_ref.astype(bool)], cy[(cy.year <= 2022) & cy.pooled_ref.astype(bool)]
nums["clsPreYearOwnMax"], nums["clsPreYearPooledMax"], nums["clsPreYearPooledN"] = p1(pre_own.alpha.max()), p1(pre_pool.alpha.max()), int(len(pre_pool))
nums["clsNullPairs"] = int(loyo.groupby(["instance", "lang"]).ngroups)


# residue check with the markdown layout removed (scripts/review_8_markdown.py)
if (RV / "review_8_markdown_summary.csv").exists():
    d = pd.read_csv(RV / "review_8_markdown_summary.csv").set_index(["label", "tier"])
    for (lab, tier), t in ((("A", "ALL"), "All"), (("A", "dialogue"), "Dial"), (("C", "ALL"), "Not")):
        r = d.loc[(lab, tier)]
        nums[f"clsMd{t}N"], nums[f"clsMd{t}Above"], nums[f"clsMd{t}Plain"] = int(r.n), int(r.above_final), int(r.above_final_plain)
        nums[f"clsMd{t}PlainShare"], nums[f"clsMd{t}PlainLo"], nums[f"clsMd{t}PlainHi"] = p0(r.above_final_plain / r.n), p0(r.plain_lo), p0(r.plain_hi)
        nums[f"clsMd{t}WithLayout"] = int(r.md_before)
    d = pd.read_csv(RV / "review_8_markdown_prevalence.csv").set_index("model").share_md
    nums["clsMdShareGenerated"], nums["clsMdSharePre"], nums["clsMdSharePost"] = p0(d["ALL"]), p1(d["pre-ChatGPT"]), p1(d["post-ChatGPT"])

out = ["% Auto-generated by scripts/make_cls.py -- do not edit by hand.\n"]
for k, v in nums.items():
    assert re.fullmatch(r"[A-Za-z]+", k), k
    out.append(f"\\newcommand{{\\{k}}}{{{v}}}\n")
(O / "numbers_cls.tex").write_text("".join(out), encoding="utf-8")
json.dump(nums, open(O / "numbers_cls.json", "w"), indent=1, default=str)
print(len(nums), "macros written to paper/numbers_cls.tex")
