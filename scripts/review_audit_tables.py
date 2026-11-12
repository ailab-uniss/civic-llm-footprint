"""Referee remarks R33, R34, R35(b): tables recomputed from the existing audit files (read-only, no API call).

  python scripts/review_audit_tables.py        -> results/review_audit/r33_*.csv, r34_*.csv, r35_*.csv

R33  retrieval experiment split by whether search was actually invoked; consumer-application spot check.
R34  per-generator table of the manual check; manual L5 figures by scenario; L6 counts by scenario.
R35  length distribution of the texts available locally.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "civic"))
from audit import FOREIGN_MARKERS, legal_refs  # noqa: E402

R = ROOT / "results"
OUT = R / "review_audit"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
norm = lambda s: s.astype(str).str.strip().str.lower()
col = lambda df, p: next(c for c in df.columns if c.startswith(p))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return (100 * lo, 100 * hi)


def ci(k, n):
    w, c = wilson(k, n), clopper(k, n)
    return {"wilson_lo": round(w[0], 1), "wilson_hi": round(w[1], 1), "cp_lo": round(c[0], 1), "cp_hi": round(c[1], 1)}


def ref_summary(g: pd.DataFrame) -> dict:
    """Counts on a set of rows of an L1_references sheet (columns A1..A5)."""
    a1, a3, a5 = col(g, "A1_"), col(g, "A3_"), col(g, "A5_")
    real = g[norm(g[a1]) == "y"]; ex = norm(real[a3]); exist = real[ex == "y"]; pe = norm(exist[a5])
    d = {"refs_checked": len(g), "false_extractor_matches": len(g) - len(real), "actual_refs": len(real),
         "exist": int((ex == "y").sum()), "non_existent": int((ex == "n").sum()), "not_verifiable": int((ex == "not verifiable").sum()),
         "pertinent": int((pe == "pertinent").sum()), "not_pertinent": int((pe == "not pertinent").sum()),
         "uncertain": int((pe == "uncertain").sum()), "pertinence_not_judged": int((~pe.isin(["pertinent", "not pertinent", "uncertain"])).sum())}
    d["pct_non_existent"] = round(100 * d["non_existent"] / len(real), 1) if len(real) else np.nan
    w = wilson(d["non_existent"], len(real)); d["non_existent_wilson_lo"], d["non_existent_wilson_hi"] = (round(w[0], 1), round(w[1], 1))
    nj = d["pertinent"] + d["not_pertinent"] + d["uncertain"]
    d["pct_pertinent_of_existing"] = round(100 * d["pertinent"] / len(exist), 1) if len(exist) else np.nan
    d["pct_not_pertinent_of_existing"] = round(100 * d["not_pertinent"] / len(exist), 1) if len(exist) else np.nan
    d["pct_not_pertinent_of_judged"] = round(100 * d["not_pertinent"] / nj, 1) if nj else np.nan
    w = wilson(d["pertinent"], len(exist)); d["pertinent_wilson_lo"], d["pertinent_wilson_hi"] = (round(w[0], 1), round(w[1], 1))
    return d


# =====================================================================================================================
print("\n================ R33 (a, b): retrieval experiment by search actually invoked")
Rt = pd.read_parquet(ROOT / "data/audit/audit_retrieval.parquet"); Rt = Rt[~Rt.finish.str.startswith("error")].copy()
Rt["n_new"] = [len([r for r in legal_refs(o, l) if r not in set(legal_refs(g, l))]) for o, g, l in zip(Rt.output, Rt.orig, Rt.lang)]
OFF = re.compile(r"eur-lex|normattiva|boe\.es|legifrance|dogc|gencat|legislation\.gov|govinfo|ecfr|federalregister|regulations\.gov|\.gov\b|\.gouv\.|comune\.|ajuntament|ayuntamiento|senato|camera\.it|parlament")  # as make_numbers.py
urls = Rt.citations.map(lambda t: [c["url"] for c in json.loads(t)])
Rt["has_citation"] = urls.map(len) > 0; Rt["has_official"] = urls.map(lambda L: any(OFF.search(u) for u in L))
Rt["n_words"] = Rt.output.str.split().str.len()
X = pd.ExcelFile(R / "manual_check" / "audit_check_retrieval.xlsx")
rp, rr = X.parse("pairs"), X.parse("L1_references")
chk = rp[["pair_id", "provider", "web_search_used"]].merge(Rt[["pair_id", "provider", "searched"]], on=["pair_id", "provider"])
print("workbook 'web_search_used' equals parquet 'searched':", bool((chk.web_search_used.astype(bool) == chk.searched.astype(bool)).all()), len(chk))
rr = rr.merge(Rt[["pair_id", "provider", "searched"]], on=["pair_id", "provider"], how="left")
rows = []
for prov in ["openai_auto", "openai_forced", "gemini"]:
    for lab, m in [("search invoked", True), ("search not invoked", False), ("all requests", None)]:
        g = Rt[Rt.provider == prov] if m is None else Rt[(Rt.provider == prov) & (Rt.searched == m)]
        q = rr[rr.provider == prov] if m is None else rr[(rr.provider == prov) & (rr.searched == m)]
        d = {"condition": prov, "subset": lab, "outputs": len(g), "new_refs_total": int(g.n_new.sum()),
             "new_refs_per_output": round(g.n_new.mean(), 2) if len(g) else np.nan,
             "pct_outputs_with_new_ref": round(100 * (g.n_new > 0).mean(), 1) if len(g) else np.nan,
             "pct_outputs_with_url_citation": round(100 * g.has_citation.mean(), 1) if len(g) else np.nan,
             "pct_outputs_citing_official_source": round(100 * g.has_official.mean(), 1) if len(g) else np.nan,
             "mean_words": round(g.n_words.mean(), 0) if len(g) else np.nan}
        d.update(ref_summary(q) if len(q) else {"refs_checked": 0})
        rows.append(d)
T = pd.DataFrame(rows)
T.to_csv(OUT / "r33a_retrieval_by_search_invoked.csv", index=False)
print(T[["condition", "subset", "outputs", "new_refs_total", "new_refs_per_output", "pct_outputs_with_new_ref", "pct_outputs_with_url_citation",
         "pct_outputs_citing_official_source", "mean_words"]].to_string(index=False))
print(T[["condition", "subset", "refs_checked", "false_extractor_matches", "actual_refs", "exist", "non_existent", "not_verifiable", "pertinent", "not_pertinent", "uncertain",
         "pct_pertinent_of_existing", "pertinent_wilson_lo", "pertinent_wilson_hi", "pct_not_pertinent_of_existing"]].to_string(index=False))
tot = rr.groupby("searched").apply(lambda g: pd.Series(ref_summary(g)))
tot.to_csv(OUT / "r33b_checked_refs_by_search_invoked_total.csv")
print("\nR33(b) checked references, all conditions pooled, by search invoked:\n", tot[["refs_checked", "actual_refs", "exist", "non_existent", "pertinent", "not_pertinent", "uncertain"]].to_string())
real_all = rr[norm(rr[col(rr, "A1_")]) == "y"]
print("existing refs: 0 non-existent of", len(real_all), "-> one-sided 95% upper bound (rule of three / Clopper-Pearson two-sided):", clopper(0, len(real_all)))
for lab, g in [("search invoked", real_all[real_all.searched]), ("search not invoked", real_all[~real_all.searched])]:
    print("  ", lab, "0 of", len(g), "CP 95%:", [round(x, 2) for x in clopper(0, len(g))])
print("share of A6 (source consulted) filled among actual refs: %.1f%%" % (100 * real_all[col(rr, "A6_")].notna().mean()))

# no-tools baselines on the same 100 originals, and paired comparison of the number of new references
M = pd.read_parquet(R / "audit_metrics.parquet", columns=["doc_id", "lang", "model", "mode", "l1_new_refs", "l1_any_new"])
s3 = M[(M["mode"] == "s3_persuasive") & M.doc_id.isin(set(Rt.doc_id))]
base = s3.groupby("model").agg(n=("l1_new_refs", "size"), refs_per_output=("l1_new_refs", "mean"), pct_with_ref=("l1_any_new", lambda s: 100 * s.mean())).round(2)
print("\nno-tools S3 on the same originals (audit_metrics.parquet):\n", base.loc[[m for m in ["gpt-5.4-mini", "gemini-3.6-flash"] if m in base.index]].to_string())
prs = []
for prov, model in [("openai_auto", "gpt-5.4-mini"), ("openai_forced", "gpt-5.4-mini"), ("gemini", "gemini-3.6-flash")]:
    b = s3[s3.model == model].drop_duplicates("doc_id").set_index("doc_id").l1_new_refs
    for lab, m in [("all requests", None), ("search invoked", True), ("search not invoked", False)]:
        g = Rt[Rt.provider == prov] if m is None else Rt[(Rt.provider == prov) & (Rt.searched == m)]
        g = g.drop_duplicates("doc_id").set_index("doc_id")
        j = g.index.intersection(b.index)
        if len(j) < 5:
            prs.append({"condition": prov, "subset": lab, "n_paired": len(j)}); continue
        dlt = g.loc[j, "n_new"] - b.loc[j]
        rng = np.random.default_rng(0); bs = [dlt.sample(len(dlt), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
        try:
            pw = stats.wilcoxon(dlt[dlt != 0]).pvalue
        except ValueError:
            pw = np.nan
        prs.append({"condition": prov, "subset": lab, "n_paired": len(j), "no_tools_refs_per_output": round(b.loc[j].mean(), 2),
                    "with_tool_refs_per_output": round(g.loc[j, "n_new"].mean(), 2), "mean_diff": round(dlt.mean(), 2),
                    "boot_lo": round(np.percentile(bs, 2.5), 2), "boot_hi": round(np.percentile(bs, 97.5), 2), "wilcoxon_p": pw})
prs = pd.DataFrame(prs); prs.to_csv(OUT / "r33a_paired_refs_vs_no_tools.csv", index=False)
print(prs.to_string(index=False))

# =====================================================================================================================
print("\n================ R33 (c): consumer-application check")
C = R / "consumer_check"
v = pd.read_csv(C / "references_verified.csv"); s = pd.read_excel(C / "spotcheck_human.xlsx")
runs = pd.read_excel(C / "runs.xlsx", sheet_name="runs"); scol = col(runs, "C2_")
runs["searched"] = norm(runs[scol]) == "y"
allr = pd.read_csv(C / "references_all.csv").merge(runs[["pair_id", "app", "searched"]], on=["pair_id", "app"])
v = v.merge(runs[["pair_id", "app", "searched"]], on=["pair_id", "app"], how="left")
print(runs.groupby("app").agg(chats=("pair_id", "size"), searched=("searched", "sum"), model=("C1_model_shown_by_app", "first")).to_string())
print("new references extracted (all), by app and search:\n", allr.groupby(["app", "searched"]).size().to_string())
rows = []
for (app, se), g in v.groupby(["app", "searched"]):
    e = g[g.A3_exists == "Y"]
    rows.append({"app": app, "search_invoked_in_chat": se, "refs_checked_by_agent": len(g), "rechecked_by_author": int(g.checked_by_author.sum()),
                 "non_existent": int((g.A3_exists == "N").sum()), "existing": len(e), "pertinent": int((e.A5_pertinent == "pertinent").sum()),
                 "not_pertinent": int((e.A5_pertinent == "not pertinent").sum()), "uncertain": int((e.A5_pertinent == "uncertain").sum())})
cs = pd.DataFrame(rows); cs.to_csv(OUT / "r33c_consumer_refs_by_search.csv", index=False); print(cs.to_string(index=False))
s["agent_class"] = np.where(s.LLM_A3_exists != "Y", "agent: non-existent", "agent: " + s.LLM_A5_pertinent.astype(str))
s["exist_agree"] = s.LLM_A3_exists == s.HUMAN_A3_exists
s["pert_comparable"] = s.HUMAN_A5_pertinent.notna() & (s.HUMAN_A3_exists == "Y")
s["pert_agree"] = s.pert_comparable & (s.LLM_A5_pertinent == s.HUMAN_A5_pertinent)
# population of agent judgements from which the spot check was drawn
v["agent_class"] = np.where(v.A3_exists_llm != "Y", "agent: non-existent", "agent: " + v.A5_pertinent_llm.astype(str))
pop = v.groupby(["agent_class", "app"]).size().unstack(fill_value=0)
sp = s.groupby(["agent_class", "app"]).agg(rechecked=("row_id", "size"), exist_agree=("exist_agree", "sum"), pert_comparable=("pert_comparable", "sum"), pert_agree=("pert_agree", "sum"))
print("agent judgements in the sample of 100 (rows) by app:\n", pop.to_string()); print("re-checked by the authors:\n", sp.to_string())
rows = []
for cl in sorted(v.agent_class.unique()):
    g = s[s.agent_class == cl]
    rows.append({"agent_judgement": cl, "in_sample_of_100": int((v.agent_class == cl).sum()), "rechecked": len(g),
                 "existence_agree": int(g.exist_agree.sum()), "pertinence_comparable": int(g.pert_comparable.sum()), "pertinence_agree": int(g.pert_agree.sum())})
rows.append({"agent_judgement": "total", "in_sample_of_100": len(v), "rechecked": len(s), "existence_agree": int(s.exist_agree.sum()),
             "pertinence_comparable": int(s.pert_comparable.sum()), "pertinence_agree": int(s.pert_agree.sum())})
sc = pd.DataFrame(rows)
for k, n, name in [("existence_agree", "rechecked", "exist"), ("pertinence_agree", "pertinence_comparable", "pert")]:
    cc = [ci(int(a), int(b)) if b else {} for a, b in zip(sc[k], sc[n])]
    sc[name + "_wilson"] = [f"{c['wilson_lo']}-{c['wilson_hi']}" if c else "" for c in cc]
    sc[name + "_clopper_pearson"] = [f"{c['cp_lo']}-{c['cp_hi']}" if c else "" for c in cc]
sc.to_csv(OUT / "r33c_consumer_spotcheck.csv", index=False); print(sc.to_string(index=False))
dis = s[s.pert_comparable & ~s.pert_agree][["row_id", "app", "language", "reference_detected", "LLM_A5_pertinent", "HUMAN_A5_pertinent"]]
print("disagreements:\n", dis.to_string(index=False))
# Cohen's kappa on the three pertinence categories (20 comparable items)
b = s[s.pert_comparable]; cats = ["pertinent", "not pertinent", "uncertain"]
tab = pd.crosstab(b.LLM_A5_pertinent, b.HUMAN_A5_pertinent).reindex(index=cats, columns=cats, fill_value=0)
po = np.trace(tab.values) / tab.values.sum(); pe = (tab.sum(1).values * tab.sum(0).values).sum() / tab.values.sum() ** 2
print("confusion (rows agent, columns authors):\n", tab.to_string(), "\nCohen kappa = %.2f" % ((po - pe) / (1 - pe)))
for app in ["ChatGPT", "Gemini"]:
    g = v[(v.app == app) & (v.A3_exists == "Y")]; k = int((g.A5_pertinent == "pertinent").sum())
    kk = int(((g.A5_pertinent == "pertinent") & g.checked_by_author).sum())
    print(app, "pertinent %d/%d" % (k, len(g)), ci(k, len(g)), "| of these 'pertinent', re-checked by an author:", kk)

# =====================================================================================================================
print("\n================ R34 (a): manual check, per generator")
X = pd.ExcelFile(R / "manual_check" / "audit_check.xlsx")
mp, mr, mc = X.parse("pairs"), X.parse("L1_references"), X.parse("L5_claims")
print("generators in the manually checked sample:", sorted(mp.generator.unique()))
print("annotator columns in the workbook:", [c for c in list(mp.columns) + list(mr.columns) + list(mc.columns) if re.search("annotat|author|rater|coder|initial", c, re.I)])
gen = pd.DataFrame({g: ref_summary(d) for g, d in list(mr.groupby("generator")) + [("ALL", mr)]}).T
gen.insert(0, "pairs_in_sample", [int((mp.generator == g).sum()) if g != "ALL" else len(mp) for g in gen.index])
gen.insert(1, "S3_pairs_in_sample", [int(((mp.generator == g) & (mp.scenario == "S3 persuasive")).sum()) if g != "ALL" else int((mp.scenario == "S3 persuasive").sum()) for g in gen.index])
gen.index.name = "generator"; gen.to_csv(OUT / "r34a_manual_refs_by_generator.csv")
print(gen[["pairs_in_sample", "refs_checked", "false_extractor_matches", "actual_refs", "non_existent", "pct_non_existent", "non_existent_wilson_lo", "non_existent_wilson_hi",
           "not_verifiable", "exist", "not_pertinent", "pct_not_pertinent_of_existing", "uncertain", "pertinent", "pertinence_not_judged"]].to_string())
bys = pd.DataFrame({g: ref_summary(d) for g, d in mr.groupby("scenario")}).T
bys.to_csv(OUT / "r34a_manual_refs_by_scenario.csv"); print(bys[["refs_checked", "actual_refs", "non_existent", "not_verifiable", "exist", "not_pertinent", "uncertain", "pertinent"]].to_string())

print("\n================ R34 (b): manual L5 by scenario")
st = [col(mc, f"B{i}_status_{i}") for i in (1, 2, 3)]; b4, b5 = col(mc, "B4_"), col(mc, "B5_")
Sc = mc.melt(id_vars=["pair_id", "scenario", "language", "generator"], value_vars=st, value_name="status").dropna(subset=["status"]); Sc["status"] = norm(Sc.status)
rows = []
for sce, g in list(mc.groupby("scenario")) + [("ALL", mc)]:
    q = Sc if sce == "ALL" else Sc[Sc.scenario == sce]
    k = int((q.status == "kept").sum()); w = wilson(k, len(q))
    rows.append({"scenario": sce, "pairs": len(g), "claims_traced": len(q), "preserved": k, "materially_changed": int((q.status == "changed").sum()),
                 "omitted": int((q.status == "omitted").sum()), "pct_preserved": round(100 * k / len(q), 1), "preserved_wilson_lo": round(w[0], 1), "preserved_wilson_hi": round(w[1], 1),
                 "pairs_all_claims_preserved": int(q.groupby("pair_id").status.apply(lambda x: (x == "kept").all()).sum()),
                 "B5_intent_preserved_Y": int((norm(g[b5]) == "y").sum()), "B5_partly": int((norm(g[b5]) == "partly").sum()), "B5_N": int((norm(g[b5]) == "n").sum()),
                 "B4_substantial_new_claims": int((norm(g[b4]) == "substantial").sum()),
                 "auto_L5_coverage_mean_same_pairs": round(g.L5_claim_coverage_automatic.mean(), 3), "auto_L5_grounded_mean_same_pairs": round(g.L5_output_grounded_automatic.mean(), 3)})
l5 = pd.DataFrame(rows); l5.to_csv(OUT / "r34b_manual_l5_by_scenario.csv", index=False); print(l5.T.to_string())
ct = pd.crosstab(Sc.scenario, Sc.status)
for a, b in [("S1 improve", "S2 notes"), ("S1 improve", "S3 persuasive"), ("S2 notes", "S3 persuasive")]:
    t = [[int(ct.loc[a].get("kept", 0)), int(ct.loc[a].sum() - ct.loc[a].get("kept", 0))], [int(ct.loc[b].get("kept", 0)), int(ct.loc[b].sum() - ct.loc[b].get("kept", 0))]]
    print("Fisher exact, preserved vs not, %s vs %s: p = %.4f (claims are clustered in pairs: indicative only)" % (a, b, stats.fisher_exact(t)[1]))
kept = Sc.groupby("pair_id").status.apply(lambda x: (x == "kept").mean())
auto = mc.set_index("pair_id").L5_claim_coverage_automatic.reindex(kept.index)
print("Spearman human kept share vs automatic coverage: rho=%.3f p=%.3f n=%d" % (*stats.spearmanr(kept, auto), len(kept)))

print("\n================ R34 (c): L6 foreign legal orders, counts by scenario")
Mf = pd.read_parquet(R / "audit_metrics.parquet")
l6 = Mf.groupby("mode").agg(outputs=("l6_any_foreign", "size"), outputs_with_foreign=("l6_any_foreign", "sum"), foreign_markers_total=("l6_foreign", "sum"))
l6["pct"] = (100 * l6.outputs_with_foreign / l6.outputs).round(3); l6.to_csv(OUT / "r34c_l6_by_scenario.csv"); print(l6.to_string())
bl = Mf.groupby(["mode", "lang"]).l6_any_foreign.agg(["sum", "size"]); bl["pct"] = (100 * bl["sum"] / bl["size"]).round(3)
bl.to_csv(OUT / "r34c_l6_by_scenario_language.csv"); print(bl.unstack("lang").to_string())
bm = Mf.groupby(["mode", "model"]).l6_any_foreign.agg(["sum", "size"]); bm["pct"] = (100 * bm["sum"] / bm["size"]).round(3)
bm.to_csv(OUT / "r34c_l6_by_scenario_generator.csv"); print(bm["sum"].unstack("mode").to_string())
am = pd.read_csv(R / "audit_mixed.csv"); print(am[am.metric == "l6_any_foreign"].to_string(index=False))
# what the markers matched (texts available locally: nine generators in pairs.parquet, Gemini rebuilt from its own file)
P = pd.read_parquet(ROOT / "data/audit/pairs.parquet", columns=["model", "lang", "mode", "doc_id", "output", "country"])
S = pd.read_parquet(ROOT / "data/audit/audit_sample.parquet")[["doc_id", "instance"]]
import yaml
cfg = yaml.safe_load(open(ROOT / "configs/instances.yaml", encoding="utf-8"))["instances"]
S["country"] = S.instance.map(lambda i: cfg.get(i, {}).get("country", "INT"))
if "gemini-3.6-flash" not in set(P.model):
    G = pd.read_parquet(ROOT / "data/audit/audit_gemini-3.6-flash.parquet", columns=["model", "lang", "mode", "doc_id", "output"]).merge(S[["doc_id", "country"]], on="doc_id")
    P = pd.concat([P, G[G.output.fillna("").str.split().str.len() >= 10]], ignore_index=True)
print("seed documents by country:", S.country.value_counts().to_dict())
hits = []
for r in P.itertuples():
    pat = FOREIGN_MARKERS.get(r.country, FOREIGN_MARKERS["INT"])
    for m in re.finditer(pat, r.output, flags=re.IGNORECASE):
        a, b = max(0, m.start() - 60), min(len(r.output), m.end() + 60)
        hits.append({"mode": r.mode, "lang": r.lang, "country": r.country, "model": r.model, "doc_id": r.doc_id, "marker": m.group(0).lower(),
                     "context": r.output[a:b].replace("\n", " ")})
H = pd.DataFrame(hits); H.to_csv(OUT / "r34c_l6_marker_matches.csv", index=False)
print("outputs with a marker, recomputed locally:", H.groupby("mode").apply(lambda g: g[["model", "doc_id"]].drop_duplicates().shape[0]).to_dict())
H["marker_norm"] = H.marker.str.replace(r"\d+", "N", regex=True).str.replace(r"\s+", " ", regex=True)
print(H.groupby(["mode", "country", "marker_norm"]).size().sort_values(ascending=False).head(40).to_string())

# =====================================================================================================================
print("\n================ R35 (b): length of the texts")
ps = pd.read_csv(R / "processed_stats.csv")
w = ps.groupby("type").apply(lambda g: pd.Series({"n_texts": int(g.n.sum()), "mean_words_weighted": round((g.mean_words * g.n).sum() / g.n.sum(), 1),
                                                    "cells": len(g), "cells_with_mean_below_80_words": int((g.mean_words < 80).sum()),
                                                    "min_cell_mean": round(g.mean_words.min(), 1), "max_cell_mean": round(g.mean_words.max(), 1)}))
w.to_csv(OUT / "r35b_mean_words_by_type_processed.csv"); print("processed corpus (texts of at least 30 words only):\n", w.to_string())
print("all types: n=%d, weighted mean words=%.1f" % (ps.n.sum(), (ps.mean_words * ps.n).sum() / ps.n.sum()))
f = ROOT / "data/interim/decide_madrid.parquet"
if f.exists():
    D = pd.read_parquet(f, columns=["type", "text", "created_at"])
    t = D.text.fillna("").astype(str).str.replace(r"<[^>]+>", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    D["n_chars"], D["n_words"] = t.str.len(), t.str.split().str.len()
    D = D[D.n_chars > 0]
    q = lambda g: pd.Series({"n_texts": len(g), "pct_below_500_chars": round(100 * (g.n_chars < 500).mean(), 1), "pct_below_30_words": round(100 * (g.n_words < 30).mean(), 1),
                             "median_chars": g.n_chars.median(), "median_words": g.n_words.median(), "mean_words": round(g.n_words.mean(), 1),
                             "pct_below_500_chars_among_30plus_words": round(100 * (g[g.n_words >= 30].n_chars < 500).mean(), 1), "n_30plus_words": int((g.n_words >= 30).sum())})
    L = pd.concat([D.groupby("type").apply(q), q(D).to_frame("all").T]); L.index.name = "type"
    L.to_csv(OUT / "r35b_length_decide_madrid_raw.csv"); print("Decide Madrid raw export (before the 30-word filter):\n", L.to_string())
for name in ["data/generated/human_sample.parquet", "data/audit/audit_sample.parquet"]:
    h = pd.read_parquet(ROOT / name, columns=["lang", "type", "text", "n_words"]); h["n_chars"] = h.text.str.len()
    print(name, "(human pre-2023 texts retained, 30-400 words): n=%d, below 500 characters: %.1f%%, median chars %.0f" % (len(h), 100 * (h.n_chars < 500).mean(), h.n_chars.median()))
    print(h.groupby("type").apply(lambda g: pd.Series({"n": len(g), "pct_below_500_chars": round(100 * (g.n_chars < 500).mean(), 1)})).to_string())
