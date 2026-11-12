"""Referee analysis 7: naturally occurring positives, i.e. real contributions that carry the residue of an assistant's reply.

A pattern set in the nine study languages (scripts/review_7_patterns.py) is searched in every non-duplicate text. The
specificity of each pattern family is measured with the pre-ChatGPT texts (period == "pre"), where assistant residue
cannot occur. The matches of the retained families after 30 Nov 2022 are real texts with a near-certain label; the share
of them that scores above the human threshold of their instance-language pair (98th percentile of the pre-ChatGPT scores,
pairs with at least 300 scored pre-ChatGPT texts) is a field check of the classifier on text written by the assistants
actually in use. Three tiers are reported apart: "dialogue" (explicit residue of the exchange: self-reference, reply
openers and closers, unfilled personal placeholders, markdown letter), "format" (unrendered chat markdown on the
platforms that do not render it) and "disclosed" (the author states the use of an assistant).

Manual labels (A = assistant residue, B = text about AI / quoting an assistant, C = human formula, false match or
undetermined) are read from scripts/review_7_labels.csv (doc_id,label,note) when the file exists.

  python scripts/review_7_natural.py results/supervised/study_scores_aug.parquet
    -> results/review/review_7_natural_matches.csv   one row per (text, family); snippet of at most 160 characters, no full text
       results/review/review_7_natural_summary.csv   all tables in long format
       results/review/log_7_natural.txt
"""
import glob
import html
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import binomtest

sys.path.insert(0, "src/civic")
sys.path.insert(0, "scripts")
from analysis import split_author_type  # noqa: E402
import review_7_patterns as RP  # noqa: E402

LANGS = ["ca", "de", "en", "es", "fi", "fr", "it", "pt", "sv"]
MIN_PRE = 300
Q = 0.98
OUT = Path("results/review")
LABELS = "scripts/review_7_labels.csv"
SNIP = 160
TRANS = {0x2019: "'", 0x2018: "'", 0x02BC: "'", 0x201C: '"', 0x201D: '"', 0x00A0: " ", 0x202F: " ", 0x200B: ""}
RX = [(f, pid, scope, re.compile(rx, re.I)) for f, pid, scope, rx in RP.P]
WS = re.compile(r"\s+")
DROPPED_PID = {pid for f, pid, _, _ in RP.P if pid in RP.DROPPED_PATTERNS or f in RP.DROPPED_FAMILIES}
TIERS = ["dialogue", "format", "disclosed"]


def scan(text):
    """All pattern matches of one text: list of (family, pattern ids, matched string, snippet, start, length of the text).
    Matched string and snippet are those of the first retained pattern of the family (of the first pattern if none is retained)."""
    t = html.unescape(text).replace("\r\n", "\n").translate(TRANS)
    n = len(t)
    head, tail, off = t[:RP.HEAD], t[-RP.TAIL:], max(0, n - RP.TAIL)
    found = {}
    for f, pid, scope, rx in RX:
        if scope == "head":
            m, base = rx.search(head), 0
        elif scope == "tail":
            m, base = rx.search(tail), off
        else:
            m, base = rx.search(t), 0
        if m is None:
            continue
        s, e = m.start() + base, m.end() + base
        kept = pid not in DROPPED_PID
        if f not in found or (kept and not found[f][6]):
            a = max(0, min(s - 50, n - SNIP))
            pids = found[f][1] if f in found else []
            found[f] = [f, pids, WS.sub(" ", t[s:e]).strip()[:100], WS.sub(" ", t[a:a + SNIP]).strip()[:SNIP], s, n, kept]
        found[f][1].append(pid)
    return [(f, "|".join(p), ms, sn, s, n) for f, p, ms, sn, s, n, _ in found.values()]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def source_group(inst):
    if inst.startswith("eu_hys"):
        return inst
    if inst.startswith("regulations_gov"):
        return "regulations_gov"
    if inst == "decide_madrid":
        return inst
    return "decidim_instances"


class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.f.write(s)
        self.o.write(s)

    def flush(self):
        self.f.flush()
        self.o.flush()


def main():
    scores_path = sys.argv[1] if len(sys.argv) > 1 else "results/supervised/study_scores_aug.parquet"
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUT / "log_7_natural.txt")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 170)
    print("scores:", scores_path)

    # ---------------------------------------------------------------- texts and pattern scan
    cols = ["doc_id", "lang", "text", "created_at", "period", "is_dup_exact", "instance", "user_type"]
    A = pd.concat([pd.read_parquet(p, columns=[c for c in cols if c in pq.read_schema(p).names])
                   for p in sorted(glob.glob("data/processed/*.parquet"))], ignore_index=True)
    A["instance"] = split_author_type(A.instance, A["user_type"] if "user_type" in A.columns else None)
    A = A[(~A.is_dup_exact) & A.text.notna() & A.lang.isin(LANGS)].drop_duplicates("doc_id").sort_values("doc_id").reset_index(drop=True)
    A["year"] = A.created_at.dt.year
    A["group"] = A.instance.map(source_group)
    with Pool(min(20, os.cpu_count() or 1)) as pool:
        res = pool.map(scan, A.text.tolist(), chunksize=500)
    A = A.drop(columns=["text"])
    rows = [(i,) + r for i, rr in enumerate(res) for r in rr]
    M = pd.DataFrame(rows, columns=["i", "family", "pattern_ids", "match", "snippet", "start", "text_len"])
    del res

    # ---------------------------------------------------------------- scores, thresholds, within-pair percentile
    S = pd.read_parquet(scores_path)
    S = S[S.set == "civic_h"][["doc_id", "score"]].drop_duplicates("doc_id")
    A = A.merge(S, on="doc_id", how="left")
    pre = A[(A.period == "pre") & A.score.notna()]
    pre_scores = {k: np.sort(g.score.values) for k, g in pre.groupby(["instance", "lang"]) if len(g) >= MIN_PRE}
    thr = {k: float(np.quantile(v, Q)) for k, v in pre_scores.items()}
    key = list(zip(A.instance, A.lang))
    n_pre_pair = pre.groupby(["instance", "lang"]).size().to_dict()      # scored pre-ChatGPT texts of the pair (threshold only if >= MIN_PRE)
    A["n_pre_pair"] = [int(n_pre_pair.get(k, 0)) for k in key]
    A["thr"] = [thr.get(k, np.nan) for k in key]
    A["has_thr"] = A.thr.notna() & A.score.notna()
    A["above"] = np.where(A.has_thr, A.score > A.thr, np.nan)
    A["pct_pair"] = [100.0 * np.searchsorted(pre_scores[k], s, side="left") / len(pre_scores[k]) if (k in pre_scores and s == s) else np.nan
                     for k, s in zip(key, A.score)]
    n_all = A.groupby("period").size()
    print("texts (non-duplicate, nine languages):", n_all.to_dict(), "| scored:", A[A.score.notna()].groupby("period").size().to_dict(),
          "| with threshold:", A[A.has_thr].groupby("period").size().to_dict(), "| pairs with threshold:", len(thr))

    # ---------------------------------------------------------------- scope, retention, labels
    M = M.join(A[["doc_id", "instance", "group", "lang", "year", "period", "score", "thr", "n_pre_pair", "has_thr", "above", "pct_pair"]], on="i")
    excl = RP.FAMILY_EXCLUDED_GROUPS
    M["in_scope"] = [g not in excl.get(f, ()) for f, g in zip(M.family, M.group)]
    M["retained"] = M.in_scope & M.pattern_ids.map(lambda s: any(p not in DROPPED_PID for p in s.split("|")))
    M["tier"] = M.family.map(lambda f: RP.FAMILY_TIER.get(f, "dialogue"))
    if os.path.exists(LABELS):
        L = pd.read_csv(LABELS, dtype=str).drop_duplicates("doc_id")
        M = M.merge(L[["doc_id", "label"]], on="doc_id", how="left")
        print("labels read:", len(L), L.label.value_counts().to_dict())
    else:
        M["label"] = ""
    M["label"] = M.label.astype(object).where(M.label.notna(), "")
    has_labels = bool((M.label != "").any())
    M = M.sort_values(["family", "period", "lang", "doc_id"]).reset_index(drop=True)
    M[["doc_id", "instance", "group", "lang", "year", "period", "family", "tier", "pattern_ids", "match", "snippet", "start", "text_len", "in_scope",
       "retained", "label", "score", "thr", "n_pre_pair", "has_thr", "above", "pct_pair"]].to_csv(OUT / "review_7_natural_matches.csv", index=False)
    unl = M[M.retained & (M.label == "")]
    print("retained matches without label:", len(unl), "(post:", int((unl.period == "post").sum()), ")")

    out = []

    def rec(table, subset, grp, **kw):
        out.append(dict(table=table, subset=subset, group=grp, **kw))

    # ---------------------------------------------------------------- 2. specificity: pre vs post rates
    npre, npost = int(n_all.get("pre", 0)), int(n_all.get("post", 0))
    print("\n=== SPECIFICITY: texts matched per family, pre vs post (rates per 10,000 texts). 'all' = every pattern of the family, 'ret' = retained "
          "patterns only. Families restricted to some source groups are counted, with their denominators, on those groups only. "
          "p: one-sided binomial test that the retained patterns do not match more often after ChatGPT than before ===")
    Mall, M = M, M[M.in_scope]
    R = M[M.retained]
    T = []
    for f in RP.FAMILY_ORDER:
        g, gr = M[M.family == f], R[R.family == f]
        Af = A[~A.group.isin(excl.get(f, ()))]
        nf = Af.groupby("period").size()
        nfl = Af.groupby(["lang", "period"]).size()
        a0, b0 = int(nf.get("pre", 0)), int(nf.get("post", 0))
        kp, ko = int((g.period == "pre").sum()), int((g.period == "post").sum())
        rp, ro = int((gr.period == "pre").sum()), int((gr.period == "post").sum())
        pv = binomtest(ro, rp + ro, b0 / (a0 + b0), alternative="greater").pvalue if rp + ro else np.nan
        status = "dropped" if f in RP.DROPPED_FAMILIES else "retained"
        T.append(dict(family=f, tier=RP.FAMILY_TIER.get(f, "dialogue"), status=status, n_pre=a0, n_post=b0, all_pre=kp, all_post=ko, ret_pre=rp, ret_post=ro,
                      ret_pre_per10k=1e4 * rp / a0, ret_post_per10k=1e4 * ro / b0, all_pre_per10k=1e4 * kp / a0, all_post_per10k=1e4 * ko / b0, p=pv))
        rec("specificity_family", status, f, n_pre=a0, n_post=b0, k_pre_all_patterns=kp, k_post_all_patterns=ko, k_pre=rp, k_post=ro,
            rate_pre_per10k=1e4 * rp / a0, rate_post_per10k=1e4 * ro / b0, rate_pre_per10k_all_patterns=1e4 * kp / a0,
            rate_post_per10k_all_patterns=1e4 * ko / b0, p_one_sided=pv)
        for lg in LANGS:
            a, b = int(nfl.get((lg, "pre"), 0)), int(nfl.get((lg, "post"), 0))
            kp2, ko2 = int(((g.period == "pre") & (g.lang == lg)).sum()), int(((g.period == "post") & (g.lang == lg)).sum())
            rp2, ro2 = int(((gr.period == "pre") & (gr.lang == lg)).sum()), int(((gr.period == "post") & (gr.lang == lg)).sum())
            rec("specificity_family_lang", lg, f, n_pre=a, n_post=b, k_pre_all_patterns=kp2, k_post_all_patterns=ko2, k_pre=rp2, k_post=ro2,
                rate_pre_per10k=1e4 * rp2 / max(a, 1), rate_post_per10k=1e4 * ro2 / max(b, 1))
    print(pd.DataFrame(T).round(4).to_string(index=False))
    for tier in TIERS + ["dialogue+format"]:
        x = R[R.tier.isin(tier.split("+"))].drop_duplicates("doc_id")
        rp, ro = int((x.period == "pre").sum()), int((x.period == "post").sum())
        print(f"  texts with any retained {tier} match: pre {rp} ({1e4 * rp / npre:.3f} per 10k), post {ro} ({1e4 * ro / npost:.3f} per 10k)")
        rec("specificity_tier", "retained", tier, n_pre=npre, n_post=npost, k_pre=rp, k_post=ro, rate_pre_per10k=1e4 * rp / npre, rate_post_per10k=1e4 * ro / npost)

    def cells(D, by, order):
        X = D.groupby(["family", by, "period"]).size().unstack(fill_value=0).reindex(columns=["pre", "post"], fill_value=0)
        X["cell"] = X.pre.astype(str) + "/" + X.post.astype(str)
        return X.cell.unstack(fill_value="0/0").reindex(RP.FAMILY_ORDER).reindex(columns=order).fillna("0/0")

    groups = sorted(A.group.unique())
    print("\n--- texts per language (pre/post):", {lg: f"{int(((A.lang == lg) & (A.period == 'pre')).sum())}/{int(((A.lang == lg) & (A.period == 'post')).sum())}" for lg in LANGS})
    print("--- per family and language of the text, RETAINED patterns: pre/post matches")
    print(cells(R, "lang", LANGS).to_string())
    print("--- per family and language of the text, ALL patterns: pre/post matches")
    print(cells(M, "lang", LANGS).to_string())
    print("\n--- texts per source group (pre/post):", {k: f"{int(((A.group == k) & (A.period == 'pre')).sum())}/{int(((A.group == k) & (A.period == 'post')).sum())}" for k in groups})
    print("--- per family and source group, RETAINED patterns: pre/post matches")
    print(cells(R, "group", groups).to_string())
    print("--- per family and source group, ALL patterns and all groups (also those out of scope for the family): pre/post matches")
    print(cells(Mall, "group", groups).to_string())
    X = Mall.groupby(["family", "group", "period"]).size().unstack(fill_value=0).reindex(columns=["pre", "post"], fill_value=0)
    for (f, gr), r in X.iterrows():
        rec("specificity_family_group_all_patterns", gr, f, k_pre=int(r.pre), n_pre=int(((A.group == gr) & (A.period == "pre")).sum()), k_post=int(r.post),
            n_post=int(((A.group == gr) & (A.period == "post")).sum()))
    print("\n--- per pattern (in-scope groups): pre/post matches and labels of the post-ChatGPT matches (a text can match several patterns)")
    E = M.assign(pid=M.pattern_ids.str.split("|")).explode("pid")
    Xp = E.groupby(["pid", "period"]).size().unstack(fill_value=0).reindex(columns=["pre", "post"], fill_value=0)
    Xp = Xp.reindex([pid for _, pid, _, _ in RP.P], fill_value=0)
    Xp["dropped"] = [pid in DROPPED_PID for _, pid, _, _ in RP.P]
    if has_labels:
        lab = E[(E.period == "post") & (E.label != "")].groupby(["pid", "label"]).size().unstack(fill_value=0)
        Xp = Xp.join(lab.add_prefix("post_"))
    Xp = Xp.fillna(0)
    print(Xp[(Xp.pre > 0) | (Xp.post > 0)].astype({c: int for c in Xp.columns if c != "dropped"}).to_string())
    print("patterns without any match:", " ".join(Xp[(Xp.pre == 0) & (Xp.post == 0)].index))
    for pid, r in Xp.iterrows():
        rec("specificity_pattern", "dropped" if r.dropped else "retained", pid, k_pre=int(r.pre), n_pre=npre, k_post=int(r.post), n_post=npost,
            **{("k_" + c[5:]): int(r[c]) for c in Xp.columns if c.startswith("post_")})

    # ---------------------------------------------------------------- 3. reading-based precision
    if has_labels:
        print("\n=== READING: labels of the matches (A assistant residue, B about AI / quoting an assistant, C human formula, false match or undetermined) ===")
        for per in ("post", "pre"):
            g = M[(M.period == per)]
            tab = g.assign(label=g.label.replace("", "unlabelled")).groupby(["family", "label"]).size().unstack(fill_value=0).reindex(RP.FAMILY_ORDER).fillna(0).astype(int)
            print(f"\n--- {per}-ChatGPT matches, ALL patterns in scope (dropped ones included)"); print(tab.to_string())
        g = R[R.period == "post"]
        print("\n--- post-ChatGPT matches of the RETAINED patterns: precision = A / labelled")
        units = [(f, g[g.family == f]) for f in RP.FAMILY_ORDER] + [(f"TIER {t} (texts)", g[g.tier == t].drop_duplicates("doc_id")) for t in TIERS] + \
                [("TIER dialogue+format (texts)", g[g.tier != "disclosed"].drop_duplicates("doc_id"))]
        for f, x in units:
            if len(x) == 0:
                continue
            nlab, ka = int((x.label != "").sum()), int((x.label == "A").sum())
            lo, hi = wilson(ka, nlab)
            print(f"  {f:30s} n={len(x):4d} labelled={nlab:4d} A={ka:4d} B={int((x.label == 'B').sum()):3d} C={int((x.label == 'C').sum()):3d} "
                  f"precision={ka / max(nlab, 1):.3f} [{lo:.3f}, {hi:.3f}]")
            rec("precision", "post_retained", f, n=len(x), n_labelled=nlab, k=ka, k_B=int((x.label == "B").sum()), k_C=int((x.label == "C").sum()),
                share=ka / max(nlab, 1), lo=lo, hi=hi)

    # ---------------------------------------------------------------- 4. final sets
    def block(sel, name):
        """sel: rows (text, family) of the post-ChatGPT matches of a set; the statistics are per text."""
        D = sel.drop_duplicates("doc_id")
        print(f"\n=== FINAL SET [{name}]: {len(D)} post-ChatGPT texts; with threshold {int(D.has_thr.sum())}; without threshold {int((~D.has_thr).sum())} ===")
        rec("final_n", name, "all", n=len(D), n_with_threshold=int(D.has_thr.sum()), n_without_threshold=int((~D.has_thr).sum()))
        if len(D) == 0:
            return
        for by in ("year", "lang", "group"):
            c = D.groupby(by).agg(n=("doc_id", "size"), n_with_threshold=("has_thr", "sum"), n_above=("above", lambda s: int(np.nansum(s))))
            c["n_with_threshold"] = c.n_with_threshold.astype(int)
            print(f"--- by {by}"); print(c.to_string())
            for k, r in c.iterrows():
                lo, hi = wilson(r.n_above, r.n_with_threshold)
                rec(f"final_by_{by}", name, str(k), n=int(r.n), n_with_threshold=int(r.n_with_threshold), k=int(r.n_above),
                    share=r.n_above / r.n_with_threshold if r.n_with_threshold else np.nan, lo=lo, hi=hi)
        H = D[D.has_thr]
        k, n = int(H.above.sum()), len(H)
        lo, hi = wilson(k, n)
        print(f"--- above the human threshold of the pair: {k}/{n} = {k / max(n, 1):.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")
        rec("final_above", name, "all", n=n, k=k, share=k / max(n, 1), lo=lo, hi=hi)
        if n:
            q = H.pct_pair.quantile([0.25, 0.5, 0.75]).round(2).tolist()
            print(f"--- within-pair percentile of the score (relative to pre-ChatGPT texts): Q1 {q[0]}, median {q[1]}, Q3 {q[2]}; "
                  f"share above the 90th / 95th / 99th percentile: {(H.pct_pair > 90).mean():.3f} / {(H.pct_pair > 95).mean():.3f} / {(H.pct_pair > 99).mean():.3f}")
            rec("final_percentile", name, "all", n=n, q1=q[0], median=q[1], q3=q[2], share_above_p90=(H.pct_pair > 90).mean(),
                share_above_p95=(H.pct_pair > 95).mean(), share_above_p99=(H.pct_pair > 99).mean())
            pairs = H.groupby(["instance", "lang"]).size().rename("w")      # baseline: all post-ChatGPT texts of the same pairs
            B = A[(A.period == "post") & A.has_thr].groupby(["instance", "lang"]).above.agg(["sum", "size"]).join(pairs, how="inner")
            kb, nb = int(B["sum"].sum()), int(B["size"].sum())
            lo, hi = wilson(kb, nb)
            wavg = float(((B["sum"] / B["size"]) * B.w).sum() / B.w.sum())
            print(f"--- baseline, all post-ChatGPT texts of the same {len(B)} pairs: {kb}/{nb} = {kb / nb:.4f} [{lo:.4f}, {hi:.4f}]; "
                  f"pairs weighted by their number of matches: {wavg:.4f}")
            rec("final_baseline_post_same_pairs", name, "all", n=nb, k=kb, share=kb / nb, lo=lo, hi=hi, share_match_weighted=wavg, n_pairs=len(B))
            Bp = A[(A.period == "pre") & A.has_thr].groupby(["instance", "lang"]).above.agg(["sum", "size"]).join(pairs, how="inner")
            print(f"--- pre-ChatGPT texts of the same pairs (2% by construction): {int(Bp['sum'].sum())}/{int(Bp['size'].sum())} = {Bp['sum'].sum() / Bp['size'].sum():.4f}")
            rec("final_baseline_pre_same_pairs", name, "all", n=int(Bp["size"].sum()), k=int(Bp["sum"].sum()), share=Bp["sum"].sum() / Bp["size"].sum())
        fams = [f for f in RP.FAMILY_ORDER if (sel.family == f).any()]
        if len(fams) > 1:
            print("--- by family (a text can belong to several families)")
        for f in fams if len(fams) > 1 else []:
            x = sel[(sel.family == f) & sel.has_thr]
            k, n = int(x.above.sum()), len(x)
            lo, hi = wilson(k, n)
            med = x.pct_pair.median() if n else np.nan
            print(f"  {f:20s} n={int((sel.family == f).sum()):4d} with threshold={n:4d} above={k:4d} share={k / max(n, 1):.3f} [{lo:.3f}, {hi:.3f}] median percentile={med:.1f}")
            rec("final_above_by_family", name, f, n_matches=int((sel.family == f).sum()), n=n, k=k, share=k / max(n, 1), lo=lo, hi=hi, median=med)

    P = R[R.period == "post"]
    A_docs = set(P[P.label == "A"].doc_id)
    for tiers in (["dialogue"], ["format"], ["dialogue", "format"], ["disclosed"], TIERS):
        sel = P[P.tier.isin(tiers)]
        block(sel, "+".join(tiers) + ", all matches")
        if has_labels:
            block(sel[sel.label == "A"], "+".join(tiers) + ", label A")
    if has_labels:   # format-only texts: chat markdown without any explicit residue of the dialogue
        dia = set(P[P.tier == "dialogue"].doc_id)
        block(P[(P.tier == "format") & ~P.doc_id.isin(dia) & (P.label == "A")], "format without dialogue residue, label A")
        block(P[(P.tier != "disclosed") & (P.label != "A") & (P.label != "")], "dialogue+format, labels B and C (not assistant residue)")

    # pre-ChatGPT matches of the retained patterns: false positives of the patterns
    for tier in TIERS:
        Dp = R[(R.period == "pre") & (R.tier == tier)].drop_duplicates("doc_id")
        Hp = Dp[Dp.has_thr]
        print(f"\npre-ChatGPT matches of the retained {tier} patterns: {len(Dp)} texts; with threshold {len(Hp)}; above {int(Hp.above.sum())}")
        rec("final_pre_matches", tier, "all", n=len(Dp), n_with_threshold=len(Hp), k=int(Hp.above.sum()))
    Ba = A[A.has_thr]
    for per in ("pre", "post"):
        x = Ba[Ba.period == per]
        lo, hi = wilson(int(x.above.sum()), len(x))
        print(f"all {per}-ChatGPT texts with a threshold that score above it: {int(x.above.sum())}/{len(x)} = {x.above.mean():.4f}")
        rec("baseline_all_pairs", per, "all", n=len(x), k=int(x.above.sum()), share=x.above.mean(), lo=lo, hi=hi)
    pd.DataFrame(out).to_csv(OUT / "review_7_natural_summary.csv", index=False)
    print("\nfamilies of each retained post-ChatGPT text (counts):")
    print(P.groupby("doc_id").family.agg(lambda s: "+".join(sorted(set(s)))).value_counts().to_string())
    print("label A texts:", len(A_docs))
    print("REVIEW7_DONE")


if __name__ == "__main__":
    main()
