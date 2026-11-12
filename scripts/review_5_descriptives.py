"""Referee analysis 5: (a) discrimination of the classifier (AUROC pre-ChatGPT human versus generated texts) and share of
human texts with a positive score; (b) length descriptives of the corpus by source; (c) share of texts removed by the
30-word minimum. -> results/review/review_5a_auroc.csv, review_5b_lengths.csv, review_5c_minwords.csv"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "scripts")
from review_common import OUT, REGION, SCORES, load  # noqa: E402

pd.set_option("display.width", 300)
COMM = list(REGION)


def auroc(neg, pos):
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].mean() - (len(pos) + 1) / 2) / len(neg))


# ---------------- (a)
H, G, W = load()
Sc = pd.read_parquet(SCORES)
hp = H[H.period == "pre"][["doc_id", "lang", "score"]]
g = Sc[Sc.set == "civic_g"]
rows = []
for lang in sorted(set(hp.lang) & set(g.lang)) + ["ALL"]:
    h = hp if lang == "ALL" else hp[hp.lang == lang]
    gl = g if lang == "ALL" else g[g.lang == lang]
    for gens, gg in (("all generators", gl), ("five commercial generators", gl[gl.model.isin(COMM)])):
        for mode, gm in [("all modes", gg), ("rewrite+generate", gg[gg["mode"].isin(["rewrite", "generate"])])] + [(m, gg[gg["mode"] == m]) for m in ("rewrite", "generate", "translate")]:
            if len(gm) < 30 or len(h) < 30:
                continue
            rows.append(dict(lang=lang, generators=gens, mode=mode, n_human_pre=len(h), n_generated=len(gm), auroc=auroc(h.score.values, gm.score.values),
                             share_human_pre_score_gt0=float((h.score > 0).mean()), share_generated_score_gt0=float((gm.score > 0).mean())))
A = pd.DataFrame(rows)
A.to_csv(OUT / "review_5a_auroc.csv", index=False)
print(A.pivot_table(index="lang", columns=["generators", "mode"], values="auroc").round(4).to_string())
print(A.drop_duplicates("lang")[["lang", "n_human_pre", "share_human_pre_score_gt0"]].round(4).to_string())
print("post-ChatGPT human share with score > 0 (for reference):", H[H.period == "post"].groupby("lang").score.apply(lambda s: (s > 0).mean()).round(4).to_dict())

# ---------------- (b)
parts = []
for p in sorted(glob.glob("data/processed/*.parquet")):
    d = pd.read_parquet(p, columns=["doc_id", "instance", "is_dup_exact", "n_words", "text", "period"])
    d = d[~d.is_dup_exact].drop_duplicates("doc_id")
    d["n_chars"] = d.text.str.len()
    parts.append(d.drop(columns="text"))
D = pd.concat(parts, ignore_index=True)
D["group"] = D.instance.map(lambda s: "eu_hys" if s.startswith("eu_hys") else "regulations_gov" if s.startswith("regulations_gov")
                            else "decide_madrid" if s == "decide_madrid" else "Decidim instances")
D["scored"] = D.doc_id.isin(set(H.doc_id))
rows = []
for scope, X in (("all non-duplicate processed texts", D), ("texts with a classifier score (study)", D[D.scored])):
    for name, x in list(X.groupby("group")) + [("ALL", X)]:
        rows.append(dict(scope=scope, group=name, n_docs=len(x), n_instances=x.instance.nunique(), share_gt190_words=float((x.n_words > 190).mean()),
                         median_words=float(x.n_words.median()), mean_words=float(x.n_words.mean()), share_lt500_chars=float((x.n_chars < 500).mean()),
                         median_chars=float(x.n_chars.median()), share_gt190_words_pre=float((x[x.period == "pre"].n_words > 190).mean()),
                         share_gt190_words_post=float((x[x.period == "post"].n_words > 190).mean())))
B = pd.DataFrame(rows)
B.to_csv(OUT / "review_5b_lengths.csv", index=False)
print(B.round(4).to_string())

# ---------------- (c) same order as preprocess.run: allow-list of declared languages, normalise, n_words, >= 30 words
sys.path.insert(0, "src/civic")
try:
    from preprocess import load_allowlist, n_words, normalise
    exact = True
except Exception as e:  # noqa: BLE001
    print("preprocess import failed:", e)
    exact = False
proc = {Path(p).stem: p for p in glob.glob("data/processed/*.parquet")}
rows = []
for p in sorted(glob.glob("data/interim/*.parquet")):
    d = pd.read_parquet(p)
    stem = Path(p).stem
    n0 = len(d)
    if n0 == 0:
        continue
    raw_lt30 = int((d.text.fillna("").str.split().str.len() < 30).sum())
    n1, lt30 = np.nan, np.nan
    if exact:
        allow = load_allowlist("configs/instances.yaml", stem)
        if allow:
            d = d[(d.type != "proposal") | (d.lang_declared.isin(allow)) | (d.lang_declared == "")]
        n1 = len(d)
        lt30 = int((d.text.map(normalise).map(n_words) < 30).sum())
    npr = len(pd.read_parquet(proc[stem], columns=["doc_id"])) if stem in proc else np.nan
    rows.append(dict(file=stem, in_processed=stem in proc, n_interim=n0, n_after_language_allowlist=n1, n_lt30_words_after_cleaning=lt30,
                     n_raw_lt30_whitespace_words=raw_lt30, n_processed=npr))
    print(rows[-1], flush=True)
Cc = pd.DataFrame(rows)
Cc["group"] = Cc.file.map(lambda s: s if s in ("eu_hys", "regulations_gov", "decide_madrid") else "Decidim instances")
tot = []
for name, x in [("ALL files with a processed output", Cc[Cc.in_processed])] + [("group=" + k, v) for k, v in Cc[Cc.in_processed].groupby("group")] + [("ALL interim files", Cc)]:
    s = x[["n_interim", "n_after_language_allowlist", "n_lt30_words_after_cleaning", "n_raw_lt30_whitespace_words", "n_processed"]].sum()
    tot.append(dict(file=name, in_processed=True, group="total", **s.to_dict()))
Cc = pd.concat([Cc, pd.DataFrame(tot)], ignore_index=True)
Cc["share_lt30_after_cleaning"] = Cc.n_lt30_words_after_cleaning / Cc.n_after_language_allowlist
Cc["share_raw_lt30_of_interim"] = Cc.n_raw_lt30_whitespace_words / Cc.n_interim
Cc["share_processed_of_interim"] = Cc.n_processed / Cc.n_interim
Cc.to_csv(OUT / "review_5c_minwords.csv", index=False)
print(Cc.round(4).to_string())
print("REVIEW5_DONE")
