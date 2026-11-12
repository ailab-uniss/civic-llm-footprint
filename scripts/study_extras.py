"""Complements of the classifier study that need the texts (run where data/processed is available).

  alpha_country.csv   Have Your Say, citizens, post-ChatGPT, without templated letters: estimate per country and language
                      (cells of at least 150 texts), with the calibration of the instance and language; the pre-ChatGPT
                      cells are the false-positive floor of each country
  high_lemmas.csv     post-hoc description of the texts above the human 98th percentile of the score: content lemmas whose
                      document frequency among them is highest relative to the pre-ChatGPT texts of the same instance
  spam_sample.txt     sample of the off-language (English) texts of three Decidim instances, for manual inspection

  python scripts/study_extras.py results/supervised/study_scores_aug.parquet results/analysis_cls4 configs/reference_weights_vendor.yaml
"""
import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src/civic")
from analysis import attach_scores, split_author_type  # noqa: E402
from score_estimator import LB_Q, ScoreReferences  # noqa: E402

scores, out = sys.argv[1], Path(sys.argv[2])
W = {str(k): float(v) for k, v in yaml.safe_load(open(sys.argv[3] if len(sys.argv) > 3 else "configs/reference_weights.yaml", encoding="utf-8"))["weights"].items()}
H = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("data/processed/*.parquet"))], ignore_index=True)
H = H[~H.is_dup_exact].drop_duplicates("doc_id").copy()
H["instance"] = split_author_type(H.instance, H["user_type"] if "user_type" in H.columns else None)
T = pd.read_parquet("results/templates.parquet")[["doc_id", "is_template"]].drop_duplicates("doc_id")
H = H.merge(T, on="doc_id", how="left"); H["is_template"] = H.is_template.fillna(False).astype(bool)
G = pd.read_parquet("data/generated/generated_processed.parquet")
G["inst"] = G.doc_id.map(H.set_index("doc_id")["instance"]).fillna(G.doc_id.str.split(":").str[0])
H, G = attach_scores(H, G, scores)
refs = ScoreReferences(H, G, W, 300)
H["year"] = H.created_at.dt.year

# ---- countries (Have Your Say, citizens)
rows = []
C = H[(H.instance == "eu_hys_citizens") & ~H.is_template & H.country.notna()]
for (country, lang), x in C.groupby(["country", "lang"]):
    est = refs.get("eu_hys_citizens", lang)
    if est is None or len(x) < 150:
        continue
    for label, y in (("pre", x[x.period == "pre"]), ("2023-24", x[(x.period == "post") & (x.year <= 2024)]), ("2025-26", x[x.year >= 2025]),
                     ("post", x[x.period == "post"])):
        if len(y) >= 150:
            r = est.estimate(y.score.values, n_boot=200)
            rows.append(dict(country=country, lang=lang, years=label, n_docs=len(y), alpha=r["alpha"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             lb=r["lb"], lb_lo=r["lb_lo"], n_consultations=y.publication_id.nunique(),
                             top_consultation_share=y.publication_id.value_counts(normalize=True).iloc[0]))
pd.DataFrame(rows).to_csv(out / "alpha_country.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(), flush=True)

# ---- lemmas of the texts above the human threshold
POS = ("ADJ", "ADV", "VERB", "NOUN")
rows = []
for inst, lang in [("decide_madrid", "es"), ("barcelona", "ca"), ("participa_gencat", "ca"), ("eu_hys_citizens", "en"),
                   ("eu_hys_citizens", "fr"), ("eu_hys_citizens", "de"), ("eu_hys_organisations", "en"),
                   ("participationcitoyenne_ccomptes", "fr"), ("regulations_gov_dot", "en"), ("eu_hys_citizens", "it")]:
    x = H[(H.instance == inst) & (H.lang == lang)]
    pre, post = x[x.period == "pre"], x[(x.period == "post") & ~x.is_template]
    thr = pre.score.quantile(LB_Q)
    hi, lo = post[post.score > thr], post[post.score <= thr]
    if len(hi) < 100:
        continue
    def df(d):
        c = Counter(t for toks in d.tokens for t in set(toks) if str(t).rsplit("_", 1)[-1] in POS or "_" not in str(t))
        return c, len(d)
    (ch, nh), (cp, npre), (cl, nl) = df(hi), df(pre), df(lo)
    for w, k in ch.items():
        if k / nh < 0.05:
            continue
        rows.append(dict(instance=inst, lang=lang, lemma=w, n_high=nh, df_high=k / nh, df_pre=cp.get(w, 0) / npre, df_low=cl.get(w, 0) / max(nl, 1),
                         ratio_pre=(k / nh + 0.005) / (cp.get(w, 0) / npre + 0.005), ratio_low=(k / nh + 0.005) / (cl.get(w, 0) / max(nl, 1) + 0.005)))
L = pd.DataFrame(rows)
L["rk"] = L.groupby(["instance", "lang"]).ratio_pre.rank(ascending=False)
L = L[L.rk <= 40].sort_values(["instance", "lang", "rk"])
L.to_csv(out / "high_lemmas.csv", index=False)
print(L[L.rk <= 15][["instance", "lang", "lemma", "df_high", "df_pre", "df_low"]].round(3).to_string(), flush=True)

# ---- off-language texts
with open(out / "spam_sample.txt", "w", encoding="utf-8") as f:
    for inst in ("participa_gencat", "participez_nanterre", "ecrivons_angers"):
        x = H[(H.instance == inst) & (H.lang == "en")]
        post = x[x.period == "post"]
        url = post.text.str.contains("(?i)www|http|[.]com|click here", regex=True).mean()
        f.write(f"\n===== {inst}: {len(x)} English texts, {len(post)} post-ChatGPT, {post.author_hash.nunique()} authors, types {post.type.value_counts().to_dict()}, "
                f"years {post.year.value_counts().sort_index().to_dict()}, share with URL-like words {url:.2f}\n")
        for t in post.sample(min(12, len(post)), random_state=1).text:
            f.write("--- " + " ".join(str(t).split())[:400] + "\n")
print("EXTRAS_DONE")
