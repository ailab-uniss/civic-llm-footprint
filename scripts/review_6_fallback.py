"""Referee analysis 6: for which study pairs the generated reference F_M is instance-specific (>= MIN_G generated texts
seeded from the instance, per generator) and for which it falls back to the language. -> results/review/review_6_fallback.csv"""
import sys

import pandas as pd

sys.path.insert(0, "scripts")
from review_common import MIN_G, OUT, REGION, ScoreReferences, gen_setup, load  # noqa: E402

H, G, W = load()
refs = ScoreReferences(H, G, W, 300)
Gf, region = gen_setup(G, W)
cnt = Gf.groupby(["inst", "lang", "model"]).size()
cnt_lang = Gf.groupby(["lang", "model"]).size()
P = pd.read_csv("results/analysis_cls5/alpha_cells_period.csv")
post = P[P.period == "post"].groupby(["instance", "lang"]).n_docs.sum()
pairs = P[["instance", "lang"]].drop_duplicates()
rows = []
for inst, lang in zip(pairs.instance, pairs.lang):
    has_inst = (inst, lang) in refs.inst      # otherwise the pair has no pre-ChatGPT text and uses the language calibration as a whole
    for m in REGION:
        n = int(cnt.get((inst, lang, m), 0))
        rows.append(dict(instance=inst, lang=lang, generator=m, n_generated_instance=n, n_generated_language=int(cnt_lang.get((lang, m), 0)),
                         reference=("own" if n >= MIN_G and has_inst else "language"), instance_calibration=has_inst,
                         human_pooled=(inst, lang) in refs.pooled or not has_inst, n_docs_post=int(post.get((inst, lang), 0))))
R = pd.DataFrame(rows)
R.to_csv(OUT / "review_6_fallback.csv", index=False)
R["own"] = R.reference == "own"
S = R.groupby("generator").apply(lambda g: pd.Series({
    "n_pairs": len(g), "n_pairs_own": int(g.own.sum()), "share_pairs_own": g.own.mean(), "n_docs_post": int(g.n_docs_post.sum()),
    "share_post_texts_own": (g.n_docs_post * g.own).sum() / g.n_docs_post.sum(), "median_n_generated_instance": g.n_generated_instance.median()})).reset_index()
allown = R.groupby(["instance", "lang"]).agg(own=("own", "all"), anyown=("own", "any"), n=("n_docs_post", "first"))
S = pd.concat([S, pd.DataFrame([
    dict(generator="ALL five own", n_pairs=len(allown), n_pairs_own=int(allown.own.sum()), share_pairs_own=allown.own.mean(), n_docs_post=int(allown.n.sum()),
         share_post_texts_own=(allown.n * allown.own).sum() / allown.n.sum()),
    dict(generator="at least one own", n_pairs=len(allown), n_pairs_own=int(allown.anyown.sum()), share_pairs_own=allown.anyown.mean(), n_docs_post=int(allown.n.sum()),
         share_post_texts_own=(allown.n * allown.anyown).sum() / allown.n.sum())])], ignore_index=True)
S.to_csv(OUT / "review_6_fallback_summary.csv", index=False)
pd.set_option("display.width", 300)
print(R.pivot_table(index=["instance", "lang", "n_docs_post"], columns="generator", values="n_generated_instance").to_string())
print(S.round(4).to_string())
print("pairs without instance calibration:", R[~R.instance_calibration][["instance", "lang"]].drop_duplicates().values.tolist())
print("REVIEW6_DONE")
