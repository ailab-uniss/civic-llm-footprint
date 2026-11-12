"""Referee analysis 8: does the residue check depend on the markdown layout itself?

(a) Share of texts with markdown layout among the generated texts (by model), the pre-ChatGPT and the post-ChatGPT texts.
(b) The real post-ChatGPT texts retained by scripts/review_7_natural.py are scored by the final classifier as they are and
    after the markdown layout has been removed (markers deleted, words kept); the share above the human threshold of the
    pair is compared between the two versions.

  python scripts/review_8_markdown.py   -> results/review/review_8_markdown_{prevalence,rescored,summary}.csv
"""
import glob
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src/civic")
from tool import score_texts  # noqa: E402

MD = re.compile(r"(\*\*[^*\n]{2,80}\*\*)|(^\s{0,3}#{1,4}\s+\S)|(^\s{0,3}[-*•]\s+\S.*\n\s{0,3}[-*•]\s+\S)|(^\s{0,3}\d{1,2}\.\s+\*\*)|(^\s*---+\s*$)", re.M)


def has_md(s: str) -> bool:
    return bool(MD.search(s or ""))


def strip_md(s: str) -> str:
    s = re.sub(r"\*\*|__|`", "", s)
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", s)
    s = re.sub(r"(?m)^\s{0,3}[-*•]\s+", "", s)
    s = re.sub(r"(?m)^\s{0,3}\d{1,2}[.)]\s+", "", s)
    s = re.sub(r"(?m)^\s*---+\s*$", "", s)
    s = re.sub(r"(?<!\*)\*(?!\*)", "", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------- (a) prevalence
G = pd.read_parquet("data/generated/generated_processed.parquet", columns=["model", "mode", "text"])
G = G[G["mode"].isin(["rewrite", "generate"])]
rows = [dict(group="generated", model=m, n=len(g), share_md=float(g.text.map(has_md).mean())) for m, g in G.groupby("model")]
rows.append(dict(group="generated", model="ALL", n=len(G), share_md=float(G.text.map(has_md).mean())))
P = pd.concat([pd.read_parquet(p, columns=["doc_id", "text", "created_at"]) for p in glob.glob("data/processed/*.parquet")], ignore_index=True)
P["post"] = pd.to_datetime(P.created_at, utc=True, errors="coerce") >= pd.Timestamp("2022-11-30", tz="UTC")
for post, g in P.groupby("post"):
    rows.append(dict(group="platform", model="post-ChatGPT" if post else "pre-ChatGPT", n=len(g), share_md=float(g.text.map(has_md).mean())))
pd.DataFrame(rows).to_csv("results/review/review_8_markdown_prevalence.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())

# ---------------------------------------------------------------- (b) rescoring without the layout
M = pd.read_csv("results/review/review_7_natural_matches.csv")
M = M[(M.period == "post") & M.retained & M.has_thr & M.label.isin(["A", "C"])]
tier = M.groupby("doc_id").tier.agg(lambda t: "dialogue" if "dialogue" in set(t) else ("disclosed" if "disclosed" in set(t) else "format"))
D = M.drop_duplicates("doc_id").set_index("doc_id")[["instance", "lang", "label", "score", "thr"]]
D["tier"] = tier
D = D.join(P.drop_duplicates("doc_id").set_index("doc_id").text, how="inner")
D["text_plain"] = D.text.map(strip_md)
D["md_before"], D["md_after"] = D.text.map(has_md), D.text_plain.map(has_md)
D["score_final"] = score_texts(D.text.tolist(), "models/xlmr_civic")
D["score_final_plain"] = score_texts(D.text_plain.tolist(), "models/xlmr_civic")
D["above_final"], D["above_final_plain"] = D.score_final > D.thr, D.score_final_plain > D.thr
D.drop(columns=["text", "text_plain"]).to_csv("results/review/review_8_markdown_rescored.csv")
S = []
for (lab, t), g in list(D.groupby(["label", "tier"])) + [((lab, "ALL"), g) for lab, g in D.groupby("label")]:
    n = len(g)
    r = dict(label=lab, tier=t, n=n, md_before=int(g.md_before.sum()), md_after=int(g.md_after.sum()),
             above_crossfit=int((g.score > g.thr).sum()), above_final=int(g.above_final.sum()), above_final_plain=int(g.above_final_plain.sum()),
             mean_score_final=float(g.score_final.mean()), mean_score_final_plain=float(g.score_final_plain.mean()))
    r["plain_lo"], r["plain_hi"] = wilson(r["above_final_plain"], n)
    S.append(r)
S = pd.DataFrame(S)
S.to_csv("results/review/review_8_markdown_summary.csv", index=False)
print(S.round(3).to_string())
print("REVIEW8_DONE")
