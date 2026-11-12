"""Per-process estimates: consultations on the portals (EU Have Your Say, Regulations.gov) and participatory
processes on the Decidim platforms.

On these platforms a half-year is often dominated by one consultation or process (e.g. 98% of the French citizen feedback
of 2024H2 on Have Your Say belongs to a single initiative; 37% of the Catalan government texts of 2024H1 to one law
consultation), so the half-year series measures the process of the moment rather than the population of participants.
Here the LLM share is estimated per process (publication on Have Your Say, docket on Regulations.gov, participatory
process / assembly / initiative on Decidim, from the space URL) with at least --min-docs texts, and the yearly series is
rebuilt as the median across the processes of that year, each process counting once whatever its size. CONSUL (Madrid)
has no process structure: its proposals form one continuous stream.

Outputs (in --out): alpha_cells_consultation.csv  (instance, lang, consultation, year, half, n_docs, alpha, ci)
                    alpha_cells_half_consult.csv / alpha_cells_year_consult.csv  (median, quartiles, weighted mean)
                    process_counts.csv            (instance, lang, period: n_docs, n_processes, top-process share)
Same reference construction and arguments as analysis.py (weights, likelihood, exclude-models).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import CHATGPT, References, attach_scores, estimate_cells, half_year, split_author_type  # noqa: E402


def consultation_id(df: pd.DataFrame) -> pd.Series:
    cid = pd.Series(np.nan, index=df.index, dtype=object)
    raw = df["instance_raw"] if "instance_raw" in df else df.instance
    hys = raw.str.startswith("eu_hys")
    if "publication_id" in df.columns:
        cid[hys] = df.loc[hys, "publication_id"].astype(str)
    reg = raw.str.startswith("regulations_gov")
    cid[reg] = df.loc[reg, "id"].astype(str).str.rsplit("-", n=1).str[0]   # docket id = comment id minus its last segment
    other = ~(hys | reg)
    if "space_url" in df.columns:   # Decidim: participatory process, assembly or initiative in the space URL
        slug = df.loc[other, "space_url"].astype(str).str.extract(r"((?:processes|assemblies|initiatives)/[^/?#]+)")[0]
        cid[other] = slug
    return cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--sample", default="data/generated/human_sample.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", default="ratio")
    ap.add_argument("--likelihood", default="occurrence")
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--min-df", type=float, default=0.003)
    ap.add_argument("--min-docs", type=int, default=150)
    ap.add_argument("--min-ref", type=int, default=300)
    ap.add_argument("--kappa", type=float, default=1000.0)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--weights", default="")
    ap.add_argument("--exclude-models", nargs="*", default=[])
    ap.add_argument("--scores", default="")
    ap.add_argument("--templates", default="", help="templated-campaign flags (templates.py): estimates also without them")
    a = ap.parse_args()
    if a.weights:
        import yaml
        a.weights = {str(k): float(v) for k, v in yaml.safe_load(open(a.weights, encoding="utf-8"))["weights"].items()}
    else:
        a.weights = None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    H = pd.concat([pd.read_parquet(p) for p in a.processed], ignore_index=True)
    if "doc_id" not in H.columns:
        H["doc_id"] = H["instance"] + ":" + H["type"] + ":" + H["id"].astype(str) + ":" + H["lang"]
    H = H[~H.is_dup_exact].drop_duplicates("doc_id").copy()
    H["instance_raw"] = H.instance
    H["instance"] = split_author_type(H.instance, H["user_type"] if "user_type" in H.columns else None)
    if a.templates:
        T = pd.read_parquet(a.templates)[["doc_id", "is_template"]].drop_duplicates("doc_id")
        H = H.merge(T, on="doc_id", how="left")
        H["is_template"] = H["is_template"].fillna(False).astype(bool)
    G = pd.read_parquet(a.gen)
    if a.exclude_models:
        G = G[~G.model.isin(a.exclude_models)]
    seed_inst = H.set_index("doc_id")["instance"]
    G["inst"] = G.doc_id.map(seed_inst).fillna(G.doc_id.str.split(":").str[0])
    S = pd.read_parquet(a.sample, columns=["doc_id", "lang", "instance"])
    S["instance"] = S.doc_id.map(seed_inst).fillna(S.instance)
    if a.scores:
        from score_estimator import ScoreReferences
        H, G = attach_scores(H, G, a.scores)
        refs = ScoreReferences(H, G, a.weights, a.min_ref)
    else:
        refs = References(H, G, S, a)

    D = H.copy()
    D["consultation"] = consultation_id(D)
    D["half"] = half_year(D.created_at)
    D["period"] = np.where(D.created_at >= CHATGPT, "post", "pre")
    pc = D.groupby(["instance", "lang", "period"]).apply(lambda g: pd.Series({
        "n_docs": len(g), "n_processes": g.consultation.nunique(),
        "top_process_share": g.consultation.value_counts(normalize=True).iloc[0] if g.consultation.notna().any() else np.nan})).reset_index()
    pc.to_csv(out / "process_counts.csv", index=False)
    D = D[D.consultation.notna()].copy()
    print("texts with a process:", len(D), "| processes:", D.groupby(["instance", "lang"]).consultation.nunique().to_dict(), flush=True)

    cells = estimate_cells(D, refs, a.min_docs, a.n_boot, ["instance", "lang", "consultation"])
    # a consultation is dated by the median timestamp of its texts
    med = D.groupby(["instance", "lang", "consultation"]).created_at.median()
    cells["created_at"] = [med.get((i, l, c)) for i, l, c in zip(cells.instance, cells.lang, cells.consultation)]
    cells["half"] = half_year(pd.to_datetime(cells.created_at, utc=True))
    cells["year"] = pd.to_datetime(cells.created_at, utc=True).dt.year
    cells["post"] = (pd.to_datetime(cells.created_at, utc=True) >= CHATGPT).astype(int)
    cells.to_csv(out / "alpha_cells_consultation.csv", index=False)

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"n_consultations": len(g), "n_docs": g.n_docs.sum(), "alpha": g.alpha.median(),
                          "q1": g.alpha.quantile(0.25), "q3": g.alpha.quantile(0.75),
                          "alpha_wmean": np.average(g.alpha, weights=g.n_docs)})
    hc = cells.groupby(["instance", "lang", "half"]).apply(agg).reset_index()
    hc["t"] = hc["half"].map(lambda h: int(h[:4]) + (0.25 if h.endswith("1") else 0.75)) - 2022.91
    hc["post"] = (hc.t > 0).astype(int)
    hc.to_csv(out / "alpha_cells_half_consult.csv", index=False)
    yc = cells.groupby(["instance", "lang", "year"]).apply(agg).reset_index()
    yc.to_csv(out / "alpha_cells_year_consult.csv", index=False)
    print(hc[hc.n_consultations >= 2].to_string())


if __name__ == "__main__":
    main()
