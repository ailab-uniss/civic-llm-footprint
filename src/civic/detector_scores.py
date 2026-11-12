"""Zero-shot detector scores arranged for per-instance calibrated quantification (an independent signal).

Binoculars needs no training, so its scores are the same in every protocol; this script only arranges them as
supervised_scores.py and study_scores.py do for the fine-tuned classifier, so that the same quantifiers
(scripts/quantify_scores.py) and the same study pipeline (analysis.py --scores) run on them unchanged.

Inputs: data/scores/scores_<detector>.parquet (generated texts, doc_id = seed|model|mode|variant) and
data/scores/corpus/bino_chunk_*.parquet (every civic text; scripts/run_binoculars_corpus.sh).
Outputs: results/supervised/scores_<detector>.parquet (benchmark roles per held-out generator and protocol) and
results/supervised/study_scores_<detector>.parquet (study format).

  python src/civic/detector_scores.py --processed data/processed/*.parquet --gen data/generated/generated_processed.parquet
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import fold_of, split_author_type  # noqa: E402
from supervised_scores import cap, is_cal  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--detector", default="binoculars")
    ap.add_argument("--cap-inst", type=int, default=1500)
    ap.add_argument("--cap-gen", type=int, default=600)
    a = ap.parse_args()
    col = f"score_{a.detector}"
    cols = ["doc_id", "lang", "period", "is_dup_exact", "instance", "created_at", "user_type"]
    A = pd.concat([pd.read_parquet(p, columns=[c for c in cols if c in pq.read_schema(p).names]) for p in a.processed],
                  ignore_index=True)
    A["instance"] = split_author_type(A.instance, A["user_type"] if "user_type" in A.columns else None)
    A = A[~A.is_dup_exact].drop_duplicates("doc_id").copy()
    A["year"] = A.created_at.dt.year
    sc = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"data/scores/corpus/{a.detector[:4]}_chunk_*.parquet"))])
    A = A.merge(sc.drop_duplicates("doc_id").rename(columns={col: "score"}), on="doc_id", how="inner")
    G = pd.read_parquet(a.gen, columns=["doc_id", "src_doc_id", "lang", "model", "mode", "variant"])
    gs = pd.read_parquet(f"data/scores/scores_{a.detector}.parquet").rename(columns={col: "score"})
    G["key"] = G.doc_id + "|" + G.model + "|" + G["mode"] + "|" + G.variant.astype(str)
    G = G.merge(gs.rename(columns={"doc_id": "key"}).drop_duplicates("key"), on="key", how="inner")
    G = G[G.lang.isin(set(A.lang))]
    A = A[A.lang.isin(set(G.lang))]
    print(len(A), "civic texts and", len(G), "generated texts with a score", flush=True)
    out = Path("results/supervised"); out.mkdir(parents=True, exist_ok=True)
    study = pd.concat([A[["doc_id", "lang", "score"]].assign(model="human", mode="", variant=-1, set="civic_h"),
                       G[["doc_id", "lang", "score", "model", "mode", "variant"]].assign(set="civic_g")], ignore_index=True)
    study["variant"] = pd.to_numeric(study.variant, errors="coerce").fillna(-1).astype(int)
    study.to_parquet(out / f"study_scores_{a.detector}.parquet", index=False)

    H = A[A.period == "pre"].copy()
    H["fold"] = fold_of(H.doc_id); H["cal"] = is_cal(H.doc_id)
    G["fold"] = fold_of(G.doc_id)
    G["cal"] = is_cal(G.src_doc_id.fillna(G.doc_id.str.split("|").str[0]))
    G["inst"] = G.doc_id.map(H.set_index("doc_id").instance).fillna(G.doc_id.str.split(":").str[0])
    G["year"] = G.doc_id.map(H.set_index("doc_id").year)
    models = sorted(G.model.unique())
    rows = []
    for m in models:
        for tag, fit_h, fit_g, test_h, test_g in (
                ("hash", H[H.fold == 0], G[G.fold == 0], H[H.fold == 1], G[(G.fold == 1) & (G.model == m)]),
                ("period", H[H.year <= 2020], G[G.year <= 2020], H[H.year.isin([2021, 2022])],
                 G[G.year.isin([2021, 2022]) & (G.model == m)])):
            parts = [("cal_h", cap(fit_h[fit_h.cal], ["lang", "instance"], a.cap_inst)),
                     ("cal_g", cap(fit_g[fit_g.cal & (fit_g.model != m)], ["lang", "inst", "model"], a.cap_gen)),
                     ("test_h", cap(test_h, ["lang", "instance"], a.cap_inst)),
                     ("test_g", cap(test_g, ["lang", "inst"], a.cap_gen))]
            for role, d in parts:
                r = pd.DataFrame({"protocol": tag, "heldout_model": m, "role": role, "doc_id": d.doc_id.values,
                                  "lang": d.lang.values, "score": d.score.values, "year": d.year.values})
                r["instance"] = d["instance"].values if "instance" in d else d["inst"].values
                r["model"] = d["model"].values if "model" in d else "human"
                r["mode"] = d["mode"].values if "mode" in d else ""
                rows.append(r)
    pd.concat(rows, ignore_index=True).to_parquet(out / f"scores_{a.detector}.parquet", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
