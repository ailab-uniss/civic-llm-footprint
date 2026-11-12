"""Shared loader of the referee analyses (review_*.py): same construction of H, G and of the calibration sets as
src/civic/analysis.py and scripts/study_extras.py. Run from the project root on the server."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, "src/civic")
from analysis import attach_scores, split_author_type  # noqa: E402
from score_estimator import MIN_G, REGION, HDy, ScoreReferences, lower_bound  # noqa: E402,F401

SCORES = "results/supervised/study_scores_aug.parquet"
WEIGHTS = "configs/reference_weights_vendor.yaml"
OUT = Path("results/review")
OUT.mkdir(parents=True, exist_ok=True)
SKIP = {"tokens", "text", "title"}


def load(extra_cols=()):
    W = {str(k): float(v) for k, v in yaml.safe_load(open(WEIGHTS, encoding="utf-8"))["weights"].items()}
    parts = []
    for p in sorted(glob.glob("data/processed/*.parquet")):
        cols = [c for c in pq.ParquetFile(p).schema_arrow.names if c not in SKIP or c in extra_cols]
        parts.append(pd.read_parquet(p, columns=cols))
    H = pd.concat(parts, ignore_index=True)
    H = H[~H.is_dup_exact].drop_duplicates("doc_id").copy()
    H["instance_raw"] = H.instance
    H["instance"] = split_author_type(H.instance, H["user_type"] if "user_type" in H.columns else None)
    T = pd.read_parquet("results/templates.parquet")[["doc_id", "is_template"]].drop_duplicates("doc_id")
    H = H.merge(T, on="doc_id", how="left")
    H["is_template"] = H.is_template.fillna(False).astype(bool)
    G = pd.read_parquet("data/generated/generated_processed.parquet", columns=["model", "lang", "mode", "variant", "doc_id"])
    G["inst"] = G.doc_id.map(H.set_index("doc_id")["instance"]).fillna(G.doc_id.str.split(":").str[0])
    H, G = attach_scores(H, G, SCORES)
    H["year"] = H.created_at.dt.year
    return H, G, W


def gen_setup(G, W):
    """Generated calibration texts as filtered in ScoreReferences (weighted + region generators, rewrite and generate)."""
    region = [m for m in REGION if m in set(G.model)]
    Gf = G[G.model.isin(set(W) | set(region))]
    Gf = Gf[Gf["mode"].isin(["rewrite", "generate"])]
    return Gf, region


def by_model(Gf, inst, lang):
    gl = Gf[Gf.lang == lang]
    gi = gl[gl.inst == inst]
    by_lang = {m: x.score.values for m, x in gl.groupby("model")}
    return {m: (gi[gi.model == m].score.values if (gi.model == m).sum() >= MIN_G else v) for m, v in by_lang.items()}
