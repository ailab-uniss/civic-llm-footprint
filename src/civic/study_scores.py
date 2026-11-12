"""Cross-fitted classifier scores for the study (4 folds).

Every document is assigned to a fold by a hash of its identifier (generated texts by the hash of their seed, so that a
seed and its rewrites share a fold). For each fold k, XLM-RoBERTa-large is fine-tuned on the pre-ChatGPT human texts and
the generated texts of the other folds (all generators except the legacy GPT-4o mini) and scores the documents of fold k:
pre- and post-ChatGPT civic texts, generated texts, and the MULTITuDE news texts (never used for training). Every score
is therefore out-of-sample, and the calibration of an instance can use all of its pre-ChatGPT texts.

  python src/civic/study_scores.py --processed data/processed/*.parquet --gen data/generated/generated_processed.parquet \
      --model models/xlm-roberta-large --out results/supervised/study_scores.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import split_author_type  # noqa: E402
from supervised_baseline import train  # noqa: E402
from supervised_scores import predict_logit  # noqa: E402

K = 4


def fold4(ids: pd.Series) -> np.ndarray:
    return np.array([int(hashlib.md5(("cal|" + str(i)).encode()).hexdigest(), 16) % K for i in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--mt-human", default="data/external/multitude/mt_human_processed_all.parquet")
    ap.add_argument("--mt-gen", default="data/external/multitude/mt_generated_processed_all.parquet")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-lang", type=int, default=3000)
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(K)))
    ap.add_argument("--augment", action="store_true", help="also train on GPT-4o mini civic texts and on MULTITuDE v3 news (2024 generators, with human news texts as negatives); MULTITuDE v1 (2023 GPT-3.5 / GPT-4) is never used for training")
    ap.add_argument("--news-per-lang", type=int, default=1000)
    ap.add_argument("--train-only", nargs="*", default=None, help="generators allowed in training (review 9: commercial models only)")
    ap.add_argument("--extra-suffix", default=".scores", help="suffix of the score files of --extra; with another suffix the extra files are scored for every fold trained")
    ap.add_argument("--save-model", default="", help="with --folds -1: train on all folds and save the classifier here")
    ap.add_argument("--extra", nargs="*", default=[], help="with --folds -1: external parquet files (column 'text') scored by the final classifier; scores written next to them as <name>.scores.parquet")
    a = ap.parse_args()
    cols = ["doc_id", "lang", "period", "is_dup_exact", "text", "instance", "created_at", "user_type"]
    A = pd.concat([pd.read_parquet(p, columns=[c for c in cols if c in pq.read_schema(p).names]) for p in a.processed],
                  ignore_index=True)
    A["instance"] = split_author_type(A.instance, A["user_type"] if "user_type" in A.columns else None)
    G = pd.read_parquet(a.gen, columns=["doc_id", "src_doc_id", "lang", "model", "mode", "variant", "text"])
    KEEP = ["doc_id", "lang", "fold", "model", "mode", "variant", "text"]
    G = G[G.text.notna()].copy()
    A = A[(~A.is_dup_exact) & A.text.notna() & A.lang.isin(set(G.lang))].drop_duplicates("doc_id").copy()
    A["fold"] = fold4(A.doc_id)
    G["fold"] = fold4(G.src_doc_id.fillna(G.doc_id.str.split("|").str[0]))
    sets = [A.assign(model="human", mode="", variant=-1)[KEEP].assign(set="civic_h"), G[KEEP].assign(set="civic_g")]
    for path, name in ((a.mt_human, "mt_h"), (a.mt_gen, "mt_g")):
        if Path(path).exists():
            names = pq.read_schema(path).names
            M = pd.read_parquet(path, columns=[c for c in ["doc_id", "lang", "model", "mode", "variant", "text"] if c in names])
            M = M[M.text.notna()].copy()
            for c, v in (("model", "human"), ("mode", ""), ("variant", -1)):
                if c not in M:
                    M[c] = v
            M["fold"] = fold4(M.doc_id)
            sets.append(M[KEEP].assign(set=name))
    S = pd.concat(sets, ignore_index=True)
    S["variant"] = pd.to_numeric(S.variant, errors="coerce").fillna(-1).astype(int)
    print("documents to score:", S.groupby("set").size().to_dict(), flush=True)
    pre = A[A.period == "pre"]
    Gtr = G if a.augment else G[G.model != "gpt-4o-mini"]
    MT = S[S.set.isin(["mt_h", "mt_g"])]
    MT = MT[(MT.set == "mt_h") | MT.model.str.startswith("mt_")]   # v3 generators (2024) only
    if a.train_only is not None:
        Gtr = Gtr[Gtr.model.isin(a.train_only)]
        MT = MT[(MT.set == "mt_h") | MT.model.str.contains("gpt-")]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in a.folds:
        sd = k if k >= 0 else 99   # k = -1: no fold left out (final classifier)
        rng = np.random.default_rng(sd)
        texts, labels = [], []
        for l in sorted(set(pre.lang) & set(Gtr.lang)):
            h, g = pre[(pre.lang == l) & (pre.fold != k)], Gtr[(Gtr.lang == l) & (Gtr.fold != k)]
            n = min(a.train_per_lang, len(h), len(g))
            if n < 50:
                continue
            texts += h.text.sample(n, random_state=sd).astype(str).tolist() + g.text.sample(n, random_state=sd).astype(str).tolist()
            labels += [0] * n + [1] * n
        if a.augment:
            for l in sorted(MT.lang.unique()):
                h = MT[(MT.lang == l) & (MT.set == "mt_h") & (MT.fold != k)]
                g = MT[(MT.lang == l) & (MT.set == "mt_g") & (MT.fold != k)]
                n = min(a.news_per_lang, len(h), len(g))
                texts += h.text.sample(n, random_state=sd).astype(str).tolist() + g.text.sample(n, random_state=sd).astype(str).tolist()
                labels += [0] * n + [1] * n
        idx = rng.permutation(len(texts))
        print(f"fold {k}: training on {len(texts)} texts", flush=True)
        model, tok = train(a.model, [texts[i] for i in idx], [labels[i] for i in idx], seed=sd)
        if k < 0:
            if a.save_model:
                model.save_pretrained(a.save_model); tok.save_pretrained(a.save_model)
            for p in a.extra:
                E = pd.read_parquet(p)
                E["score"] = predict_logit(model, tok, E.text.astype(str).tolist())
                E.drop(columns="text").to_parquet(Path(p).with_suffix(a.extra_suffix + ".parquet"), index=False)
                print("scored", p, len(E), flush=True)
            continue
        d = S[S.fold == k]
        s = predict_logit(model, tok, d.text.astype(str).tolist())
        rows.append(d.drop(columns="text").assign(score=s))
        pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
        print(f"fold {k}: scored {len(d)}", flush=True)
        if a.extra_suffix != ".scores":
            for p in a.extra:
                E = pd.read_parquet(p)
                E["score"] = predict_logit(model, tok, E.text.astype(str).tolist())
                E.drop(columns="text").to_parquet(Path(p).with_suffix(a.extra_suffix + ".parquet"), index=False)
                print("scored", p, len(E), flush=True)
        del model; torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
