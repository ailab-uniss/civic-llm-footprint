"""Classifier scores for quantification under shift (review M7 and the forward-validation analysis).

A multilingual XLM-RoBERTa-large is fine-tuned on pre-ChatGPT human texts versus generated texts; every document needed by
the quantifiers is scored once and saved, so that the quantifiers (classify and count, adjusted counts with language- or
instance-level rates, median sweep, distribution matching, EM) are compared offline on the same scores.

Protocols (per held-out generator m*; the legacy GPT-4o mini is never used for training):
  hash    fit = pre-ChatGPT texts of fold 0, test = fold 1 (benchmark.py split)
  period  fit = texts of 2020 and earlier (generated texts seeded from them), test = 2021-2022 texts (forward validation)
Within the fit set a calibration part (one quarter, by a second hash) is never used for training: it supplies the human
and generated score distributions of each language and instance, as the pre-ChatGPT texts of an instance do in the study.
  study   fit = all pre-ChatGPT texts, all generators (no held-out): scores of every document of the corpus.

Roles saved: cal_h, cal_g (calibration), test_h (human test texts), test_g (texts of m*), corpus (study protocol).

  python src/civic/supervised_scores.py --processed data/processed/*.parquet --gen data/generated/generated_processed.parquet \
      --model models/xlm-roberta-large --out results/supervised/scores.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import fold_of, split_author_type  # noqa: E402
from supervised_baseline import train  # noqa: E402


def is_cal(ids: pd.Series) -> np.ndarray:
    return np.array([int(hashlib.md5(("cal|" + str(i)).encode()).hexdigest(), 16) % 4 == 0 for i in ids])


def predict_logit(model, tok, texts, bs=128, max_len=256):
    model.eval(); out = []
    order = np.argsort([len(t) for t in texts])          # length-sorted batches: faster
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = [texts[j] for j in order[i:i + bs]]
            enc = tok(b, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**enc).logits.float()
            out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    s = np.concatenate(out) if out else np.array([])
    res = np.empty_like(s); res[order] = s
    return res


def cap(df, by, k, seed=3):
    return df.sample(frac=1.0, random_state=seed).groupby(by, dropna=False).head(k)


def run(a, H, G, fit_h, fit_g, test_h, test_g, train_models, tag, heldout, out_rows, news=None):
    rng = np.random.default_rng(0)
    cal_h, tr_h = fit_h[fit_h.cal], fit_h[~fit_h.cal]
    gg = fit_g[fit_g.model.isin(train_models)]
    cal_g, tr_g = gg[gg.cal], gg[~gg.cal]
    texts, labels = [], []
    for l in sorted(set(tr_h.lang) & set(tr_g.lang)):
        h, g = tr_h[tr_h.lang == l], tr_g[tr_g.lang == l]
        k = min(a.train_per_lang, len(h), len(g))
        if k < 50:
            continue
        texts += h.text.sample(k, random_state=1).astype(str).tolist() + g.text.sample(k, random_state=1).astype(str).tolist()
        labels += [0] * k + [1] * k
    if news is not None:   # news texts of 2024 generators (MULTITuDE v3) and human news: training only
        for l in sorted(news.lang.unique()):
            h, g = news[(news.lang == l) & (news.model == "human")], news[(news.lang == l) & (news.model != "human")]
            k = min(a.news_per_lang, len(h), len(g))
            texts += h.text.sample(k, random_state=1).astype(str).tolist() + g.text.sample(k, random_state=1).astype(str).tolist()
            labels += [0] * k + [1] * k
    idx = rng.permutation(len(texts))
    model, tok = train(a.model, [texts[i] for i in idx], [labels[i] for i in idx])
    parts = [("cal_h", cap(cal_h, ["lang", "instance"], a.cap_inst)),
             ("cal_g", cap(cal_g, ["lang", "inst", "model"], a.cap_gen)),
             ("test_h", cap(test_h, ["lang", "instance"], a.cap_inst)) if test_h is not None else None,
             ("test_g", cap(test_g, ["lang", "inst"], a.cap_gen)) if test_g is not None else None]
    for p in parts:
        if p is None or not len(p[1]):
            continue
        role, d = p
        s = predict_logit(model, tok, d.text.astype(str).tolist())
        r = pd.DataFrame({"protocol": tag, "heldout_model": heldout, "role": role, "doc_id": d.doc_id.values,
                          "lang": d.lang.values, "score": s})
        r["instance"] = d["instance"].values if "instance" in d else d["inst"].values
        r["year"] = d["year"].values
        r["model"] = d["model"].values if "model" in d else "human"
        r["mode"] = d["mode"].values if "mode" in d else ""
        out_rows.append(r)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-lang", type=int, default=2500)
    ap.add_argument("--cap-inst", type=int, default=1500)
    ap.add_argument("--cap-gen", type=int, default=600)
    ap.add_argument("--heldout", nargs="*", default=None)
    ap.add_argument("--protocols", nargs="+", default=["hash", "period"])
    ap.add_argument("--augment", action="store_true", help="train also on the legacy generators (GPT-4o, GPT-4o mini) and on MULTITuDE v3 news texts (2024 generators), as the study classifier (study_scores.py --augment)")
    ap.add_argument("--news-per-lang", type=int, default=1000)
    ap.add_argument("--mt-human", default="data/external/multitude/mt_human_processed_all.parquet")
    ap.add_argument("--mt-gen", default="data/external/multitude/mt_generated_processed_all.parquet")
    ap.add_argument("--train-only", nargs="*", default=None, help="generators allowed in training and calibration (review 9: commercial models only); the news texts are then restricted to the GPT generators of MULTITuDE v3")
    ap.add_argument("--groups", nargs="*", default=[], help="name=model1,model2: generators held out together (e.g. one vendor)")
    ap.add_argument("--hold", nargs="*", default=[], help="attr=value (mode=rewrite, variant=2): generated texts with this value are "
                    "excluded from training and calibration and form the test component, for all generators")
    a = ap.parse_args()
    cols = ["doc_id", "lang", "period", "is_dup_exact", "text", "instance", "created_at"]
    import pyarrow.parquet as pq
    A = pd.concat([pd.read_parquet(p, columns=[c for c in cols + ["user_type"] if c in pq.read_schema(p).names])
                   for p in a.processed], ignore_index=True)
    A["instance"] = split_author_type(A.instance, A["user_type"] if "user_type" in A.columns else None)
    A = A[(~A.is_dup_exact) & A.text.notna()].drop_duplicates("doc_id").copy()
    A["year"] = A.created_at.dt.year
    H = A[A.period == "pre"].copy()
    H["fold"] = fold_of(H.doc_id); H["cal"] = is_cal(H.doc_id)
    G = pd.read_parquet(a.gen, columns=["doc_id", "src_doc_id", "lang", "model", "mode", "variant", "text"])
    G = G[G.text.notna()].copy()
    G["fold"] = fold_of(G.doc_id)
    seed = G.src_doc_id.fillna(G.doc_id.str.split("|").str[0])
    G["cal"] = is_cal(seed)
    G["inst"] = G.doc_id.map(H.set_index("doc_id").instance).fillna(G.doc_id.str.split(":").str[0])
    G["year"] = G.doc_id.map(H.set_index("doc_id").year)
    models = sorted(G.model.unique())
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    news = None
    if a.augment:
        mh = pd.read_parquet(a.mt_human, columns=["doc_id", "lang", "text"]).assign(model="human")
        mg = pd.read_parquet(a.mt_gen, columns=["doc_id", "lang", "text", "model"])
        news = pd.concat([mh, mg[mg.model.str.startswith("mt_")]], ignore_index=True)
        news = news[news.text.notna()]
        if a.train_only is not None:
            news = news[(news.model == "human") | news.model.str.contains("gpt-")]
    cases = [(m, [m]) for m in (a.heldout if a.heldout is not None else models)]
    cases += [("group:" + g.split("=")[0], g.split("=")[1].split(",")) for g in a.groups]

    def save():
        pd.concat(rows, ignore_index=True).to_parquet(out, index=False)

    for h in a.hold:    # held-out generation mode or prompt variant
        attr, val = h.split("=")
        is_out = G[attr].astype(str) == val
        refm = [x for x in models if a.augment or x != "gpt-4o-mini"]
        for tag, fh, fg, th, tg in (("hash", H[H.fold == 0], G[(G.fold == 0) & ~is_out], H[H.fold == 1], G[(G.fold == 1) & is_out]),
                                    ("period", H[H.year <= 2020], G[(G.year <= 2020) & ~is_out], H[H.year.isin([2021, 2022])],
                                     G[G.year.isin([2021, 2022]) & is_out])):
            if tag in a.protocols:
                print(f"[{tag}] heldout {attr}:{val}", flush=True)
                model, tok = run(a, H, G, fh, fg, th, tg[tg.model.isin(refm)], refm, tag, f"{attr}:{val}", rows, news)
                del model; torch.cuda.empty_cache(); save()
    for m, out_models in cases:
        refm = [x for x in models if x not in out_models and (a.augment or x != "gpt-4o-mini")
                and (a.train_only is None or x in a.train_only)]
        if "hash" in a.protocols:
            print(f"[hash] heldout {m}", flush=True)
            model, tok = run(a, H, G, H[H.fold == 0], G[G.fold == 0], H[H.fold == 1],
                             G[(G.fold == 1) & G.model.isin(out_models)], refm, "hash", m, rows, news)
            del model; torch.cuda.empty_cache(); save()
        if "period" in a.protocols:
            print(f"[period] heldout {m}", flush=True)
            model, tok = run(a, H, G, H[H.year <= 2020], G[G.year <= 2020], H[H.year.isin([2021, 2022])],
                             G[G.year.isin([2021, 2022]) & G.model.isin(out_models)], refm, "period", m, rows, news)
            del model; torch.cuda.empty_cache(); save()
    if "study" in a.protocols:
        print("[study] all generators", flush=True)
        refm = [x for x in models if x != "gpt-4o-mini"]
        model, tok = run(a, H, G, H, G, None, None, refm, "study", "none", rows)
        C = A[A.lang.isin(set(G.lang))].copy()
        s = predict_logit(model, tok, C.text.astype(str).tolist())
        rows.append(pd.DataFrame({"protocol": "study", "heldout_model": "none", "role": "corpus", "doc_id": C.doc_id.values,
                                  "lang": C.lang.values, "score": s, "instance": C.instance.values, "year": C.year.values,
                                  "model": "human", "mode": ""}))
        # in-sample flag: the study scores of training texts are not used for calibration (cal part only)
        save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
