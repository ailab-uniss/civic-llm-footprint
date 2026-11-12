"""Supervised baseline requested in review: a multilingual classifier fine-tuned on the same reference texts, used for
quantification with the standard corrections (classify & count, probabilistic CC, adjusted CC, probabilistic ACC,
EM/SLD), under the same protocols as benchmark.py and benchmark_robust.py.

For each held-out generator m*, XLM-RoBERTa-large (one multilingual model) is fine-tuned on pre-ChatGPT human texts versus
texts of the other generators (the legacy GPT-4o mini is never used for training); 10% of the training texts are kept
apart to estimate, per language, the true- and false-positive rates and the training prior.
  main      training on fold 0; mixtures per language of fold-1 human texts and fold-1 texts of m* (benchmark.py protocol)
  instance  same model; mixtures per instance of that instance's fold-1 human texts and m* texts derived from fold-1
            seeds of the instance (of the language when fewer than 60) -- the composition-shift protocol
  period    a second model trained on texts of 2020 and earlier (and generated texts seeded from them); mixtures per
            instance of 2021-2022 human texts and m* texts seeded from 2021-2022 texts -- the temporal-shift protocol

  python src/civic/supervised_baseline.py --processed data/processed/*.parquet --gen data/generated/generated_processed.parquet \
      --model models/xlm-roberta-large --out results/supervised/supervised.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import fold_of, split_author_type  # noqa: E402

ALPHAS = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50]
ALPHAS_SHIFT = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def predict(model, tok, texts, bs=64, max_len=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.float()
            out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train(model_path, texts, labels, epochs=1, bs=16, lr=1e-5, max_len=256, seed=0):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).cuda()
    idx = np.random.default_rng(seed).permutation(len(texts))
    batches = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * len(batches) * epochs), len(batches) * epochs)
    model.train()
    for _ in range(epochs):
        for j, b in enumerate(batches):
            enc = tok([texts[i] for i in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor([labels[i] for i in b], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            if j % 1000 == 0:
                print(f"  step {j}/{len(batches)} loss {loss.item():.3f}", flush=True)
    return model, tok


def quantify(p, tpr, fpr, ptpr, pfpr, prior):
    cc = float((p >= 0.5).mean()); pcc = float(p.mean())
    acc = (cc - fpr) / (tpr - fpr) if tpr - fpr > 0.05 else cc
    pacc = (pcc - pfpr) / (ptpr - pfpr) if ptpr - pfpr > 0.05 else pcc
    q = prior   # EM / SLD (Saerens et al. 2002)
    for _ in range(200):
        w1, w0 = q / prior, (1 - q) / (1 - prior)
        post = p * w1 / (p * w1 + (1 - p) * w0)
        qn = float(post.mean())
        if abs(qn - q) < 1e-6:
            break
        q = qn
    clip = lambda v: float(min(1.0, max(0.0, v)))
    return {"cc": clip(cc), "pcc": clip(pcc), "acc": clip(acc), "pacc": clip(pacc), "sld": clip(q)}


def fit_and_rates(a, Htr, Gtr, langs, rng):
    tr_t, tr_y, tr_l = [], [], []
    for l in langs:
        h, g = Htr[Htr.lang == l], Gtr[Gtr.lang == l]
        k = min(a.train_per_lang, len(h), len(g))
        if k < 50:
            continue
        tr_t += h.text.sample(k, random_state=1).astype(str).tolist() + g.text.sample(k, random_state=1).astype(str).tolist()
        tr_y += [0] * k + [1] * k; tr_l += [l] * (2 * k)
    idx = rng.permutation(len(tr_t)); cut = int(0.9 * len(idx))
    model, tok = train(a.model, [tr_t[i] for i in idx[:cut]], [tr_y[i] for i in idx[:cut]])
    vi = idx[cut:]
    pv = predict(model, tok, [tr_t[i] for i in vi]); yv = np.array([tr_y[i] for i in vi]); lv = np.array([tr_l[i] for i in vi])

    def rates(mask):
        p, y = pv[mask], yv[mask]
        return (float((p[y == 1] >= 0.5).mean()), float((p[y == 0] >= 0.5).mean()), float(p[y == 1].mean()), float(p[y == 0].mean()), float(y.mean()))
    glob = rates(np.ones(len(yv), bool))
    per = {l: (rates(lv == l) if (lv == l).sum() >= 100 and 0 < yv[lv == l].mean() < 1 else glob) for l in langs}
    return model, tok, per


def mixtures(ph, pg, rl, alphas, reps, n_max, rng, base, out):
    if len(ph) < 100 or len(pg) < 30:
        return
    for alpha in alphas:
        for r in range(reps):
            n = min(n_max, len(ph), int(len(pg) / max(alpha, 0.5)))
            k = int(round(alpha * n))
            p = np.concatenate([ph[rng.choice(len(ph), n - k, replace=False)], pg[rng.choice(len(pg), k, replace=False)] if k else np.array([])])
            for meth, v in quantify(p, *rl).items():
                out.append(dict(base, alpha=alpha, rep=r, n=n, method=f"xlmr+{meth}", alpha_hat=v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", nargs="+", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-lang", type=int, default=2500)
    ap.add_argument("--pool-per-lang", type=int, default=2000)
    ap.add_argument("--pool-per-inst", type=int, default=1500)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--heldout", nargs="*", default=None)
    ap.add_argument("--exps", nargs="+", default=["main", "instance", "period"])
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cols = ["doc_id", "lang", "period", "is_dup_exact", "text", "instance", "created_at"]
    H = pd.concat([pd.read_parquet(p) for p in a.processed], ignore_index=True)
    H["instance"] = split_author_type(H.instance, H["user_type"] if "user_type" in H.columns else None)
    H = H[cols]
    H = H[(H.period == "pre") & (~H.is_dup_exact) & H.text.notna()].drop_duplicates("doc_id").copy()
    H["fold"] = fold_of(H.doc_id); H["year"] = H.created_at.dt.year
    G = pd.read_parquet(a.gen, columns=["doc_id", "lang", "model", "mode", "text"])
    G = G[G.text.notna()].copy()
    G["fold"] = fold_of(G.doc_id)
    G["inst"] = G.doc_id.map(H.set_index("doc_id").instance).fillna(G.doc_id.str.split(":").str[0])
    G["year"] = G.doc_id.map(H.set_index("doc_id").year)
    langs = sorted(set(H.lang) & set(G.lang))
    models = sorted(G.model.unique())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res = []
    for m in (a.heldout or models):
        refm = [x for x in models if x not in (m, "gpt-4o-mini")]
        if {"main", "instance"} & set(a.exps):
            print(f"heldout {m}: fold-0 model", flush=True)
            model, tok, per = fit_and_rates(a, H[H.fold == 0], G[(G.fold == 0) & G.model.isin(refm)], langs, rng)
            for l in langs:
                h1 = H[(H.lang == l) & (H.fold == 1)]
                g1 = G[(G.lang == l) & (G.fold == 1) & (G.model == m)]
                if len(h1) < 100 or len(g1) < 30:
                    continue
                if "main" in a.exps:
                    hs = h1.sample(min(a.pool_per_lang, len(h1)), random_state=2); gs = g1.sample(min(a.pool_per_lang, len(g1)), random_state=2)
                    mixtures(predict(model, tok, hs.text.astype(str).tolist()), predict(model, tok, gs.text.astype(str).tolist()),
                             per[l], ALPHAS, a.reps, 1000, rng, dict(exp="main", lang=l, heldout_model=m, setting="main"), res)
                if "instance" in a.exps:
                    pg_all = None
                    for inst, hi in h1.groupby("instance"):
                        if len(hi) < 150 or (H[(H.lang == l) & (H.fold == 0)].instance == inst).sum() < 300:
                            continue
                        hi = hi.sample(min(a.pool_per_inst, len(hi)), random_state=3)
                        gi = g1[g1.inst == inst]
                        src = "instance"
                        if len(gi) < 60:
                            if pg_all is None:
                                ga = g1.sample(min(a.pool_per_lang, len(g1)), random_state=2)
                                pg_all = predict(model, tok, ga.text.astype(str).tolist())
                            pg, src = pg_all, "language"
                        else:
                            pg = predict(model, tok, gi.text.astype(str).tolist())
                        mixtures(predict(model, tok, hi.text.astype(str).tolist()), pg, per[l], ALPHAS_SHIFT, 5, 500, rng,
                                 dict(exp="instance", lang=l, heldout_model=m, setting="held out", instance=inst, llm_source=src), res)
            del model; torch.cuda.empty_cache()
            pd.DataFrame(res).to_parquet(a.out, index=False)
        if "period" in a.exps:
            print(f"heldout {m}: <=2020 model", flush=True)
            model, tok, per = fit_and_rates(a, H[H.year <= 2020], G[(G.year <= 2020) & G.model.isin(refm)], langs, rng)
            for l in langs:
                ht = H[(H.lang == l) & H.year.isin([2021, 2022])]
                gt = G[(G.lang == l) & G.year.isin([2021, 2022]) & (G.model == m)]
                if len(gt) < 30:
                    continue
                pg_all = None
                for inst, hi in ht.groupby("instance"):
                    if len(hi) < 150 or ((H.lang == l) & (H.year <= 2020) & (H.instance == inst)).sum() < 300:
                        continue
                    hi = hi.sample(min(a.pool_per_inst, len(hi)), random_state=3)
                    gi = gt[gt.inst == inst]
                    src = "instance"
                    if len(gi) < 60:
                        if pg_all is None:
                            pg_all = predict(model, tok, gt.sample(min(a.pool_per_lang, len(gt)), random_state=2).text.astype(str).tolist())
                        pg, src = pg_all, "language"
                    else:
                        pg = predict(model, tok, gi.text.astype(str).tolist())
                    mixtures(predict(model, tok, hi.text.astype(str).tolist()), pg, per[l], ALPHAS_SHIFT, 5, 500, rng,
                             dict(exp="period", lang=l, heldout_model=m, setting="held out", instance=inst, llm_source=src), res)
            del model; torch.cuda.empty_cache()
            pd.DataFrame(res).to_parquet(a.out, index=False)
        print(f"heldout {m}: done", flush=True)
    df = pd.DataFrame(res)
    df["err"] = (df.alpha_hat - df.alpha) * 100
    print(df.groupby(["exp", "method"]).err.agg(mae=lambda e: e.abs().mean(), bias="mean").round(2).to_string())


if __name__ == "__main__":
    main()
