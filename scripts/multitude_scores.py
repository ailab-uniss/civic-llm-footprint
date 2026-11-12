"""Temporal transfer of the classifier estimator: MULTITuDE news texts (never used for training).

Calibration as for a new platform instance: F_H = scores of one hash half of the human news texts, F_M = scores of the
news texts generated now (generation mode) by the reference generators (market-share weights, as in the study; equal
weights over the other current generators as a check). Test: mixtures of the other half of the human texts and the texts
of a held-out generator (the 2023-2024 MULTITuDE generators and GPT-4o mini of 2024), per language.

  python scripts/multitude_scores.py -> results/multitude/cls_benchmark.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "civic"))
from analysis import fold_of  # noqa: E402
from score_estimator import HDy  # noqa: E402

ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
W = {str(k): float(v) for k, v in yaml.safe_load(open("configs/reference_weights_vendor.yaml", encoding="utf-8"))["weights"].items()}
CURRENT = ["claude-haiku-5-5", "gpt-5.4-mini", "eurollm-9b", "gpt-oss-20b", "llama-3.1-8b", "mistral-small-24b", "qwen2.5-7b",
           "salamandra-7b"]
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "results/supervised/study_scores.parquet")
d = pd.read_parquet(SRC)
d = d[d.set.isin(["mt_h", "mt_g"])]
H = d[d.set == "mt_h"].copy(); H["half"] = fold_of(H.doc_id)
G = d[(d.set == "mt_g") & (d["mode"] == "generate")]
held = [m for m in G.model.unique() if m not in CURRENT]
rng = np.random.default_rng(0)
rows = []
for lang in sorted(H.lang.unique()):
    hc, ht = H[(H.lang == lang) & (H.half == 0)].score.values, H[(H.lang == lang) & (H.half == 1)].score.values
    gl = G[G.lang == lang]
    for m in held + CURRENT:
        pg = gl[gl.model == m].score.values
        if len(pg) < 100 or len(ht) < 200:
            continue
        refs = {"weighted": {g: W[g] for g in CURRENT if g in W and g != m},
                "equal": {g: 1.0 for g in CURRENT if g != m}}
        ests = {k: HDy(hc, {g: gl[gl.model == g].score.values for g in w}, w) for k, w in refs.items()}
        for alpha in ALPHAS:
            for r in range(10):
                n = min(1000, len(ht), int(len(pg) / max(alpha, 0.5)))
                k = int(round(alpha * n))
                s = np.concatenate([ht[rng.choice(len(ht), n - k, replace=False)],
                                    pg[rng.choice(len(pg), k, replace=False)] if k else np.array([])])
                for ref, e in ests.items():
                    o = e.estimate(s)
                    rows.append(dict(lang=lang, heldout_model=m, reference=ref, alpha=alpha, rep=r, n=n,
                                     alpha_hat=o["alpha"], lb=o["lb"], current=m in CURRENT))
R = pd.DataFrame(rows)
Path("results/multitude").mkdir(parents=True, exist_ok=True)
R.to_parquet(f"results/multitude/cls_benchmark{SRC.stem.replace('study_scores', '')}.parquet", index=False)
R["err"] = (R.alpha_hat - R.alpha) * 100
x = R[(R.alpha >= 0.05) & (R.reference == "weighted")]
print(x.groupby("heldout_model").err.agg(mae=lambda e: e.abs().mean(), bias="mean").round(1).sort_values("mae").to_string())
x = x.assign(rec=x.alpha_hat / x.alpha, rec_lb=x.lb / x.alpha, viol=x.lb > x.alpha + 0.01)
print(x.groupby("heldout_model")[["rec", "rec_lb", "viol"]].mean().round(2).to_string())
print("floor (alpha=0):", round(R[(R.alpha == 0) & (R.reference == "weighted")].alpha_hat.mean() * 100, 2))
