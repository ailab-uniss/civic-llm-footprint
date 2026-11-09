"""Create synthetic result files with the real schemas, to exercise figures.py / make_tables.py before real results exist.
python tests/make_fake_results.py results_fake && python src/civic/figures.py --results results_fake --out results_fake/fig
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results_fake")
out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
langs = ["ca", "fr", "fi", "en", "es", "it"]
models = ["gpt-oss-20b", "gemma-2-9b-it", "qwen2.5-7b", "gpt-5.4-mini"]
alphas = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5]
rows = []
for lang in langs:
    for sel in ["ratio", "pos", "all"]:
        for scen in ["all", "rewrite", "generate", "translate", "short"]:
            for m in models:
                for a in alphas:
                    for r in range(10):
                        sd = 0.01 + 0.02 * (scen == "short")
                        est = a + rng.normal(0.005, sd)
                        rows.append(dict(lang=lang, selection=sel, scenario=scen, heldout_model=m, alpha=a, rep=r, n=1000,
                                         alpha_hat=max(0, est), ci_lo=max(0, est - 2 * sd), ci_hi=est + 2 * sd, vocab=1000, method="distributional"))
                        if sel == "ratio":
                            for det in ["pangram", "binoculars", "fastdetect"]:
                                for meth, bias, s in [("raw", 0.08, 0.02), ("rogan_gladen", 0.0, 0.04), ("bayes", 0.0, 0.04)]:
                                    e = a + bias + rng.normal(0, s)
                                    rows.append(dict(lang=lang, selection=sel, scenario=scen, heldout_model=m, alpha=a, rep=r, n=1000,
                                                     alpha_hat=max(0, e), ci_lo=max(0, e - 2 * s), ci_hi=e + 2 * s, vocab=None, method=f"{det}+{meth}"))
df = pd.DataFrame(rows)
df.to_parquet(out / "benchmark.parquet", index=False)
sys.path.insert(0, "src/civic")
from benchmark import summarise  # noqa: E402
summarise(df).to_csv(out / "benchmark.summary.csv", index=False)

# cells
insts = {"ca": ["barcelona", "participa_gencat", "espluguesparticipa_diba"], "fr": ["participation_bordeaux_metropole", "ecrivons_angers", "oye_participer_lyon", "jeparticipe_toulouse"],
         "fi": ["osallistu_tuusula", "asukasbudjetti_turku"], "en": ["meta_decidim"], "es": ["barcelona"], "it": []}
cells = []
for lang, ii in insts.items():
    for inst in ii:
        for y in range(2018, 2027):
            for h in (1, 2):
                t = y + (0.25 if h == 1 else 0.75) - 2022.91
                a = 0.01 + (0.06 * min(max(t, 0), 3) / 3) + rng.normal(0, 0.01)
                cells.append(dict(instance=inst, lang=lang, half=f"{y}H{h}", n_docs=int(rng.integers(150, 2000)), alpha=max(0, a), ci_lo=max(0, a - 0.02), ci_hi=a + 0.02,
                                  mean_words=90, median_words=70, n_authors=100, campaign_share=0.05, t=t, post=int(t > 0)))
pd.DataFrame(cells).to_csv(out / "alpha_cells_half.csv", index=False)
per = []
for lang, ii in insts.items():
    for inst in ii:
        for typ in ["proposal", "comment"]:
            for period, a in [("pre", 0.01), ("post", 0.07)]:
                a = a + rng.normal(0, 0.01)
                per.append(dict(instance=inst, lang=lang, type=typ, period=period, n_docs=800, alpha=max(0, a), ci_lo=max(0, a - 0.02), ci_hi=a + 0.02))
pd.DataFrame(per).to_csv(out / "alpha_cells_period.csv", index=False)
print("fake results in", out)
