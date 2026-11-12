"""Bounding the generator-switch artefact in the yearly series (review M3).

For each year, the assistants plausibly in use are a set of generators whose recovery factor f_g(alpha) =
E[alpha_hat]/alpha was measured on MULTITuDE against a reference of today's models (Table tab:multitude). An observed
estimate a_hat is then compatible with a_hat / (f_g / f_GPT-5.4 mini), the estimate that the same texts would give if written by the generator of the reference, for every plausible g; the range over g bounds the
artefact. Plausible sets: 2023 GPT-3.5 / GPT-4 (2023 versions); 2024 GPT-3.5 (Jan 2024) / GPT-4o mini; 2025 GPT-4o mini /
GPT-5.4 mini; 2026 GPT-5.4 mini / GPT-4o mini.

  python scripts/generator_bounds.py -> results/generator_bounds.csv
"""
import numpy as np
import pandas as pd

mb = pd.read_parquet("results/multitude/benchmark_all.parquet")
mb = mb[(mb.method == "distributional") & (mb.selection == "ratio") & (mb.likelihood == "length") & (mb.scenario == "generate") & (mb.alpha > 0)]
f = (mb.groupby(["heldout_model", "alpha"]).alpha_hat.mean() / mb.groupby(["heldout_model", "alpha"]).alpha.mean()).unstack()
PLAUS = {2023: ["mt1_gpt-3.5-turbo", "mt1_gpt-4"], 2024: ["mt_gpt-3.5-turbo-0125", "gpt-4o-mini"],
         2025: ["gpt-4o-mini", "gpt-5.4-mini"], 2026: ["gpt-5.4-mini", "gpt-4o-mini"]}


def factor(g, a_hat):
    """Recovery of generator g relative to GPT-5.4 mini (the dominant generator of the study reference), averaged over
    true shares 0.1-0.3; the ratio cancels the higher floor of the news domain."""
    cols = [c for c in f.columns if 0.1 <= c <= 0.3]
    return float((f.loc[g, cols] / f.loc["gpt-5.4-mini", cols]).mean())


y = pd.read_csv("results/alpha_cells_instance_year.csv")
series = {"Madrid": y[(y.instance == "decide_madrid") & (y.lang == "es")].set_index("year").alpha}
pooled = y[y.year >= 2023].groupby("year").apply(lambda g: (g.alpha * g.n_docs).sum() / g.n_docs.sum())
series["All instances (pooled)"] = pooled
rows = []
for name, s in series.items():
    for yr, a_hat in s.items():
        if yr not in PLAUS or a_hat < 0.005:
            if yr in PLAUS:
                rows.append(dict(series=name, year=yr, alpha_hat=a_hat, lo=a_hat, hi=a_hat))
            continue
        vals = [a_hat / factor(g, a_hat) for g in PLAUS[yr]]
        rows.append(dict(series=name, year=yr, alpha_hat=a_hat, lo=min(vals), hi=max(vals)))
R = pd.DataFrame(rows)
R.to_csv("results/generator_bounds.csv", index=False)
print((R.set_index(["series", "year"]) * 100).round(1).to_string())
