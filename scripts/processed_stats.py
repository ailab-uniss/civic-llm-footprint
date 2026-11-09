"""Corpus statistics after pre-processing -> results/processed_stats.csv and paper/tables/tab_instances.tex.
Run on the server inside the venv:  python scripts/processed_stats.py
Only instances/languages that satisfy the inclusion criteria are kept for the study:
  >= MIN_PRE pre-ChatGPT texts and >= MIN_POST post-ChatGPT texts in the same language (any type).
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import yaml

MIN_PRE, MIN_POST = 300, 200
cfg = yaml.safe_load(open("configs/instances.yaml", encoding="utf-8"))["instances"]
rows = []
for p in sorted(glob.glob("data/processed/*.parquet")):
    df = pd.read_parquet(p, columns=["instance", "lang", "type", "period", "created_at", "is_dup_exact", "n_words"])
    df = df[~df.is_dup_exact]
    for (inst, lang, typ, per), g in df.groupby(["instance", "lang", "type", "period"]):  # a file may hold several instances
        rows.append(dict(instance=inst, lang=lang, type=typ, period=per, n=len(g),
                         first=g.created_at.min().date(), last=g.created_at.max().date(), mean_words=g.n_words.mean()))
st = pd.DataFrame(rows)
tot = st.groupby(["instance", "lang", "period"]).n.sum().unstack(fill_value=0)
ok = tot[(tot.get("pre", 0) >= MIN_PRE) & (tot.get("post", 0) >= MIN_POST)].index
st["included"] = [((i, l) in set(ok)) for i, l in zip(st.instance, st.lang)]
Path("results").mkdir(exist_ok=True)
st.to_csv("results/processed_stats_all.csv", index=False)
st[st.included].drop(columns="included").to_csv("results/processed_stats.csv", index=False)

# LaTeX table: one row per included instance x language (all text types pooled: proposals, comments, feedback)
inc = st[st.included]
t = inc.pivot_table(index=["instance", "lang"], columns="period", values="n", aggfunc="sum", fill_value=0)
first = inc.groupby(["instance", "lang"])["first"].min()
last = inc.groupby(["instance", "lang"])["last"].max()
KIND = {"city": "city", "metro": "metropolis", "regional": "region", "national": "national", "party": "party",
        "community": "community", "eu-consultation": "EU consultation", "us-rulemaking": "US rulemaking"}
lines = [r"\begin{table*}[t]",
         r"\caption{Instances and languages included in the study, with the number of texts (proposals, comments and "
         r"consultation feedback pooled) retained after pre-processing and exact-duplicate removal, before (pre) and "
         r"after (post) 30 November 2022.}",
         r"\label{tab:instances}", r"\centering\small", r"\begin{tabular}{llllrrl}", r"\toprule",
         r"Instance & Country & Lang. & Kind & Pre & Post & Period \\", r"\midrule"]
for (inst, lang), r in t.sort_values("pre", ascending=False).iterrows():
    c = cfg.get(inst, {})
    kind = KIND.get(c.get("kind", ""), c.get("kind", ""))
    lines.append(f"{c.get('name', inst)} & {c.get('country', '')} & {lang} & {kind} & "
                 f"{int(r.get('pre', 0)):,} & {int(r.get('post', 0)):,} & "
                 f"{str(first[(inst, lang)])[:7]}--{str(last[(inst, lang)])[:7]} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
Path("paper/tables").mkdir(parents=True, exist_ok=True)
Path("paper/tables/tab_instances.tex").write_text("\n".join(lines).replace(",", "{,}").replace("{{,}}", ","), encoding="utf-8")
print(st[st.included].groupby(["lang", "period"]).n.sum().unstack())
print("included instance-language pairs:", len(ok))
