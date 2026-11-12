"""Seeds for the instance-specific fill of GPT-4o and GPT-4o mini (11 Nov 2026): for every instance-language pair of the
study (EU Have Your Say split by respondent type, as in analysis.py) in which a model has fewer than TARGET generated texts
(rewrite + generate) seeded from the pair, pick up to N_SEEDS further seeds of the pair from the human sample.

  python scripts/build_fill_sample.py   -> data/generated/fill_inst_<model>.parquet (same columns as human_sample.parquet)
"""
import glob
import sys

import pandas as pd

sys.path.insert(0, "src/civic")
from analysis import split_author_type  # noqa: E402

TARGET, N_SEEDS = 70, 40
H = pd.read_parquet("data/generated/human_sample.parquet")
ut = pd.concat([pd.read_parquet(p, columns=["doc_id", "user_type"]) for p in glob.glob("data/processed/eu_hys*.parquet")]).drop_duplicates("doc_id")
ut = ut.set_index("doc_id").user_type


def study_instance(doc_ids: pd.Series, instance: pd.Series) -> pd.Series:
    return split_author_type(instance.reset_index(drop=True), doc_ids.map(ut).reset_index(drop=True)).set_axis(doc_ids.index)


H["sinst"] = study_instance(H.doc_id, H.instance)
G = pd.read_parquet("data/generated/generated_processed.parquet", columns=["doc_id", "model", "mode", "lang"])
G = G[G["mode"].isin(["rewrite", "generate"])]
inst_of = H.drop_duplicates("doc_id").set_index("doc_id").sinst
G["sinst"] = G.doc_id.map(inst_of)
study = pd.read_csv("results/analysis_cls5/alpha_cells_period.csv")
pairs = set(zip(study.instance, study.lang))
for model in (sys.argv[1:] or ["gpt-4o", "gpt-4o-mini"]):
    g = G[G.model == model]
    have = g.groupby(["sinst", "lang"]).size()
    used = set(g.doc_id)
    rows, rep = [], []
    for (inst, lang), h in H.groupby(["sinst", "lang"]):
        if (inst, lang) not in pairs or inst.endswith("_unknown"):
            continue
        n = int(have.get((inst, lang), 0))
        if n >= TARGET:
            continue
        cand = h[~h.doc_id.isin(used)]
        pick = cand.sample(min(len(cand), N_SEEDS), random_state=0)
        rows.append(pick)
        rep.append((inst, lang, n, len(pick)))
    out = pd.concat(rows).drop(columns="sinst")
    out.to_parquet(f"data/generated/fill_inst_{model}.parquet", index=False)
    print(model, "pairs filled:", len(rep), "seeds:", len(out), "calls:", 2 * len(out), "mean words:", round(out.n_words.mean()))
    print(pd.DataFrame(rep, columns=["instance", "lang", "have", "new_seeds"]).to_string())
