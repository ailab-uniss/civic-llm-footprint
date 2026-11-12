"""Referee remark R18: does the L2 loss under S2 (writing from notes) measure the note extractor or the assistant?

The notes of S2 are produced deterministically by generate.make_notes (no model): the first clause of the first five
sentences, at most 14 words each. This script regenerates them, checks them against the prompts actually sent, and
computes, with the entity and number extraction of audit.py (spaCy NER labels LOC/GPE/FAC/ORG/PER/MISC on text[:5000],
lower-cased strings; NUM_RE):
  (1) the share of the original's entities / numbers that are present in the notes;
  (2) the retention in the S2 output conditional on presence / absence in the notes;
  (3) the retention of the same two subsets of items under S1 and S3 (the model sees the full text), as a benchmark;
  (4) a validation of the local spaCy pipelines against the stored l2_ents_kept / l2_nums_kept of audit_metrics.parquet.
Read-only on existing files; writes results/review_audit/r18_*.csv. No API call.

  <venv with spaCy 3.8 + *_core_*_md 3.8.0>/python scripts/review_audit_r18_notes.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "civic"))
from audit import NUM_RE  # noqa: E402

OUT = ROOT / "results" / "review_audit"
OUT.mkdir(parents=True, exist_ok=True)
SPACY_MODELS = {"ca": "ca_core_news_md", "es": "es_core_news_md", "fr": "fr_core_news_md", "it": "it_core_news_md",
                "fi": "fi_core_news_md", "en": "en_core_web_md"}   # as in preprocess.SPACY_MODELS
LABELS = ("LOC", "GPE", "FAC", "ORG", "PER", "MISC")               # as in audit.NER.ents


def make_notes(text: str, max_points: int = 5, max_words: int = 14) -> str:
    """Verbatim copy of generate.make_notes."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 4]
    pts = []
    for s in sents[:max_points]:
        w = re.split(r"[,;:]", s)[0].split()
        pts.append("- " + " ".join(w[:max_words]))
    return "\n".join(pts) if pts else "- " + " ".join(text.split()[:max_words])


def ner_sets(texts: list[str], langs: list[str]) -> list[set]:
    import spacy
    res: list = [None] * len(texts)
    langs = np.asarray(langs)
    for lang in sorted(set(langs)):
        nlp = spacy.load(SPACY_MODELS[lang], disable=["lemmatizer"])
        idx = np.where(langs == lang)[0]
        docs = nlp.pipe((texts[i][:5000] for i in idx), batch_size=64)
        for i, doc in zip(idx, docs):
            res[i] = {e.text.lower().strip() for e in doc.ents if e.label_ in LABELS}
        print("  NER", lang, len(idx), flush=True)
    return res


def load_pairs() -> pd.DataFrame:
    S = pd.read_parquet(ROOT / "data/audit/audit_sample.parquet")[["doc_id", "text", "lang"]].rename(columns={"text": "orig"})
    P = pd.read_parquet(ROOT / "data/audit/pairs.parquet", columns=["model", "lang", "mode", "doc_id", "prompt", "output", "orig"])
    # the local pairs.parquet predates the Gemini run: rebuild the Gemini pairs as run_audit.sh does
    if "gemini-3.6-flash" not in set(P.model):
        G = pd.read_parquet(ROOT / "data/audit/audit_gemini-3.6-flash.parquet", columns=["model", "lang", "mode", "doc_id", "prompt", "output"])
        G = G.merge(S[["doc_id", "orig"]], on="doc_id")
        G = G[G.output.fillna("").str.split().str.len() >= 10]
        P = pd.concat([P, G], ignore_index=True)
    return P.reset_index(drop=True), S


def main():
    P, S = load_pairs()
    M = pd.read_parquet(ROOT / "results/audit_metrics.parquet")
    print("pairs", len(P), "metrics", len(M))
    S["notes"] = S.orig.map(make_notes)
    # check: the regenerated notes are the ones in the S2 prompts actually sent
    s2 = P[P["mode"] == "s2_notes"].merge(S[["doc_id", "notes"]], on="doc_id")
    in_prompt = np.array([n in p for n, p in zip(s2.notes, s2.prompt)])
    print("S2 prompts containing the regenerated notes verbatim: %d / %d" % (in_prompt.sum(), len(s2)))

    S["E_o"] = ner_sets(S.orig.tolist(), S.lang.tolist())
    S["E_n"] = ner_sets(S.notes.tolist(), S.lang.tolist())
    S["N_o"] = S.orig.map(lambda t: set(NUM_RE.findall(t)))
    S["N_n"] = S.notes.map(lambda t: set(NUM_RE.findall(t)))
    S["notes_l"] = S.notes.str.lower()
    S["E_o_in_ner"] = [o & n for o, n in zip(S.E_o, S.E_n)]                       # entity also extracted from the notes
    S["E_o_in_str"] = [{e for e in o if e in t} for o, t in zip(S.E_o, S.notes_l)]  # entity string occurs in the notes
    S["N_o_in"] = [o & n for o, n in zip(S.N_o, S.N_n)]
    S["w_orig"] = S.orig.str.split().str.len(); S["w_notes"] = S.notes.str.split().str.len()
    S["n_sent"] = S.orig.map(lambda t: len([s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if len(s.split()) >= 4]))

    # ---- (1) presence in the notes, per document (each document has one set of notes, shared by all generators)
    def cov(g: pd.DataFrame) -> pd.Series:
        ge, gn = g[g.E_o.map(len) > 0], g[g.N_o.map(len) > 0]
        return pd.Series({
            "n_docs": len(g), "mean_words_orig": g.w_orig.mean(), "mean_words_notes": g.w_notes.mean(),
            "share_docs_more_than_5_sentences": (g.n_sent > 5).mean(),
            "n_docs_with_entity": len(ge), "n_entities_orig": int(ge.E_o.map(len).sum()),
            "ent_in_notes_string_micro": ge.E_o_in_str.map(len).sum() / ge.E_o.map(len).sum(),
            "ent_in_notes_string_macro": (ge.E_o_in_str.map(len) / ge.E_o.map(len)).mean(),
            "ent_in_notes_ner_micro": ge.E_o_in_ner.map(len).sum() / ge.E_o.map(len).sum(),
            "ent_in_notes_ner_macro": (ge.E_o_in_ner.map(len) / ge.E_o.map(len)).mean(),
            "share_docs_no_entity_in_notes_string": (ge.E_o_in_str.map(len) == 0).mean(),
            "n_docs_with_number": len(gn), "n_numbers_orig": int(gn.N_o.map(len).sum()),
            "num_in_notes_micro": gn.N_o_in.map(len).sum() / gn.N_o.map(len).sum(),
            "num_in_notes_macro": (gn.N_o_in.map(len) / gn.N_o.map(len)).mean(),
            "share_docs_no_number_in_notes": (gn.N_o_in.map(len) == 0).mean()})
    C = pd.concat([S.groupby("lang").apply(cov), cov(S).to_frame("all").T])
    C.index.name = "lang"
    C.round(4).to_csv(OUT / "r18_notes_coverage.csv")
    print(C.round(3).T.to_string())

    # ---- NER on all outputs (as audit.py), then validation against the stored metrics
    P["E_out"] = ner_sets(P.output.tolist(), P.lang.tolist())
    P["N_out"] = P.output.map(lambda t: set(NUM_RE.findall(t)))
    P["out_l"] = P.output.str.lower()
    D = P.merge(S[["doc_id", "E_o", "N_o", "E_o_in_ner", "E_o_in_str", "N_o_in"]], on="doc_id")
    D["ents_kept_local"] = [len(o & e) / len(o) if o else np.nan for o, e in zip(D.E_o, D.E_out)]
    D["nums_kept_local"] = [len(o & e) / len(o) if o else np.nan for o, e in zip(D.N_o, D.N_out)]
    V = D.merge(M[["doc_id", "lang", "model", "mode", "l2_ents_kept", "l2_nums_kept", "l2_ents_orig"]], on=["doc_id", "lang", "model", "mode"], how="inner")
    val = []
    for mode, g in list(V.groupby("mode")) + [("all", V)]:
        a, b = g.l2_ents_kept, g.ents_kept_local
        ok = a.notna() & b.notna()
        val.append({"mode": mode, "n_pairs_matched": len(g), "stored_l2_ents_kept": a.mean(), "local_l2_ents_kept": b.mean(),
                    "share_pairs_identical_ents": float((np.abs(a[ok] - b[ok]) < 1e-9).mean()),
                    "nan_pattern_identical": float((a.isna() == b.isna()).mean()),
                    "stored_l2_nums_kept": g.l2_nums_kept.mean(), "local_l2_nums_kept": g.nums_kept_local.mean(),
                    "share_pairs_identical_nums": float((np.abs(g.l2_nums_kept - g.nums_kept_local).dropna() < 1e-9).mean())})
    val = pd.DataFrame(val); val.round(4).to_csv(OUT / "r18_validation_vs_stored_metrics.csv", index=False)
    print(val.round(4).to_string())

    # ---- (2)/(3) retention conditional on presence in the notes, per scenario
    def kept(items, out_set, out_l, how):
        return {e for e in items if (e in out_set if how == "ner" else e in out_l)}
    rows = []
    for r in D.itertuples():
        for notes_def, inn in (("string", r.E_o_in_str), ("ner", r.E_o_in_ner)):
            out_ = r.E_o - inn
            rows.append(("entities", notes_def, r.lang, r.model, r.mode, len(r.E_o), len(inn), len(out_),
                         len(kept(inn, r.E_out, r.out_l, "ner")), len(kept(out_, r.E_out, r.out_l, "ner")),
                         len(kept(inn, r.E_out, r.out_l, "str")), len(kept(out_, r.E_out, r.out_l, "str"))))
        nin, nout = r.N_o_in, r.N_o - r.N_o_in
        rows.append(("numbers", "regex", r.lang, r.model, r.mode, len(r.N_o), len(nin), len(nout),
                     len(nin & r.N_out), len(nout & r.N_out), len(nin & r.N_out), len(nout & r.N_out)))
    T = pd.DataFrame(rows, columns=["item", "notes_presence_def", "lang", "model", "mode", "n_orig", "n_in_notes", "n_not_in_notes",
                                    "kept_in_notes", "kept_not_in_notes", "kept_in_notes_strmatch", "kept_not_in_notes_strmatch"])

    def agg(g: pd.DataFrame) -> pd.Series:
        g = g[g.n_orig > 0]
        gi, go = g[g.n_in_notes > 0], g[g.n_not_in_notes > 0]
        return pd.Series({
            "n_pairs": len(g), "n_items_orig": int(g.n_orig.sum()), "n_items_in_notes": int(g.n_in_notes.sum()),
            "share_items_in_notes": g.n_in_notes.sum() / g.n_orig.sum(),
            "retention_all_macro(audit.py)": ((g.kept_in_notes + g.kept_not_in_notes) / g.n_orig).mean(),
            "retention_all_micro": (g.kept_in_notes + g.kept_not_in_notes).sum() / g.n_orig.sum(),
            "retention_if_in_notes_micro": gi.kept_in_notes.sum() / gi.n_in_notes.sum(),
            "retention_if_in_notes_macro": (gi.kept_in_notes / gi.n_in_notes).mean(),
            "retention_if_not_in_notes_micro": go.kept_not_in_notes.sum() / go.n_not_in_notes.sum(),
            "retention_if_not_in_notes_macro": (go.kept_not_in_notes / go.n_not_in_notes).mean(),
            "retention_if_in_notes_micro_strmatch": gi.kept_in_notes_strmatch.sum() / gi.n_in_notes.sum(),
            "retention_if_not_in_notes_micro_strmatch": go.kept_not_in_notes_strmatch.sum() / go.n_not_in_notes.sum(),
            "upper_bound_if_all_in_notes_kept_micro": g.n_in_notes.sum() / g.n_orig.sum()})
    keys = ["item", "notes_presence_def", "mode"]
    A = pd.concat([T.groupby(keys).apply(agg).reset_index().assign(lang="all"),
                   T.groupby(keys + ["lang"]).apply(agg).reset_index()], ignore_index=True)
    A = A[keys + ["lang"] + [c for c in A.columns if c not in keys + ["lang"]]]
    A.round(4).to_csv(OUT / "r18_conditional_retention.csv", index=False)
    G = T[T["mode"] == "s2_notes"].groupby(["item", "notes_presence_def", "model"]).apply(agg).reset_index()
    G.round(4).to_csv(OUT / "r18_conditional_retention_s2_by_generator.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    show = ["item", "notes_presence_def", "mode", "lang", "n_pairs", "share_items_in_notes", "retention_all_macro(audit.py)", "retention_all_micro",
            "retention_if_in_notes_micro", "retention_if_not_in_notes_micro", "retention_if_in_notes_micro_strmatch"]
    print(A[show].round(3).to_string())
    print(G[["item", "notes_presence_def", "model", "n_pairs", "retention_all_macro(audit.py)", "retention_if_in_notes_micro", "retention_if_not_in_notes_micro"]].round(3).to_string())


if __name__ == "__main__":
    main()
