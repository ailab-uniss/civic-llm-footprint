"""Package for reproducing the consumer condition by hand: the same persuasive prompts pasted into the chat applications
(ChatGPT, Gemini, optionally Claude) with default settings, by a person with ordinary accounts.

  python scripts/consumer_check_package.py build  -> ../consumer_check/consumer_check.xlsx (prompts + answer sheet)
  python scripts/consumer_check_package.py score  -> extracts the legal references from the pasted answers (L1 regex) and
                                                     writes L1_references for the manual verification, plus a summary
A subset of the 100 manually checked originals is used (4 per language, 20 in all), so that the no-tools and the
API-with-search outputs of the same originals are available for comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "civic"))
from audit import legal_refs  # noqa: E402

OUT = Path("../consumer_check")
APPS = ["ChatGPT", "Gemini"]
PER_LANG = 4


def build():
    OUT.mkdir(exist_ok=True)
    pairs = pd.read_csv("../audit_manual_check/pairs.csv")
    P = pd.read_parquet("data/audit/pairs.parquet", columns=["lang", "mode", "orig", "prompt"])
    s3 = P[P["mode"] == "s3_persuasive"].drop_duplicates("orig").set_index("orig")
    rng = np.random.default_rng(11)
    sel = []
    for lang, g in pairs.groupby("language"):
        pos = g[g.stratum == "s3_pos"]; neg = g[g.stratum != "s3_pos"]
        k = rng.choice(pos.index, min(3, len(pos)), replace=False).tolist() + rng.choice(neg.index, 1, replace=False).tolist()
        sel.append(g.loc[k])
    S = pd.concat(sel)
    rows = []
    for r in S.itertuples():
        prompt = s3.loc[r.original_text, "prompt"] if r.original_text in s3.index else ""
        for app in APPS:
            rows.append({"pair_id": r.pair_id, "language": r.language, "app": app, "prompt_to_paste": prompt,
                         "C1_model_shown_by_app": "", "C2_searched_web (Y/N)": "", "C3_answer_text": "", "C4_sources_shown_by_app": "",
                         "C5_date": "", "C_notes": ""})
    sheet = pd.DataFrame(rows)
    readme = pd.DataFrame({"README": [
        "For each row: open the app in a NEW chat (no memory, no custom instructions, default model, logged in with an ordinary account).",
        "Paste the text of 'prompt_to_paste' exactly, without additions. Wait for the full answer.",
        "C1: the model name the app shows (e.g. 'GPT-5.4', 'Gemini 3.6 Flash'). C2: did the app show that it searched the web (sources, 'Searching...' indicator)?",
        "C3: paste the whole answer. C4: paste the list of sources/links shown by the app, if any. C5: date of the run.",
        "Do not re-run a prompt after seeing the answer; one answer per row. Return the workbook with the same name."]})
    with pd.ExcelWriter(OUT / "consumer_check.xlsx", engine="openpyxl") as w:
        readme.to_excel(w, sheet_name="README", index=False); sheet.to_excel(w, sheet_name="runs", index=False)
        from openpyxl.styles import Alignment
        for name, df in [("README", readme), ("runs", sheet)]:
            ws = w.sheets[name]
            for j, c in enumerate(df.columns, start=1):
                ws.column_dimensions[ws.cell(1, j).column_letter].width = 70 if c in ("prompt_to_paste", "C3_answer_text", "C4_sources_shown_by_app", "README") else min(26, max(12, len(c) + 2))
                for i in range(2, len(df) + 2):
                    ws.cell(i, j).alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "B2"
    print(len(sheet), "runs:", sheet.groupby(["language", "app"]).size().to_dict())


def context(text: str, ref: str, width: int = 160) -> str:
    i = text.lower().find(ref.lower())
    if i < 0:
        return ""
    a, b = max(0, i - width), min(len(text), i + len(ref) + width)
    return ("…" if a > 0 else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")


def score():
    X = pd.ExcelFile(OUT / "consumer_check.xlsx"); runs = X.parse("runs")
    pairs = pd.read_csv("../audit_manual_check/pairs.csv").set_index("pair_id")
    runs = runs[runs.C3_answer_text.notna()].copy()
    runs["orig"] = runs.pair_id.map(pairs.original_text)
    runs["new_refs"] = [[r for r in legal_refs(str(o), l) if r not in set(legal_refs(str(g), l))] for o, g, l in zip(runs.C3_answer_text, runs.orig, runs.language)]
    runs["n_new"] = runs.new_refs.map(len); runs["n_words"] = runs.C3_answer_text.astype(str).str.split().str.len()
    searched = runs[[c for c in runs.columns if c.startswith("C2_")][0]].astype(str).str.strip().str.lower().eq("y")
    print(runs.assign(searched=searched).groupby("app").agg(n=("pair_id", "size"), searched=("searched", "mean"), refs=("n_new", "mean"), share=("n_new", lambda s: (s > 0).mean()), words=("n_words", "mean")).round(2))
    refs = []
    for r in runs.itertuples():
        for ref in r.new_refs:
            refs.append({"pair_id": r.pair_id, "language": r.language, "app": r.app, "reference_detected": ref, "context_in_output": context(str(r.C3_answer_text), ref),
                         "sources_shown_by_app": r.C4_sources_shown_by_app, "A1_present_in_output (Y/N)": "", "A2_identifiable (Y/N/ambiguous)": "",
                         "A3_exists (Y/N/not verifiable)": "", "A4_jurisdiction_compatible (Y/N/not determinable)": "",
                         "A5_pertinent (pertinent/not pertinent/uncertain)": "", "A6_source_checked": "", "A_notes": ""})
    pd.DataFrame(refs).to_csv(OUT / "L1_references_consumer.csv", index=False)
    print("references to verify:", len(refs))


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()
