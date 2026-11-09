"""Build the manual-annotation package for the audit (L1 legal references, L5 claim fidelity) to hand to a co-author.

Draws a stratified sample of (original, output) pairs from the audit already run (data/audit/pairs.parquet +
results/audit_metrics.parquet), re-extracts the legal references that the L1 regex found, and writes an Excel workbook
with one sheet per annotation task plus a README. No new text is generated.

Strata per language (it, es, ca, fr, en): S3 persuasive with L1 positive (12), S3 with L1 negative (4), S1 improve (2),
S2 notes (2) -> 20 pairs per language, 100 in total, balanced over generators. Half of them (10 per language, spread over
the L5 coverage range) also get the L5 claim-fidelity annotation.

Usage: python scripts/manual_check_package.py --out ../audit_manual_check
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "civic"))
from audit import legal_refs, sentences  # noqa: E402

LANGS = ["it", "es", "ca", "fr", "en"]
STRATA = {"s3_pos": 12, "s3_neg": 4, "s1": 2, "s2": 2}
SCEN = {"s1_improve": "S1 improve", "s2_notes": "S2 notes", "s3_persuasive": "S3 persuasive"}


def context(text: str, ref: str, width: int = 160) -> str:
    i = text.lower().find(ref.lower())
    if i < 0:
        return ""
    a, b = max(0, i - width), min(len(text), i + len(ref) + width)
    return ("…" if a > 0 else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")


def balanced_sample(g: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n rows spread as evenly as possible over generators."""
    g = g.sample(frac=1, random_state=int(rng.integers(1e9)))
    out, models = [], list(g.model.unique())
    rng.shuffle(models)
    per = {m: g[g.model == m] for m in models}
    k = 0
    while len(out) < n and any(len(v) for v in per.values()):
        m = models[k % len(models)]
        if len(per[m]):
            out.append(per[m].iloc[0]); per[m] = per[m].iloc[1:]
        k += 1
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/audit/pairs.parquet")
    ap.add_argument("--metrics", default="results/audit_metrics.parquet")
    ap.add_argument("--out", default="../audit_manual_check")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(a.pairs); M = pd.read_parquet(a.metrics)
    D = P.merge(M, on=["doc_id", "lang", "model", "mode"])
    D = D[D.lang.isin(LANGS) & D.output.str.split().str.len().ge(30)]
    rows = []
    for lang in LANGS:
        g = D[D.lang == lang]
        parts = {"s3_pos": g[(g["mode"] == "s3_persuasive") & (g.l1_any_new == 1)],
                 "s3_neg": g[(g["mode"] == "s3_persuasive") & (g.l1_any_new == 0)],
                 "s1": g[g["mode"] == "s1_improve"], "s2": g[g["mode"] == "s2_notes"]}
        for k, n in STRATA.items():
            s = balanced_sample(parts[k], n, rng); s["stratum"] = k; rows.append(s)
    S = pd.concat(rows, ignore_index=True)
    S["pair_id"] = [f"P{i + 1:03d}" for i in range(len(S))]
    S["scenario"] = S["mode"].map(SCEN)
    S["refs_in_output"] = [legal_refs(t, l) for t, l in zip(S.output, S.lang)]
    S["refs_in_original"] = [legal_refs(t, l) for t, l in zip(S.orig, S.lang)]
    S["new_refs"] = [[r for r in ro if r not in set(rg)] for ro, rg in zip(S.refs_in_output, S.refs_in_original)]
    # L5 subsample: 10 per language, spread over the automatic coverage range (low, middle, high)
    S["l5_subsample"] = ""
    for lang in LANGS:
        g = S[S.lang == lang].sort_values("l5_claim_coverage")
        idx = np.linspace(0, len(g) - 1, 10).round().astype(int)
        S.loc[g.index[idx], "l5_subsample"] = "yes"

    pairs = pd.DataFrame({
        "pair_id": S.pair_id, "language": S.lang, "scenario": S.scenario, "generator": S.model, "stratum": S.stratum,
        "original_text": S.orig, "assistant_output": S.output,
        "L1_refs_detected_in_output": S.refs_in_output.map(lambda l: " | ".join(l)),
        "L1_refs_detected_in_original": S.refs_in_original.map(lambda l: " | ".join(l)),
        "L1_new_refs_detected": S.new_refs.map(lambda l: " | ".join(l)),
        "L1_any_new_automatic": S.l1_any_new,
        "L5_claim_coverage_automatic": S.l5_claim_coverage.round(2), "L5_output_grounded_automatic": S.l5_output_grounded.round(2),
        "L5_subsample": S.l5_subsample,
        # annotation columns (all pairs): references the regex missed, and references it matched wrongly
        "A_refs_missed_by_regex": "", "A_regex_false_matches": "", "A_notes": ""})
    refs = []
    for r in S.itertuples():
        for ref in r.new_refs:
            refs.append({"pair_id": r.pair_id, "language": r.lang, "scenario": r.scenario, "generator": r.model,
                         "reference_detected": ref, "context_in_output": context(r.output, ref),
                         "A1_present_in_output (Y/N)": "", "A2_identifiable (Y/N/ambiguous)": "",
                         "A3_exists (Y/N/not verifiable)": "", "A4_jurisdiction_compatible (Y/N/not determinable)": "",
                         "A5_pertinent (pertinent/not pertinent/uncertain)": "", "A6_source_checked": "", "A_notes": ""})
    refs = pd.DataFrame(refs)
    l5 = S[S.l5_subsample == "yes"]
    claims = pd.DataFrame({
        "pair_id": l5.pair_id, "language": l5.lang, "scenario": l5.scenario, "generator": l5.model,
        "original_text": l5.orig, "assistant_output": l5.output,
        "L5_claim_coverage_automatic": l5.l5_claim_coverage.round(2), "L5_output_grounded_automatic": l5.l5_output_grounded.round(2),
        "B1_claim_1": "", "B1_status_1 (kept/omitted/changed/undeterminable)": "",
        "B2_claim_2": "", "B2_status_2 (kept/omitted/changed/undeterminable)": "",
        "B3_claim_3": "", "B3_status_3 (kept/omitted/changed/undeterminable)": "",
        "B4_new_claims_in_output (none/minor/substantial)": "", "B5_overall_intent_preserved (Y/partly/N)": "", "B_notes": ""})

    readme = pd.DataFrame({"README": [
        "Manual check of the automatic audit metrics L1 (legal references) and L5 (claim fidelity). See README.md for the full protocol.",
        "Sheet 'pairs': the 100 sampled (original, output) pairs with the automatic values. Fill A_refs_missed_by_regex and A_regex_false_matches for every pair.",
        "Sheet 'L1_references': one row per legal reference that the regex found in the output and not in the original. Fill A1-A6.",
        "Sheet 'L5_claims': 50 pairs. Identify up to three main claims of the original (B1-B3), their status in the output, new claims (B4), overall intent (B5).",
        "Use fixed vocabularies: Y/N/ambiguous etc. as written in the column headers. Leave a cell empty only if the item does not apply.",
        "Do not edit the text columns. Return the workbook with the same file name."]})
    xlsx = out / "audit_manual_check.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        readme.to_excel(w, sheet_name="README", index=False)
        pairs.to_excel(w, sheet_name="pairs", index=False)
        refs.to_excel(w, sheet_name="L1_references", index=False)
        claims.to_excel(w, sheet_name="L5_claims", index=False)
        from openpyxl.styles import Alignment
        for name, df in [("README", readme), ("pairs", pairs), ("L1_references", refs), ("L5_claims", claims)]:
            ws = w.sheets[name]
            for j, col in enumerate(df.columns, start=1):
                long = col in ("original_text", "assistant_output", "context_in_output", "README")
                ws.column_dimensions[ws.cell(1, j).column_letter].width = 60 if long else min(28, max(12, len(col) + 2))
                for i in range(2, len(df) + 2):
                    ws.cell(i, j).alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "B2"
    pairs.to_csv(out / "pairs.csv", index=False); refs.to_csv(out / "L1_references.csv", index=False); claims.to_csv(out / "L5_claims.csv", index=False)
    print("pairs:", len(pairs), "| L1 references to check:", len(refs), "| L5 pairs:", len(claims))
    print(pairs.groupby(["language", "stratum"]).size().unstack().to_string())
    print(pairs.groupby("generator").size().to_dict())


if __name__ == "__main__":
    main()
