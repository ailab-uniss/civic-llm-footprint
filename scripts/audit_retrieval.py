"""Retrieval-enabled audit: the S3 'persuasive, with references to the applicable laws' scenario re-run on the 100
manually checked pairs with assistants that can search the web (OpenAI web_search tool; Gemini Google-Search grounding).

Measures whether retrieval removes the invented legal references found in the no-tools audit (L1), and prepares the
same annotation workbook for the co-author (one row per new reference, with the context and the URLs the model cited).

Usage (PC, keys from ../.env):  python scripts/audit_retrieval.py run     -> data/audit/audit_retrieval.parquet
                                python scripts/audit_retrieval.py package -> ../audit_manual_check_retrieval/
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "civic"))
from audit import legal_refs  # noqa: E402

OUT = Path("data/audit/audit_retrieval.parquet")
PKG = Path("../audit_manual_check")
MODELS = {"openai": "gpt-5.4-mini", "gemini": "gemini-3.6-flash"}
# conditions: the search tool is available and the model decides (auto), or a search is forced (forced); Gemini grounding
CONDITIONS = {"openai_auto": ("openai", False), "openai_forced": ("openai", True), "gemini": ("gemini", None)}


def load_keys():
    for line in Path("../.env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())


def jobs() -> pd.DataFrame:
    pairs = pd.read_csv(PKG / "pairs.csv")
    P = pd.read_parquet("data/audit/pairs.parquet", columns=["doc_id", "lang", "mode", "orig", "prompt"])
    s3 = P[P["mode"] == "s3_persuasive"].drop_duplicates("orig").set_index("orig")
    rows = []
    for r in pairs.itertuples():
        m = s3.loc[r.original_text] if r.original_text in s3.index else None
        if m is None:
            print("no S3 prompt for", r.pair_id); continue
        rows.append(dict(pair_id=r.pair_id, lang=r.language, doc_id=m.doc_id, orig=r.original_text, prompt=m.prompt))
    return pd.DataFrame(rows)


def call_openai(client, prompt: str, force: bool = False) -> dict:
    kw = {"tool_choice": {"type": "web_search"}} if force else {}
    r = client.responses.create(model=MODELS["openai"], tools=[{"type": "web_search"}], input=prompt, **kw)
    cites = []
    for item in getattr(r, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            for a in getattr(c, "annotations", []) or []:
                if getattr(a, "type", "") == "url_citation":
                    cites.append({"url": getattr(a, "url", ""), "title": getattr(a, "title", "")})
    searched = any(getattr(item, "type", "") == "web_search_call" for item in getattr(r, "output", []) or [])
    return dict(output=r.output_text or "", citations=json.dumps(cites, ensure_ascii=False), searched=searched, finish="stop")


def call_gemini(client, prompt: str) -> dict:
    from google.genai import types
    r = client.models.generate_content(model=MODELS["gemini"], contents=prompt,
                                       config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]))
    cites, searched = [], False
    try:
        gm = r.candidates[0].grounding_metadata
        searched = bool(gm and (gm.web_search_queries or gm.grounding_chunks))
        for ch in (gm.grounding_chunks or []) if gm else []:
            if ch.web:
                cites.append({"url": ch.web.uri or "", "title": ch.web.title or ""})
    except Exception:  # noqa: BLE001
        pass
    return dict(output=r.text or "", citations=json.dumps(cites, ensure_ascii=False), searched=searched, finish="stop")


def run():
    load_keys()
    from openai import OpenAI
    from google import genai
    oa, ge = OpenAI(), genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    J = jobs()
    done, rows = set(), []
    if OUT.exists():
        prev = pd.read_parquet(OUT); prev = prev[~prev.finish.str.startswith("error")]
        rows = prev.to_dict("records"); done = set(zip(prev.pair_id, prev.provider))
    todo = [(r, prov) for r in J.itertuples() for prov in CONDITIONS if (r.pair_id, prov) not in done]
    print(len(todo), "calls")

    def one(r, prov):
        vendor, force = CONDITIONS[prov]
        for att in range(4):
            try:
                res = call_openai(oa, r.prompt, force) if vendor == "openai" else call_gemini(ge, r.prompt)
                return dict(pair_id=r.pair_id, lang=r.lang, doc_id=r.doc_id, provider=prov, model=MODELS[vendor], prompt=r.prompt, orig=r.orig, **res)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                if "402" in err or "RESOURCE_EXHAUSTED" in err:
                    break
                time.sleep(3 * (att + 1))
        return dict(pair_id=r.pair_id, lang=r.lang, doc_id=r.doc_id, provider=prov, model=MODELS[vendor], prompt=r.prompt, orig=r.orig,
                    output="", citations="[]", searched=False, finish=f"error:{err[:100]}")

    with ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(one, r, prov) for r, prov in todo]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if i % 20 == 0:
                pd.DataFrame(rows).to_parquet(OUT, index=False); print(f"  {i}/{len(todo)}", flush=True)
    D = pd.DataFrame(rows); D.to_parquet(OUT, index=False)
    print(D.groupby("provider").agg(n=("output", "size"), searched=("searched", "mean"), errors=("finish", lambda s: s.str.startswith("error").sum())))


def context(text: str, ref: str, width: int = 160) -> str:
    i = text.lower().find(ref.lower())
    if i < 0:
        return ""
    a, b = max(0, i - width), min(len(text), i + len(ref) + width)
    return ("…" if a > 0 else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")


def package():
    D = pd.read_parquet(OUT); D = D[~D.finish.str.startswith("error")].copy()
    D["refs_out"] = [legal_refs(t, l) for t, l in zip(D.output, D.lang)]
    D["refs_orig"] = [legal_refs(t, l) for t, l in zip(D.orig, D.lang)]
    D["new_refs"] = [[r for r in a if r not in set(b)] for a, b in zip(D.refs_out, D.refs_orig)]
    D["n_words"] = D.output.str.split().str.len()
    out = Path("../audit_manual_check_retrieval"); out.mkdir(exist_ok=True)
    pairs = pd.DataFrame({"pair_id": D.pair_id, "language": D.lang, "provider": D.provider, "model": D.model, "web_search_used": D.searched,
                          "original_text": D.orig, "assistant_output": D.output, "n_words_output": D.n_words,
                          "L1_new_refs_detected": D.new_refs.map(lambda l: " | ".join(l)), "n_new_refs": D.new_refs.map(len),
                          "urls_cited_by_model": D.citations.map(lambda s: " | ".join(dict.fromkeys(c["url"] for c in json.loads(s)))),
                          "A_refs_missed_by_regex": "", "A_regex_false_matches": "", "A_notes": ""}).sort_values(["pair_id", "provider"])
    refs = []
    for r in D.itertuples():
        urls = " | ".join(dict.fromkeys(c["url"] for c in json.loads(r.citations)))
        for ref in r.new_refs:
            refs.append({"pair_id": r.pair_id, "language": r.lang, "provider": r.provider, "reference_detected": ref,
                         "context_in_output": context(r.output, ref), "urls_cited_by_model": urls,
                         "A1_present_in_output (Y/N)": "", "A2_identifiable (Y/N/ambiguous)": "", "A3_exists (Y/N/not verifiable)": "",
                         "A4_jurisdiction_compatible (Y/N/not determinable)": "", "A5_pertinent (pertinent/not pertinent/uncertain)": "",
                         "A6_source_checked": "", "A_notes": ""})
    refs = pd.DataFrame(refs)
    # keep the annotation load bounded: all OpenAI references, a random subset of at most 150 Gemini references
    gem = refs[refs.provider == "gemini"]
    if len(gem) > 150:
        refs = pd.concat([refs[refs.provider != "gemini"], gem.sample(150, random_state=3)]).sort_values(["pair_id", "provider"])
    readme = pd.DataFrame({"README": [
        "Same task as the first check (sheet L1_references, columns A1-A6), on the persuasive scenario re-run with web search enabled.",
        "Sheet 'pairs': 100 originals x 3 conditions (gpt-5.4-mini with the search tool available / forced; gemini-3.6-flash with the Google Search tool).",
        "Sheet 'L1_references': all OpenAI references and a random subset of 150 Gemini references.",
        "'urls_cited_by_model' lists the pages the assistant retrieved: start the verification there, then the official databases.",
        "Fill A_refs_missed_by_regex and A_regex_false_matches on sheet 'pairs' as before. Do not edit the text columns."]})
    with pd.ExcelWriter(out / "audit_check_retrieval.xlsx", engine="openpyxl") as w:
        readme.to_excel(w, sheet_name="README", index=False); pairs.to_excel(w, sheet_name="pairs", index=False); refs.to_excel(w, sheet_name="L1_references", index=False)
        from openpyxl.styles import Alignment
        for name, df in [("README", readme), ("pairs", pairs), ("L1_references", refs)]:
            ws = w.sheets[name]
            for j, c in enumerate(df.columns, start=1):
                ws.column_dimensions[ws.cell(1, j).column_letter].width = 60 if c in ("original_text", "assistant_output", "context_in_output", "README", "urls_cited_by_model") else min(28, max(12, len(c) + 2))
                for i in range(2, len(df) + 2):
                    ws.cell(i, j).alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "B2"
    pairs.to_csv(out / "pairs.csv", index=False); refs.to_csv(out / "L1_references.csv", index=False)
    base = pd.read_csv(PKG / "pairs.csv")
    base_s3 = base[base.scenario == "S3 persuasive"]
    print("no-tools S3 (same originals): new refs per output", round(base_s3.L1_new_refs_detected.fillna("").map(lambda s: len([x for x in s.split(" | ") if x])).mean(), 2))
    print(pairs.groupby("provider").agg(outputs=("pair_id", "size"), searched=("web_search_used", "mean"), refs_per_output=("n_new_refs", "mean"),
                                        share_with_ref=("n_new_refs", lambda s: (s > 0).mean()), words=("n_words_output", "mean")).round(2))
    print("references to check:", len(refs))


if __name__ == "__main__":
    {"run": run, "package": package}[sys.argv[1]]()
