"""Monitoring tool for administrations: estimate the LLM share in any Decidim instance from its open-data export.

  python tool.py analyze --url https://<instance> --refs models/refs --out report_dir
  python tool.py analyze --zip export.zip --instance myinst --refs models/refs --out report_dir

Produces alpha estimates (with bootstrap CIs) per language x half-year x type, a campaign summary and an HTML report.
Reference distributions (one JSON per language) are produced by `build_refs` from the project data.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator import DistributionalEstimator  # noqa: E402
from ingest import ingest_zip  # noqa: E402
from preprocess import run as preprocess_run  # noqa: E402
from analysis import half_year, to_sets  # noqa: E402


def download(url: str, dest: Path):
    r = requests.get(url.rstrip("/") + "/open-data/download", timeout=900, verify=False, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)


def build_refs(processed_glob: str, gen_path: str, out_dir: Path, selection="ratio", top_k=1000, min_df=0.003,
               likelihood="occurrence"):
    import glob
    H = pd.concat([pd.read_parquet(p) for p in glob.glob(processed_glob)], ignore_index=True)
    G = pd.read_parquet(gen_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in sorted(set(H.lang) & set(G.lang)):
        h = H[(H.lang == lang) & (H.period == "pre") & (~H.is_dup_exact)]
        g = G[G.lang == lang]
        by_model = {m: to_sets(d.tokens) for m, d in g.groupby("model")}
        by_model = {m: d for m, d in by_model.items() if len(d) >= 200}
        if len(h) < 300 or not by_model:
            continue
        est = DistributionalEstimator(selection=selection, top_k=top_k, min_df=min_df, likelihood=likelihood).fit_reference(to_sets(h.tokens), by_model)
        est.save(out_dir / f"ref_{lang}.json")
        print(lang, "vocab", len(est.ref.vocab), "n_h", est.ref.n_h, "n_m", est.ref.n_m)


def analyze(df: pd.DataFrame, refs_dir: Path, min_docs: int = 150, n_boot: int = 300) -> pd.DataFrame:
    rows = []
    df = df[~df.is_dup_exact].copy()
    df["half"] = half_year(df.created_at)
    for lang, g in df.groupby("lang"):
        p = refs_dir / f"ref_{lang}.json"
        if not p.exists():
            continue
        est = DistributionalEstimator.load(p)
        for (half, typ), gg in g.groupby(["half", "type"]):
            if len(gg) < min_docs:
                continue
            r = est.estimate(to_sets(gg.tokens), n_boot=n_boot)
            rows.append(dict(lang=lang, half=half, type=typ, n_docs=len(gg), alpha=r["alpha"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
        if len(g) >= min_docs:
            for period, gg in g.groupby("period"):
                if len(gg) >= min_docs:
                    r = est.estimate(to_sets(gg.tokens), n_boot=n_boot)
                    rows.append(dict(lang=lang, half=period, type="all", n_docs=len(gg), alpha=r["alpha"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
    return pd.DataFrame(rows)


def report(res: pd.DataFrame, instance: str, out: Path):
    rows = "".join(
        f"<tr><td>{html.escape(str(r.lang))}</td><td>{html.escape(str(r.half))}</td><td>{html.escape(str(r.type))}</td>"
        f"<td>{r.n_docs}</td><td>{100 * r.alpha:.1f}%</td><td>{100 * r.ci_lo:.1f}–{100 * r.ci_hi:.1f}%</td></tr>"
        for r in res.itertuples())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>LLM footprint report – {html.escape(instance)}</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto;color:#0b0b0b}}table{{border-collapse:collapse}}td,th{{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}}
.note{{color:#52514e;font-size:0.9em}}</style></head><body>
<h1>Estimated share of LLM-influenced contributions</h1><p><b>Instance:</b> {html.escape(instance)}</p>
<p class="note">Estimates are population-level (corpus) shares with 95% bootstrap confidence intervals. They must never be used to judge individual contributions or authors. Cells with fewer than 150 texts are not reported.</p>
<table><thead><tr><th>Language</th><th>Period</th><th>Type</th><th>Texts</th><th>LLM share</th><th>95% CI</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    out.write_text(doc, encoding="utf-8")


def main():
    import urllib3
    urllib3.disable_warnings()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-refs")
    b.add_argument("--processed", default="data/processed/*.parquet")
    b.add_argument("--gen", default="data/generated/generated_processed.parquet")
    b.add_argument("--out", default="models/refs")
    a_ = sub.add_parser("analyze")
    a_.add_argument("--url")
    a_.add_argument("--zip")
    a_.add_argument("--instance", default=None)
    a_.add_argument("--refs", default="models/refs")
    a_.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build-refs":
        build_refs(a.processed, a.gen, Path(a.out))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inst = a.instance or (a.url.split("//")[-1].split("/")[0] if a.url else Path(a.zip).stem)
    zip_path = Path(a.zip) if a.zip else out / "export.zip"
    if a.url and not a.zip:
        download(a.url, zip_path)
    df = ingest_zip(zip_path, inst)
    df.to_parquet(out / "interim.parquet", index=False)
    preprocess_run(out / "interim.parquet", out / "processed.parquet", instances_cfg=None)
    P = pd.read_parquet(out / "processed.parquet")
    res = analyze(P, Path(a.refs))
    res.to_csv(out / "alpha.csv", index=False)
    report(res, inst, out / "report.html")
    print(res.to_string())


if __name__ == "__main__":
    main()
