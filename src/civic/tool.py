"""Monitoring tool for administrations: estimate the share of LLM-written contributions in a Decidim instance from its
open-data export, with the classifier estimator of the paper.

  python tool.py analyze --url https://<instance> --out report_dir
  python tool.py analyze --zip export.zip --instance myinst --out report_dir

The texts of the export are scored by the released classifier (models/xlmr_civic). The share of each slice (language x
year, and language x period x contribution type) is the mixing proportion between the scores of the instance's own texts
written before 30 November 2022 and those of generated texts (models/refs/generated_scores.parquet, cross-fitted scores of
the reference texts of the paper), with the vendor weights of configs/reference_weights_vendor.yaml. An instance with fewer
than 300 earlier texts in a language is calibrated on the human scores of the language (models/refs/human_scores.parquet),
which is less accurate and is marked in the report. Every estimate carries the interval calibrated on the forward-validation
benchmark (models/refs/calibration_coefs.csv) and the generator-free lower bound.

`build-refs` produces the three reference files from the project data (study scores, processed and generated texts).
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import ingest_zip  # noqa: E402
from preprocess import run as preprocess_run  # noqa: E402
from score_estimator import MIN_G, REGION, HDy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MIN_REF, MIN_DOCS = 300, 150


def download(url: str, dest: Path):
    r = requests.get(url.rstrip("/") + "/open-data/download", timeout=900, verify=False, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)


def build_refs(scores: str, processed_glob: str, gen_path: str, coefs: str, out_dir: Path, cap: int = 20000):
    """Reference files of the tool: scores of the generated texts (rewrite and generate modes) and of a sample of the
    pre-ChatGPT human texts of each language, both cross-fitted, and the coefficients of the calibrated interval."""
    import glob
    S = pd.read_parquet(scores)
    H = pd.concat([pd.read_parquet(p, columns=["doc_id", "lang", "period", "is_dup_exact", "created_at"]) for p in glob.glob(processed_glob)])
    H = H[(H.period == "pre") & ~H.is_dup_exact].drop_duplicates("doc_id")
    h = H.merge(S[S.set == "civic_h"][["doc_id", "score"]].drop_duplicates("doc_id"), on="doc_id")
    h["year"] = h.created_at.dt.year
    h = h.sample(frac=1, random_state=0).groupby("lang").head(cap)
    g = S[(S.set == "civic_g") & S["mode"].isin(["rewrite", "generate"])][["lang", "model", "mode", "score"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    h[["lang", "year", "score"]].to_parquet(out_dir / "human_scores.parquet", index=False)
    g.to_parquet(out_dir / "generated_scores.parquet", index=False)
    pd.read_csv(coefs).to_csv(out_dir / "calibration_coefs.csv", index=False)
    print(h.groupby("lang").size().to_string(), g.groupby(["lang", "model"]).size().unstack().to_string(), sep="\n")


def score_texts(texts: list[str], model_dir: str, bs: int = 64, max_len: int = 256) -> np.ndarray:
    """Classifier score (difference of the two logits) of every text; GPU when available."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.empty(len(texts))
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            lg = model(**enc).logits.float()
            out[idx] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
    return out


def band(a: float, coef: pd.Series) -> tuple[float, float]:
    lo = a - max(coef.hi0 + coef.hi1 * a, 0)
    hi = a - min(coef.lo0 + coef.lo1 * a, 0)
    return float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1))


def analyze(df: pd.DataFrame, refs_dir: Path, weights: dict, min_docs: int = MIN_DOCS, n_boot: int = 300) -> pd.DataFrame:
    Hs = pd.read_parquet(refs_dir / "human_scores.parquet")
    Gs = pd.read_parquet(refs_dir / "generated_scores.parquet")
    coefs = pd.read_csv(refs_dir / "calibration_coefs.csv").set_index("calibration")
    df = df[~df.is_dup_exact].copy()
    df["year"] = df.created_at.dt.year
    rows = []
    for lang, g in df.groupby("lang"):
        gl = Gs[Gs.lang == lang]
        by = {m: x.score.values for m, x in gl.groupby("model") if len(x) >= MIN_G}
        if not by:
            continue
        pre = g[g.period == "pre"]
        own = len(pre) >= MIN_REF
        if own:
            sh, groups = pre.score.values, pre.year.values
        else:
            hl = Hs[Hs.lang == lang]
            sh, groups = hl.score.values, hl.year.values
        est = HDy(sh, by, {m: w for m, w in weights.items() if m in by}, groups=groups, region=[m for m in REGION if m in by])
        coef = coefs.loc["own" if own else "pooled"]
        cells = [(str(y), "all", gg) for y, gg in g.groupby("year")] + [(per, typ, gg) for (per, typ), gg in g.groupby(["period", "type"])]
        for label, typ, gg in cells:
            if len(gg) < min_docs:
                continue
            r = est.estimate(gg.score.values, n_boot=n_boot)
            lo, hi = band(r["alpha"], coef)
            if r.get("ci_lo") is not None and r.get("ci_hi") is not None:  # union with the bootstrap interval (small slices)
                lo, hi = min(lo, float(r["ci_lo"])), max(hi, float(r["ci_hi"]))
            rows.append(dict(lang=lang, slice=label, type=typ, n_docs=len(gg), alpha=r["alpha"], lo=lo, hi=hi,
                             lower_bound=r["lb_lo"] if own else np.nan,  # the bound needs the human scores of the instance itself
                             own_calibration=own, n_calibration=len(sh)))
    return pd.DataFrame(rows)


def report(res: pd.DataFrame, instance: str, out: Path):
    rows = "".join(
        f"<tr><td>{html.escape(str(r.lang))}</td><td>{html.escape(str(r.slice))}</td><td>{html.escape(str(r.type))}</td>"
        f"<td>{r.n_docs}</td><td>{100 * r.alpha:.1f}%</td><td>{100 * r.lo:.1f}–{100 * r.hi:.1f}%</td><td>{f'{100 * r.lower_bound:.1f}%' if pd.notna(r.lower_bound) else '–'}</td>"
        f"<td>{'own texts' if r.own_calibration else 'language pool'}</td></tr>"
        for r in res.itertuples())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>LLM footprint report – {html.escape(instance)}</title>
<style>body{{font-family:sans-serif;max-width:960px;margin:2em auto;color:#0b0b0b}}table{{border-collapse:collapse}}td,th{{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}}
.note{{color:#52514e;font-size:0.9em}}</style></head><body>
<h1>Estimated share of contributions written or rewritten with an LLM</h1><p><b>Instance:</b> {html.escape(instance)}</p>
<p class="note">Estimates are shares of a corpus slice and must never be used to judge individual contributions or authors. The interval is the
95% error band measured in forward validation with an unseen generator; the lower bound does not depend on any generator. A lightly edited text
counts for about half a text. Slices with fewer than {MIN_DOCS} texts are not reported; slices before December 2022 show the false-positive floor
of the instance. Calibration on the language pool (fewer than {MIN_REF} earlier texts of the instance) is less accurate, and the lower bound is not computed for it.</p>
<table><thead><tr><th>Language</th><th>Slice</th><th>Type</th><th>Texts</th><th>Share</th><th>95% interval</th><th>Lower bound</th><th>Calibration</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    out.write_text(doc, encoding="utf-8")


def main():
    import urllib3
    urllib3.disable_warnings()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-refs")
    b.add_argument("--scores", default="results/supervised/study_scores_aug.parquet")
    b.add_argument("--processed", default="data/processed/*.parquet")
    b.add_argument("--gen", default="data/generated/generated_processed.parquet")
    b.add_argument("--coefs", default="results/supervised/calibration_coefs.csv")
    b.add_argument("--out", default="models/refs")
    a_ = sub.add_parser("analyze")
    a_.add_argument("--url")
    a_.add_argument("--zip")
    a_.add_argument("--instance", default=None)
    a_.add_argument("--classifier", default=str(ROOT / "models" / "xlmr_civic"))
    a_.add_argument("--refs", default=str(ROOT / "models" / "refs"))
    a_.add_argument("--weights", default=str(ROOT / "configs" / "reference_weights_vendor.yaml"))
    a_.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build-refs":
        build_refs(a.scores, a.processed, a.gen, a.coefs, Path(a.out))
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
    P["score"] = score_texts(P.text.astype(str).tolist(), a.classifier)
    P.drop(columns=[c for c in ("tokens",) if c in P]).to_parquet(out / "scored.parquet", index=False)
    weights = {str(k): float(v) for k, v in yaml.safe_load(open(a.weights, encoding="utf-8"))["weights"].items()}
    res = analyze(P, Path(a.refs), weights)
    res.to_csv(out / "alpha.csv", index=False)
    report(res, inst, out / "report.html")
    print(res.to_string())


if __name__ == "__main__":
    main()
