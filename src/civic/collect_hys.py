"""Collect public feedback from the EU 'Have Your Say' portal (Better Regulation API, public, no key).

Initiatives are listed through searchInitiatives; for every publication (feedback period) with at least --min
feedback items, all feedback is fetched (dateFeedback, text, language, userType, country, organization) and mapped
to the unified schema with extra columns user_type, country, publication_type.

  python collect_hys.py --out data/interim/eu_hys.parquet --min 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import clean_html, hash_author  # noqa: E402

BASE = "https://ec.europa.eu/info/law/better-regulation"
S = requests.Session()
S.headers["User-Agent"] = "civic-llm-footprint research collector (academic use)"


def get(url: str, retries: int = 5, wait: float = 0.3):
    for i in range(retries):
        try:
            r = S.get(url, timeout=120)
            if r.status_code == 200:
                time.sleep(wait)
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 * (i + 1))
    return None


def topic_str(topics) -> str:
    """Topics come as strings or as dicts with a 'name'/'label'/'id' key, depending on the endpoint."""
    out = []
    for t in topics or []:
        if isinstance(t, dict):
            t = t.get("name") or t.get("label") or t.get("id") or ""
        if t:
            out.append(str(t))
    return "|".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min", type=int, default=30, help="minimum feedback items per publication")
    ap.add_argument("--max-pages", type=int, default=100, help="max feedback pages (x100) per publication")
    a = ap.parse_args()
    out = Path(a.out)
    rows, done_pubs = [], set()
    if out.exists():
        prev = pd.read_parquet(out)
        rows = prev.to_dict("records")
        done_pubs = set(prev.publication_id.astype(str))
    cache = out.with_suffix(".initiatives.json")
    if cache.exists():
        inits = json.loads(cache.read_text())
    else:
        page, inits = 0, []
        while True:
            d = get(f"{BASE}/brpapi/searchInitiatives?page={page}&size=100&language=EN")
            if not d:
                break
            p = d["initiativeResultDtoPage"]
            inits += p["content"]
            if p.get("last") or page >= p.get("totalPages", 0):
                break
            page += 1
        cache.write_text(json.dumps(inits))
    print(f"{len(inits)} initiatives", flush=True)
    for k, it in enumerate(inits):
        gid = int(it["id"])
        g = get(f"{BASE}/brpapi/groupInitiatives/{gid}")
        if not g:
            continue
        title = g.get("shortTitle") or it.get("shortTitle") or ""
        for pub in g.get("publications", []):
            pid = str(pub.get("id"))
            try:
                total = int(pub.get("totalFeedback") or 0)
            except ValueError:
                total = 0
            if total < a.min or pid in done_pubs:
                continue
            fp = 0
            while fp < a.max_pages:
                fb = get(f"{BASE}/api/allFeedback?publicationId={pid}&page={fp}&size=100")
                if not fb or not fb.get("content"):
                    break
                for c in fb["content"]:
                    text = clean_html(c.get("feedback") or "")
                    if len(text) < 5:
                        continue
                    name = " ".join(x for x in [c.get("firstName"), c.get("surname"), c.get("organization")] if x)
                    rows.append(dict(instance="eu_hys", lang_declared=str(c.get("language") or "").lower(), type="feedback",
                                     id=str(c.get("id")), author_hash=hash_author("eu_hys", name) if name else "",
                                     created_at=c.get("dateFeedback"), text=text, title=title,
                                     space_url=f"{BASE}/have-your-say/initiatives/{gid}", state="", supports="",
                                     n_comments="", source="hys_api", user_type=c.get("userType") or "",
                                     country=c.get("country") or "", publication_type=pub.get("type") or "",
                                     publication_id=pid, initiative_id=str(gid), topics=topic_str(it.get("topics"))))
                if fb.get("last", True):
                    break
                fp += 1
            done_pubs.add(pid)
            print(f"[{k}/{len(inits)}] {gid} pub {pid} ({pub.get('type')}): {total} feedback -> rows {len(rows)}", flush=True)
        if k % 25 == 0 and rows:
            pd.DataFrame(rows).to_parquet(out, index=False)
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], format="%Y/%m/%d %H:%M:%S", errors="coerce", utc=True)
    df = df.dropna(subset=["created_at"])
    df.to_parquet(out, index=False)
    print(df.groupby([df.lang_declared, df.created_at.dt.year]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
