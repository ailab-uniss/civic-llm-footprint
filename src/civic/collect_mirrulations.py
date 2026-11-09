"""Collect U.S. public comments (Regulations.gov) from the Mirrulations open-data mirror on S3 (anonymous access).

For each agency and year, dockets are listed, their comment counts measured, and dockets with between
--min and --max comments are sampled until --per-year comments are reached. Comment JSON files are read
directly from S3 and converted to the unified schema (instance = agency, type = 'comment', author_hash from
the comment's first/last name or organization when present).

  python collect_mirrulations.py --agencies EPA DOT FDA USDA DOI --years 2019 2026 --per-year 6000 --out data/interim/regulations_gov.parquet
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import clean_html, hash_author  # noqa: E402

BUCKET = "mirrulations"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=32), region_name="us-east-1")


def list_prefixes(prefix: str) -> list[str]:
    out = []
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        out += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
    return out


def list_keys(prefix: str, limit: int | None = None) -> list[str]:
    out = []
    pag = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET, Prefix=prefix):
        out += [o["Key"] for o in page.get("Contents", [])]
        if limit and len(out) >= limit:
            break
    return out


def read_json(key: str) -> dict | None:
    try:
        b = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        return json.loads(b)
    except Exception:  # noqa: BLE001
        return None


def comment_row(d: dict, agency: str, docket: str) -> dict | None:
    a = (d or {}).get("data", {}).get("attributes", {})
    text = clean_html(a.get("comment") or "")
    if len(text) < 5:
        return None
    name = " ".join(x for x in [a.get("firstName"), a.get("lastName"), a.get("organization")] if x) or ""
    return dict(instance=f"regulations_gov_{agency.lower()}", lang_declared="en", type="comment", id=d["data"]["id"],
                author_hash=hash_author("regulations_gov", name) if name else "",
                created_at=a.get("postedDate") or a.get("receiveDate"), text=text, title=a.get("title") or "",
                space_url=f"https://www.regulations.gov/docket/{docket}", state="", supports="",
                n_comments=str(a.get("duplicateComments") or ""), source="mirrulations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agencies", nargs="+", default=["EPA", "DOT", "FDA", "USDA", "DOI"])
    ap.add_argument("--years", nargs=2, type=int, default=[2019, 2026])
    ap.add_argument("--per-year", type=int, default=6000, help="target comments per agency-year")
    ap.add_argument("--min", type=int, default=200)
    ap.add_argument("--max", type=int, default=4000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for agency in a.agencies:
        dockets = [p.rstrip("/").split("/")[-1] for p in list_prefixes(f"raw-data/{agency}/")]
        for year in range(a.years[0], a.years[1] + 1):
            cand = [d for d in dockets if re.search(rf"-{year}-\d+$", d)]
            rng.shuffle(cand)
            got = 0
            for d in cand:
                if got >= a.per_year:
                    break
                prefix = f"raw-data/{agency}/{d}/text-{d}/comments/"
                keys = list_keys(prefix, limit=a.max + 1)
                n = len(keys)
                if n < a.min or n > a.max:
                    continue
                with ThreadPoolExecutor(16) as ex:
                    docs = list(ex.map(read_json, keys))
                k = 0
                for doc in docs:
                    r = comment_row(doc, agency, d)
                    if r:
                        rows.append(r)
                        k += 1
                got += k
                print(f"{agency} {year} {d}: {n} files, {k} comments with text (year total {got})", flush=True)
            pd.DataFrame(rows).to_parquet(a.out, index=False)
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at"])
    df.to_parquet(a.out, index=False)
    print(df.groupby([df.instance, df.created_at.dt.year]).size().to_string())


if __name__ == "__main__":
    main()
