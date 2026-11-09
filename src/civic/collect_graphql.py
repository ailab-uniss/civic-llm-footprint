"""Collect proposals, comments and debates from a Decidim instance through its public GraphQL API.

Used for instances whose open-data ZIP export is unavailable (e.g. decidim.barcelona returns HTTP 500).
Output follows the unified schema of ingest.py, with source='graphql'.
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

COMMENT_FIELDS = "id body createdAt author { id }"
COMMENTS_NESTED = "comments { %s comments { %s } }" % (COMMENT_FIELDS, COMMENT_FIELDS)

Q_SPACES = """{ participatoryProcesses { id slug } assemblies { id slug } }"""

Q_COMPONENTS = """query($id: ID!) { %s(id: $id) { id components { id __typename } } }"""

Q_PROPOSALS = """query($id: ID!, $after: String) { component(id: $id) { ... on Proposals {
  proposals(first: 50, after: $after) { pageInfo { hasNextPage endCursor } nodes {
    id publishedAt state voteCount totalCommentsCount author { id }
    title { translations { locale text } } body { translations { locale text } }
    %s } } } } }""" % COMMENTS_NESTED

Q_DEBATES = """query($id: ID!, $after: String) { component(id: $id) { ... on Debates {
  debates(first: 50, after: $after) { pageInfo { hasNextPage endCursor } nodes {
    id createdAt author { id } title { translations { locale text } }
    description { translations { locale text } } %s } } } } }""" % COMMENTS_NESTED


class Client:
    def __init__(self, base: str, sleep: float = 0.3):
        self.url = base.rstrip("/") + "/api"
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "civic-llm-footprint research collector (academic use)"
        self.sleep = sleep

    def q(self, query: str, variables: dict | None = None, retries: int = 5):
        for i in range(retries):
            try:
                r = self.s.post(self.url, json={"query": query, "variables": variables or {}},
                                timeout=120, verify=False)
                if r.status_code == 200:
                    j = r.json()
                    if "errors" in j and not j.get("data"):
                        raise RuntimeError(j["errors"][:1])
                    time.sleep(self.sleep)
                    return j.get("data")
                print(f"HTTP {r.status_code}; retry {i}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"error {e}; retry {i}", file=sys.stderr)
            time.sleep(2 * (i + 1))
        raise RuntimeError("GraphQL failed")


def flatten_comments(comments, instance, space_url, rows, ctype="comment"):
    for c in comments or []:
        body = clean_html(c.get("body"))
        if len(body) >= 5:
            rows.append(dict(instance=instance, lang_declared="", type=ctype, id=str(c["id"]),
                             author_hash=hash_author(instance, (c.get("author") or {}).get("id")),
                             created_at=c.get("createdAt"), text=body, title="", space_url=space_url,
                             state="", supports="", n_comments="", source="graphql"))
        flatten_comments(c.get("comments"), instance, space_url, rows, ctype)


def collect(base: str, instance: str, out: Path):
    cl = Client(base)
    spaces = cl.q(Q_SPACES)
    space_list = [("participatoryProcess", s["id"], s["slug"]) for s in spaces.get("participatoryProcesses") or []]
    space_list += [("assembly", s["id"], s["slug"]) for s in spaces.get("assemblies") or []]
    print(f"{instance}: {len(space_list)} spaces", file=sys.stderr)
    rows = []
    for kind, sid, slug in space_list:
        try:
            d = cl.q(Q_COMPONENTS % kind, {"id": sid})
        except Exception as e:  # noqa: BLE001
            print(f"skip space {kind} {sid}: {e}", file=sys.stderr)
            continue
        comps = ((d or {}).get(kind) or {}).get("components") or []
        space_url = f"{base}/{'processes' if kind == 'participatoryProcess' else 'assemblies'}/{slug}"
        for comp in comps:
            if comp["__typename"] not in ("Proposals", "Debates"):
                continue
            after = None
            n = 0
            while True:
                query = Q_PROPOSALS if comp["__typename"] == "Proposals" else Q_DEBATES
                try:
                    d = cl.q(query, {"id": comp["id"], "after": after})
                except Exception as e:  # noqa: BLE001
                    print(f"skip component {comp['id']}: {e}", file=sys.stderr)
                    break
                key = "proposals" if comp["__typename"] == "Proposals" else "debates"
                page = ((d or {}).get("component") or {}).get(key)
                if not page:
                    break
                for node in page["nodes"]:
                    if not node:
                        continue
                    n += 1
                    if key == "proposals":
                        titles = {t["locale"]: t["text"] for t in (node.get("title") or {}).get("translations", [])}
                        for tr in (node.get("body") or {}).get("translations", []):
                            body = clean_html(tr["text"])
                            if len(body) < 5:
                                continue
                            rows.append(dict(instance=instance, lang_declared=tr["locale"].lower(), type="proposal",
                                             id=str(node["id"]),
                                             author_hash=hash_author(instance, (node.get("author") or {}).get("id")),
                                             created_at=node.get("publishedAt"), text=body,
                                             title=clean_html(titles.get(tr["locale"], "")), space_url=space_url,
                                             state=node.get("state") or "", supports=str(node.get("voteCount") or ""),
                                             n_comments=str(node.get("totalCommentsCount") or ""), source="graphql"))
                        flatten_comments(node.get("comments"), instance, space_url, rows, "comment")
                    else:
                        flatten_comments(node.get("comments"), instance, space_url, rows, "debate_comment")
                if page["pageInfo"]["hasNextPage"]:
                    after = page["pageInfo"]["endCursor"]
                else:
                    break
            print(f"  {slug} {comp['__typename']} {comp['id']}: {n} items, total rows {len(rows)}", file=sys.stderr)
        # checkpoint
        pd.DataFrame(rows).to_parquet(out, index=False)
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at"])
    df.to_parquet(out, index=False)
    print(f"{instance}: {len(df)} rows; {df.created_at.min()} .. {df.created_at.max()}", file=sys.stderr)
    print(df.groupby(["type"]).size().to_string(), file=sys.stderr)


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    collect(a.base, a.instance, Path(a.out))
