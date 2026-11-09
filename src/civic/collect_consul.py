"""Convert the open-data CSV files of a CONSUL Democracy installation (proposals.csv, comments.csv, debates.csv,
as served under /system/api/) to the unified schema used for Decidim exports.

  python -I src/civic/collect_consul.py --dir <folder with the csv files> --instance decide_madrid --lang es \
      --base-url https://decide.madrid.es --out data/interim/decide_madrid.parquet

Files are ';'-separated, cp1252/latin-1 encoded, with dd/mm/yyyy dates. Comments carry no author; proposals and
debates carry a user name that is hashed. Comment types are 'comment' (on proposals) and 'debate_comment'.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import clean_html, hash_author  # noqa: E402


def read_csv(p: Path) -> pd.DataFrame:
    raw = p.read_bytes()
    try:
        raw.decode("utf-8")
        enc = "utf-8"
    except UnicodeDecodeError:
        enc = "cp1252"
    df = pd.read_csv(p, sep=";", quotechar='"', encoding=enc, encoding_errors="replace", on_bad_lines="skip",
                     dtype=str, engine="python")
    df["created_at"] = pd.to_datetime(df["created_at"].str.slice(0, 10), format="%d/%m/%Y", errors="coerce", utc=True)
    return df.dropna(subset=["created_at"])


def joined(*parts) -> str:
    return "\n".join(clean_html(x) for x in parts if isinstance(x, str) and x.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    rows = []

    P = read_csv(d / "proposals.csv")
    for r in P.itertuples(index=False):
        rows.append(dict(instance=a.instance, lang_declared=a.lang, type="proposal", id=str(r.id),
                         author_hash=hash_author(a.instance, r.author_name) if isinstance(r.author_name, str) else "",
                         created_at=r.created_at, text=joined(r.summary, r.description), title=clean_html(r.title or ""),
                         space_url=f"{a.base_url}/proposals/{r.id}", state="retired" if isinstance(r.retired_at, str) else "",
                         supports=str(r.cached_votes_up or ""), n_comments=str(r.comments_count or ""), source="consul_csv"))
    if (d / "debates.csv").exists():
        D = read_csv(d / "debates.csv")
        for r in D.itertuples(index=False):
            rows.append(dict(instance=a.instance, lang_declared=a.lang, type="debate", id=str(r.id),
                             author_hash=hash_author(a.instance, r.author_name) if isinstance(r.author_name, str) else "",
                             created_at=r.created_at, text=joined(r.description), title=clean_html(r.title or ""),
                             space_url=f"{a.base_url}/debates/{r.id}", state="", supports=str(r.cached_votes_up or ""),
                             n_comments=str(r.comments_count or ""), source="consul_csv"))
    if (d / "comments.csv").exists():
        C = read_csv(d / "comments.csv")
        kind = {"Proposal": ("comment", "proposals"), "Debate": ("debate_comment", "debates")}
        for r in C.itertuples(index=False):
            typ, path = kind.get(r.commentable_type, ("comment", "proposals"))
            rows.append(dict(instance=a.instance, lang_declared=a.lang, type=typ, id=str(r.id), author_hash="",
                             created_at=r.created_at, text=clean_html(r.body or ""), title="",
                             space_url=f"{a.base_url}/{path}/{r.commentable_id}", state="",
                             supports=str(r.cached_votes_up or ""), n_comments="", source="consul_csv"))
    df = pd.DataFrame(rows)
    df = df[df.text.str.len() >= 5]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(df.groupby([df.type, df.created_at.dt.year]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
