"""Ingest Decidim open-data exports (ZIP of ';'-separated CSVs) into a unified parquet schema.

Unified schema (one row per text in a declared language):
  instance, lang_declared, type ('proposal'|'comment'|'debate_comment'), id, author_hash,
  created_at (UTC), text, title, space_url, state, supports, n_comments, source ('opendata'|'graphql')

Author identifiers are pseudonymised at import time with a salted SHA-256 hash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

csv.field_size_limit(2**31 - 1)  # sys.maxsize overflows on Windows

SALT = os.environ.get("CIVIC_SALT", "civic-llm-footprint-2026")
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def hash_author(instance: str, author_id) -> str:
    if author_id is None or str(author_id).strip() in ("", "0", "nan"):
        return ""
    return hashlib.sha256(f"{SALT}|{instance}|{author_id}".encode()).hexdigest()[:16]


def clean_html(s) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("<br>", "\n").replace("<br/>", "\n").replace("</p>", "\n")
    s = TAG_RE.sub(" ", s)
    s = (s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
          .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " "))
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def read_csv_bytes(b: bytes) -> pd.DataFrame:
    txt = b.decode("utf-8", errors="replace")
    if not txt.strip():
        return pd.DataFrame()
    first = txt.split("\n", 1)[0]
    sep = ";" if first.count(";") >= first.count(",") else ","
    return pd.read_csv(io.StringIO(txt), sep=sep, dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_MINIMAL, engine="python", on_bad_lines="skip")


def lang_columns(df: pd.DataFrame, base: str) -> list[tuple[str, list[str]]]:
    """Return [(lang, [columns])] for columns like 'body/fi' or structured bodies 'body/<section>/fi'
    (skipping machine translations). Several columns of the same language are concatenated."""
    groups: dict[str, list[str]] = {}
    for c in df.columns:
        if "machine_translations" in c:
            continue
        m = re.fullmatch(rf"{base}/(?:[^/]+/)?([a-zA-Z]{{2}}(?:-[a-zA-Z]{{2}})?)", c)
        if m:
            groups.setdefault(m.group(1).lower()[:2], []).append(c)
    return list(groups.items())


def joined(r, cols) -> str:
    return chr(10).join(clean_html(r.get(c, "")) for c in cols if str(r.get(c, "")).strip())


LOCALISED = {  # Portuguese (Brasil Participativo) -> canonical
    "titulo": "title", "corpo": "body", "autor": "author", "publicado_em": "published_at", "criado_em": "created_at",
    "estado": "state", "apoios": "supports", "comentarios": "comments", "data_publicacao": "published_at",
    "espaco_participativo": "participatory_space", "localidade": "locale", "nome_processo_participativo": "space_name",
    "url_processo_participativo": "participatory_space/url",
}


def localise_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        parts = c.split("/")
        parts[0] = LOCALISED.get(parts[0], parts[0])
        if parts[0] == "author" and len(parts) > 1 and parts[1] in ("id", "nome", "name"):
            parts[1] = "id" if parts[1] == "id" else "name"
        new = "/".join(parts)
        if new != c:
            ren[c] = new
    return df.rename(columns=ren) if ren else df


def parse_proposals(df: pd.DataFrame, instance: str) -> pd.DataFrame:
    rows = []
    df = localise_columns(df)
    bodies = lang_columns(df, "body")
    titles = dict(lang_columns(df, "title"))
    date_col = next((c for c in ("published_at", "created_at") if c in df.columns), None)
    for _, r in df.iterrows():
        for lang, bcols in bodies:
            body = joined(r, bcols)
            if len(body) < 5:
                continue
            title = joined(r, titles.get(lang, []))
            rows.append(dict(
                instance=instance, lang_declared=lang, type="proposal", id=str(r.get("id", "")),
                author_hash=hash_author(instance, r.get("author/id")),
                created_at=r.get(date_col, "") if date_col else "",
                text=body, title=title,
                space_url=r.get("participatory_space/url", ""), state=r.get("state", ""),
                supports=r.get("supports", r.get("proposal_votes_count", "")),
                n_comments=r.get("comments", ""), source="opendata",
            ))
    return pd.DataFrame(rows)


def parse_comments(df: pd.DataFrame, instance: str, ctype: str) -> pd.DataFrame:
    df = localise_columns(df)
    if df.empty or "body" not in df.columns:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        body = clean_html(r.get("body", ""))
        if len(body) < 5:
            continue
        rows.append(dict(
            instance=instance, lang_declared=str(r.get("locale", "")).lower(), type=ctype,
            id=str(r.get("id", "")), author_hash=hash_author(instance, r.get("author/id")),
            created_at=r.get("created_at", ""), text=body, title="",
            space_url=r.get("root_commentable_url", ""), state="", supports="",
            n_comments="", source="opendata",
        ))
    return pd.DataFrame(rows)


def ingest_zip(zip_path: Path, instance: str) -> pd.DataFrame:
    parts = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            low = name.lower()
            if low.endswith("proposals.csv") and "comment" not in low:
                parts.append(parse_proposals(read_csv_bytes(z.read(name)), instance))
            elif low.endswith("proposal_comments.csv"):
                parts.append(parse_comments(read_csv_bytes(z.read(name)), instance, "comment"))
            elif low.endswith("debate_comments.csv"):
                parts.append(parse_comments(read_csv_bytes(z.read(name)), instance, "debate_comment"))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at"])
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 0]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = ingest_zip(Path(a.zip), a.instance)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    if df.empty:
        print(f"{a.instance}: EMPTY")
        return
    s = df.groupby(["type", "lang_declared"]).size()
    print(f"{a.instance}: {len(df)} rows; {df.created_at.min().date()} .. {df.created_at.max().date()}")
    print(s.to_string())


if __name__ == "__main__":
    main()
