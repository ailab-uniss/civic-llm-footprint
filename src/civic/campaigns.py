"""Coordinated-campaign detection: lexical near-duplicates (MinHash/LSH) and semantic near-duplicates
(multilingual embeddings + cosine threshold), clustered with union-find within an instance.

A cluster is labelled a *campaign* when it has >= min_authors distinct authors and >= min_texts texts
posted within a window of <= max_days days.

Output: parquet with columns doc_id, cluster_id, cluster_size, cluster_authors, cluster_span_days,
        is_campaign, dup_kind ('lexical' | 'semantic' | '')
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

EMB_MODEL = "BAAI/bge-m3"


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def shingles(text: str, k: int = 5) -> set[str]:
    w = text.lower().split()
    if len(w) < k:
        return {" ".join(w)}
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def lexical_pairs(texts: list[str], threshold: float = 0.6, num_perm: int = 128, k: int = 5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mhs = []
    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for s in shingles(t, k):
            m.update(s.encode("utf-8"))
        mhs.append(m)
        lsh.insert(str(i), m)
    pairs = set()
    for i, m in enumerate(mhs):
        for j in lsh.query(m):
            j = int(j)
            if j > i:
                pairs.add((i, j))
    return pairs


def semantic_pairs(texts: list[str], threshold: float = 0.9, batch: int = 32, device: str = "cuda"):
    import faiss
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device=device)
    model.max_seq_length = 512  # civic texts are short; the 8k default makes the attention mask of a batch exceed GPU memory
    model.half()
    emb = model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    pairs = set()
    k = min(20, len(texts))
    for s in range(0, len(texts), 2048):
        D, I = index.search(emb[s:s + 2048], k)
        for r in range(D.shape[0]):
            i = s + r
            for d, j in zip(D[r], I[r]):
                if j != i and d >= threshold:
                    pairs.add((min(i, j), max(i, j)))
    return pairs, emb


def cluster(df: pd.DataFrame, lex_thr: float, sem_thr: float, min_authors: int, min_texts: int, max_days: int,
            use_semantic: bool = True, device: str = "cuda") -> tuple[pd.DataFrame, np.ndarray | None]:
    df = df.reset_index(drop=True)
    texts = df["text"].tolist()
    uf = UnionFind(len(df))
    kind = np.array([""] * len(df), dtype=object)
    lp = lexical_pairs(texts, lex_thr)
    for i, j in lp:
        uf.union(i, j)
        kind[i] = kind[j] = "lexical"
    emb = None
    if use_semantic and len(df) > 1:
        sp, emb = semantic_pairs(texts, sem_thr, device=device)
        for i, j in sp:
            uf.union(i, j)
            if kind[i] == "":
                kind[i] = "semantic"
            if kind[j] == "":
                kind[j] = "semantic"
    df["cluster_id"] = [uf.find(i) for i in range(len(df))]
    df["dup_kind"] = kind
    g = df.groupby("cluster_id")
    df["cluster_size"] = g["doc_id"].transform("size")
    df["cluster_authors"] = g["author_hash"].transform(lambda s: s[s != ""].nunique())
    span = g["created_at"].transform(lambda s: (s.max() - s.min()).days)
    df["cluster_span_days"] = span
    df["is_campaign"] = (df.cluster_size >= min_texts) & (df.cluster_authors >= min_authors) & (span <= max_days)
    return df, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="processed parquet (one instance or all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb-out", default="")
    ap.add_argument("--lex-thr", type=float, default=0.6)
    ap.add_argument("--sem-thr", type=float, default=0.9)
    ap.add_argument("--min-authors", type=int, default=3)
    ap.add_argument("--min-texts", type=int, default=5)
    ap.add_argument("--max-days", type=int, default=30)
    ap.add_argument("--no-semantic", action="store_true")
    a = ap.parse_args()
    df = pd.read_parquet(a.inp)
    if "doc_id" not in df.columns:
        df["doc_id"] = df["instance"] + ":" + df["type"] + ":" + df["id"].astype(str) + ":" + df["lang"]
    outs, embs = [], []
    for inst, g in df.groupby("instance"):
        print(f"{inst}: {len(g)} texts")
        res, emb = cluster(g[["doc_id", "instance", "lang", "type", "author_hash", "created_at", "text"]].copy(),
                           a.lex_thr, a.sem_thr, a.min_authors, a.min_texts, a.max_days, not a.no_semantic)
        res["cluster_id"] = inst + ":" + res["cluster_id"].astype(str)
        outs.append(res.drop(columns=["text"]))
        if emb is not None and a.emb_out:
            embs.append(pd.DataFrame({"doc_id": res.doc_id, "emb": list(emb)}))
    out = pd.concat(outs, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    if embs:
        pd.concat(embs, ignore_index=True).to_parquet(a.emb_out, index=False)
    print(out.groupby("instance")[["is_campaign"]].mean().to_string())


if __name__ == "__main__":
    main()
