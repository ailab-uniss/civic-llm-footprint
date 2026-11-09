"""End-to-end smoke test of benchmark / analysis / campaigns on small synthetic data (no GPU, no spaCy)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "civic"))

VOCAB = [f"w{i}|NOUN" for i in range(400)]
LLM_WORDS = [f"l{i}|ADJ" for i in range(60)]


def make_doc(rng, llm=False, n=40):
    words = list(rng.choice(VOCAB, n))
    if llm:
        words += list(rng.choice(LLM_WORDS, 8))
    return words


def fake_processed(rng, n=1200, langs=("fr", "ca"), insts=("a", "b")):
    rows = []
    for i in range(n):
        lang = langs[i % 2]
        inst = insts[(i // 2) % 2]
        post = i % 3 == 0
        ts = pd.Timestamp("2024-03-01", tz="UTC") if post else pd.Timestamp("2021-03-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 600)))
        llm = post and rng.random() < 0.3
        toks = make_doc(rng, llm)
        rows.append(dict(instance=inst, lang=lang, type="proposal" if i % 5 else "comment", id=str(i), author_hash=f"h{i % 50}",
                         created_at=ts, text=" ".join(t.split("|")[0] for t in toks), n_words=len(toks), is_dup_exact=False,
                         period="post" if post else "pre", tokens=toks, doc_id=f"{inst}:p:{i}:{lang}"))
    return pd.DataFrame(rows)


def fake_generated(rng, H, models=("m1", "m2", "m3")):
    rows = []
    for m in models:
        for r in H[H.period == "pre"].itertuples():
            for mode in ("rewrite", "generate"):
                toks = make_doc(rng, True)
                rows.append(dict(doc_id=r.doc_id, src_doc_id=r.doc_id, lang=r.lang, model=m, mode=mode, variant=0,
                                 text="x", n_words=len(toks), tokens=toks))
    return pd.DataFrame(rows)


def test_benchmark_and_analysis(tmp_path):
    from benchmark import run_language, summarise
    import analysis
    rng = np.random.default_rng(0)
    H = fake_processed(rng)
    G = fake_generated(rng, H)
    res = run_language("fr", H[H.lang == "fr"], G[G.lang == "fr"], None, n_mix=300, reps=2, n_boot=20, rng=rng,
                       selections=["ratio"], modes_scenarios={"all": ["rewrite", "generate"], "short": ["rewrite", "generate"]})
    df = pd.DataFrame(res)
    assert len(df) > 0
    s = summarise(df)
    assert (s.mae < 0.1).all(), s
    from types import SimpleNamespace
    G["inst"] = G.doc_id.map(H.set_index("doc_id").instance)
    S = H[H.period == "pre"][["doc_id", "lang", "instance"]]
    args = SimpleNamespace(selection="ratio", top_k=200, min_df=0.003, likelihood="occurrence", min_ref=50, kappa=100.0, weights=None)
    ests = analysis.References(H, G, S, args)
    assert set(ests.base) == {"fr", "ca"}
    D = H.copy()
    D["half"] = analysis.half_year(D.created_at)
    D["post"] = (D.created_at >= analysis.CHATGPT).astype(int)
    D["t"] = (D.created_at - analysis.CHATGPT).dt.days / 365.25
    cells = analysis.estimate_cells(D, ests, 50, 20, ["instance", "lang", "period"])
    assert len(cells) > 0
    post = cells[cells.period == "post"].alpha.mean()
    pre = cells[cells.period == "pre"].alpha.mean()
    assert post > pre + 0.1, (pre, post)
    hom = analysis.homogenisation(D, None, ["instance", "lang", "period"], min_docs=20)
    assert "distinct2" in hom
    txt = analysis.mixed_model(cells.assign(post=(cells.period == "post").astype(int), type="proposal"))
    assert "post" in txt


def test_campaigns_lexical():
    from campaigns import cluster
    rng = np.random.default_rng(1)
    base = "the council should build a new bike lane along the main street to reduce traffic and pollution in our neighbourhood"
    rows = []
    for i in range(30):
        rows.append(dict(doc_id=f"d{i}", instance="x", lang="en", type="comment", author_hash=f"a{i}",
                         created_at=pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i % 5),
                         text=base if i < 12 else " ".join(rng.choice(VOCAB, 40))))
    df, _ = cluster(pd.DataFrame(rows), 0.6, 0.9, 3, 5, 30, use_semantic=False)
    assert df.is_campaign.sum() == 12
