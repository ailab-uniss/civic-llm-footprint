"""Audit of LLM failure modes in civic writing (hypotheses L1-L7), computed on (original, output) pairs.

Metrics
  L1 legal references : count of normative references in the output (regex per language) and, where a
                        verification index is available, the share that cannot be verified. Without an index
                        we report the *rate of newly introduced references* (absent from the original).
  L2 specificity      : share of location entities (LOC/GPE/FAC from spaCy NER) and numbers of the original
                        preserved in the output; count of new location entities.
  L3 homogenisation   : mean pairwise cosine similarity among outputs vs. among originals (per lang/model/scenario),
                        distinct-2 ratio.
  L4 inflation        : words(output)/words(original); argumentative units approximated by sentence count.
  L5 intent           : multilingual NLI entailment of the original's key claims by the output (mDeBERTa XNLI);
                        stance preservation approximated by entailment both ways (bidirectional).
  L6 jurisdiction     : share of legal references pointing to a foreign legal system (pattern lists per country).
  L7 language quality : share of outputs whose detected language != requested language; proxy for translationese:
                        rate of English loan tokens.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---- L1/L6: legal reference patterns -----------------------------------------------------------
LEGAL_PATTERNS = {
    "it": r"(?:art(?:icolo|\.)\s*\d+[\w\-]*|legge\s+(?:n\.?\s*)?\d+(?:/\d{2,4})?|l\.\s*\d+/\d{2,4}|d\.?\s?lgs\.?\s*(?:n\.?\s*)?\d+(?:/\d{2,4})?|d\.?p\.?r\.?\s*(?:n\.?\s*)?\d+(?:/\d{2,4})?|decreto\s+legislativo\s+(?:n\.?\s*)?\d+|d\.?m\.?\s*\d+|regolamento\s+(?:ue|ce)\s*\d+/\d{2,4}|direttiva\s+\d{2,4}/\d+(?:/ue|/ce)?|costituzione|codice\s+(?:civile|della\s+strada|penale)|t\.?u\.?e\.?l\.?|d\.l\.\s*\d+/\d{2,4})",
    "es": r"(?:art(?:ículo|iculo|\.)\s*\d+[\w\.\-]*|ley\s+(?:orgánica\s+)?\d+/\d{2,4}|real\s+decreto(?:\s+legislativo)?\s+\d+/\d{2,4}|rd\s*\d+/\d{2,4}|decreto\s+(?:ley\s+)?\d+/\d{2,4}|reglamento\s+\(?(?:ue|ce)\)?\s*\d+/\d{2,4}|directiva\s+\d{2,4}/\d+|constitución\s+española|ordenanza\s+(?:municipal\s+)?(?:de\s+)?[\w\s]{3,30}|lrbrl|lbrl|ley\s+de\s+bases)",
    "ca": r"(?:art(?:icle|\.)\s*\d+[\w\.\-]*|llei\s+(?:orgànica\s+)?\d+/\d{2,4}|reial\s+decret(?:\s+legislatiu)?\s+\d+/\d{2,4}|decret\s+(?:llei\s+)?\d+/\d{2,4}|reglament\s+\(?(?:ue|ce)\)?\s*\d+/\d{2,4}|directiva\s+\d{2,4}/\d+|constitució|estatut\s+d'autonomia|ordenança\s+(?:municipal\s+)?(?:de\s+)?[\w\s]{3,30}|llei\s+de\s+bases)",
    "fr": r"(?:art(?:icle|\.)\s*[lrd]?\.?\s*\d+[\w\.\-]*|loi\s+(?:n°\s*)?\d{2,4}-\d+|décret\s+(?:n°\s*)?\d{2,4}-\d+|code\s+(?:de\s+l'urbanisme|de\s+l'environnement|général\s+des\s+collectivités\s+territoriales|de\s+la\s+route|civil|pénal|de\s+la\s+santé\s+publique)|cgct|plu[i]?\b|règlement\s+\(?(?:ue|ce)\)?\s*(?:n°\s*)?\d+/\d{2,4}|directive\s+\d{2,4}/\d+|loi\s+(?:climat|elan|sru|notre|maptam|3ds|lom|egalim|alur)\b|constitution)",
    "fi": r"(?:\d+\s*§|laki\s+[\w\-]+\s*\(\d+/\d{4}\)|\(\d+/\d{4}\)|\d+/\d{4}\s*§?|maankäyttö-\s*ja\s+rakennuslaki|kuntalaki|perustuslaki|tieliikennelaki|ympäristönsuojelulaki|rakennuslaki|asetus\s+\d+/\d{4}|eu:n\s+asetus\s+\d+/\d{4}|direktiivi\s+\d{4}/\d+)",
    "en": r"(?:section\s+\d+[\w\.\-]*|§\s*\d+|article\s+\d+[\w\.\-]*|(?:local\s+law|law)\s+(?:no\.?\s*)?\d+(?:\s+of\s+\d{4})?|title\s+\d+|chapter\s+\d+|u\.?s\.?c\.?\s*§?\s*\d+|c\.?f\.?r\.?\s*(?:part\s+)?\d+|zoning\s+resolution|administrative\s+code|executive\s+order\s+\d+|ada\b|nepa\b|ceqr\b|ulurp\b|regulation\s+\(?(?:eu|ec)\)?\s*\d+/\d{4}|directive\s+\d{4}/\d+)",
    "pt": r"(?:art(?:igo|\.)\s*\d+[\w\.\-º]*|lei\s+(?:n[ºo°]\.?\s*)?\d+[\.\d]*(?:/\d{2,4})?|decreto(?:-lei)?\s+(?:n[ºo°]\.?\s*)?\d+[\.\d]*(?:/\d{2,4})?|constituição\s+federal|estatuto\s+da\s+cidade|plano\s+diretor|código\s+(?:civil|penal|de\s+trânsito)|lei\s+orgânica|resolução\s+\d+)",
}
# foreign-jurisdiction markers: references to another legal order than the instance's country
FOREIGN_MARKERS = {   # every marker is a whole word/phrase (\b...\b), so that e.g. "ada" cannot match "cada"
    "IT": r"(?:u\.?s\.?c\.?|c\.?f\.?r\.?|section\s+\d+|real\s+decreto|ley\s+\d+/\d{4}|code\s+de\s+l'urbanisme|cgct|loi\s+n°|boe|ada)",
    "ES": r"(?:u\.?s\.?c\.?|c\.?f\.?r\.?|section\s+\d+|d\.?lgs\.?|legge\s+n|code\s+de\s+l'urbanisme|cgct|loi\s+n°|gazzetta\s+ufficiale|ada)",
    "FR": r"(?:u\.?s\.?c\.?|c\.?f\.?r\.?|section\s+\d+|d\.?lgs\.?|legge\s+n|real\s+decreto|ley\s+\d+/\d{4}|boe|gazzetta|ada)",
    "FI": r"(?:u\.?s\.?c\.?|c\.?f\.?r\.?|section\s+\d+|d\.?lgs\.?|legge\s+n|real\s+decreto|ley\s+\d+/\d{4}|code\s+de\s+l'urbanisme|loi\s+n°|ada)",
    "CH": r"(?:u\.?s\.?c\.?|c\.?f\.?r\.?|section\s+\d+|d\.?lgs\.?|legge\s+n|real\s+decreto|ley\s+\d+/\d{4}|cgct|code\s+général\s+des\s+collectivités|ada)",
    "US": r"(?:d\.?lgs\.?|legge\s+n|real\s+decreto|ley\s+\d+/\d{4}|code\s+de\s+l'urbanisme|cgct|loi\s+n°|directive\s+\d{4}/\d+|regulation\s+\(?(?:eu|ec)\)?|gdpr)",
    "INT": r"(?:$^)",
    "CA": r"(?:u\.?s\.?c\.?|c\.?f\.?r\.?|d\.?lgs\.?|legge\s+n|real\s+decreto|ley\s+\d+/\d{4}|cgct)",
    "BR": r"(?:u\.?s\.?c\.?|c\.?f\.?r\.?|section\s+\d+|d\.?lgs\.?|legge\s+n|real\s+decreto|ley\s+\d+/\d{4}|cgct|loi\s+n°)",
}
FOREIGN_MARKERS = {k: (r"\b" + v + r"\b" if k != "INT" else v) for k, v in FOREIGN_MARKERS.items()}
NUM_RE = re.compile(r"\b\d[\d\.,]*\b")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def legal_refs(text: str, lang: str) -> list[str]:
    pat = LEGAL_PATTERNS.get(lang)
    if not pat:
        return []
    return [m.group(0).strip().lower() for m in re.finditer(pat, text, flags=re.IGNORECASE)]


def foreign_refs(text: str, country: str) -> int:
    pat = FOREIGN_MARKERS.get(country, FOREIGN_MARKERS["INT"])
    return len(re.findall(pat, text, flags=re.IGNORECASE))


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[\.\!\?])\s+|\n+", text.strip()) if len(s.split()) >= 3]


class NER:
    def __init__(self):
        self.nlps = {}

    def ents(self, text: str, lang: str) -> set[str]:
        import spacy
        from preprocess import SPACY_MODELS
        if lang not in self.nlps:
            try:
                self.nlps[lang] = spacy.load(SPACY_MODELS[lang], disable=["lemmatizer"])
            except Exception:  # noqa: BLE001
                self.nlps[lang] = None
        nlp = self.nlps[lang]
        if nlp is None:
            return set()
        doc = nlp(text[:5000])
        return {e.text.lower().strip() for e in doc.ents if e.label_ in ("LOC", "GPE", "FAC", "ORG", "PER", "MISC")}


class NLI:
    MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.MODEL)
        self.m = AutoModelForSequenceClassification.from_pretrained(self.MODEL).to(device).eval()
        self.device = device
        self.ent_idx = [i for i, l in self.m.config.id2label.items() if l.lower().startswith("entail")][0]

    def entail(self, premises: list[str], hypotheses: list[str], batch: int = 32) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(premises), batch):
                enc = self.tok(premises[i:i + batch], hypotheses[i:i + batch], return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(self.device)
                p = torch.softmax(self.m(**enc).logits, -1)[:, self.ent_idx]
                out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.array([])


def pair_metrics(orig: str, out: str, lang: str, country: str, ner: NER | None) -> dict:
    o_refs, n_refs = legal_refs(orig, lang), legal_refs(out, lang)
    new_refs = [r for r in n_refs if r not in set(o_refs)]
    o_nums = set(NUM_RE.findall(orig))
    n_nums = set(NUM_RE.findall(out))
    d = dict(
        l1_refs_out=len(n_refs), l1_refs_orig=len(o_refs), l1_new_refs=len(new_refs), l1_any_new=int(len(new_refs) > 0),
        l6_foreign=foreign_refs(out, country), l6_any_foreign=int(foreign_refs(out, country) > 0),
        l4_len_ratio=len(WORD_RE.findall(out)) / max(1, len(WORD_RE.findall(orig))),
        l4_sent_ratio=len(sentences(out)) / max(1, len(sentences(orig))),
        l2_nums_kept=(len(o_nums & n_nums) / len(o_nums)) if o_nums else np.nan,
    )
    if ner is not None:
        oe, ne = ner.ents(orig, lang), ner.ents(out, lang)
        d.update(l2_ents_orig=len(oe), l2_ents_kept=(len(oe & ne) / len(oe)) if oe else np.nan,
                 l2_new_ents=len(ne - oe))
    return d


def _distinct2(texts) -> float:
    bigrams, total = set(), 0
    for t in texts:
        w = WORD_RE.findall(t.lower())
        total += max(len(w) - 1, 0)
        bigrams.update(zip(w[:-1], w[1:]))
    return len(bigrams) / total if total else np.nan


def _mean_offdiag(S: np.ndarray, idx: np.ndarray) -> float:
    sub = S[np.ix_(idx, idx)]
    mask = idx[:, None] != idx[None, :]   # a resampled original is never paired with itself
    return float(sub[mask].mean())


def homogenisation_paired(df: pd.DataFrame, device: str = "cuda", n_boot: int = 200, min_docs: int = 50,
                          seed: int = 0) -> pd.DataFrame:
    """L3 on the paired design: for each lang x model x scenario, the mean pairwise cosine similarity (BGE-M3) among the
    outputs minus that among their own originals (same doc_ids, so topic is held fixed), with a bootstrap CI over
    originals, and the distinct-2 ratio of outputs and originals. Because outputs are longer than originals and both
    statistics depend on length, every quantity is also computed on outputs truncated to the length of their original."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    model.max_seq_length = 512
    model.half()
    enc = lambda xs: model.encode(list(xs), batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False).astype("float32")
    df = df.copy()
    df["n_orig"] = df.orig.map(lambda t: len(WORD_RE.findall(t)))
    df["out_trunc"] = [" ".join(o.split()[:max(n, 1)]) for o, n in zip(df.output, df.n_orig)]
    origs = df.drop_duplicates("doc_id").set_index("doc_id").orig
    E_orig = pd.Series(list(enc(origs.values)), index=origs.index)
    E_out, E_trc = enc(df.output.values), enc(df.out_trunc.values)
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(["lang", "model", "mode"]):
        if len(g) < min_docs:
            continue
        pos = g.index.to_numpy()
        Eo = np.stack(E_orig.loc[g.doc_id].values)
        S_o = Eo @ Eo.T
        res = {"lang": key[0], "model": key[1], "mode": key[2], "n_docs": len(g)}
        for tag, E in [("full", E_out), ("trunc", E_trc)]:
            Eg = E[df.index.get_indexer(pos)]
            S_g = Eg @ Eg.T
            allidx = np.arange(len(g))
            co, cg = _mean_offdiag(S_o, allidx), _mean_offdiag(S_g, allidx)
            boots = []
            for _ in range(n_boot):
                b = rng.integers(0, len(g), len(g))
                boots.append(_mean_offdiag(S_g, b) - _mean_offdiag(S_o, b))
            res.update({f"cos_out_{tag}": cg, f"dcos_{tag}": cg - co,
                        f"dcos_{tag}_lo": float(np.percentile(boots, 2.5)), f"dcos_{tag}_hi": float(np.percentile(boots, 97.5))})
        res["cos_orig"] = _mean_offdiag(S_o, np.arange(len(g)))
        res["d2_orig"], res["d2_out_full"], res["d2_out_trunc"] = (_distinct2(g.orig), _distinct2(g.output), _distinct2(g.out_trunc))
        res["len_ratio"] = float(g.output.map(lambda t: len(WORD_RE.findall(t))).sum() / max(1, g.n_orig.sum()))
        rows.append(res)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="parquet with columns: lang, model, mode(scenario), doc_id, orig, output, country")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--no-homog", action="store_true", help="skip L3 (paired homogenisation, needs BGE-M3)")
    ap.add_argument("--homog-only", action="store_true", help="compute only L3")
    a = ap.parse_args()
    df = pd.read_parquet(a.pairs).reset_index(drop=True)
    homog_out = Path(a.out).parent / "audit_homog.csv"
    if not a.no_homog:
        H = homogenisation_paired(df)
        homog_out.parent.mkdir(parents=True, exist_ok=True)
        H.to_csv(homog_out, index=False)
        print(H.groupby(["model", "mode"])[["dcos_full", "dcos_trunc", "d2_orig", "d2_out_trunc"]].mean().round(3).to_string())
        if a.homog_only:
            return
    ner = None if a.no_ner else NER()
    rows = []
    for r in df.itertuples():
        rows.append({"doc_id": r.doc_id, "lang": r.lang, "model": r.model, "mode": r.mode,
                     **pair_metrics(r.orig, r.output, r.lang, r.country, ner)})
    M = pd.DataFrame(rows)
    if not a.no_nli:
        nli = NLI()
        # L5: do the output's sentences entail the original's claims? premise = output, hypothesis = each original sentence
        prem, hyp, idx = [], [], []
        for i, r in enumerate(df.itertuples()):
            for s in sentences(r.orig)[:8]:
                prem.append(r.output[:3000]); hyp.append(s); idx.append(i)
        e = nli.entail(prem, hyp)
        cov = pd.Series(e).groupby(pd.Series(idx)).mean()
        M["l5_claim_coverage"] = cov.reindex(range(len(df))).values
        # reverse direction: original entails output sentences (added content)
        prem, hyp, idx = [], [], []
        for i, r in enumerate(df.itertuples()):
            for s in sentences(r.output)[:8]:
                prem.append(r.orig[:3000]); hyp.append(s); idx.append(i)
        e2 = nli.entail(prem, hyp)
        M["l5_output_grounded"] = pd.Series(e2).groupby(pd.Series(idx)).mean().reindex(range(len(df))).values
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    M.to_parquet(a.out, index=False)
    print(M.groupby(["lang", "model", "mode"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
