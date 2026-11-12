"""Pre-processing: language identification, filtering, lemmatisation and POS tagging.

Input : unified parquet from ingest.py / collect_graphql.py
Output: parquet with extra columns
    lang (ISO 639-1 detected), lang_conf, n_words, is_dup_exact, tokens (list of 'lemma|POS'),
    period ('pre' if created before 2022-11-30 else 'post'), quarter (YYYYQn)

Lemmatisation uses spaCy pipelines for the supported languages and falls back to
lower-cased surface forms (with POS='X') for unsupported ones.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lingua import Language, LanguageDetectorBuilder

CHATGPT_RELEASE = pd.Timestamp("2022-11-30", tz="UTC")

SPACY_MODELS = {
    "ca": "ca_core_news_md", "es": "es_core_news_md", "fr": "fr_core_news_md", "it": "it_core_news_md",
    "fi": "fi_core_news_md", "en": "en_core_web_md", "sv": "sv_core_news_md", "de": "de_core_news_md",
    "nl": "nl_core_news_md", "pt": "pt_core_news_md", "el": "el_core_news_md", "ja": "ja_core_news_md",
    "pl": "pl_core_news_md", "ro": "ro_core_news_md", "da": "da_core_news_md", "nb": "nb_core_news_md",
}
LINGUA_LANGS = [Language.CATALAN, Language.SPANISH, Language.FRENCH, Language.ITALIAN, Language.FINNISH,
                Language.ENGLISH, Language.SWEDISH, Language.GERMAN, Language.DUTCH, Language.PORTUGUESE,
                Language.BASQUE, Language.DANISH, Language.POLISH, Language.GREEK,
                Language.JAPANESE, Language.ROMANIAN, Language.ARABIC, Language.RUSSIAN, Language.UKRAINIAN]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]+")
WORD_RE = re.compile(r"\w+", re.UNICODE)
CONTENT_POS = {"ADJ", "ADV", "VERB", "NOUN", "PROPN"}


GID_RE = re.compile(r"gid://[\w\-]+/Decidim::\w+/\d+")


def normalise(text: str) -> str:
    t = GID_RE.sub("@user", text)  # Decidim mention markup
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def n_words(text: str) -> int:
    return len(WORD_RE.findall(text))


class Lemmatiser:
    def __init__(self):
        self.nlps = {}

    def get(self, lang: str):
        import spacy
        if lang in self.nlps:
            return self.nlps[lang]
        model = SPACY_MODELS.get(lang)
        nlp = None
        if model:
            try:
                nlp = spacy.load(model, disable=["ner", "parser"])
                nlp.max_length = 2_000_000
            except OSError:
                nlp = None
        self.nlps[lang] = nlp
        return nlp

    def tokens(self, texts: list[str], lang: str, batch_size: int = 64, n_process: int = 1) -> list[list[str]]:
        nlp = self.get(lang)
        if nlp is None:
            return [[w.lower() + "|X" for w in WORD_RE.findall(t)] for t in texts]
        out = []
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            out.append([f"{tok.lemma_.lower()}|{tok.pos_}" for tok in doc
                        if tok.is_alpha and not tok.is_stop and tok.pos_ in CONTENT_POS and len(tok.lemma_) > 1])
        return out


def detect_languages(texts: list[str]) -> tuple[list[str], list[float]]:
    det = LanguageDetectorBuilder.from_languages(*LINGUA_LANGS).build()
    langs, confs = [], []
    for t in texts:
        res = det.compute_language_confidence_values(t[:2000])
        if res:
            langs.append(res[0].language.iso_code_639_1.name.lower())
            confs.append(float(res[0].value))
        else:
            langs.append("")
            confs.append(0.0)
    return langs, confs


def load_allowlist(cfg_path: str | None, instance: str) -> list[str] | None:
    if not cfg_path:
        return None
    import yaml
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))["instances"]
    ent = cfg.get(instance)
    if not ent or ent.get("langs") in (None, "all"):
        return None
    return [str(x) for x in ent["langs"]]


def detect_languages_full(texts: list[str]) -> list[str]:
    """Language among all the languages of the detector. A text in a language outside LINGUA_LANGS (for instance Czech,
    Slovak or Hungarian feedback on the EU portal) is assigned by the restricted detector to the nearest candidate with full
    confidence; a text is kept only when this second identification agrees with the first."""
    det = LanguageDetectorBuilder.from_all_languages().build()
    res = det.detect_languages_in_parallel_of([t[:2000] for t in texts])
    return [r.iso_code_639_1.name.lower() if r is not None else "" for r in res]


def run(inp: Path, out: Path, min_words: int = 30, min_conf: float = 0.9, n_process: int = 4,
        instances_cfg: str | None = None):
    df = pd.read_parquet(inp)
    if "doc_id" not in df.columns:
        df["doc_id"] = df["instance"] + ":" + df["type"] + ":" + df["id"].astype(str) + ":" + df["lang_declared"]
    allow = load_allowlist(instances_cfg, inp.stem) if len(df) else None
    if allow:
        # proposals: keep only official-language columns (platform machine translations are dropped);
        # comments carry the author's locale, keep them all (language is verified by LID below)
        df = df[(df.type != "proposal") | (df.lang_declared.isin(allow)) | (df.lang_declared == "")]
    df["text"] = df["text"].map(normalise)
    df["n_words"] = df["text"].map(n_words)
    df = df[df["n_words"] >= min_words].copy()
    if df.empty:
        print(f"{inp.name}: no text >= {min_words} words")
        df.to_parquet(out, index=False)
        return
    langs, confs = detect_languages(df["text"].tolist())
    df["lang"], df["lang_conf"] = langs, confs
    # keep high-confidence; also accept when detector agrees with declared language at lower confidence
    keep = (df["lang_conf"] >= min_conf) | ((df["lang"] == df["lang_declared"]) & (df["lang_conf"] >= 0.6))
    df = df[keep & (df["lang"] != "")].copy()
    df = df[np.array(detect_languages_full(df["text"].tolist())) == df["lang"].values].copy()
    df["text_hash"] = df["text"].map(lambda s: hashlib.md5(s.lower().encode()).hexdigest())
    df["is_dup_exact"] = df.duplicated("text_hash", keep="first")
    df["period"] = (df["created_at"] >= CHATGPT_RELEASE).map({True: "post", False: "pre"})
    df["quarter"] = df["created_at"].dt.to_period("Q").astype(str)
    df["year"] = df["created_at"].dt.year
    lem = Lemmatiser()
    toks = pd.Series([None] * len(df), index=df.index, dtype=object)
    for lang, g in df.groupby("lang"):
        t = lem.tokens(g["text"].tolist(), lang, n_process=n_process)
        toks.loc[g.index] = pd.Series(t, index=g.index, dtype=object)
    df["tokens"] = toks
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{inp.name}: {len(df)} rows kept")
    print(df.groupby(["lang", "type", "period"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-words", type=int, default=30)
    ap.add_argument("--n-process", type=int, default=4)
    ap.add_argument("--instances", default="configs/instances.yaml")
    a = ap.parse_args()
    run(Path(a.inp), Path(a.out), a.min_words, n_process=a.n_process, instances_cfg=a.instances)
