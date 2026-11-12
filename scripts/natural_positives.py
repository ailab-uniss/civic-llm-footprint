"""Naturally occurring positives: contributions that carry leftovers of an assistant's reply.

Texts with assistant boilerplate ("Certainly! Here is ...", "as an AI language model", unfilled placeholders such as
"[Your Name]") are near-certain cases of pasted assistant output (cf. the Academ-AI collection for the scholarly
literature). They are few, but they are real post-ChatGPT texts with a known label, written with the assistants actually
in use: the share of them that scores above the 98th percentile of the instance's pre-ChatGPT texts is a field estimate
of the recall of the lower bound. Hits before December 2022 measure the false-positive rate of the patterns.

  python scripts/natural_positives.py results/supervised/study_scores.parquet -> results/natural_positives.csv
"""
import glob
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "src/civic")
from analysis import split_author_type  # noqa: E402

BOILER = [
    r"as an ai (language )?model", r"as a large language model", r"i am an ai\b", r"i'm an ai\b",
    r"(certainly|sure|of course|absolutely)[!,.]? here('s| is| are)", r"here('s| is) (a|an|the|your) (revised|rewritten|improved|polished|draft|"
    r"refined|more formal|concise|possible|sample|suggested)", r"i hope this (helps|message finds)", r"feel free to (adjust|modify|customi[sz]e|edit)",
    r"come modello linguistico", r"in quanto (modello|intelligenza artificiale)", r"(certo|certamente|ecco)[!,.]? ecco", r"ecco (una|la) (versione|bozza|proposta|"
    r"riscrittura)", r"spero (che questo|di esserti|ti sia)", r"sentiti liber[oa] di",
    r"como modelo de lenguaje", r"(claro|por supuesto)[!,.]? aqu[ií] (tienes|est[aá])", r"aqu[ií] tienes (una|la|el|un) ", r"espero que (esto|te) (sea|ayude|sirva)",
    r"en tant que mod[eè]le de langage", r"en tant qu'(ia|intelligence artificielle)", r"(bien s[uû]r|certainement)\s?[!,.]? voici", r"voici une (version|proposition|"
    r"reformulation)", r"j'esp[eè]re que cela (vous|t')", r"n'h[eé]site[zs]? pas [aà] (modifier|adapter|ajuster)",
    r"als ki-(sprach)?modell", r"als k[uü]nstliche intelligenz", r"(gerne|nat[uü]rlich)[!,.]? hier ist", r"hier ist (eine|ein|der|die) ([uü]berarbeitete|verbesserte|"
    r"m[oö]gliche|formellere)", r"ich hoffe,? (das|dies) hilft",
    r"com a model de llenguatge", r"(i tant|és clar)[!,.]? aqu[ií] tens", r"aqu[ií] tens (una|la|el|un) ", r"espero que (aix[oò]|et) (sigui|ajudi)",
    r"como modelo de linguagem", r"(claro|com certeza)[!,.]? aqui est[aá]", r"aqui est[aá] (uma|a|o|um) (vers[aã]o|proposta|rascunho)", r"espero que isso ajude",
    r"som en ai-(spr[aå]k)?modell", r"h[aä]r [aä]r (en|ett) (reviderad|f[oö]rb[aä]ttrad|omskriv)", r"teko[aä]lyn[aä]", r"kielimallina",
]
PLACEHOLDER = [r"\[(your|tu|su|il tuo|votre|ton|ihr|dein|el teu|o seu|seu|ditt) (full )?(name|nombre|nome|nom|namn)\]", r"\[(name|nome|nombre|nom|ort|city|"
               r"citt[aà]|ciudad|ville|date|data|fecha|datum)\]", r"\[insert [^\]]{3,40}\]", r"\[(inserire|inserisci|insertar|ins[eé]rer|einf[uü]gen) [^\]]{3,40}\]"]
RX_B = re.compile("|".join(BOILER), re.I)
RX_P = re.compile("|".join(PLACEHOLDER), re.I)

cols = ["doc_id", "lang", "text", "created_at", "period", "is_dup_exact", "instance", "user_type"]
A = pd.concat([pd.read_parquet(p, columns=[c for c in cols if c in pq.read_schema(p).names])
               for p in sorted(glob.glob("data/processed/*.parquet"))], ignore_index=True)
A["instance"] = split_author_type(A.instance, A["user_type"] if "user_type" in A.columns else None)
A = A[(~A.is_dup_exact) & A.text.notna()].drop_duplicates("doc_id")
S = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "results/supervised/study_scores.parquet")
A = A.merge(S[S.set == "civic_h"][["doc_id", "score"]].drop_duplicates("doc_id"), on="doc_id")
A["boiler"] = A.text.str.contains(RX_B)
A["placeholder"] = A.text.str.contains(RX_P)
A["hit"] = A.boiler | A.placeholder
pre = A[A.period == "pre"]
thr = pre.groupby(["instance", "lang"]).score.quantile(0.98).rename("thr")
n_pre = pre.groupby(["instance", "lang"]).size().rename("n_pre")
A = A.join(thr, on=["instance", "lang"]).join(n_pre, on=["instance", "lang"])
A = A[A.n_pre >= 300]
A["above"] = A.score > A.thr
A["year"] = A.created_at.dt.year
print("texts:", A.groupby("period").size().to_dict())
print("hits per 10,000 texts:", (A.groupby("period").hit.mean() * 1e4).round(2).to_dict(), "| n hits:", A.groupby("period").hit.sum().to_dict())
for k in ("boiler", "placeholder", "hit"):
    h = A[A[k]]
    print(f"\n{k}: share above the instance's human 98th percentile")
    print(h.groupby("period").above.agg(["mean", "size"]).round(3).to_string())
print("\npost-ChatGPT hits by year:"); print(A[A.hit & (A.period == "post")].groupby("year").above.agg(["mean", "size"]).round(2).to_string())
print("\npost-ChatGPT hits by language:"); print(A[A.hit & (A.period == "post")].groupby("lang").above.agg(["mean", "size"]).round(2).to_string())
print("\nall post-ChatGPT texts above threshold:", round(float(A[A.period == "post"].above.mean()), 3))
H = A[A.hit].copy()
H["match"] = H.text.map(lambda t: (RX_B.search(t) or RX_P.search(t)).group(0))
H["snippet"] = H.text.str[:160]
H[["doc_id", "instance", "lang", "year", "period", "boiler", "placeholder", "match", "score", "thr", "above", "snippet"]].to_csv(
    "results/natural_positives.csv", index=False)
print("\nexamples (post, below threshold):")
for _, r in H[(H.period == "post") & ~H.above].head(8).iterrows():
    print(" ", r.lang, r.instance, "|", r.match, "|", r.snippet[:110].replace("\n", " "))
print("\nexamples (pre):")
for _, r in H[H.period == "pre"].head(8).iterrows():
    print(" ", r.lang, r.instance, "|", r.match, "|", r.snippet[:110].replace("\n", " "))
