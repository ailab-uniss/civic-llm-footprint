"""Subset of RAID (Dugan et al., ACL 2024; MIT licence) for the out-of-domain test of the lower bound on the assistants
of 2023: English Reddit posts and reviews, human texts and texts of ChatGPT (gpt-3.5-turbo-0613), GPT-4 (gpt-4-0613) and
GPT-3 (text-davinci-002), without adversarial attacks. The 11 GB training file is streamed and filtered, not stored.

  python scripts/raid_subset.py -> data/external/raid/raid_subset.parquet
"""
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
out = Path("data/external/raid"); out.mkdir(parents=True, exist_ok=True)
keep, n = [], 0
for ch in pd.read_csv(URL, chunksize=50000):
    n += len(ch)
    x = ch[(ch.attack == "none") & ch.domain.isin(["reddit", "reviews"]) & ch.model.isin(["human", "chatgpt", "gpt4", "gpt3"])]
    if len(x):
        keep.append(x[["id", "source_id", "model", "decoding", "repetition_penalty", "domain", "title", "generation"]])
    if n % 500000 == 0:
        print(n, "rows read,", sum(len(k) for k in keep), "kept", flush=True)
R = pd.concat(keep, ignore_index=True)
R.to_parquet(out / "raid_subset.parquet", index=False)
print(R.groupby(["domain", "model", "decoding"], dropna=False).size().to_string())
print("RAID_DONE")
