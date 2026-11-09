"""Per-document AI-text detectors used as baselines (aggregated at corpus level by the estimator module).

- binoculars : Hans et al. (ICML 2024). score = log-PPL(observer) / cross-PPL(observer, performer).
               Lower = more likely machine. We store ai_score = -score.
- fastdetect : Fast-DetectGPT (Bao et al., ICLR 2024), analytic conditional probability curvature with
               a single scoring model. Higher = more likely machine.
- pangram    : Open Pangram EditLens (pangram/editlens_Llama-3.2-3B), a sequence classifier/regressor;
               score in [0,1], 1 = fully AI.

Usage: python detectors.py --detector binoculars --inp texts.parquet --out scores.parquet [--text-col text]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

OBSERVER = "Qwen/Qwen2.5-1.5B"
PERFORMER = "Qwen/Qwen2.5-1.5B-Instruct"
PANGRAM = "pangram/editlens_Llama-3.2-3B"
PANGRAM_BASE = "unsloth/Llama-3.2-3B"  # ungated mirror of meta-llama/Llama-3.2-3B (same weights)


def _load(name, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map="cuda").eval()
    return tok, m


@torch.no_grad()
def binoculars_scores(texts: list[str], batch: int = 16, max_len: int = 512) -> np.ndarray:
    tok, obs = _load(OBSERVER)
    _, perf = _load(PERFORMER)
    out = []
    for i in tqdm(range(0, len(texts), batch), desc="binoculars"):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
        lo = obs(**enc).logits[:, :-1].float()
        lp = perf(**enc).logits[:, :-1].float()
        tgt = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].float()
        # log-ppl of observer on the text
        logp_o = torch.log_softmax(lo, -1)
        nll = -logp_o.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        ppl = (nll * mask).sum(1) / mask.sum(1)
        # cross-ppl: expected observer NLL under performer's next-token distribution
        p_perf = torch.softmax(lp, -1)
        xent = -(p_perf * logp_o).sum(-1)
        xppl = (xent * mask).sum(1) / mask.sum(1)
        out.append((ppl / xppl).cpu().numpy())
    b = np.concatenate(out)
    return -b  # higher = more AI-like


@torch.no_grad()
def fastdetect_scores(texts: list[str], batch: int = 16, max_len: int = 512) -> np.ndarray:
    tok, m = _load(OBSERVER)
    out = []
    for i in tqdm(range(0, len(texts), batch), desc="fast-detectgpt"):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
        logits = m(**enc).logits[:, :-1].float()
        tgt = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].float()
        logp = torch.log_softmax(logits, -1)
        p = logp.exp()
        lp_x = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mu = (p * logp).sum(-1)                      # E[log p(x~)]
        var = (p * logp**2).sum(-1) - mu**2          # Var[log p(x~)]
        num = ((lp_x - mu) * mask).sum(1)
        den = torch.sqrt((var * mask).sum(1)).clamp_min(1e-6)
        out.append((num / den).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def pangram_scores(texts: list[str], batch: int = 8, max_len: int = 1024) -> np.ndarray:
    """EditLens (Thai et al., 2025) is distributed as a LoRA adapter on Llama-3.2-3B whose classification head
    is a LayerNorm followed by a bias-free linear layer over n_buckets ordinal classes (bucket 0 = human,
    last bucket = fully AI); the ungated mirror of the base weights is used. As in the authors' inference
    script, the score is the expected bucket index normalised to [0, 1]."""
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class NormedLinear(torch.nn.Module):  # EditLens head (scripts/train.py of github.com/pangramlabs/EditLens)
        def __init__(self, hidden, n, device=None, dtype=None):
            super().__init__()
            self.norm = torch.nn.LayerNorm(hidden, device=device, dtype=dtype)
            self.linear = torch.nn.Linear(hidden, n, bias=False, device=device, dtype=dtype)

        def forward(self, x):
            return self.linear(self.norm(x))

    ad = hf_hub_download(PANGRAM, "adapter_model.safetensors")
    with safe_open(ad, "pt") as f:
        score_keys = [k for k in f.keys() if "score" in k and k.endswith("linear.weight")]
        n_labels = f.get_slice(score_keys[0]).get_shape()[0]
    # the adapter's tokenizer_config was written by transformers 5 ("TokenizersBackend"); the base tokenizer is identical
    tok = AutoTokenizer.from_pretrained(PANGRAM_BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForSequenceClassification.from_pretrained(PANGRAM_BASE, num_labels=n_labels, torch_dtype=torch.bfloat16,
                                                              device_map="cuda")
    base.config.pad_token_id = tok.pad_token_id
    base.score = NormedLinear(base.config.hidden_size, n_labels, device=base.score.weight.device, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(base, PANGRAM).eval()
    print(f"pangram: {n_labels} bucket(s)")
    buckets = torch.arange(n_labels, dtype=torch.float32, device="cuda")
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch), desc="pangram"):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
            probs = torch.softmax(m(**enc).logits.float(), -1)
            s = probs[:, 0] if n_labels == 1 else (probs @ buckets) / (n_labels - 1)
            out.append(s.cpu().numpy())
    return np.concatenate(out)


DETECTORS = {"binoculars": binoculars_scores, "fastdetect": fastdetect_scores, "pangram": pangram_scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=DETECTORS, required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--id-cols", nargs="+", default=["doc_id"])
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    df = pd.read_parquet(a.inp)
    prev = None
    if Path(a.out).exists():  # resumable: keep scores of texts already seen, score only the new ones
        prev = pd.read_parquet(a.out)
        df = df[~df[a.id_cols[0]].isin(prev[a.id_cols[0]])]
        print(f"{len(prev)} already scored, {len(df)} new")
    res = df[[c for c in a.id_cols if c in df.columns]].copy()
    if len(df):
        texts = df[a.text_col].fillna("").astype(str).tolist()
        res[f"score_{a.detector}"] = DETECTORS[a.detector](texts, batch=a.batch)
    if prev is not None:
        res = pd.concat([prev, res], ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out, index=False)
    print(f"{a.detector}: {len(res)} scored -> {a.out}")


if __name__ == "__main__":
    main()
