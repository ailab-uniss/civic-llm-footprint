"""Generate LLM reference texts with vLLM, simulating how citizens use LLMs on participation platforms.

Modes: rewrite (improve a human text), generate (write from a title/topic), translate (translate a
human text from another language into the target language and polish it).

Usage:
  python generate.py sample  --processed data/processed --out data/generated --n 2000 --langs ca es fr it fi en
  python generate.py run     --model Qwen/Qwen2.5-7B-Instruct --tag qwen2.5-7b --sample data/generated --out data/generated
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import yaml

CHATGPT_RELEASE = pd.Timestamp("2022-11-30", tz="UTC")
MODES = ["rewrite", "generate", "translate"]
PLATFORM_NAMES = {"helsinki": "OmaStadi", "barcelona": "Decidim Barcelona", "nyc": "Participate NYC",
                  "gencat": "participa.gencat.cat", "lausanne": "participer.lausanne.ch",
                  "eu_hys": "Have Your Say", "regulations_gov": "Regulations.gov"}


def prompt_family(instance: str) -> str:
    """Consultation portals (EU Have Your Say, Regulations.gov) use the consultation prompt variants
    (`rewrite_consult`, `generate_consult`); participation platforms use the municipal ones."""
    if instance.startswith("multitude"):   # news articles (MULTITuDE temporal-transfer test): `rewrite_news`, `generate_news`
        return "_news"
    return "_consult" if instance.startswith(("eu_hys", "regulations_gov")) else ""


def make_notes(text: str, max_points: int = 5, max_words: int = 14) -> str:
    """Key points a citizen might give an assistant: the first clause of up to max_points sentences."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 4]
    pts = []
    for s in sents[:max_points]:
        w = re.split(r"[,;:]", s)[0].split()
        pts.append("- " + " ".join(w[:max_words]))
    return "\n".join(pts) if pts else "- " + " ".join(text.split()[:max_words])


def first_sentence(text: str, max_words: int = 25) -> str:
    s = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    w = s.split()
    return " ".join(w[:max_words]) + ("…" if len(w) > max_words else "")


def make_sample(processed_dir: Path, out_dir: Path, n: int, langs: list[str], seed: int = 0,
                min_words: int = 30, max_words: int = 400):
    """Draw up to n human pre-ChatGPT texts per language (stratified over instances and types)."""
    rng = random.Random(seed)
    frames = []
    for p in sorted(processed_dir.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["instance", "lang", "type", "id", "created_at", "text", "title",
                                         "n_words", "is_dup_exact", "period"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.period == "pre") & (~df.is_dup_exact) & (df.n_words >= min_words) & (df.n_words <= max_words)
            & (df.lang.isin(langs))]
    out = []
    for lang, g in df.groupby("lang"):
        # proportional allocation over (instance, type) with a floor, then random within strata
        strata = g.groupby(["instance", "type"]).size()
        total = min(n, len(g))
        alloc = (strata / strata.sum() * total).round().astype(int).clip(lower=min(50, total))
        rows = []
        for (inst, typ), k in alloc.items():
            gg = g[(g.instance == inst) & (g.type == typ)]
            idx = list(gg.index)
            rng.shuffle(idx)
            rows.extend(idx[:k])
        rows = rows[:total]
        out.append(g.loc[rows])
    sample = pd.concat(out, ignore_index=True)
    if "doc_id" not in sample.columns:   # the seed keeps the identifier of its processed text: generated texts inherit its fold
        sample["doc_id"] = sample["instance"] + ":" + sample["type"] + ":" + sample["id"].astype(str) + ":" + sample["lang"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_dir / "human_sample.parquet", index=False)
    print(sample.groupby(["lang", "instance", "type"]).size().to_string())
    return sample


def build_prompts(sample: pd.DataFrame, prompts: dict, modes: list[str], seed: int = 0,
                  n_translate_frac: float = 1.0) -> list[dict]:
    rng = random.Random(seed)
    jobs = []
    by_lang = {l: g for l, g in sample.groupby("lang")}
    for lang, g in by_lang.items():
        if lang not in prompts:
            continue
        P = prompts[lang]
        others = [l for l in by_lang if l != lang and l in prompts]
        for _, r in g.iterrows():
            platform = PLATFORM_NAMES.get(r.instance, PLATFORM_NAMES.get(r.instance.rsplit("_", 1)[0], r.instance))
            fam = prompt_family(r.instance) if "rewrite" + prompt_family(r.instance) in P else ""
            title = r.title if isinstance(r.title, str) and len(r.title.split()) >= 3 else first_sentence(r.text)
            n_words = int(max(40, min(300, r.n_words)))
            for mode in modes:
                if mode == "translate":
                    if not others or rng.random() > n_translate_frac:
                        continue
                    src_lang = rng.choice(others)
                    src = by_lang[src_lang].sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
                    v = rng.randrange(len(P["translate"]))
                    prompt = P["translate"][v].format(text=src.text, src_lang_name=prompts[src_lang]["lang_name"])
                    jobs.append(dict(lang=lang, mode=mode, variant=v, doc_id=r.doc_id, src_doc_id=src.doc_id,
                                     prompt=prompt))
                elif mode == "rewrite":
                    v = rng.randrange(len(P["rewrite" + fam]))
                    jobs.append(dict(lang=lang, mode=mode, variant=v, doc_id=r.doc_id, src_doc_id=r.doc_id,
                                     prompt=P["rewrite" + fam][v].format(text=r.text, platform=platform)))
                elif mode == "generate":
                    v = rng.randrange(len(P["generate" + fam]))
                    jobs.append(dict(lang=lang, mode=mode, variant=v, doc_id=r.doc_id, src_doc_id=r.doc_id,
                                     prompt=P["generate" + fam][v].format(title=title, platform=platform, n_words=n_words)))
                else:  # custom scenario prompts (audit): key = mode name in yaml
                    v = rng.randrange(len(P[mode]))
                    jobs.append(dict(lang=lang, mode=mode, variant=v, doc_id=r.doc_id, src_doc_id=r.doc_id,
                                     prompt=P[mode][v].format(text=r.text, title=title, platform=platform,
                                                              n_words=n_words, notes=make_notes(r.text),
                                                              city=platform)))
    return jobs


def run_model(model: str, tag: str, jobs: list[dict], out_path: Path, max_model_len: int = 4096,
              temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 900, gpu_util: float = 0.90,
              batch: int = 512, dtype: str = "auto", seed: int = 0, quantization: str | None = None,
              enforce_eager: bool = False, max_num_seqs: int = 128):
    from vllm import LLM, SamplingParams
    kw = {}
    if quantization:
        kw["quantization"] = quantization
    llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=gpu_util,
              seed=seed, enable_prefix_caching=True, trust_remote_code=True, enforce_eager=enforce_eager,
              max_num_seqs=max_num_seqs, **kw)
    is_gptoss = "gpt-oss" in model.lower()
    chat_kw = {"chat_template_kwargs": {"reasoning_effort": "low"}} if is_gptoss else {}
    if is_gptoss:
        max_tokens = max(max_tokens, 1400)  # reasoning channel + final answer
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)

    def final_text(t: str) -> str:
        # gpt-oss (harmony format): "analysis...assistantfinal<answer>" once special tokens are skipped
        if is_gptoss:
            if "assistantfinal" in t:
                return t.rsplit("assistantfinal", 1)[1].strip()
            return ""  # reasoning did not finish: discard
        return t
    done = set()
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        done = set(zip(prev.doc_id, prev["mode"], prev.variant))  # prev.mode is DataFrame.mode
        rows = prev.to_dict("records")
    else:
        rows = []
    todo = [j for j in jobs if (j["doc_id"], j["mode"], j["variant"]) not in done]
    print(f"{tag}: {len(todo)} prompts to run ({len(done)} already done)")
    t0 = time.time()
    n_tok = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        msgs = [[{"role": "user", "content": j["prompt"]}] for j in chunk]
        outs = llm.chat(msgs, sp, use_tqdm=False, **chat_kw)
        for j, o in zip(chunk, outs):
            text = final_text(o.outputs[0].text)
            n_tok += len(o.outputs[0].token_ids)
            rows.append(dict(model=tag, **j, output=text, n_tokens=len(o.outputs[0].token_ids),
                             finish=o.outputs[0].finish_reason))
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        el = time.time() - t0
        print(f"  {i + len(chunk)}/{len(todo)}  {n_tok / el:.0f} tok/s  elapsed {el / 60:.1f} min", flush=True)
    print(f"{tag}: done in {(time.time() - t0) / 60:.1f} min")


def run_api(model: str, tag: str, jobs: list[dict], out_path: Path, provider: str = "openai",
            temperature: float = 0.7, max_tokens: int = 900, workers: int = 8, limit_per_lang: int = 0):
    """Commercial generator through an OpenAI-compatible chat API (OpenAI, OpenRouter, Anthropic or Gemini key).
    Used, when budget allows, as an additional *held-out* generator in the benchmark."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    # every provider is reached through its OpenAI-compatible chat endpoint
    bases = {"openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
             "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/"),
             "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/")}
    if provider in bases:
        key, base = bases[provider]
        client = OpenAI(api_key=os.environ[key], base_url=base)
    else:
        client = OpenAI()
    tok_kw = "max_completion_tokens" if provider == "openai" else "max_tokens"
    rows, done = [], set()
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        prev = prev[~prev.finish.fillna("").str.startswith("error")]  # failed calls (rate limits) are retried
        rows = prev.to_dict("records")
        done = set(zip(prev.doc_id, prev["mode"], prev.variant))
    todo = [j for j in jobs if (j["doc_id"], j["mode"], j["variant"]) not in done]
    if limit_per_lang:
        # the limit counts outputs already present (resumed runs), so a (lang, mode) never exceeds it in total
        cnt = {k: int(v) for k, v in prev.groupby(["lang", "mode"]).size().items()} if done else {}
        keep = []
        for j in todo:
            k = (j["lang"], j["mode"])
            if cnt.get(k, 0) < limit_per_lang:
                keep.append(j)
                cnt[k] = cnt.get(k, 0) + 1
        todo = keep
    print(f"{tag}: {len(todo)} API calls")

    use_temp = {"v": True}

    def call(j):
        err = ""
        for att in range(5):
            try:
                kw = {"temperature": temperature} if use_temp["v"] else {}
                kw[tok_kw] = max_tokens
                if provider == "gemini":  # no hidden reasoning: the token budget is for the visible answer
                    kw["extra_body"] = {"reasoning_effort": "none"}
                elif provider == "anthropic":
                    kw["extra_body"] = {"thinking": {"type": "disabled"}}
                r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": j["prompt"]}], **kw)
                return dict(model=tag, **j, output=r.choices[0].message.content or "",
                            n_tokens=r.usage.completion_tokens if r.usage else None, finish=r.choices[0].finish_reason)
            except Exception as e:  # noqa: BLE001
                err = str(e)
                if "temperature" in err and use_temp["v"]:
                    use_temp["v"] = False  # reasoning models accept only the default temperature
                    continue
                time.sleep(2 * (att + 1))
        return dict(model=tag, **j, output="", n_tokens=None, finish=f"error:{err[:80]}")

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(call, j) for j in todo]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if i % 100 == 0:
                pd.DataFrame(rows).to_parquet(out_path, index=False)
                print(f"  {i}/{len(todo)}", flush=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"{tag}: done")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--processed", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--langs", nargs="+", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--sample", required=True, help="parquet with human sample (doc_id, lang, text, title, ...)")
    r.add_argument("--prompts", default="configs/prompts.yaml")
    r.add_argument("--modes", nargs="+", default=MODES)
    r.add_argument("--out", required=True)
    r.add_argument("--max-model-len", type=int, default=4096)
    r.add_argument("--max-tokens", type=int, default=900)
    r.add_argument("--gpu-util", type=float, default=0.90)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--dtype", default="auto")
    r.add_argument("--backend", default="vllm", choices=["vllm", "openai", "openrouter", "anthropic", "gemini"])
    r.add_argument("--quantization", default=None, help="vLLM quantization, e.g. bitsandbytes for bnb-4bit checkpoints")
    r.add_argument("--enforce-eager", action="store_true", help="skip CUDA graphs (saves memory)")
    r.add_argument("--max-num-seqs", type=int, default=128)
    r.add_argument("--limit-per-lang", type=int, default=0, help="API backend: max prompts per (lang, mode)")
    a = ap.parse_args()
    if a.cmd == "sample":
        make_sample(Path(a.processed), Path(a.out), a.n, a.langs)
    else:
        sample = pd.read_parquet(a.sample)
        prompts = yaml.safe_load(open(a.prompts, encoding="utf-8"))
        jobs = build_prompts(sample, prompts, a.modes)
        if a.limit:  # stratified: an equal share per language (jobs are ordered language, seed, mode)
            by_lang: dict[str, list] = {}
            for j in jobs:
                by_lang.setdefault(j["lang"], []).append(j)
            k = max(1, a.limit // len(by_lang))
            jobs = [j for L in by_lang.values() for j in L[:k]]
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if a.backend == "vllm":
            run_model(a.model, a.tag, jobs, out, a.max_model_len, max_tokens=a.max_tokens, gpu_util=a.gpu_util,
                      dtype=a.dtype, quantization=a.quantization, enforce_eager=a.enforce_eager,
                      max_num_seqs=a.max_num_seqs)
        else:
            run_api(a.model, a.tag, jobs, out, provider=a.backend, max_tokens=a.max_tokens,
                    limit_per_lang=a.limit_per_lang)


if __name__ == "__main__":
    main()
