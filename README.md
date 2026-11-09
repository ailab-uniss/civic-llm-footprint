# civic-llm-footprint

Code, benchmark and monitoring tool for the paper *Large Language Models in Civic Participation: Multilingual
Corpus-Level Measurement and an Audit of Assisted Contributions* (submitted to IEEE Access).

The framework estimates, at the **corpus level**, the share α of texts written or modified with large language models
in citizen contributions to participation platforms (Decidim open data), separately for each language, using only
open models that run on a single consumer GPU. It never classifies individual texts or authors.

## Layout

| Path | Content |
|---|---|
| `src/civic/ingest.py` | Decidim open-data ZIP → unified parquet (pseudonymised authors) |
| `src/civic/collect_graphql.py` | Fallback collector through the public GraphQL API |
| `src/civic/preprocess.py` | Cleaning, language ID (lingua), lemma+POS tokens (spaCy) |
| `src/civic/generate.py` | LLM reference texts with vLLM (rewrite / generate / translate; audit scenarios); optional API backend |
| `src/civic/estimator.py` | Distributional estimator (mixture MLE, vocabulary selection, bootstrap CIs); Rogan–Gladen and Bayesian detector corrections |
| `src/civic/detectors.py` | Binoculars, Fast-DetectGPT, Open Pangram (EditLens) per-document scores |
| `src/civic/benchmark.py` | Synthetic mixtures with known α, leave-one-model-out, stress scenarios, baselines |
| `src/civic/campaigns.py` | Coordinated-campaign detection (MinHash/LSH + BGE-M3 + union-find) |
| `src/civic/analysis.py` | Cross-country study: α per cell, ITS, mixed models, placebo, homogenisation |
| `src/civic/audit.py` | Failure-mode metrics L1–L7 on paired original/assisted texts |
| `src/civic/tool.py` | `analyze --url https://<decidim-instance>` → HTML report for administrations |
| `src/civic/figures.py` | Publication figures |
| `configs/` | Prompts (9 languages), audit scenarios, instance registry |
| `scripts/` | Server pipeline (`run_pipeline.sh`, `run_audit.sh`), numbers/tables generation |

## Quick start

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install vllm==0.11.0 "transformers<5" pandas pyarrow datasketch sentence-transformers faiss-cpu \
    statsmodels spacy lingua-language-detector scikit-learn pyyaml requests matplotlib
python -m spacy download fr_core_news_md   # one model per language used
# 1. data
python src/civic/ingest.py --zip export.zip --instance mycity --out data/interim/mycity.parquet
python src/civic/preprocess.py --inp data/interim/mycity.parquet --out data/processed/mycity.parquet
# 2. reference distributions (pre-2023 human seeds + open-model rewrites)
python src/civic/generate.py sample --processed data/processed --out data/generated --n 1500 --langs fr
python src/civic/generate.py run --model Qwen/Qwen2.5-7B-Instruct --tag qwen2.5-7b --sample data/generated/human_sample.parquet --out data/generated/gen_qwen2.5-7b.parquet
# 3. estimate
python src/civic/tool.py build-refs && python src/civic/tool.py analyze --zip export.zip --instance mycity --out report/
```

## Ethics

Estimates are population-level and carry bootstrap confidence intervals. They must not be used to judge individual
contributions or participants (see the paper's guidelines). Author identifiers are hashed with a salt at import time.

## Licence

Code: Apache-2.0. Benchmark metadata: CC-BY-4.0 (texts remain under the licence of the originating platform).
