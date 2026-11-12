"""Build the Zenodo dataset archive (no texts, no per-text scores of platform contributions): benchmark index and
mixtures, per-cell estimates of the study, comparison methods, audit summaries and the reference files of the tool.

  python scripts/build_zenodo.py --out release/civic-llm-footprint-data [--with-model]

Run where the corrected data are (the server): it needs data/generated/{human_sample,generated_processed}.parquet,
results/ (supervised, robust, multitude, intensity, analysis_cls5, excess, audit files), results/study_cls (written by
scripts/make_cls.py) and models/refs (written by `tool.py build-refs`). `--with-model` also copies models/xlmr_civic.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

README = """# civic-llm-footprint: benchmark, estimates and classifier

Data accompanying the paper "LLM-Assisted Writing in Civic Participation: Multilingual Corpus-Level Estimation Across Platforms" (submitted to IEEE Access).
Code: https://github.com/ailab-uniss/civic-llm-footprint. DOI of this record: 10.5281/zenodo.24091498. Licence: CC-BY-4.0.

No text of a platform contribution is redistributed, and no score of a single contribution written after November 2022
is included: the estimates describe corpus slices and must not be used to judge individual contributions or
participants. Human texts remain available from the open-data endpoints of the originating platforms (Decidim and
CONSUL exports, European Commission "Have Your Say", Regulations.gov) under their licences, and can be retrieved with the
identifiers below and the collectors of the code repository. Generated texts can be regenerated with the prompts in
configs/prompts.yaml.

## benchmark/
- human_seeds.csv: pre-ChatGPT human texts used as seeds of the generated texts. instance, lang, type, id (identifier on
  the platform), doc_id (internal identifier), created_at, n_words, fold (0-3: cross-fitting fold, md5("cal|" + doc_id)
  mod 4; a generated text has the fold of its seed).
- generated_index.csv: one row per generated text. model, lang, mode (rewrite / generate / translate), variant (prompt
  variant), doc_id (identifier of the seed), n_words, fold.
- mixtures.parquet: every mixture of the benchmark with the generator (or the vendor group) held out, and its estimates.
  exp (main = language protocol, instance, period = forward protocol), lang, instance, heldout_model, alpha (true share),
  rep, n (texts), method, alpha_hat, ci_lo, ci_hi (bootstrap), own (calibration on the instance's own texts), lb, lb_lo
  (generator-free lower bound and its one-sided 95% limit), gmin, gmax (single-generator range).
  Methods (suffix _lang = language calibration, _inst = instance calibration): hdyv (study estimator: HDy, vendor
  weights with the ChatGPT share split over three OpenAI models), hdyx (equal weights, all generators), hdyw (vendor
  weights, current models only), hdyc (equal weights, commercial generators), acc_sp (adjusted count at 2% false
  positives), ms (median sweep), em (EM prior shift), cc (classify and count), lb (lower bound), bbe (positive-unlabelled
  mixture proportion).
- mixtures_heldout_mode_prompt.parquet: the same with one mode or one prompt variant held out for all generators.
- calibration_coefs.csv, calibration_coverage.csv: coefficients of the interval calibrated on forward-validation errors
  (quantile regression of the error on the estimate, own / pooled calibration) and its coverage with each generator left
  out of the fit.
- external/multitude_mixtures.parquet: MULTITuDE news (assistants of 2023 and generators of the study held out);
  external/raid_lower_bound.csv: recovered fraction of the lower bound on RAID; external/intensity.csv: recovered
  fraction by editing intensity.
- comparison/: lexical distributional estimator under the instance and forward protocols (lexical_instance_forward.parquet)
  and at the language level together with the detectors with Rogan-Gladen and Bayesian corrections
  (lexical_detectors_language.parquet, with summary), Binoculars score with the HDy quantifier (binoculars_hdy.parquet).

## estimates/
Estimated share of texts written or rewritten with an LLM (alpha), per cell, with band_lo and band_hi (interval
calibrated on forward-validation errors), lb_lo (one-sided 95% limit of the generator-free lower bound; 0 for the cells
calibrated on the language pool, for which the bound is not computed), ci_lo and ci_hi (bootstrap, sampling error only),
alpha_gmin and alpha_gmax (range over single commercial generators), pooled_ref (calibration on the language pool),
template_share and campaign_share:
- cells_period.csv (instance x language x type x pre/post), cells_instance_year.csv, cells_half.csv, cells_consultation.csv
  (one publication, docket or participatory process), and the *_notemplate.csv versions without templated letters.
- null_calibration.csv: estimates on pre-ChatGPT texts excluded from the calibration (false-positive floors).
- alpha_template.csv, alpha_campaign.csv: inside and outside templated letters and near-duplicate campaigns.
- alpha_country.csv: citizens' feedback on Have Your Say by country and language.
- alpha_single_generator.csv, alpha_pooled_vs_adapted.csv, equal_weights/: sensitivity to the reference and to the
  level of the calibration.
- its.csv, mixed_model.txt, placebo.txt: interrupted time series, mixed model, placebo break.
- citizens_organisations.csv, classifier_vs_lexical.csv, homogenisation.csv, high_lemmas.csv, process_counts.csv,
  processed_stats.csv.
- benchmark_pairs.csv: forward benchmark by instance-language pair (coverage of the interval, estimate at alpha = 0,
  mean absolute error; Salamandra excluded).
- lexical/: cells of the lexical distributional estimator (comparison); excess/: excess-vocabulary indicator.

## audit/
- audit_metrics.csv: failure-mode metrics per original/output pair (no texts): doc_id, lang, model, mode (scenario),
  L1 legal references, L6 foreign references, L4 length ratios, L2 numbers and named entities kept / added, L5 claim
  coverage and groundedness.
- audit_homog.csv (L3), audit_l7.csv (language of the output), audit_mixed.csv (mixed-effects models).

## models/
- refs/: reference files of the monitoring tool. human_scores.parquet (lang, year, score: cross-fitted scores of a sample
  of pre-ChatGPT texts, without identifiers), generated_scores.parquet (lang, model, mode, score), calibration_coefs.csv.
- xlmr_civic/ (separate file xlmr_civic.tar of the record): XLM-RoBERTa-large fine-tuned on all folds, used by
  `python src/civic/tool.py analyze`.
"""


def fold4(i: str) -> int:
    return int(hashlib.md5(("cal|" + str(i)).encode()).hexdigest(), 16) % 4


def cp(src: Path, dst: Path, missing: list):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    else:
        missing.append(str(src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release/civic-llm-footprint-data")
    ap.add_argument("--analysis", default="results/analysis_cls5")
    ap.add_argument("--with-model", action="store_true")
    a = ap.parse_args()
    O, R, A, S = Path(a.out), Path("results"), Path(a.analysis), Path("results/supervised")
    if O.exists():
        shutil.rmtree(O)
    O.mkdir(parents=True)
    miss: list = []

    # ------------------------------------------------------------------ benchmark
    H = pd.read_parquet("data/generated/human_sample.parquet")
    H = H[[c for c in ("instance", "lang", "type", "id", "doc_id", "created_at", "n_words") if c in H]].copy()
    H["fold"] = H.doc_id.map(fold4)
    (O / "benchmark").mkdir()
    H.to_csv(O / "benchmark/human_seeds.csv", index=False)
    G = pd.read_parquet("data/generated/generated_processed.parquet")
    G = G[[c for c in ("model", "lang", "mode", "variant", "doc_id", "n_words") if c in G]].copy()
    G["fold"] = G.doc_id.map(fold4)
    G.to_csv(O / "benchmark/generated_index.csv", index=False)
    for src, dst in [(S / "quant_aug.parquet", "benchmark/mixtures.parquet"),
                     (S / "quant_hold.parquet", "benchmark/mixtures_heldout_mode_prompt.parquet"),
                     (S / "calibration_coefs.csv", "benchmark/calibration_coefs.csv"),
                     (S / "calibration_coverage.csv", "benchmark/calibration_coverage.csv"),
                     (R / "multitude/cls_benchmark_aug.parquet", "benchmark/external/multitude_mixtures.parquet"),
                     (R / "raid_lb.csv", "benchmark/external/raid_lower_bound.csv"),
                     (R / "intensity/cls_intensity.csv", "benchmark/external/intensity.csv"),
                     (R / "robust/robust.parquet", "benchmark/comparison/lexical_instance_forward.parquet"),
                     (R / "robust/robust.summary.csv", "benchmark/comparison/lexical_instance_forward.summary.csv"),
                     (R / "benchmark.parquet", "benchmark/comparison/lexical_detectors_language.parquet"),
                     (R / "benchmark.summary.csv", "benchmark/comparison/lexical_detectors_language.summary.csv"),
                     (S / "quant_binoculars.parquet", "benchmark/comparison/binoculars_hdy.parquet")]:
        cp(src, O / dst, miss)

    # ------------------------------------------------------------------ estimates of the study
    E = O / "estimates"
    for f in sorted((R / "study_cls").glob("*.csv")):       # cells with the calibrated interval (scripts/make_cls.py)
        cp(f, E / f.name, miss)
    for f in ["null_calibration.csv", "alpha_template.csv", "alpha_campaign.csv", "alpha_country.csv", "alpha_single_generator.csv",
              "alpha_pooled_vs_adapted.csv", "its.csv", "mixed_model.txt", "placebo.txt", "homogenisation.csv", "high_lemmas.csv",
              "process_counts.csv", "alpha_cells_year_consult.csv"]:
        cp(A / f, E / f, miss)
    for f in sorted(Path(str(A) + "_equal").glob("alpha_cells_*.csv")):
        cp(f, E / "equal_weights" / f.name, miss)
    cp(R / "processed_stats.csv", E / "processed_stats.csv", miss)
    for f in ["alpha_cells_period.csv", "alpha_cells_instance_year.csv", "alpha_cells_half.csv", "null_calibration.csv"]:
        cp(R / f, E / "lexical" / f, miss)
    for f in ["excess.csv", "excess_words.csv"]:
        cp(R / "excess" / f, E / "excess" / f, miss)

    # ------------------------------------------------------------------ audit
    if (R / "audit_metrics.parquet").exists():
        (O / "audit").mkdir(exist_ok=True)
        pd.read_parquet(R / "audit_metrics.parquet").to_csv(O / "audit/audit_metrics.csv", index=False)
    else:
        miss.append("results/audit_metrics.parquet")
    for f in ["audit_homog.csv", "audit_l7.csv", "audit_mixed.csv"]:
        cp(R / f, O / "audit" / f, miss)

    # ------------------------------------------------------------------ tool references and classifier
    for f in sorted(Path("models/refs").glob("*")):
        cp(f, O / "models/refs" / f.name, miss)
    if a.with_model:
        shutil.copytree("models/xlmr_civic", O / "models/xlmr_civic")
    (O / "README.md").write_text(README, encoding="utf-8")
    for m in miss:
        print("missing", m)
    print("built", O, sum(p.stat().st_size for p in O.rglob("*") if p.is_file()) // 2**20, "MB,", sum(1 for p in O.rglob("*") if p.is_file()), "files")


if __name__ == "__main__":
    main()
