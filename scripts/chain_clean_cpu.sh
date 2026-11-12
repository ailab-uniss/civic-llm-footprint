#!/bin/bash
# CPU side of the clean rerun: waits for the outputs of scripts/chain_clean_gpu.sh and runs the study, the quantifier
# benchmark and the external tests on them.
cd "$(dirname "$0")/.."
source ~/venvs/civic/bin/activate
GEN=data/generated/generated_processed.parquet
S=results/supervised
SC=$S/study_scores_aug.parquet
W=configs/reference_weights_vendor.yaml
python scripts/processed_stats.py > logs/processed_stats.log 2>&1
python src/civic/templates.py --processed data/processed/*.parquet --out results/templates.parquet > logs/templates.log 2>&1
while ! grep -q STUDY_SCORES_DONE logs/clean_gpu.log; do sleep 120; done
echo "=== study $(date +%H:%M)"
extra="--campaigns results/campaigns.parquet --templates results/templates.parquet"
[ -s results/embeddings.parquet ] && extra="$extra --emb results/embeddings.parquet"
python src/civic/analysis.py --processed data/processed/*.parquet --gen $GEN $extra --scores $SC --weights $W \
    --out results/analysis_cls5 2>&1 | tail -3 &
python src/civic/consultations.py --processed data/processed/*.parquet --gen $GEN --scores $SC --templates results/templates.parquet \
    --weights $W --out results/analysis_cls5 2>&1 | tail -2 &
python src/civic/analysis.py --processed data/processed/*.parquet --gen $GEN $extra --scores $SC --n-boot 0 \
    --out results/analysis_cls5_equal 2>&1 | tail -2 &
python scripts/intensity_scores.py $SC > logs/intensity_clean.log 2>&1 &
python scripts/multitude_scores.py $SC > logs/multitude_clean.log 2>&1 &
wait
python scripts/study_extras.py $SC results/analysis_cls5 $W > logs/extras5.log 2>&1
echo "STUDY5_DONE $(date +%H:%M)"
while ! grep -q BENCH_SCORES_DONE logs/clean_gpu.log; do sleep 120; done
echo "=== quantifiers $(date +%H:%M)"
python scripts/quantify_scores.py $S/scores_aug.parquet 16 > logs/quant_aug_clean.log 2>&1
python scripts/calibrate_scores.py $S/quant_aug.parquet hdyv_inst > logs/calibrate_clean.log 2>&1
echo "QUANT5_DONE $(date +%H:%M)"
while ! grep -q HOLD_DONE logs/clean_gpu.log; do sleep 120; done
python scripts/quantify_scores.py $S/scores_hold.parquet 12 > logs/quant_hold_clean.log 2>&1
echo "CPU_CHAIN_DONE $(date +%H:%M)"
