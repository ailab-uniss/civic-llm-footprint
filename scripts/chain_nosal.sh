#!/bin/bash
# Is Salamandra 7B useful in the training set of the classifier? Forward benchmark with every other generator held out
# in turn and Salamandra never in training, compared with the full training (results/supervised/quant_aug.parquet).
cd "$(dirname "$0")/.."
source ~/venvs/civic/bin/activate
XLMR=models/xlm-roberta-large
GEN=data/generated/generated_processed.parquet
S=results/supervised
ALL="gpt-5.4-mini gemini-3.6-flash claude-haiku-5-5 gpt-4o gpt-4o-mini gpt-oss-20b mistral-small-24b gemma-2-9b-it llama-3.1-8b eurollm-9b qwen2.5-7b"
echo "=== forward benchmark, training without Salamandra $(date +%H:%M)"
python src/civic/supervised_scores.py --processed data/processed/*.parquet --gen $GEN --model $XLMR --out $S/scores_nosal.parquet --augment --protocols period --heldout $ALL --train-only $ALL 2>&1 | grep -E "^\[|^done|Error|Traceback"
python scripts/quantify_scores.py $S/scores_nosal.parquet 8 > logs/quant_nosal.log 2>&1
echo "=== fold 0 classifier without Salamandra $(date +%H:%M)"
python src/civic/study_scores.py --processed data/processed/*.parquet --gen $GEN --model $XLMR --out $S/study_scores_nosal_f0.parquet --augment --folds 0 --train-only $ALL --extra data/external/raid/raid_texts.parquet --extra-suffix .scores_nosal 2>&1 | grep -E "^fold|^scored|^done|Error|Traceback"
python scripts/multitude_scores.py $S/study_scores_nosal_f0.parquet > logs/multitude_nosal.log 2>&1
echo "NOSAL_DONE $(date +%H:%M)"
