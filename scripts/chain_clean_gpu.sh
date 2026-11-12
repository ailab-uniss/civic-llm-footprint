#!/bin/bash
# Clean rerun of the classifier pipeline after scripts/fix_data.py (wrong-language texts removed, seed identifiers re-keyed):
# cross-fitted study scores, final classifier with the RAID scores, benchmark scores, held-out mode and prompt variant.
cd "$(dirname "$0")/.."
source ~/venvs/civic/bin/activate
XLMR=models/xlm-roberta-large
GEN=data/generated/generated_processed.parquet
S=results/supervised
echo "=== cross-fitted scores $(date +%H:%M)"
python src/civic/study_scores.py --processed data/processed/*.parquet --gen $GEN --model $XLMR \
    --out $S/study_scores_aug.parquet --augment 2>&1 | grep -E "^fold|^done|^documents|Error|Traceback"
echo "STUDY_SCORES_DONE $(date +%H:%M)"
echo "=== benchmark scores $(date +%H:%M)"
python src/civic/supervised_scores.py --processed data/processed/*.parquet --gen $GEN --model $XLMR \
    --out $S/scores_aug.parquet --augment \
    --groups openai=gpt-4o,gpt-4o-mini,gpt-5.4-mini,gpt-oss-20b 2>&1 | grep -E "^\[|^done|Error|Traceback"
echo "BENCH_SCORES_DONE $(date +%H:%M)"
echo "=== held-out mode and prompt variant $(date +%H:%M)"
python src/civic/supervised_scores.py --processed data/processed/*.parquet --gen $GEN --model $XLMR \
    --out $S/scores_hold.parquet --augment --heldout --protocols hash \
    --hold mode=rewrite mode=generate mode=translate variant=0 variant=1 variant=2 2>&1 | grep -E "^\[|^done|Error|Traceback"
echo "HOLD_DONE $(date +%H:%M)"
echo "=== final classifier $(date +%H:%M)"
python src/civic/study_scores.py --processed data/processed/*.parquet --gen $GEN --model $XLMR \
    --out $S/_unused.parquet --augment --folds -1 --save-model models/xlmr_civic \
    --extra data/external/raid/raid_texts.parquet 2>&1 | grep -E "^fold|^scored|Error|Traceback"
echo "GPU_CHAIN_DONE $(date +%H:%M)"
