#!/bin/bash
# Review 9: is the classifier as accurate when trained on the commercial generators only (no open-weight models)?
cd "$(dirname "$0")/.."
source ~/venvs/civic/bin/activate
XLMR=models/xlm-roberta-large
GEN=data/generated/generated_processed.parquet
S=results/supervised
COMM="gpt-5.4-mini gemini-3.6-flash claude-haiku-5-5 gpt-4o gpt-4o-mini"
echo "=== forward benchmark, commercial-only training $(date +%H:%M)"
python src/civic/supervised_scores.py --processed data/processed/*.parquet --gen $GEN --model $XLMR \
    --out $S/scores_comm.parquet --augment --protocols period --heldout $COMM --train-only $COMM 2>&1 | grep -E "^\[|^done|Error|Traceback"
python scripts/quantify_scores.py $S/scores_comm.parquet 8 > logs/quant_comm.log 2>&1
echo "=== fold 0 classifier, commercial-only training $(date +%H:%M)"
python src/civic/study_scores.py --processed data/processed/*.parquet --gen $GEN --model $XLMR \
    --out $S/study_scores_comm_f0.parquet --augment --folds 0 --train-only $COMM \
    --extra data/external/raid/raid_texts.parquet --extra-suffix .scores_comm 2>&1 | grep -E "^fold|^scored|^done|Error|Traceback"
python -c "
import pandas as pd
d=pd.read_parquet('$S/study_scores_aug.parquet'); d[d.fold==0].to_parquet('$S/study_scores_aug_f0cmp.parquet',index=False)"
python scripts/multitude_scores.py $S/study_scores_comm_f0.parquet > logs/multitude_comm.log 2>&1
python scripts/multitude_scores.py $S/study_scores_aug_f0cmp.parquet > logs/multitude_f0cmp.log 2>&1
echo "REVIEW9_DONE $(date +%H:%M)"
