set -e
BEAM_SIZE=6
MAX_LEN_B=128
MIN_LEN=10
LEN_PEN=1.0
DATA=/scratch/tw2112/codes/ablation/wiki_weight

DATA_PATH=$DATA/pos_bin
MODEL_PATH=$DATA/ckpt_nli/checkpoint_best.pt
RESULT_PATH=./outdir/wiki_weight_nli


fairseq-generate $DATA_PATH \
    --path $MODEL_PATH --results-path $RESULT_PATH \
    --task translation \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
    --no-repeat-ngram-size 3 \
    --batch-size 32 --fp16 \
    --truncate-source --gen-subset test;


python  convert_bart_result.py --generate-dir $RESULT_PATH