set -e
export BART_PATH=/scratch/tw2112/codes/models/bart.large/model.pt
#export BART_PATH=/scratch/tw2112/codes/ablation/xsum/ckpt_ablation2/checkpoint_best.pt
DATA=/scratch/tw2112/codes/ablation/xsum
TOTAL_NUM_UPDATES=40000
WARMUP_UPDATES=500
LR=3e-05
#TOTAL_NUM_UPDATES=10000
#WARMUP_UPDATES=500
#LR=1e-05


MAX_TOKENS=2048
UPDATE_FREQ=4
SAVE_PATH=$DATA/ckpt_ablation4

DATA_DIR=$DATA/pos_binarized
NEG_DIR=$DATA/neg_binarized

LOGFILE=log/log.txt

fairseq-train $DATA_DIR \
    --negative-data $NEG_DIR \
    --lambda-neg-config 0.5 \
    --restore-file $BART_PATH \
    --save-dir $SAVE_PATH \
    --max-tokens $MAX_TOKENS \
    --task nli_with_neg \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --log-file $LOGFILE \
    --user-dir ./ ;