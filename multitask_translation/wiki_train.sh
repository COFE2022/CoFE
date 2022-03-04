set -e
export BART_PATH=/scratch/tw2112/codes/models/bart.large/model.pt
#BART_PATH=/scratch/tw2112/codes/ablation/wiki_weight/ckpt_nli/checkpoint1.pt
DATA=/scratch/tw2112/codes/ablation/wiki_weight



TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
END_LR=6e-6
MAX_TOKENS=2048
UPDATE_FREQ=4
#LR=1e-05
SAVE_EVERY=2000

SAVE_PATH=$DATA/ckpt_nli

DATA_DIR=$DATA/pos_bin
NEG_DIR=$DATA/neg_bin
NLI_DIR=$DATA/nli_bin

NEG_COEF=1
NLI_COEF=0.8
NLI_SIZE=500000


LOGFILE=log/log_wiki_nli.txt

#fairseq-train $DATA_DIR \
#  --negative-data $NEG_DIR \
#   --lambda-neg-config 0.44 \
#    --restore-file $BART_PATH \
#    --save-dir $SAVE_PATH \
#    --max-tokens $MAX_TOKENS \
#    --task nli_with_neg \
#    --source-lang source --target-lang target \
#    --truncate-source \
#    --layernorm-embedding \
#    --share-all-embeddings \
#    --share-decoder-input-output-embed \
#    --reset-optimizer --reset-dataloader --reset-meters \
#    --required-batch-size-multiple 16 \
#    --arch bart_large \
#    --criterion label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --dropout 0.1 --attention-dropout 0.1 \
#    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#    --clip-norm 0.1 \
#    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES  --end-learning-rate $END_LR \
#    --fp16 --update-freq $UPDATE_FREQ \
#    --skip-invalid-size-inputs-valid-test \
#    --find-unused-parameters \
#    --log-file $LOGFILE \
#    --user-dir ./ ;

fairseq-train $DATA_DIR \
    --negative-data $NEG_DIR \
    --lambda-neg-config $NEG_COEF \
    --nli-data $NLI_DIR \
    --nli-size  $NLI_SIZE \
    --lambda-nli-config $NLI_COEF \
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
    --log-interval 50 \
    --save-interval-updates $SAVE_EVERY \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy_noflatten  \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES --end-learning-rate $END_LR \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --log-file $LOGFILE \
    --user-dir ./ \
    --batch-size 20 ;