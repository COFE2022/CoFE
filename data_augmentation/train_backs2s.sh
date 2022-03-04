#!/bin/bash
set -e

while getopts ':g:k:n:' args;do
case $args in
    g)
        echo 'num of gpus' $OPTARG
        GPUS=$OPTARG
        ;;
    k)
        echo 'num of fold' $OPTARG
        K=$OPTARG
        ;;
    n)
        echo 'num of split_num' $OPTARG
        N=$OPTARG
        ;;
    ?)
        echo error
        exit 1
    ;;
esac
done

echo $GPUS $K $N



MODEL=facebook/bart-large
TASK=WIKI_BACK
TASK="$TASK"_"$K"_"$N"
OUTDIR=$SCRATCH/exp_out/$TASK
DATA_DIR="$SCRATCH"/datas/wikihow

rm -rf "${OUTDIR}*"

echo $TASK


python s2s.py --model_name_or_path \
$MODEL \
--dataset_name \
wikihow \
--dataset_config_name \
sep \
--data_dir \
$DATA_DIR \
--output_dir \
$OUTDIR \
--gpus $GPUS \
--num_beams 6 \
--per_device_train_batch_size \
12 \
--per_device_eval_batch_size \
24 \
--max_source_length 200 \
--max_target_length 562 \
--val_max_target_length 562 \
--text_column headline \
--summary_column text \
--label_smoothing \
0.1 \
--clip_norm \
0.1 \
--num_warmup_steps \
500 \
--lr_scheduler_type \
polynomial \
--weight_decay \
0.01 \
--val_interval 0.5 \
--patience 4 \
--task_name \
$TASK \
--fold "$K" \
--split_num "$N"
