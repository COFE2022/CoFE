#!/bin/bash
set -e
echo original parameters=[$@]


K=
N=
DATASET_CONFIG_NAME=



TASK_NAME=
DATASET_NAME=
CKPTOUT=
TEXT_COLUMN=
SUMMARY_COLUMN=
VAL_LEN=
SPLIT=

GPUS=4
NODES=1
MODEL_NAME_OR_PATH=
TOKENIZER_NAME_OR_PATH=facebook/bart-large
NUM_BEAMS=6
EVAL_BATCH=20
SRC_LEN=1024
TGT_LEN=256
KEEP_VAL=100

LABEL_SMOOTHING=0.1
CLIP_NORM=0.1
NUM_WARMUP_STEPS=500
LR_SCHEDULER_TYPE=polynomial
WEIGHT_DECAY=0.01
VAL_INTERVAL=0.25
PATIENCE=4

ARGS=`getopt -o :g:k:n: --long task_name:,gpus:,model_name_or_path:,tokenizer_name_or_path:,nodes:,\
dataset_name:,dataset_config_name:,data_dir:,CKPTOUT:,\
num_beams:,eval_batch:,src_len:,tgt_len:,val_len:,\
validation_file:,split:,\
text_column:,summary_column:,\
fold:,split_num:,keep_val:,save_name:\
   -n "$0" -- "$@"`
if [ $? != 0 ]; then
    echo "Terminating..."
    exit 1
fi

echo ARGS=[$ARGS]
#将规范化后的命令行参数分配至位置参数（$1,$2,...)
eval set -- "${ARGS}"
echo formatted parameters=[$@]

while true
do
    case "$1" in
#        -a|--along)
#            echo "Option a";
#            shift

        --task_name)
            echo "Option task_name, argument $2";
            TASK_NAME=$2
            shift 2
            ;;
#            ;;
        -g|--gpus)
            echo "Option gpus, argument $2";
            GPUS="$2"
            shift 2
            ;;

        --model_name_or_path)
            echo "Option MODEL_NAME_OR_PATH, argument $2";
            MODEL_NAME_OR_PATH=$2
            shift 2
            ;;
        --tokenizer_name_or_path)
            echo "Option TOKENIZER_NAME_OR_PATH, argument $2";
            TOKENIZER_NAME_OR_PATH=$2
            shift 2
            ;;


        --dataset_name)
            echo "Option dataset_name, argument $2";
            DATASET_NAME=$2
            shift 2
            ;;

        --dataset_config_name)
            echo "Option dataset_config_name, argument $2";
            DATASET_CONFIG_NAME=$2
            shift 2
            ;;

        --data_dir)
            echo "Option data_dir, argument $2";
            DATA_DIR=$2
            shift 2
            ;;

        --split)
            echo "Option SPLIT, argument $2";
            SPLIT=$2
            shift 2
            ;;



        --ckptout)
            echo "Option ckptout, argument $2";
            CKPTOUT=$2
            shift 2
            ;;
        --num_beams)
            echo "Option num_beams, argument $2";
            NUM_BEAMS=$2
            shift 2
            ;;

        --train_batch)
            echo "Option train_batch, argument $2";
            TRAIN_BATCH=$2
            shift 2
            ;;

        --eval_batch)
            echo "Option eval_batch, argument $2";
            EVAL_BATCH=$2
            shift 2
            ;;

        --src_len)
            echo "Option src_len, argument $2";
            SRC_LEN=$2
            shift 2
            ;;

        --tgt_len)
            echo "Option tgt_len, argument $2";
            TGT_LEN=$2
            shift 2
            ;;

        --val_len)
            echo "Option val_len, argument $2";
            VAL_LEN=$2
            shift 2
            ;;

        --text_column)
            echo "Option TEXT_COLUMN, argument $2";
            TEXT_COLUMN=$2
            shift 2
            ;;
        --summary_column)
            echo "Option SUMMARY_COLUMN, argument $2";
            SUMMARY_COLUMN=$2
            shift 2
            ;;
        --label_smoothing)
            echo "Option label_smoothing, argument $2";
            LABEL_SMOOTHING=$2
            shift 2
            ;;

        --clip_norm)
            echo "Option clip_norm, argument $2";
            CLIP_NORM=$2
            shift 2
            ;;


        --num_warmup_steps)
            echo "Option num_warmup_steps, argument $2";
            NUM_WARMUP_STEPS=$2
            shift 2
            ;;


        --lr_scheduler_type)
            echo "Option lr_scheduler_type, argument $2";
            LR_SCHEDULER_TYPE=$2
            shift 2
            ;;


        --weight_decay)
            echo "Option weight_decay, argument $2";
            WEIGHT_DECAY=$2
            shift 2
            ;;



        --val_interval)
            echo "Option val_interval, argument $2";
            VAL_INTERVAL=$2
            shift 2
            ;;


        --patience)
            echo "Option patience, argument $2";
            PATIENCE=$2
            shift 2
            ;;

        --nodes)
            echo "Option nodes, argument $2";
            NODES=$2
            shift 2
            ;;


        --keep_val)
            echo "Option keep_val, argument $2";
            KEEP_VAL=$2
            shift 2
            ;;



        -k|--fold)
            case "$2" in
                "")
                    echo "Option fold, no argument";
                    shift 2
                    ;;
                *)
                    echo "Option fold, argument $2";
                    K=$2
                    shift 2;
                    ;;
            esac
            ;;

        -n|--split_num)
            case "$2" in
                "")
                    echo "Option csplit_num, no argument";
                    shift 2
                    ;;
                *)
                    echo "Option split_num, argument $2";
                    N=$2
                    shift 2;
                    ;;
            esac
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done




#处理剩余的参数
echo remaining parameters=[$@]
echo \$1=[$1]
echo \$2=[$2]

if [ -z "$TASK_NAME" ]; then
    echo TASK NAME can not be empty
    exit 1
fi

#echo "=======================fake doc postprocessing============================"
FAKE_DOC_CSV="${TASK_NAME}_fake_doc_"$K"-"$N".csv"

##generate fake sum
echo "=======================generate fake sum============================"
TASK2="${TASK_NAME}s2s"
TASK2="$TASK2"_"$K"_"$N"
OUTDIR2=$SCRATCH/exp_out/$TASK2

sbr2="$(python get_best_model_path.py --root "${OUTDIR2}" )"

echo $sbr2
MODEL2=$sbr2

MODEL_CACHE2=$OUTDIR2/MODEL
echo "$MODEL_CACHE2"
mkdir -p "$MODEL_CACHE2"
#
JSON_FILE2="${TASK_NAME}_btrans_fake_sum_"$K"-"$N".txt"
DATA_OUT="$PWD""/${TASK_NAME}_data"
mkdir -p "$DATA_OUT"
echo $JSON_FILE2
GENERATE_TASK_NAME2="GEN"$TASK2""
echo $GENERATE_TASK_NAME2

if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "only run 1 time extract_model_s2s"
    python extract_model_s2s.py --model_name_or_path $MODEL2 --output_dir $MODEL_CACHE2
fi

GENERATE_TASK_NAME2="GEN"$TASK2""
echo $GENERATE_TASK_NAME2
#
python decode_s2s.py \
--model_name_or_path \
$MODEL_CACHE2 \
--validation_file $DATA_OUT/$FAKE_DOC_CSV \
--split validation \
--gpus $GPUS \
--num_nodes $NODES \
--num_beams $NUM_BEAMS \
--per_device_eval_batch_size \
$EVAL_BATCH \
--max_source_length $SRC_LEN \
--max_target_length $TGT_LEN  \
--text_column $TEXT_COLUMN \
--summary_column $SUMMARY_COLUMN \
--task_name \
$GENERATE_TASK_NAME2 \
--fold $K \
--split_num $N \
--num_nodes $NODES \
--save_name $DATA_OUT/$JSON_FILE2
#
#
#
#
#post processing of fake sum's json, data aligment
#echo "=======================post processing of fake sum's json, data aligment============================"
#
fake_sum_csv="${TASK_NAME}_fake_sum_"$K"-"$N".csv"
echo $fake_sum_csv


if [ "$SLURM_PROCID" -eq 0 ]; then
echo "only run 1 time fakesum_postprocessing"
python fakesum_postprocessing.py \
--dataset_name \
$DATASET_NAME \
--dataset_config_name \
$DATASET_CONFIG_NAME \
--data_dir \
$DATA_DIR \
--split $SPLIT \
--text_column $TEXT_COLUMN \
--summary_column $SUMMARY_COLUMN \
--generated_sum $DATA_OUT/$JSON_FILE2 \
--output  $DATA_OUT/$fake_sum_csv

echo "saved as $DATA_OUT/$fake_sum_csv"

fi



#cd data
#echo "$(head -1 *csv)"
#cd ..
#
#
#
#


