#!/bin/bash
set -e

echo original parameters=[$@]


ARGS=`getopt -o g:k:c:: --long name:,nodes: -n "$0" -- "$@"`
if [ $? != 0 ]; then
    echo "Terminating..."
    exit 1
fi

echo ARGS=[$ARGS]
#将规范化后的命令行参数分配至位置参数（$1,$2,...)
eval set -- "${ARGS}"
echo formatted parameters=[$@]


GPUS=4
NODES=2
K=5
NAME=RUN
while true
do
    case "$1" in
        -g)
            echo "Option g argument $2";
            GPUS=$2
            shift 2
            ;;

        --nodes)
            echo "Option nodes, argument $2";
            NODES=$2
            shift 2
            ;;

        -k)
            echo "Option K, argument $2";
            K=$2
            shift 2
            ;;
        --name)
            echo "Option name, argument $2";
            NAME=$2
            shift 2
            ;;
#        -c|--clong)
#            case "$2" in
#                "")
#                    echo "Option c, no argument";
#                    shift 2
#                    ;;
#                *)
#                    echo "Option c, argument $2";
#                    shift 2;
#                    ;;
#            esac
#            ;;
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

K=$K
for i in $( seq 1 $(($K-1 )))
do
    SPLIT=$i
    echo "$K"
    echo "$GPUS"
    echo "$SPLIT"
    sbr1="$(sbatch --job-name=${NAME}${i} --nodes=$NODES --gres=gpu:"$GPUS"  --ntasks-per-node="$GPUS" --export=ALL,FOLD="$K",GPUS="$GPUS",SPLIT="$i",NODES=$NODES \
     train_s2s.slurm )"
    #
    if [[ "$sbr1" =~ Submitted\ batch\ job\ ([0-9]+) ]];
    then
      jobid1=${BASH_REMATCH[1]}
      echo "${BASH_REMATCH[1]}"
    else
      echo "sbatch failed"
      exit 1
    fi



    sbr2="$(sbatch --job-name=${NAME}BACK${i} --nodes=$NODES --gres=gpu:"$GPUS"  --ntasks-per-node="$GPUS" --export=ALL,FOLD="$K",GPUS="$GPUS",SPLIT="$i",NODES=$NODES \
     train_backs2s.slurm )"

    if [[ "$sbr2" =~ Submitted\ batch\ job\ ([0-9]+) ]];
    then
      jobid2=${BASH_REMATCH[1]}
      echo "${BASH_REMATCH[1]}"
    else
      echo "sbatch failed"
      exit 1
    fi




done

