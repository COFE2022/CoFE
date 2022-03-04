#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --output=./log/%j_%x.out
#SBATCH --error=./log/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --account=cds
#SBATCH -c 4
#SBATCH --mail-user=tw2112@nyu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL




srun singularity exec --nv  --overlay $SCRATCH/overlay2/overlay-50G-10M.ext3:ro   /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
cd
source /ext3/env.sh
conda activate s2s
cd /home/tw2112/codes/s2s/src


sh getargs_genfakedoc.sh --task giga --dataset_name gigaword -k ${FOLD} -n $((${SPLIT})) -g ${GPUS} --nodes $NODES --split train --src_len 256 --tgt_len 1024 --text_column summary --summary_column document -k
"

srun singularity exec --nv  --overlay $SCRATCH/overlay2/overlay-50G-10M.ext3:ro   /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
cd
source /ext3/env.sh
conda activate s2s
cd /home/tw2112/codes/s2s/src


sh getargs_genfakesum.sh --task giga --dataset_name gigaword -k ${FOLD} -n $((${SPLIT})) -g ${GPUS} --nodes $NODES --split train --text_column document --summary_column summary --eval_batch 50
"

