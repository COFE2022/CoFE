#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --output=./log/%j_%x.out
#SBATCH --error=./log/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --account=cds
#SBATCH -c 4
#SBATCH --mail-user=tw2112@nyu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL



srun singularity exec --nv  --overlay $SCRATCH/overlay2/overlay-50G-10M.ext3:ro   /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
cd
source /ext3/env.sh
conda activate
cd ~/codes/s2s/src


sh getargs.sh -g "${GPUS}" -k "${FOLD}" -n "$((${SPLIT}))" --nodes "$NODES" --dataset_name xsum --task_name xsums2s --train_batch 6 --eval_batch 12
"
